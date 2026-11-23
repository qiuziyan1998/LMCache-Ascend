# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Selected-KV retirement gating and checkpoint hook restoration."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import gc
import unittest
import weakref

# Third Party
from lmcache.v1.prefill_admission import PrefillAdmissionBudget
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)


def _make_worker_impl(
    rank: int,
    events: list[str],
    unretired_fn: Any,
    *,
    budget: PrefillAdmissionBudget | None = None,
) -> tuple[LMCacheAscendConnectorV1Impl, dict[str, int]]:
    impl = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    impl._prefill_admission = budget
    impl._prefill_admission_enabled = True
    impl._prefill_admission_retired = set()
    impl._prefill_admission_last_report = None
    impl._prefill_admission_report_time = -10.0
    impl._cold_load_coordinator = None
    impl._force_layerwise_prefill_store = False
    impl._layerwise_save_storers = {}
    impl._worker_retrieve_state = {}
    impl._deferred_latent_pending = set()
    impl.selected_kv_source = SimpleNamespace(
        unretired_selected_request_ids=unretired_fn
    )
    impl._vllm_config = SimpleNamespace(  # type: ignore[assignment]
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384)
    )
    impl.config = SimpleNamespace(  # type: ignore[assignment]
        enable_shared_cpu_cache=True,
        get_extra_config_value=lambda _k, default=None: default,
    )
    impl._drain_dense_load_retirements = lambda: events.append(  # type: ignore[method-assign]
        f"r{rank}:drain_dense"
    )

    pool = {"capacity": 100 << 20, "available": 64 << 20}

    def _get_capacity() -> dict[str, int]:
        events.append(f"r{rank}:get_capacity")
        return dict(pool)

    impl.lmcache_engine = SimpleNamespace(  # type: ignore[assignment]
        lookup_unpin=lambda _req_id: None,
        metadata=SimpleNamespace(
            worker_id=rank,
            is_first_rank=lambda r=rank: r == 0,
            chunk_size=1024,
            runtime_kv_group_layer_counts=(79, 22),
            get_shapes=lambda: [torch.Size([1024]), torch.Size([256])],
            get_dtypes=lambda: [torch.bfloat16, torch.uint8],
        ),
        storage_manager=SimpleNamespace(
            local_cpu_backend=SimpleNamespace(
                get_prefill_admission_capacity=_get_capacity,
                get_memory_allocator=lambda: SimpleNamespace(align_bytes=512),
            )
        ),
    )
    return impl, pool


class TestPrefillAdmissionSelectedRetirement(unittest.TestCase):
    def test_4rank_prefill_admission_budget_waits_for_all_ranks_and_pops_on_retire(
        self,
    ) -> None:
        clock = [100.0]
        events: list[str] = []
        active_by_rank: list[set[str]] = [
            set(),
            set(),
            {"req_a"},
            {"req_a"},
        ]
        budget = PrefillAdmissionBudget(world_size=4)
        impls: list[LMCacheAscendConnectorV1Impl] = []
        pools: list[dict[str, int]] = []
        for rank in range(4):

            def _unretired(r: int = rank) -> frozenset[str]:
                events.append(f"r{r}:unretired_selected")
                return frozenset(active_by_rank[r])

            impl, pool = _make_worker_impl(
                rank,
                events,
                _unretired,
                budget=budget if rank == 0 else None,
            )
            impls.append(impl)
            pools.append(pool)

        with patch(
            "lmcache_ascend.integration.vllm.vllm_v1_adapter.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            # Initial poll populates allocator facts across all 4 ranks.
            for impl in impls:
                budget.update(impl._build_prefill_admission_report())
            self.assertTrue(budget.ready)

            # Polling within <0.25s with empty retired set uses cached snapshot
            # deterministically without re-scanning capacity.
            events.clear()
            clock[0] = 100.10
            cached = impls[0]._build_prefill_admission_report()
            self.assertEqual(cached[0]["retired"], ())
            self.assertEqual(cached[0]["pool"], pools[0])
            self.assertEqual(events, [])

            self.assertFalse(impls[0].has_pending_control())
            admitted, err, _ = budget.admit("req_a", 2048)
            self.assertEqual((admitted, err), (True, None))
            self.assertIn("req_a", budget.reservations)
            budget.finish("req_a")
            self.assertIn("req_a", budget.retiring)
            self.assertTrue(impls[0].has_pending_control())

            # Exercise _release_finished_worker_requests directly with
            # _force_layerwise_prefill_store=False and
            # _prefill_admission_enabled=True to populate retired req IDs.
            for impl in impls:
                impl._release_finished_worker_requests(["req_a"])
                self.assertIn("req_a", impl._prefill_admission_retired)

            events.clear()
            clock[0] = 100.50
            reports_step1 = tuple(
                r for impl in impls for r in impl._build_prefill_admission_report()
            )
            budget.update(reports_step1)
            self.assertEqual(
                events[:3],
                ["r0:drain_dense", "r0:unretired_selected", "r0:get_capacity"],
            )
            self.assertEqual(reports_step1[0]["retired"], ("req_a",))
            self.assertEqual(reports_step1[1]["retired"], ("req_a",))
            self.assertEqual(reports_step1[2]["retired"], ())
            self.assertEqual(reports_step1[3]["retired"], ())
            self.assertIn("req_a", budget.reservations)
            self.assertIn("req_a", budget.retiring)
            self.assertEqual(budget.retired_ranks["req_a"], {0, 1})

            # Rank 2 retires -> still held until Rank 3 also retires.
            active_by_rank[2].clear()
            clock[0] = 100.80
            for impl in impls:
                budget.update(impl._build_prefill_admission_report())
            self.assertEqual(budget.retired_ranks["req_a"], {0, 1, 2})
            self.assertIn("req_a", budget.reservations)

            # Rank 3 retires -> all 4 TP ranks confirmed; reservation pops cleanly.
            active_by_rank[3].clear()
            clock[0] = 101.10
            for impl in impls:
                budget.update(impl._build_prefill_admission_report())
            self.assertNotIn("req_a", budget.reservations)
            self.assertNotIn("req_a", budget.retiring)
            self.assertNotIn("req_a", budget.retired_ranks)
            self.assertFalse(impls[0].has_pending_control())

    def test_4rank_quarantined_passive_rank_retains_admission_budget_reservation(
        self,
    ) -> None:
        clock = [200.0]
        events: list[str] = []
        budget = PrefillAdmissionBudget(world_size=4)
        quarantined = [False]
        impls: list[LMCacheAscendConnectorV1Impl] = []
        for rank in range(4):

            def _unretired(r: int = rank) -> frozenset[str]:
                events.append(f"r{r}:unretired")
                if r == 3 and quarantined[0]:
                    # Fail-closed quarantine: unclosed step retains req_q.
                    return frozenset({"req_q"})
                return frozenset()

            impls.append(
                _make_worker_impl(
                    rank,
                    events,
                    _unretired,
                    budget=budget if rank == 0 else None,
                )[0]
            )

        with patch(
            "lmcache_ascend.integration.vllm.vllm_v1_adapter.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            for impl in impls:
                budget.update(impl._build_prefill_admission_report())
            admitted, err, _ = budget.admit("req_q", 4096)
            self.assertEqual((admitted, err), (True, None))
            budget.finish("req_q")

            quarantined[0] = True
            for impl in impls:
                impl._prefill_admission_retired.add("req_q")

            clock[0] = 201.0
            rpts = tuple(
                r for impl in impls for r in impl._build_prefill_admission_report()
            )
            budget.update(rpts)
            self.assertEqual([r["retired"] for r in rpts[:3]], [("req_q",)] * 3)
            self.assertEqual(rpts[3]["retired"], ())
            self.assertEqual(impls[3]._prefill_admission_retired, {"req_q"})
            # Rank 0's /dev/shm reservation MUST remain pinned while Rank 3 is
            # quarantined with an unretired step.
            self.assertIn("req_q", budget.reservations)
            self.assertIn("req_q", budget.retiring)
            self.assertEqual(budget.retired_ranks["req_q"], {0, 1, 2})

    def test_activate_checkpoint_io_chains_and_restores_hooks(self) -> None:
        calls: list[Any] = []

        class _FakeWorker:
            def __init__(self) -> None:
                self.jobs: list[str] = ["job0"]
                self.restore_owners: dict[str, Any] = {}

            def cancel(self, req_id: str, gen: int | None = None) -> None:
                calls.append(("cancel", req_id, gen))

            def release_restore(self, *args: Any) -> None:
                calls.append(("release", args))

            def seal(self, seal: Any) -> None:
                calls.append(("seal", seal))

            def poll(self) -> tuple[str, ...]:
                return ("ckpt-ready",)

        class _FakeEngine:
            def __init__(self, worker: _FakeWorker) -> None:
                self.checkpoint_worker = worker

            def get_finished_stores(self, finished_req_ids: set[str]) -> set[str]:
                calls.append(("cls_stores", sorted(finished_req_ids)))
                return {"cls-done"}

        class _FakeImpl(LMCacheAscendConnectorV1Impl):
            def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
                calls.append(("cls_start", forward_context, kwargs))

            def build_connector_worker_meta(self) -> Any:
                calls.append(("cls_meta",))
                return SimpleNamespace(
                    descriptors={"d": [1]},
                    remote_fill_results={},
                    prefill_admission=(),
                )

        def _make_subject() -> tuple[_FakeImpl, _FakeWorker]:
            worker = _FakeWorker()
            impl = _FakeImpl.__new__(_FakeImpl)
            impl.lmcache_engine = _FakeEngine(worker)  # type: ignore[assignment]
            impl._parent = SimpleNamespace(  # type: ignore[assignment]
                _get_connector_metadata=lambda: SimpleNamespace(
                    preemption_cancels=(("r0", 2),),
                    preemption_releases=(),
                    preemption_seals=(),
                )
            )
            return impl, worker

        # (a) No pre-existing instance hooks: drain deletes wrappers,
        # revealing class methods.
        impl_a, worker_a = _make_subject()
        self.assertNotIn("start_load_kv", impl_a.__dict__)
        impl_a._activate_checkpoint_io()
        self.assertIn("start_load_kv", impl_a.__dict__)
        self.assertIn("build_connector_worker_meta", impl_a.__dict__)
        self.assertIn("get_finished_stores", impl_a.lmcache_engine.__dict__)

        impl_a.start_load_kv("ctx_a", flag=True)
        out_stores_a = impl_a.lmcache_engine.get_finished_stores({"req_1"})
        self.assertEqual(out_stores_a, {"cls-done"})
        self.assertIn(("cls_start", "ctx_a", {"flag": True}), calls)
        self.assertIn(("cls_stores", ["req_1"]), calls)

        worker_a.jobs.clear()
        meta_a = impl_a.build_connector_worker_meta()
        self.assertEqual(meta_a.checkpoint_results, ("ckpt-ready",))
        self.assertNotIn("start_load_kv", impl_a.__dict__)
        self.assertNotIn("build_connector_worker_meta", impl_a.__dict__)
        self.assertNotIn("get_finished_stores", impl_a.lmcache_engine.__dict__)
        self.assertNotIn("_checkpoint_io_originals", impl_a.__dict__)

        # (b) Pre-existing instance hooks installed: chained while active,
        # restored on drain.
        calls.clear()
        impl_b, worker_b = _make_subject()

        def orig_start(fc: Any, **kw: Any) -> None:
            calls.append(("inst_start", fc, kw))

        def orig_meta() -> Any:
            calls.append(("inst_meta",))
            return SimpleNamespace(
                descriptors={"m": [2]},
                remote_fill_results={},
                prefill_admission=(),
            )

        def orig_stores(req_ids: set[str]) -> set[str]:
            calls.append(("inst_stores", sorted(req_ids)))
            return {"inst-done"}

        impl_b.start_load_kv = orig_start  # type: ignore[method-assign]
        impl_b.build_connector_worker_meta = orig_meta  # type: ignore[method-assign]
        impl_b.lmcache_engine.get_finished_stores = orig_stores  # type: ignore[method-assign]

        impl_b._activate_checkpoint_io()
        impl_b.start_load_kv("ctx_b", x=9)
        self.assertEqual(
            impl_b.lmcache_engine.get_finished_stores({"req_2"}),
            {"inst-done"},
        )
        self.assertIn(("inst_start", "ctx_b", {"x": 9}), calls)
        self.assertIn(("inst_stores", ["req_2"]), calls)

        worker_b.jobs.clear()
        meta_b = impl_b.build_connector_worker_meta()
        self.assertEqual(meta_b.checkpoint_results, ("ckpt-ready",))
        self.assertIn(("inst_meta",), calls)
        self.assertIs(impl_b.__dict__.get("start_load_kv"), orig_start)
        self.assertIs(impl_b.__dict__.get("build_connector_worker_meta"), orig_meta)
        self.assertIs(
            impl_b.lmcache_engine.__dict__.get("get_finished_stores"),
            orig_stores,
        )
        self.assertNotIn("_checkpoint_io_originals", impl_b.__dict__)

        # (c) Weakref refcount-only reclamation while _activate_checkpoint_io
        # is still armed and cyclic GC is disabled.
        impl_c, _ = _make_subject()
        impl_c._activate_checkpoint_io()
        self.assertIn("start_load_kv", impl_c.__dict__)
        impl_weak = weakref.ref(impl_c)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            del impl_c
            self.assertIsNone(impl_weak())
        finally:
            if was_enabled:
                gc.enable()


if __name__ == "__main__":
    unittest.main()
