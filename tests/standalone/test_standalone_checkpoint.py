# SPDX-License-Identifier: Apache-2.0
"""Standalone checkpoint dispatch, data retention and P/D isolation on CPU."""

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from test_local_checkpoint_restore import adapter, plan, implementation
from test_checkpoint_writeback import (
    deployment as checkpoint_deployment,
    spill,
    assert_rows,
)
from test_preemption_checkpoint import api as checkpoint_api

api = checkpoint_api
deployment = checkpoint_deployment
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("ready", [True, False])
def test_standalone_indexer_waits_for_normalization_then_loads_both_prompt_and_tail(
    ready,
):
    calls = []
    obj, p = adapter(calls), plan()
    obj.config.pd_role = None
    obj.lmcache_engine.load_group1_pages_direct = lambda *a: pytest.fail(
        "direct-HBM reader"
    )
    obj.lmcache_engine.finish_checkpoint_prefix = lambda *a: pytest.fail(
        "direct-prefix collective"
    )
    entered = Event()

    class Gate(Future):
        def result(self, *args, **kwargs):
            entered.set()
            return super().result(*args, **kwargs)

    gate = p["latent_shared_ready"] = Gate()
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(obj._run_dsa_cold_indexer_load, p, None)
        try:
            assert entered.wait(2)
            assert not calls and not job.done()
        finally:
            if ready:
                gate.set_result(None)
            else:
                gate.set_exception(ValueError("normalization failed"))
        if ready:
            result = job.result(2)
            assert calls == [("cpu_tail", 11, [True] * 11)]
            assert result[0].all() and result[1] == "completed-cpu-fence"
        else:
            with pytest.raises(ValueError, match="normalization failed"):
                job.result(2)
            assert not calls


@pytest.mark.parametrize(
    "mode", ["p2p_preferred", "persistent_serial", "persistent_parallel_prefetch"]
)
def test_standalone_actual_indexer_loader_preserves_full_mask_and_device_fence(mode):
    """Run both adapter implementations; mock only transfer/state collaborators."""
    calls, p = [], plan()
    p["planned_at"] = 0
    ready, waiting = Future(), Event()
    objects, tensors = [], []
    state = NS(
        cached_memory_objs_indexer=objects,
        cached_tensors_indexer=tensors,
        cache_kwargs=lambda *a, **kw: {"cached_memory_objs": objects},
        clear_group=lambda group: objects.clear(),
    )
    base = implementation(
        "../LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py",
        "LMCacheConnectorV1Impl",
        {"_run_dsa_cold_indexer_load"},
        object,
        serving_perf_enabled=lambda: False,
        WorkerRetrieveState=lambda **kw: state,
    )
    cls = implementation(
        "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        {"_run_dsa_cold_indexer_load"},
        base,
    )

    def retrieve(tokens, mask, **kw):
        assert p["latent_shared_ready"].done()
        assert tokens == list(range(11)) and mask.all()
        assert kw["kv_group"] == 1
        assert torch.equal(kw["slot_mapping"], p["indexer_slots_cpu"])
        calls.append("retrieve")
        kw["cached_memory_objs"].append([object()])
        kw["_dense_load_readiness_out"].append(ready)
        yield None
        yield None
        yield mask

    def fence(event):
        assert event is ready
        waiting.set()
        event.result(timeout=5)

    obj = cls()
    obj.config = NS(pd_role=None, dsa_group1_load_mode=mode)
    obj.lmcache_engine = NS(
        gpu_connector=NS(stage_dense_load_tensor=lambda tensor, **kw: tensor),
        supports_dense_sparse_cache_retention=lambda: True,
        retrieve_layer=retrieve,
        prefetch_shared_layer_pages=lambda *a, **kw: calls.append("prefetch"),
    )
    obj._dense_load_source_owners = lambda state: ()
    obj._release_dense_load_source_owners = lambda *a, **kw: None
    obj._synchronize_dsa_cold_dense_readiness = fence
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(obj._run_dsa_cold_indexer_load, p, None)
        try:
            assert waiting.wait(3)
            assert not job.done()
        finally:
            ready.set_result(None)
        result = job.result(3)
    assert result[0] is p["token_mask"] and result[1] is ready
    assert calls == (
        ["prefetch", "retrieve"]
        if mode == "persistent_parallel_prefetch"
        else ["retrieve"]
    )


@pytest.mark.parametrize("evict", [False, True])
def test_standalone_writeback_reuses_local_and_spilled_paired_data(deployment, evict):
    engine, create, _ = deployment
    engine.config.pd_role = None
    engine.config.enable_remote_lmcache_store = False
    engine.config.dsa_group1_load_mode = "p2p_preferred"
    worker = create()
    if evict:
        spill(worker, engine)
        for key in tuple(engine.backend.pages):
            assert engine.backend.evict(key)
    assert worker.local.available("r", 1, 11) == 11
    assert not engine.gets  # Lookup must not fetch payloads.
    _, owners = worker.local.normalize("r", 1, list(range(11)), None)
    manifest = worker.local.manifest("r", 1)
    end, groups = worker.local.acquire(manifest, 11)
    try:
        assert end == 11
        assert_rows(groups)
        assert bool(engine.gets) == evict
    finally:
        worker.local.release(groups)
        for owner in owners:
            owner.ref_count_down()


@pytest.mark.parametrize(
    "role,checkpoint,remote,expected",
    [
        (None, True, False, True),
        (None, False, False, False),
        ("sender", True, False, False),
        ("receiver", True, False, False),
        ("sender", False, True, True),
        ("receiver", False, True, True),
    ],
)
def test_paired_lookup_activation_does_not_enable_remote_fill(
    role, checkpoint, remote, expected
):
    cls = implementation(
        "../LMCache-NPU/lmcache/v1/cache_engine.py",
        "LMCacheEngine",
        {"_remote_fill_pair_lookup_enabled"},
        object,
        mooncake_layer_pages_enabled=lambda config: True,
    )
    obj = cls()
    obj.config = NS(
        pd_role=role,
        decode_preemption_checkpoint=checkpoint,
        enable_remote_lmcache_store=remote,
        dsa_two_groups=True,
    )
    obj.use_layerwise = True
    obj._should_use_shared_layerwise_retrieve = lambda group: True
    assert obj._remote_fill_pair_lookup_enabled() == expected
    assert obj.config.enable_remote_lmcache_store == remote
    assert obj.config.pd_role == role


def test_standalone_keeps_full_prefill_memory_budget():
    cls = implementation(
        "lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py",
        "LMCacheAscendConnectorV1Dynamic",
        {"_compact_policy_from_config"},
        object,
    )
    config = NS(
        pd_role=None,
        dsa_group1_load_mode="p2p_preferred",
        enable_dsa_cold_compact_load=True,
        decode_preemption_checkpoint=True,
        enable_shared_cpu_cache=True,
        extra_config={},
    )
    vllm = NS(kv_transfer_config=NS(kv_role="kv_both", kv_load_failure_policy="fail"))
    assert cls._compact_policy_from_config(vllm, config) is None


def test_standalone_direct_prefill_store_role_gate_remains_open():
    source = ast.parse(
        (ROOT / "lmcache_ascend/v1/cache_engine.py").read_text(encoding="utf-8")
    )
    expression = next(
        n.value
        for n in ast.walk(source)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_direct_store_enabled"
            for t in n.targets
        )
    )
    for role, expected in ((None, True), ("sender", True), ("receiver", False)):
        extra = {"mooncake_direct_npu_prefill_store": True, "use_ascend_direct": True}
        config = NS(
            pd_role=role,
            enable_remote_lmcache_store=False,
            get_extra_config_value=extra.get,
        )
        obj = NS(
            config=config,
            is_store_async=True,
            _force_layerwise_prefill_store=False,
            _is_passive=lambda: False,
        )
        assert (
            eval(
                compile(ast.Expression(expression), "direct_store", "eval"),
                {"self": obj, "mooncake_layer_pages_enabled": lambda c: True},
            )
            == expected
        )
