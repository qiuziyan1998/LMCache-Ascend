# SPDX-License-Identifier: Apache-2.0
"""
P3: Multi-process scheduler/worker integration tests.

Worker: LMCacheEngine + lookup server + connector lifecycle (pins, retrieve state).
Scheduler: LMCacheLookupClient over ZMQ (same split as production).

Run with:
    PYTHONHASHSEED=0 pytest tests/integration/vllm/test_scheduler_worker_multiprocess_consistency.py
"""

# Standard
import os
import sys

# Third Party
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("lmcache")

from scheduler_worker_multiprocess_harness import (
    _make_sparse_req_meta,
    start_multiprocess_harness,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PYTHONHASHSEED") is None,
    reason=(
        "PYTHONHASHSEED must be set for consistent hashing between "
        "scheduler lookup client and worker lookup server."
    ),
)


@pytest.fixture
def mp_harness():
    harness = start_multiprocess_harness()
    try:
        yield harness
    finally:
        harness.shutdown()


class TestSchedulerWorkerLookupPinSplit:
    """Lookup pins live only on the worker; scheduler uses ZMQ lookup client."""

    def test_scheduler_lookup_pins_on_worker_not_scheduler(
        self, mp_harness
    ) -> None:
        req_id = "mp-req-lookup-pin"
        hits = mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        assert hits["hits"] == 256

        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is True


class TestSparseDecodeDeferredUnpinAcrossProcesses:
    """Sparse decode keeps lookup pins across decode steps on the worker."""

    def test_pins_survive_multiple_sparse_decode_wait_for_save_steps(
        self, mp_harness
    ) -> None:
        req_id = "mp-req-defer-unpin"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        mp_harness.worker_call({"op": "save_worker_state", "req_id": req_id})

        sparse_meta = _make_sparse_req_meta(req_id, can_load=True)
        for _ in range(3):
            mp_harness.worker_call(
                {"op": "wait_for_save", "requests": [sparse_meta]}
            )
            pin_state = mp_harness.worker_call(
                {"op": "has_lookup_pins", "req_id": req_id}
            )
            assert pin_state["has_pins"] is True

        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id in keys["keys"]


class TestNonSparseUnpinBoundary:
    """Non-sparse requests unpin on wait_for_save even when pins were created remotely."""

    def test_wait_for_save_unpins_when_not_sparse_decode(self, mp_harness) -> None:
        req_id = "mp-req-non-sparse-unpin"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is True

        normal_meta = _make_sparse_req_meta(
            req_id, can_load=True, is_sparse_decode=False
        )
        mp_harness.worker_call(
            {"op": "wait_for_save", "requests": [normal_meta]}
        )

        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is False


class TestSchedulerWorkerFinishConsistency:
    """When scheduler drops a request, worker prune/finish must release pins."""

    def test_prune_after_scheduler_drops_request_unpins_and_clears_state(
        self, mp_harness
    ) -> None:
        req_id = "mp-req-prune-finish"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        mp_harness.worker_call({"op": "save_worker_state", "req_id": req_id})

        sparse_meta = _make_sparse_req_meta(req_id, can_load=True)
        mp_harness.worker_call(
            {"op": "wait_for_save", "requests": [sparse_meta]}
        )

        # Scheduler no longer includes req_id in active metadata (missed finish).
        mp_harness.worker_call({"op": "prune", "active_req_ids": []})

        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is False
        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id not in keys["keys"]

    def test_request_finished_unpins_when_scheduler_signals_finish(
        self, mp_harness
    ) -> None:
        pytest.importorskip("vllm")
        req_id = "mp-req-request-finished"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        mp_harness.worker_call({"op": "save_worker_state", "req_id": req_id})

        mp_harness.worker_call({"op": "request_finished", "req_id": req_id})

        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is False
        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id not in keys["keys"]


class TestPreemptWorkerConsistency:
    """Worker preempt path must clear state and pins without scheduler lookup."""

    def test_preempt_clears_worker_state_and_lookup_pins(self, mp_harness) -> None:
        req_id = "mp-req-preempt"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        mp_harness.worker_call({"op": "save_worker_state", "req_id": req_id})

        mp_harness.worker_call({"op": "preempt", "req_ids": [req_id]})

        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is False
        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id not in keys["keys"]


class TestZombieMetadataBoundary:
    """Worker state outlives one metadata step; next empty active set prunes it."""

    def test_zombie_worker_state_pruned_when_metadata_has_no_active_requests(
        self, mp_harness
    ) -> None:
        req_id = "mp-req-zombie"
        mp_harness.scheduler_call(
            {
                "op": "lookup",
                "token_ids": mp_harness.prompt_tokens,
                "lookup_id": req_id,
            }
        )
        mp_harness.worker_call({"op": "save_worker_state", "req_id": req_id})

        # Step with active sparse decode metadata.
        sparse_meta = _make_sparse_req_meta(req_id, can_load=True)
        mp_harness.worker_call(
            {"op": "wait_for_save", "requests": [sparse_meta]}
        )
        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id in keys["keys"]

        # Next step: scheduler sends metadata with no requests (zombie on worker).
        mp_harness.worker_call({"op": "wait_for_save", "requests": []})
        mp_harness.worker_call({"op": "prune", "active_req_ids": []})

        keys = mp_harness.worker_call({"op": "worker_state_keys"})
        assert req_id not in keys["keys"]
        pin_state = mp_harness.worker_call(
            {"op": "has_lookup_pins", "req_id": req_id}
        )
        assert pin_state["has_pins"] is False
