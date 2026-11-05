# SPDX-License-Identifier: Apache-2.0
"""Actual Ascend dispatch and drain implementations, with no device runtime."""

from threading import Condition
from types import SimpleNamespace as NS

import pytest

from test_local_checkpoint_restore import implementation


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_direct_stores_drained_once_before_capture_and_state_release(capture, failure):
    calls, logs = [], []
    engine_type = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"wait_for_pending_stores"},
        object,
    )
    engine = engine_type()

    def direct(ids):
        calls.append("direct")
        if failure:
            raise RuntimeError("transfer failed")
        return set(ids)

    engine.wait_for_direct_stores = direct
    engine.is_store_async = True
    engine._store_cv, engine._pending_store_reqs = Condition(), set()
    engine.checkpoint_worker = NS(capture=lambda *a: calls.append("capture"))
    engine.enable_checkpoint_prefix_agreement = lambda: None
    engine.drop_direct_store_states = lambda ids: calls.append("drop-store")
    adapter_type = implementation(
        "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        {"handle_preemptions"},
        object,
        logger=NS(debug=lambda *a: None, info=lambda *a: logs.append(a)),
    )
    obj = adapter_type()
    meta = NS(
        preemption_captures=(NS(req_id="r"),) if capture else (), preemption_cancels=()
    )
    obj.lmcache_engine = engine
    obj._parent = NS(
        has_connector_metadata=lambda: True, _get_connector_metadata=lambda: meta
    )
    obj._activate_checkpoint_io = lambda: None
    obj._direct_group_caches = lambda: {}
    obj._block_size, obj._worker_retrieve_state = 128, {}
    obj._drop_worker_retrieve_state = lambda rid: calls.append("drop-state")
    obj.store_async, obj.kv_role, obj._unfenced_live_stores = True, "kv_both", {}
    if failure:
        with pytest.raises(RuntimeError, match="transfer failed"):
            obj.handle_preemptions({"r"})
        assert calls == (["direct"] if capture else ["drop-state", "direct"])
    else:
        obj.handle_preemptions({"r"})
        assert calls == (
            ["direct", "capture", "drop-state", "drop-store"]
            if capture
            else ["drop-state", "direct", "drop-store"]
        )
        assert logs[0][1] == ["r"]  # Preserve the original drain result for logging.
