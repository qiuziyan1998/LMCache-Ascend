# SPDX-License-Identifier: Apache-2.0
"""P0: Ascend handle_preemptions must unpin lookup pins and drop worker retrieve state."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")

adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState


def _make_adapter(**kwargs):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = kwargs.get("store_async", False)
    adapter.kv_role = kwargs.get("kv_role", "kv_both")
    engine = kwargs.get("lmcache_engine", MagicMock())
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._worker_retrieve_state = {
        "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
        "req-2": WorkerRetrieveState(metadata_warm=True),
    }
    return adapter, engine


def test_handle_preemptions_unpins_and_drops_worker_state() -> None:
    adapter, engine = _make_adapter()

    adapter.handle_preemptions({"req-1", "req-2"})

    engine.lookup_unpin.assert_any_call("req-1")
    engine.lookup_unpin.assert_any_call("req-2")
    assert adapter._worker_retrieve_state == {}


def test_handle_preemptions_without_engine_is_safe() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = False
    adapter.kv_role = "kv_both"
    adapter._manager = SimpleNamespace(lmcache_engine=None)
    adapter._worker_retrieve_state = {"req-1": WorkerRetrieveState(metadata_warm=True)}

    adapter.handle_preemptions({"req-1"})
    assert adapter._worker_retrieve_state == {"req-1"}
