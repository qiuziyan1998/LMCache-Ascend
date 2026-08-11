# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest


def _import_and_patch_vllm_connector():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    lmcache_ascend = pytest.importorskip("lmcache_ascend")
    lmcache_ascend._patch_vllm_v1_adapter()
    return LMCacheConnectorV1


def _make_adapter(adapter_mod, *, store_async, kv_role, lmcache_engine):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = kv_role
    adapter._manager = SimpleNamespace(lmcache_engine=lmcache_engine)
    return adapter


def test_lmcache_connector_delegates_preemptions_after_ascend_patch():
    """Ascend patches the outer vLLM connector to delegate preemptions."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    preempted_req_ids = {"req-1", "req-2"}
    connector.handle_preemptions(preempted_req_ids)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        preempted_req_ids
    )


def test_lmcache_connector_patch_advertises_dsa_index_support():
    """Ascend's vLLM patch must make LMCache receive DSA indexer KV caches."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    assert getattr(LMCacheConnectorV1, "supports_dsa_index_lmcache", False) is True


def test_lmcache_connector_patch_reports_layerwise_callbacks():
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    connector = object.__new__(LMCacheConnectorV1)

    connector._lmcache_engine = SimpleNamespace(use_layerwise=True)
    assert connector.uses_layerwise_model_callbacks is True

    connector._lmcache_engine = SimpleNamespace(use_layerwise=False)
    assert connector.uses_layerwise_model_callbacks is False


def test_lmcache_connector_patch_advertises_staged_sfa_sparse_load():
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    connector = object.__new__(LMCacheConnectorV1)

    connector._lmcache_engine = SimpleNamespace(
        use_layerwise=True,
        kv_role="kv_both",
        config=SimpleNamespace(
            dsa_two_groups=True,
            enable_sparse_attention=True,
        ),
    )
    assert connector.supports_staged_sfa_sparse_load is True

    connector._lmcache_engine.kv_role = "kv_producer"
    assert connector.supports_staged_sfa_sparse_load is False

    connector._lmcache_engine.kv_role = None
    assert connector.supports_staged_sfa_sparse_load is False

    connector._lmcache_engine.kv_role = "kv_both"
    connector._lmcache_engine.use_layerwise = False
    assert connector.supports_staged_sfa_sparse_load is False

    connector._lmcache_engine.use_layerwise = True
    connector._lmcache_engine.config.dsa_two_groups = False
    assert connector.supports_staged_sfa_sparse_load is False

    connector._lmcache_engine.config.dsa_two_groups = True
    connector._lmcache_engine.config.enable_sparse_attention = False
    assert connector.supports_staged_sfa_sparse_load is False


@pytest.mark.parametrize(
    ("kv_role", "expected"),
    (
        ("kv_both", True),
        ("kv_consumer", True),
        ("kv_producer", False),
        (None, False),
        ("unknown", False),
    ),
)
def test_dynamic_connector_advertises_staged_sparse_load_by_role(
    kv_role, expected
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    connector_mod = pytest.importorskip(
        "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
    connector = object.__new__(
        connector_mod.LMCacheAscendConnectorV1Dynamic
    )
    connector._lmcache_engine = SimpleNamespace(
        use_layerwise=True,
        kv_role=kv_role,
        config=SimpleNamespace(
            dsa_two_groups=True,
            enable_sparse_attention=True,
        ),
    )

    assert connector.supports_staged_sfa_sparse_load is expected
    connector._lmcache_engine.config.enable_sparse_attention = False
    assert connector.supports_staged_sfa_sparse_load is False


def test_lmcache_connector_preemption_patch_handles_no_inner_impl():
    """The Ascend patch should tolerate inner implementations without a hook."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = object()

    connector.handle_preemptions({"req-1"})


def test_ascend_adapter_drains_pending_stores_for_async_producer():
    """Async non-consumer workers must drain pending stores before reuse."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-1"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    preempted_req_ids = {"req-1", "req-2"}
    adapter.handle_preemptions(preempted_req_ids)

    lmcache_engine.wait_for_pending_stores.assert_called_once_with(preempted_req_ids)
    lmcache_engine.drop_direct_store_states.assert_called_once_with(
        preempted_req_ids
    )


@pytest.mark.parametrize(
    ("store_async", "kv_role", "has_engine"),
    [
        (False, "kv_both", True),
        (True, "kv_consumer", True),
        (True, "kv_both", False),
    ],
)
def test_ascend_adapter_skips_preemption_drain_when_not_required(
    store_async, kv_role, has_engine
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock() if has_engine else None
    adapter = _make_adapter(
        adapter_mod,
        store_async=store_async,
        kv_role=kv_role,
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions({"req-1"})

    if has_engine:
        lmcache_engine.wait_for_pending_stores.assert_not_called()
