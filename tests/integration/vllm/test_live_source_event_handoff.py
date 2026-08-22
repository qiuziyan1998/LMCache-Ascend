# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for prefix-hit live-source event propagation."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
base_adapter_mod = pytest.importorskip("lmcache.integration.vllm.vllm_v1_adapter")
handoff_mod = pytest.importorskip("vllm_ascend.live_source_handoff")


def _request(
    req_id: str,
    *,
    live_source_requested: bool = True,
    is_last_prefill: bool = True,
):
    return SimpleNamespace(
        req_id=req_id,
        live_source_requested=live_source_requested,
        is_last_prefill=is_last_prefill,
    )


def test_handoff_request_ids_include_only_final_live_exports() -> None:
    requests = [
        _request("live-final"),
        _request("live-partial", is_last_prefill=False),
        _request("persistent", live_source_requested=False),
        _request("live-final"),
    ]

    assert adapter_mod.LMCacheAscendConnectorV1Impl._live_source_handoff_request_ids(
        requests
    ) == ("live-final",)


def test_start_load_arms_handoff_after_base_load_setup() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    context = SimpleNamespace(attn_metadata={}, additional_kwargs={})

    with (
        patch.object(
            base_adapter_mod.LMCacheConnectorV1Impl,
            "start_load_kv",
        ) as base_start,
        patch.object(adapter_mod, "cold_start_perf_log"),
    ):
        adapter.start_load_kv(context)

    base_start.assert_called_once_with(context)
    state = context.additional_kwargs["lmcache_ascend_live_source_event_handoff_v1"]
    assert state == ("req-1",)


def test_capture_retains_exact_event_for_deferred_mtp_finalization() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    event = object()
    context = SimpleNamespace(
        additional_kwargs={handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: ("req-1",)},
        attn_metadata={
            "layer-0": SimpleNamespace(reshape_cache_event=event),
            "layer-78": SimpleNamespace(),
        },
    )

    assert adapter.capture_live_source_event_handoff(context)
    request_ids, retained_event = metadata._live_source_event_handoff
    assert request_ids == ("req-1",)
    assert retained_event is event
    assert handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY not in context.additional_kwargs


def test_dbo_capture_fails_closed() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    context = SimpleNamespace(
        additional_kwargs={handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: ("req-1",)},
        attn_metadata=[{"layer-0": SimpleNamespace(reshape_cache_event=object())}],
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")


def test_duplicate_capture_discards_retained_event() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace(_live_source_event_handoff=(("req-1",), object()))
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    context = SimpleNamespace(
        additional_kwargs={handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: ("req-1",)},
        attn_metadata={"layer-0": SimpleNamespace(reshape_cache_event=object())},
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")


def test_finish_save_batch_passes_handoff_event_to_live_descriptor() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    event = object()
    request = _request("req-1")
    handoff = (("req-1",), event)
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = MagicMock()
    adapter._latest_live_source_ready_event = None
    adapter._latest_live_source_ready_event_source = "missing"
    adapter._latest_direct_source_ready_events = {}
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._indexer_layer_names = ["index-0", "index-78"]
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter._direct_store_step_supported = True
    adapter._direct_store_observed_layers = set()
    adapter._completed_layerwise_stores = {}
    adapter._unfenced_live_stores = {"req-1": request}
    metadata = SimpleNamespace(_live_source_event_handoff=handoff)
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with patch.object(adapter_mod, "cold_start_perf_log"):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        source_ready_event=event,
        source_ready_event_source=("forward_context.sfa_reshape_cache_event"),
        source_ready_events=(event,),
    )


def test_finish_save_batch_handoff_supersedes_partial_callback_fence() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    callback_event = object()
    handoff_event = object()
    request = _request("req-1")
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = MagicMock()
    adapter._latest_live_source_ready_event = callback_event
    adapter._latest_live_source_ready_event_source = (
        "attn_metadata.reshape_cache_event"
    )
    adapter._latest_direct_source_ready_events = {"layer-0": callback_event}
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._indexer_layer_names = ["index-0", "index-78"]
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter._direct_store_step_supported = True
    adapter._direct_store_observed_layers = set()
    adapter._completed_layerwise_stores = {}
    adapter._unfenced_live_stores = {"req-1": request}
    metadata = SimpleNamespace(
        _live_source_event_handoff=(("req-1",), handoff_event)
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with patch.object(adapter_mod, "cold_start_perf_log"):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        source_ready_event=handoff_event,
        source_ready_event_source=("forward_context.sfa_reshape_cache_event"),
        source_ready_events=(handoff_event,),
    )


def test_finish_save_batch_mismatched_handoff_does_not_complete_fence() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    callback_event = object()
    request = _request("req-1")
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = MagicMock()
    adapter._latest_live_source_ready_event = callback_event
    adapter._latest_live_source_ready_event_source = (
        "attn_metadata.reshape_cache_event"
    )
    adapter._latest_direct_source_ready_events = {"layer-0": callback_event}
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._indexer_layer_names = ["index-0", "index-78"]
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter._direct_store_step_supported = True
    adapter._direct_store_observed_layers = set()
    adapter._completed_layerwise_stores = {}
    adapter._unfenced_live_stores = {"req-1": request}
    metadata = SimpleNamespace(
        _live_source_event_handoff=(("other-request",), object())
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with patch.object(adapter_mod, "cold_start_perf_log"):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        source_ready_event=callback_event,
        source_ready_event_source=("attn_metadata.reshape_cache_event"),
        source_ready_events=(),
    )


def test_prefix_hit_request_mismatch_fails_closed() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-78"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("new-request")])
    event = object()
    context = SimpleNamespace(
        additional_kwargs={handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: ("old-request",)},
        attn_metadata={"layer-78": SimpleNamespace(reshape_cache_event=event)},
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")
