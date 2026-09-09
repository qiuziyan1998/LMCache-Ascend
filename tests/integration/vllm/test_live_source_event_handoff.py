# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for prefix-hit live-source event propagation."""

# Standard
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
base_adapter_mod = pytest.importorskip("lmcache.integration.vllm.vllm_v1_adapter")
handoff_mod = pytest.importorskip("vllm_ascend.live_source_handoff")


@pytest.mark.parametrize(
    "perf,content", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fence_preserves_readiness_with_independent_diagnostic_modes(
    monkeypatch: pytest.MonkeyPatch,
    perf: bool,
    content: bool,
) -> None:
    monkeypatch.setattr(adapter_mod, "serving_perf_enabled", lambda: perf)
    monkeypatch.setattr(adapter_mod, "npu_content_diagnostics_enabled", lambda: content)
    clock = MagicMock(
        side_effect=[1.0, 1.125]
        if perf or content
        else AssertionError("disabled clock")
    )
    monkeypatch.setattr(adapter_mod.time, "perf_counter", clock)
    perf_log = MagicMock()
    content_log = MagicMock()
    monkeypatch.setattr(adapter_mod, "serving_perf_log", perf_log)
    monkeypatch.setattr(adapter_mod, "log_npu_content_diagnostic_event", content_log)
    event = SimpleNamespace(synchronize=MagicMock())
    fence = SimpleNamespace(
        event=event, event_source="producer", ready_at_finalize=True
    )
    finalize = MagicMock()
    adapter = SimpleNamespace(
        _live_source_ready_fences={"r": fence},
        lmcache_engine=SimpleNamespace(finalize_live_source_readiness=finalize),
        _query_source_ready_event=MagicMock(return_value=True),
    )
    adapter_mod.LMCacheAscendConnectorV1Impl._fence_live_source_descriptors(adapter)
    event.synchronize.assert_called_once_with()
    finalize.assert_called_once_with(["r"])
    assert adapter._live_source_ready_fences == {}
    assert clock.call_count == (2 if perf or content else 0)
    assert perf_log.call_count == int(perf)
    assert content_log.call_count == int(content)
    if perf:
        assert perf_log.call_args.kwargs["wait_ms"] == 125.0
    if content:
        assert content_log.call_args.kwargs["wait_ms"] == 125.0


def _request(
    req_id: str,
    *,
    live_source_requested: bool = True,
    is_last_prefill: bool = True,
    remote_fill_qualified: bool | None = None,
    token_count: int = 1,
):
    if remote_fill_qualified is None:
        remote_fill_qualified = live_source_requested
    return SimpleNamespace(
        req_id=req_id,
        live_source_requested=live_source_requested,
        is_last_prefill=is_last_prefill,
        token_ids=list(range(token_count)),
        _lmcache_remote_fill_qualified=remote_fill_qualified,
    )


def test_final_deferred_targets_include_only_final_requests() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.config = SimpleNamespace(remote_fill_submission_mode="final_deferred")
    adapter._remote_store_requested = True
    requests = [
        _request("live-final"),
        _request("live-partial", is_last_prefill=False),
        _request(
            "remote-final",
            live_source_requested=False,
            remote_fill_qualified=True,
        ),
        _request(
            "remote-partial",
            live_source_requested=False,
            remote_fill_qualified=True,
            is_last_prefill=False,
        ),
        _request("live-final"),
    ]

    assert adapter._producer_fence_handoff_targets(requests) == (
        ("live-final", 1),
        ("remote-final", 1),
    )


def test_per_chunk_targets_include_qualified_nonfinal_frontier() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.config = SimpleNamespace(remote_fill_submission_mode="per_chunk")
    adapter._remote_store_requested = True
    request = _request(
        "remote-partial",
        live_source_requested=False,
        remote_fill_qualified=True,
        is_last_prefill=False,
        token_count=1024,
    )

    assert adapter._producer_fence_handoff_targets([request]) == (
        ("remote-partial", 1024),
    )


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
        patch.object(adapter_mod, "serving_perf_log"),
    ):
        adapter.start_load_kv(context)

    base_start.assert_called_once_with(context)
    state = context.additional_kwargs["lmcache_ascend_live_source_event_handoff_v1"]
    assert state == (("req-1", 1),)


def test_capture_retains_exact_event_for_deferred_mtp_finalization() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    event = object()
    context = SimpleNamespace(
        additional_kwargs={
            handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: (("req-1", 1),)
        },
        attn_metadata={
            "layer-0": SimpleNamespace(reshape_cache_event=event),
            "layer-78": SimpleNamespace(),
        },
    )

    assert adapter.capture_live_source_event_handoff(context)
    targets, retained_event = metadata._live_source_event_handoff
    assert targets == (("req-1", 1),)
    assert retained_event is event
    assert handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY not in context.additional_kwargs


def test_dbo_capture_fails_closed() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    context = SimpleNamespace(
        additional_kwargs={
            handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: (("req-1", 1),)
        },
        attn_metadata=[{"layer-0": SimpleNamespace(reshape_cache_event=object())}],
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")


def test_duplicate_capture_discards_retained_event() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace(
        _live_source_event_handoff=((("req-1", 1),), object())
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-0"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    context = SimpleNamespace(
        additional_kwargs={
            handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: (("req-1", 1),)
        },
        attn_metadata={"layer-0": SimpleNamespace(reshape_cache_event=object())},
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")


def test_finish_save_batch_passes_handoff_event_to_live_descriptor() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    event = object()
    request = _request("req-1")
    handoff = ((("req-1", 1),), event)
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

    with (
        patch.object(adapter_mod, "serving_perf_enabled", return_value=True),
        patch.object(adapter_mod, "serving_perf_log") as perf_log,
    ):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        finish_batch=True,
        source_ready_event=event,
        source_ready_event_source=("forward_context.sfa_reshape_cache_event"),
        source_ready_events=(event,),
    )
    perf_log.assert_any_call(
        adapter_mod.logger,
        "remote_fill_producer_fence_decision",
        req_id="req-1",
        pending_sync_wait_ms=ANY,
        handoff_status="adopted",
        expected_layer_count=4,
        observed_layer_count=0,
        event_layer_count=0,
        callback_fence_complete=False,
        complete_fence_count=1,
        source_event_present=True,
        event_source="forward_context.sfa_reshape_cache_event",
        accepted_store_end=1,
        submission_mode="final_deferred",
        remote_fill_eligible=True,
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
    adapter._direct_store_observed_layers = {"layer-0"}
    adapter._completed_layerwise_stores = {}
    adapter._unfenced_live_stores = {"req-1": request}
    metadata = SimpleNamespace(
        _live_source_event_handoff=((("req-1", 1),), handoff_event)
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with (
        patch.object(adapter_mod, "serving_perf_enabled", return_value=True),
        patch.object(adapter_mod, "serving_perf_log") as perf_log,
    ):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        finish_batch=True,
        source_ready_event=handoff_event,
        source_ready_event_source=("forward_context.sfa_reshape_cache_event"),
        source_ready_events=(handoff_event,),
    )
    perf_log.assert_any_call(
        adapter_mod.logger,
        "remote_fill_producer_fence_decision",
        req_id="req-1",
        pending_sync_wait_ms=ANY,
        handoff_status="adopted",
        expected_layer_count=4,
        observed_layer_count=1,
        event_layer_count=1,
        callback_fence_complete=False,
        complete_fence_count=1,
        source_event_present=True,
        event_source="forward_context.sfa_reshape_cache_event",
        accepted_store_end=1,
        submission_mode="final_deferred",
        remote_fill_eligible=True,
    )


def test_finish_save_batch_mismatched_handoff_does_not_authorize_remote_fill(
) -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    callback_event = object()
    request = _request("req-1")
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = MagicMock()
    adapter._latest_live_source_ready_event = callback_event
    adapter._latest_live_source_ready_event_source = (
        "attn_metadata.reshape_cache_event"
    )
    adapter._latest_direct_source_ready_events = {
        name: callback_event
        for name in ("layer-0", "layer-78", "index-0", "index-78")
    }
    adapter._latent_layer_names = ["layer-0", "layer-78"]
    adapter._indexer_layer_names = ["index-0", "index-78"]
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter._direct_store_step_supported = True
    adapter._direct_store_observed_layers = {
        "layer-0",
        "layer-78",
        "index-0",
        "index-78",
    }
    adapter._completed_layerwise_stores = {}
    adapter._unfenced_live_stores = {"req-1": request}
    metadata = SimpleNamespace(
        _live_source_event_handoff=((("other-request", 1),), object())
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with (
        patch.object(adapter_mod, "serving_perf_enabled", return_value=True),
        patch.object(adapter_mod, "serving_perf_log") as perf_log,
    ):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        finish_batch=True,
        source_ready_event=callback_event,
        source_ready_event_source=("attn_metadata.reshape_cache_event"),
        source_ready_events=(callback_event,),
    )
    perf_log.assert_any_call(
        adapter_mod.logger,
        "remote_fill_producer_fence_decision",
        req_id="req-1",
        pending_sync_wait_ms=ANY,
        handoff_status="target_mismatch",
        expected_layer_count=4,
        observed_layer_count=4,
        event_layer_count=4,
        callback_fence_complete=True,
        complete_fence_count=1,
        source_event_present=True,
        event_source="attn_metadata.reshape_cache_event",
        accepted_store_end=1,
        submission_mode="final_deferred",
        remote_fill_eligible=True,
    )


def test_finish_save_batch_logs_absent_handoff_without_completing_fence() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    request = _request("req-1")
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
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._direct_prefill_requests = MagicMock(return_value=[request])
    adapter._submit_direct_prefill_requests = MagicMock()

    with (
        patch.object(adapter_mod, "serving_perf_enabled", return_value=True),
        patch.object(adapter_mod, "serving_perf_log") as perf_log,
    ):
        adapter._finish_save_batch({})

    adapter._submit_direct_prefill_requests.assert_called_once_with(
        [request],
        set(),
        finish_batch=True,
        source_ready_event=None,
        source_ready_event_source="missing",
        source_ready_events=(),
    )
    perf_log.assert_any_call(
        adapter_mod.logger,
        "remote_fill_producer_fence_decision",
        req_id="req-1",
        pending_sync_wait_ms=ANY,
        handoff_status="absent",
        expected_layer_count=4,
        observed_layer_count=0,
        event_layer_count=0,
        callback_fence_complete=False,
        complete_fence_count=0,
        source_event_present=False,
        event_source="missing",
        accepted_store_end=1,
        submission_mode="final_deferred",
        remote_fill_eligible=True,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail_wait", [False, True])
def test_pending_sync_timing_is_gated_and_preserves_the_wait(
    monkeypatch, enabled, fail_wait
):
    clock = MagicMock(side_effect=[1.0, 1.125])
    monkeypatch.setattr(adapter_mod.time, "perf_counter", clock)
    monkeypatch.setattr(adapter_mod, "serving_perf_enabled", lambda: enabled)
    log = MagicMock()
    monkeypatch.setattr(adapter_mod, "serving_perf_log", log)
    wait = MagicMock()
    request = SimpleNamespace(
        req_id="r",
        token_ids=[1],
        is_last_prefill=False,
        _lmcache_remote_fill_qualified=True,
    )
    adapter = SimpleNamespace(
        kv_role="kv_producer",
        config=SimpleNamespace(dsa_two_groups=True),
        lmcache_engine=SimpleNamespace(wait_for_pending_sync_stores=wait),
        _direct_store_observed_layers=set(),
        _latent_layer_names=[],
        _indexer_layer_names=[],
        _completed_layerwise_stores={},
        _parent=SimpleNamespace(_get_connector_metadata=SimpleNamespace),
        _direct_prefill_requests=lambda: [request],
        _producer_fence_handoff_targets=lambda requests: (),
        _submit_direct_prefill_requests=MagicMock(),
    )
    if fail_wait:
        wait.side_effect = RuntimeError("original wait failure")
        adapter._completed_layerwise_stores = {("r", 0): object()}
        with pytest.raises(RuntimeError, match="original wait failure"):
            adapter_mod.LMCacheAscendConnectorV1Impl._finish_save_batch(adapter, {})
        assert adapter._completed_layerwise_stores == {}
        assert clock.call_count == int(enabled)
        log.assert_not_called()
        adapter._submit_direct_prefill_requests.assert_not_called()
        return
    adapter_mod.LMCacheAscendConnectorV1Impl._finish_save_batch(adapter, {})
    wait.assert_called_once_with()
    assert clock.call_count == (2 if enabled else 0)
    assert log.call_count == int(enabled)
    if enabled:
        assert log.call_args.kwargs["pending_sync_wait_ms"] == 125.0
    adapter._submit_direct_prefill_requests.assert_called_once()


def test_frontier_change_between_arm_and_capture_fails_closed() -> None:
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    metadata = SimpleNamespace()
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._latent_layer_names = ["layer-78"]
    adapter._direct_prefill_requests = MagicMock(return_value=[_request("req-1")])
    event = object()
    context = SimpleNamespace(
        additional_kwargs={
            handoff_mod.LIVE_SOURCE_EVENT_HANDOFF_KEY: (("req-1", 2),)
        },
        attn_metadata={"layer-78": SimpleNamespace(reshape_cache_event=event)},
    )

    assert not adapter.capture_live_source_event_handoff(context)
    assert not hasattr(metadata, "_live_source_event_handoff")
