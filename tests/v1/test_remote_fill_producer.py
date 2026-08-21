# SPDX-License-Identifier: Apache-2.0
"""Hardware-independent tests for the prefiller remote-fill lifecycle."""

# Standard
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict
from threading import Condition, RLock
from types import SimpleNamespace
from typing import Any
import logging
import time

import pytest

from lmcache.v1.remote_fill import (
    AbortRequest,
    ArmWindowRequest,
    ControlPage,
    DestinationNativeState,
    DestinationPageDescriptor,
    FinishRequest,
    NegotiateRequest,
    OpenRequest,
    OperationKind,
    PageDisposition,
    PagePreparationStatus,
    RemoteFillResponse,
    ReportTransferCompleteRequest,
    ReplyLostError,
    ReserveWindowRequest,
    ResultCode,
    StatusRequest,
    TerminalOutcome,
    WindowState,
    WindowStatus,
    destination_descriptor_digest,
    seal_descriptor,
)
from lmcache.v1.remote_fill.native import (
    DirectPushPageSource,
    DirectPushSourcePlan,
    NativeDirectPushPreSubmitError,
    NativeDirectPushResult,
    NativeDirectPushTerminalError,
)

# First Party
from lmcache_ascend.v1.cache_engine import (
    AscendLMCacheEngine,
    _DirectPageBatch,
    _DirectStoreRequestState,
)
from lmcache_ascend.v1.remote_fill_producer import (
    RemoteFillFatalError,
    RemoteFillHandoff,
    RemoteFillProducerMetrics,
    RemoteFillProducerSession,
    RemoteFillNegotiationCache,
    RemoteFillStaticSpec,
    RemoteFillTerminalResult,
    parse_remote_fill_handoff,
)


_SECRET = b"s" * 32
_EPOCH = 17
_GENERATION = 23
_SESSION = "decoder-global-te"


def test_remote_fill_rejects_vllm_sleep_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lmcache_ascend.integration.vllm import vllm_v1_adapter

    monkeypatch.setattr(vllm_v1_adapter, "remote_fill_h0_qualified", lambda: True)
    config = SimpleNamespace(enable_remote_lmcache_store=True)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True)
    )

    with pytest.raises(ValueError, match="incompatible with vLLM sleep mode"):
        vllm_v1_adapter._validate_remote_fill_sleep_mode(config, vllm_config)

    vllm_config.model_config.enable_sleep_mode = False
    vllm_v1_adapter._validate_remote_fill_sleep_mode(config, vllm_config)


def _handoff(**overrides: Any) -> RemoteFillHandoff:
    values = dict(
        transfer_id="transfer-1",
        request_attempt=2,
        source_engine_id="prefiller",
        destination_engine_id="decoder",
        destination_engine_epoch=_EPOCH,
        control_endpoint="tcp://decoder:19001",
        destination_dp_rank=1,
        shared_cache_generation=_GENERATION,
        destination_tp_size=8,
        destination_dp_size=2,
        global_te_push=True,
        token_hash_algorithm="sha256",
        python_hash_seed="",
        descriptor_verification_capability=_SECRET.hex(),
    )
    values.update(overrides)
    return RemoteFillHandoff(**values)


def _static_spec() -> RemoteFillStaticSpec:
    return RemoteFillStaticSpec(
        cache_namespace_tag="deployment-a",
        layout_tag="payload-v3",
        model_artifact_id="model-sha256",
        chunk_size=1024,
        model_layout="mla-dsa-layer-page-v3",
        group_dimensions=(576, 128),
        layer_count=79,
        save_only_first_rank=True,
        shared_group1=True,
        tp_size=8,
        dp_size=2,
        global_te_push=True,
        token_hash_algorithm="sha256",
        python_hash_seed="",
    )


def _pages() -> tuple[ControlPage, ControlPage]:
    return (
        ControlPage(
            canonical_key="group0-key",
            kv_group=0,
            chunk_index=0,
            chunk_start=0,
            chunk_end=1024,
            valid_tokens=1024,
            destination_tp_rank=0,
            expected_bytes=128,
            layer_count=79,
            layout_tag="payload-v3",
        ),
        ControlPage(
            canonical_key="group1-key",
            kv_group=1,
            chunk_index=0,
            chunk_start=0,
            chunk_end=1024,
            valid_tokens=1024,
            destination_tp_rank=0,
            expected_bytes=64,
            layer_count=79,
            layout_tag="payload-v3",
        ),
    )


class _ScriptedClient:
    def __init__(
        self,
        *,
        reserve_dispositions: tuple[PageDisposition, PageDisposition],
        descriptor_dp_rank: int = 1,
        descriptor_tp_rank: int | None = None,
        fail_arm: bool = False,
        arm_status_armed: bool = False,
        fail_reserve: bool = False,
        fail_report_replies: int = 0,
        report_status_native_state: DestinationNativeState | None = None,
        fail_finish_replies: int = 0,
        fatal_on: type | None = None,
    ) -> None:
        self.reserve_dispositions = reserve_dispositions
        self.descriptor_dp_rank = descriptor_dp_rank
        self.descriptor_tp_rank = descriptor_tp_rank
        self.fail_arm = fail_arm
        self.arm_status_armed = arm_status_armed
        self.fail_reserve = fail_reserve
        self.fail_report_replies = fail_report_replies
        self.report_status_native_state = report_status_native_state
        self.fail_finish_replies = fail_finish_replies
        self.finish_committed = False
        self.fatal_on = fatal_on
        self.requests: list[Any] = []
        self.closed = False

    @staticmethod
    def _kind(request: Any) -> OperationKind:
        return {
            NegotiateRequest: OperationKind.NEGOTIATE,
            OpenRequest: OperationKind.OPEN,
            ReserveWindowRequest: OperationKind.RESERVE_WINDOW,
            ArmWindowRequest: OperationKind.ARM_WINDOW,
            ReportTransferCompleteRequest: OperationKind.REPORT_TRANSFER_COMPLETE,
            FinishRequest: OperationKind.FINISH,
            AbortRequest: OperationKind.ABORT,
            StatusRequest: OperationKind.STATUS,
        }[type(request)]

    def _response(self, request: Any, code: ResultCode, **kwargs: Any) -> Any:
        if type(request) is self.fatal_on:
            kwargs["fatal_restart_required"] = True
            kwargs["terminal_outcome"] = TerminalOutcome.FATAL_RESTART
        return RemoteFillResponse(
            operation=self._kind(request),
            operation_id=request.common.operation_id,
            code=code,
            destination_engine_epoch=_EPOCH,
            shared_cache_generation=_GENERATION,
            **kwargs,
        )

    def execute(self, request: Any) -> Any:
        self.requests.append(request)
        if isinstance(request, NegotiateRequest):
            return self._response(request, ResultCode.OK)
        if isinstance(request, OpenRequest):
            return self._response(
                request,
                ResultCode.ACCEPTED,
                remote_session=_SESSION,
            )
        if isinstance(request, ReserveWindowRequest):
            if self.fail_reserve:
                raise TimeoutError("RESERVE response unavailable")
            results = tuple(
                PagePreparationStatus(
                    canonical_key=page.canonical_key,
                    kv_group=page.kv_group,
                    chunk_index=page.chunk_index,
                    disposition=disposition,
                )
                for page, disposition in zip(
                    request.control_pages,
                    self.reserve_dispositions,
                    strict=True,
                )
            )
            descriptors = []
            attempt = "native-attempt"
            for page, disposition in zip(
                request.control_pages,
                self.reserve_dispositions,
                strict=True,
            ):
                if disposition is not PageDisposition.ALLOCATED:
                    continue
                descriptors.append(
                    seal_descriptor(
                        _SECRET,
                        DestinationPageDescriptor(
                            canonical_key=page.canonical_key,
                            chunk_index=page.chunk_index,
                            remote_session=_SESSION,
                            destination_ptr=0x100000 + page.kv_group * 0x1000,
                            destination_length=page.expected_bytes,
                            reservation_id=f"reservation-{page.kv_group}",
                            window_id=request.window_id,
                            kv_group=page.kv_group,
                            transfer_id="transfer-1",
                            request_attempt=2,
                            destination_dp_rank=self.descriptor_dp_rank,
                            destination_tp_rank=(
                                page.destination_tp_rank
                                if self.descriptor_tp_rank is None
                                else self.descriptor_tp_rank
                            ),
                            destination_engine_epoch=_EPOCH,
                            shared_cache_generation=_GENERATION,
                            manifest_digest=request.manifest_digest,
                            native_transfer_attempt_id=attempt,
                            expires_at=time.time() + 60,
                            capability_mac="",
                        ),
                    )
                )
            descriptors_tuple = tuple(descriptors)
            has_missing = PageDisposition.MISSING in self.reserve_dispositions
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED if has_missing else ResultCode.OK,
                native_transfer_attempt_id=attempt if descriptors else "",
                destination_descriptor_digest=(
                    destination_descriptor_digest(descriptors_tuple)
                    if descriptors_tuple
                    else ""
                ),
                descriptors=descriptors_tuple,
                page_results=results,
            )
        if isinstance(request, ArmWindowRequest):
            if self.fail_arm:
                raise TimeoutError("ARM response lost")
            return self._response(request, ResultCode.OK)
        if isinstance(request, StatusRequest):
            if request.window_id < 0 and self.finish_committed:
                return self._response(
                    request,
                    ResultCode.OK,
                    terminal_outcome=TerminalOutcome.LOCAL_FULL,
                )
            armed = self.arm_status_armed
            native_state = self.report_status_native_state or (
                DestinationNativeState.ARMED
                if armed
                else DestinationNativeState.NOT_ARMED
            )
            return self._response(
                request,
                ResultCode.OK,
                windows=(
                    WindowStatus(
                        window_id=request.window_id,
                        state=(WindowState.ARMED if armed else WindowState.RESERVED),
                        native_state=native_state,
                        native_transfer_attempt_id="native-attempt",
                        page_count=2,
                        total_bytes=192,
                        expired_unarmed=False,
                        fatal_restart_required=False,
                    ),
                ),
            )
        if isinstance(request, ReportTransferCompleteRequest):
            if self.fail_report_replies > 0:
                self.fail_report_replies -= 1
                raise ReplyLostError("REPORT_TRANSFER_COMPLETE reply lost")
            return self._response(request, ResultCode.OK)
        if isinstance(request, FinishRequest):
            self.finish_committed = True
            if self.fail_finish_replies > 0:
                self.fail_finish_replies -= 1
                raise ReplyLostError("FINISH reply lost")
            return self._response(
                request,
                ResultCode.OK,
                terminal_outcome=TerminalOutcome.LOCAL_FULL,
            )
        if isinstance(request, AbortRequest):
            return self._response(request, ResultCode.TERMINAL)
        raise AssertionError(type(request))

    def close(self) -> None:
        self.closed = True


def _session(
    client: _ScriptedClient,
    *,
    native_hard_timeout_seconds: float = 120.0,
    negotiation_cache: RemoteFillNegotiationCache | None = None,
) -> RemoteFillProducerSession:
    return RemoteFillProducerSession(
        request_id="request-1",
        handoff=_handoff(),
        static_spec=_static_spec(),
        client=client,
        secret=_SECRET,
        planned_window_count_hint=1,
        required_store_end_hint=1024,
        native_hard_timeout_seconds=native_hard_timeout_seconds,
        negotiation_cache=negotiation_cache,
    )


def test_static_negotiation_is_cached_for_decoder_epoch() -> None:
    cache = RemoteFillNegotiationCache()
    first_client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.EXISTING)
    )
    second_client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.EXISTING)
    )

    assert _session(first_client, negotiation_cache=cache).open()
    assert _session(second_client, negotiation_cache=cache).open()

    assert (
        sum(isinstance(item, NegotiateRequest) for item in first_client.requests)
        == 1
    )
    assert not any(
        isinstance(item, NegotiateRequest) for item in second_client.requests
    )
    assert sum(isinstance(item, OpenRequest) for item in second_client.requests) == 1


def _source_plan() -> DirectPushSourcePlan:
    return DirectPushSourcePlan(
        pages=(
            DirectPushPageSource(
                canonical_key="group0-key",
                kv_group=0,
                source_ptrs=(100,),
                source_lengths=(128,),
            ),
            DirectPushPageSource(
                canonical_key="group1-key",
                kv_group=1,
                source_ptrs=(200,),
                source_lengths=(64,),
            ),
        ),
        owners=(object(),),
        producer_events=(object(),),
    )


def test_parse_handoff_preserves_tp_dp_and_hash_identity() -> None:
    handoff = _handoff()
    parsed = parse_remote_fill_handoff({"lmcache.remote_fill": asdict(handoff)})
    assert parsed == handoff
    assert parsed.destination_dp_rank == 1
    assert parsed.destination_tp_size == 8


@pytest.mark.parametrize(
    "capability",
    ("", "not-hex", "AB" * 32, "ab" * 31),
)
def test_parse_handoff_rejects_noncanonical_verification_capability(
    capability: str,
) -> None:
    raw = asdict(_handoff())
    raw["descriptor_verification_capability"] = capability

    with pytest.raises(ValueError, match="verification capability"):
        parse_remote_fill_handoff({"lmcache.remote_fill": raw})


def test_transfer_arms_only_allocated_descriptor_subset() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.ALLOCATED)
    )
    session = _session(client)
    submitted: list[tuple[str, DirectPushSourcePlan, tuple[Any, ...]]] = []

    def submitter(**kwargs: Any) -> Future:
        submitted.append(
            (
                kwargs["remote_session"],
                kwargs["source_plan"],
                kwargs["destination_descriptors"],
            )
        )
        future: Future = Future()
        future.set_result(
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=0,
                vector_count=1,
                transferred_bytes=64,
                elapsed_ms=1.0,
                source_event_wait_ms=2.0,
                source_registration_ms=1.0,
                native_slot_wait_ms=3.0,
                native_started_monotonic=10.0,
                native_ended_monotonic=10.001,
            )
        )
        return future

    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=submitter,
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )
    assert result.direct_satisfied and result.armed
    assert result.submitted_bytes == 64
    assert result.existing_pages == 1
    assert result.reserve_seconds >= 0
    assert result.arm_seconds >= 0
    assert result.source_event_wait_seconds == pytest.approx(0.002)
    assert result.source_registration_seconds == pytest.approx(0.001)
    assert result.native_slot_wait_seconds == pytest.approx(0.003)
    assert result.native_seconds == pytest.approx(0.001)
    assert result.report_seconds >= 0
    assert result.native_started_monotonic == 10.0
    assert result.native_ended_monotonic == 10.001
    assert submitted[0][0] == _SESSION
    assert [page.kv_group for page in submitted[0][1].pages] == [1]
    assert [descriptor.kv_group for descriptor in submitted[0][2]] == [1]
    report = next(
        request
        for request in client.requests
        if isinstance(request, ReportTransferCompleteRequest)
    )
    assert report.completed_bytes == 64
    terminal = session.finish(
        required_store_end=1024,
        persistent_common_end=1024,
    )
    assert terminal.outcome == "LOCAL_FULL"
    assert any(isinstance(item, FinishRequest) for item in client.requests)


def test_lost_finish_reply_recovers_committed_local_full_with_status() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.EXISTING),
        fail_finish_replies=2,
    )
    session = _session(client)
    result = session.probe_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
    )
    assert result.direct_satisfied

    terminal = session.finish(
        required_store_end=1024,
        persistent_common_end=1024,
    )

    assert terminal.outcome == "LOCAL_FULL"
    assert sum(isinstance(item, FinishRequest) for item in client.requests) == 2
    assert any(
        isinstance(item, StatusRequest) and item.window_id == -1
        for item in client.requests
    )


def test_probe_hole_latches_persistent_only_without_arm() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.MISSING)
    )
    session = _session(client)
    result = session.probe_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
    )
    assert not result.direct_satisfied
    assert result.reason == "cached-prefix hole"
    assert not session.direct_viable
    assert not any(isinstance(item, ArmWindowRequest) for item in client.requests)
    reserve_count = sum(
        isinstance(item, ReserveWindowRequest) for item in client.requests
    )
    later = session.transfer_window(
        window_id=1,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=lambda **_kwargs: pytest.fail("latched path must not submit"),
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )
    assert not later.direct_satisfied
    assert (
        sum(isinstance(item, ReserveWindowRequest) for item in client.requests)
        == reserve_count
    )


def test_wrong_destination_dp_binding_is_rejected_before_arm() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        descriptor_dp_rank=0,
    )
    session = _session(client)
    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=lambda **_kwargs: pytest.fail("native submit must not run"),
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )
    assert not result.direct_satisfied
    assert result.reason == "descriptor validation failed"
    assert not any(isinstance(item, ArmWindowRequest) for item in client.requests)


def test_wrong_destination_tp_binding_is_rejected_before_arm() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        descriptor_tp_rank=1,
    )
    session = _session(client)
    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=lambda **_kwargs: pytest.fail("native submit must not run"),
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )
    assert not result.direct_satisfied
    assert result.reason == "descriptor validation failed"
    assert not any(isinstance(item, ArmWindowRequest) for item in client.requests)


def test_lost_arm_ack_is_fatal_and_never_submits_native() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_arm=True,
    )
    session = _session(client)
    with pytest.raises(RemoteFillFatalError, match="ARM_WINDOW"):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=lambda **_kwargs: pytest.fail("native submit must not run"),
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )
    assert session.fatal_restart_required


def test_lost_arm_ack_uses_status_and_submits_exact_armed_attempt() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_arm=True,
        arm_status_armed=True,
    )
    session = _session(client)
    submitted: list[str] = []

    def submitter(**kwargs: Any) -> Future:
        submitted.append(kwargs["activation"].attempt)
        future: Future = Future()
        future.set_result(
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=0,
                vector_count=2,
                transferred_bytes=192,
                elapsed_ms=1.0,
            )
        )
        return future

    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=submitter,
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )

    assert result.direct_satisfied and result.armed
    assert submitted == ["native-attempt"]
    assert sum(isinstance(item, ArmWindowRequest) for item in client.requests) == 1
    assert sum(isinstance(item, StatusRequest) for item in client.requests) == 1


def test_ambiguous_armed_native_write_is_fatal_and_never_cpu_retried() -> None:
    class _AmbiguousNativeError(RuntimeError):
        terminal_future = object()

    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED)
    )
    session = _session(client)

    def submitter(**_kwargs: Any) -> Future:
        future: Future = Future()
        future.set_exception(_AmbiguousNativeError("native result unknown"))
        return future

    with pytest.raises(RemoteFillFatalError, match="unknown terminal state"):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=submitter,
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )

    assert session.fatal_restart_required
    assert not any(
        isinstance(item, ReportTransferCompleteRequest) for item in client.requests
    )


def test_outer_native_future_hard_timeout_is_fatal_and_retains_owners() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED)
    )
    session = _session(client, native_hard_timeout_seconds=0.01)
    pending: Future = Future()
    retained: dict[str, Any] = {}

    def submitter(**kwargs: Any) -> Future:
        retained["source_plan"] = kwargs["source_plan"]
        return pending

    with pytest.raises(RemoteFillFatalError, match="hard terminal bound"):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=submitter,
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )

    assert session.fatal_restart_required
    assert not pending.cancelled()
    assert retained["source_plan"].owners
    assert not any(
        isinstance(item, ReportTransferCompleteRequest) for item in client.requests
    )


def test_ambiguous_native_waits_for_original_terminal_without_resubmit() -> None:
    class _AmbiguousNativeError(RuntimeError):
        def __init__(self, terminal_future: Future) -> None:
            super().__init__("native result initially unavailable")
            self.terminal_future = terminal_future

    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED)
    )
    session = _session(client)
    submitted = 0

    def submitter(**_kwargs: Any) -> Future:
        nonlocal submitted
        submitted += 1
        terminal: Future = Future()
        terminal.set_result(
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=0,
                vector_count=2,
                transferred_bytes=192,
                elapsed_ms=1.0,
            )
        )
        initial: Future = Future()
        initial.set_exception(_AmbiguousNativeError(terminal))
        return initial

    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=submitter,
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )

    assert result.direct_satisfied and result.armed
    assert submitted == 1


def test_ambiguous_native_unknown_terminal_exception_is_fatal() -> None:
    class _AmbiguousNativeError(RuntimeError):
        def __init__(self, terminal_future: Future) -> None:
            super().__init__("native result initially unavailable")
            self.terminal_future = terminal_future

    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED)
    )
    session = _session(client)
    submitted = 0

    def submitter(**_kwargs: Any) -> Future:
        nonlocal submitted
        submitted += 1
        terminal: Future = Future()
        terminal.set_exception(RuntimeError("unknown native terminal exception"))
        initial: Future = Future()
        initial.set_exception(_AmbiguousNativeError(terminal))
        return initial

    with pytest.raises(RemoteFillFatalError, match="unknown terminal state"):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=submitter,
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )

    assert session.fatal_restart_required
    assert submitted == 1
    assert not any(
        isinstance(item, ReportTransferCompleteRequest) for item in client.requests
    )


@pytest.mark.parametrize(
    ("native_result", "status_state", "direct_satisfied"),
    (
        (
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=0,
                vector_count=2,
                transferred_bytes=192,
                elapsed_ms=1.0,
            ),
            DestinationNativeState.TERMINAL_SUCCESS,
            True,
        ),
        (
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=-1,
                vector_count=2,
                transferred_bytes=0,
                elapsed_ms=1.0,
            ),
            DestinationNativeState.TERMINAL_FAILURE,
            False,
        ),
    ),
)
def test_lost_terminal_report_is_resolved_by_exact_status(
    native_result: NativeDirectPushResult,
    status_state: DestinationNativeState,
    direct_satisfied: bool,
) -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_report_replies=2,
        report_status_native_state=status_state,
    )
    session = _session(client)
    submitted = 0

    def submitter(**_kwargs: Any) -> Future:
        nonlocal submitted
        submitted += 1
        future: Future = Future()
        if native_result.return_code == 0:
            future.set_result(native_result)
        else:
            future.set_exception(NativeDirectPushTerminalError(native_result))
        return future

    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=submitter,
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )

    assert result.direct_satisfied is direct_satisfied
    assert submitted == 1
    assert sum(
        isinstance(item, ReportTransferCompleteRequest) for item in client.requests
    ) == 2
    assert sum(isinstance(item, StatusRequest) for item in client.requests) == 1


def test_lost_terminal_report_with_mismatched_status_is_fatal() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_report_replies=2,
        report_status_native_state=DestinationNativeState.TERMINAL_FAILURE,
    )
    session = _session(client)

    def submitter(**_kwargs: Any) -> Future:
        future: Future = Future()
        future.set_result(
            NativeDirectPushResult(
                native_transfer_attempt_id="native-attempt",
                return_code=0,
                vector_count=2,
                transferred_bytes=192,
                elapsed_ms=1.0,
            )
        )
        return future

    with pytest.raises(RemoteFillFatalError, match="terminal native report"):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=submitter,
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )

    assert session.fatal_restart_required


def test_known_pre_submit_native_failure_reports_terminal_failure() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED)
    )
    session = _session(client)

    def submitter(**_kwargs: Any) -> Future:
        future: Future = Future()
        future.set_exception(NativeDirectPushPreSubmitError("source fence failed"))
        return future

    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=submitter,
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )

    assert not result.direct_satisfied
    report = next(
        item
        for item in client.requests
        if isinstance(item, ReportTransferCompleteRequest)
    )
    assert report.native_return_code == -1
    assert report.completed_bytes == 0


def test_decoder_fatal_response_is_never_downgraded_to_fallback() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fatal_on=NegotiateRequest,
    )
    session = _session(client)

    with pytest.raises(RemoteFillFatalError, match="unknown terminal state"):
        session.open()

    assert session.fatal_restart_required


def test_abort_releases_open_nonfatal_transaction() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.MISSING)
    )
    session = _session(client)
    result = session.probe_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
    )
    assert not result.direct_satisfied
    session.abort("request preempted")
    assert any(isinstance(item, AbortRequest) for item in client.requests)


def test_prearm_control_failure_finishes_persistent_only() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_reserve=True,
    )
    session = _session(client)
    result = session.transfer_window(
        window_id=0,
        source_generation=44,
        control_pages=_pages(),
        source_plan=_source_plan(),
        submitter=lambda **_kwargs: pytest.fail("pre-arm failure must not submit"),
        activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
    )

    assert not result.direct_satisfied
    assert session.prearm_control_unknown
    terminal = session.finish(
        required_store_end=1024,
        persistent_common_end=1024,
    )
    assert terminal.outcome == "PERSISTENT_ONLY"
    assert not any(isinstance(item, ArmWindowRequest) for item in client.requests)


def test_final_partial_page_is_probed_as_an_exact_pair() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.EXISTING, PageDisposition.EXISTING)
    )
    session = _session(client)
    pages = tuple(
        ControlPage(
            canonical_key=f"partial-group{group}",
            kv_group=group,
            chunk_index=4,
            chunk_start=4096,
            chunk_end=4113,
            valid_tokens=17,
            destination_tp_rank=0,
            expected_bytes=17 * (128 if group else 576) * 2 * 79,
            layer_count=79,
            layout_tag="payload-v3",
        )
        for group in (0, 1)
    )

    result = session.probe_window(
        window_id=0,
        source_generation=44,
        control_pages=pages,
    )
    terminal = session.finish(
        required_store_end=4113,
        persistent_common_end=4113,
        final_partial_valid_tokens=17,
    )

    assert result.direct_satisfied
    assert terminal.outcome == "LOCAL_FULL"
    reserve = next(
        item for item in client.requests if isinstance(item, ReserveWindowRequest)
    )
    assert [(page.kv_group, page.valid_tokens) for page in reserve.control_pages] == [
        (0, 17),
        (1, 17),
    ]


def test_disabled_engine_path_creates_no_remote_work() -> None:
    state = _DirectStoreRequestState()
    engine = SimpleNamespace(config=SimpleNamespace(enable_remote_lmcache_store=False))

    enabled = AscendLMCacheEngine._remote_fill_prepare_request(
        engine,
        "request",
        {"lmcache.remote_fill": asdict(_handoff())},
        state,
    )

    assert enabled is False
    assert state.remote_fill_handoff is None
    assert not hasattr(engine, "_remote_fill_producer_executor")
    assert not hasattr(engine, "_remote_fill_queue_lock")
    assert not hasattr(engine, "_remote_fill_client_factory")


def test_malformed_handoff_is_visible_and_uses_persistent_fallback(caplog) -> None:
    state = _DirectStoreRequestState()
    engine = SimpleNamespace(config=SimpleNamespace(enable_remote_lmcache_store=True))

    with caplog.at_level(logging.WARNING):
        enabled = AscendLMCacheEngine._remote_fill_prepare_request(
            engine,
            "request",
            {"lmcache.remote_fill": {"transfer_id": 7}},
            state,
        )

    assert enabled is False
    assert state.remote_fill_handoff is None
    assert '"code":"RF-P-001"' in caplog.text
    assert '"action":"PERSISTENT_ONLY"' in caplog.text


def test_producer_metrics_are_bounded_and_pointer_free() -> None:
    metrics = RemoteFillProducerMetrics()
    metrics.start_attempt()
    metrics.observe("reserve_seconds", 0.25)
    metrics.observe("arm_seconds", 0.01)
    metrics.observe("native_slot_wait_seconds", 0.02)
    metrics.observe("report_seconds", 0.03)
    metrics.observe("finish_control_seconds", 0.04)
    metrics.add_gauge("inflight_windows", 1)
    metrics.add_bytes("submitted_bytes", 192)
    metrics.existing_pages(1)
    metrics.abandon("a request-specific secret pointer 0x1234")
    metrics.finish_attempt("LOCAL_FULL")

    snapshot = metrics.snapshot()
    rendered = repr(snapshot)
    assert snapshot["started_total"] == 1
    assert snapshot["attempts_total"]["LOCAL_FULL:none"] == 1
    assert snapshot["direct_abandoned_total"] == {"other": 1}
    assert snapshot["timers"]["reserve_seconds"]["total_seconds"] == 0.25
    assert snapshot["timers"]["arm_seconds"]["total_seconds"] == 0.01
    assert snapshot["timers"]["native_slot_wait_seconds"]["total_seconds"] == 0.02
    assert snapshot["timers"]["report_seconds"]["total_seconds"] == 0.03
    assert snapshot["timers"]["finish_control_seconds"]["total_seconds"] == 0.04
    assert snapshot["bytes"]["submitted_bytes"] == 192
    assert "0x1234" not in rendered
    assert "request-specific" not in rendered


def test_global_producer_inflight_bytes_are_bounded_across_requests() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(
        remote_fill_max_inflight_bytes=100,
        remote_fill_max_bytes_per_request=400,
        remote_fill_max_inflight_windows_per_request=4,
        # This decoder-side reservation budget must not relax P retention.
        remote_fill_max_reserved_bytes=16 * 1024**3,
    )
    first = _DirectStoreRequestState()
    second = _DirectStoreRequestState()

    assert engine._remote_fill_acquire_queue_capacity(first, 60)
    assert not engine._remote_fill_acquire_queue_capacity(second, 50)
    assert first.remote_fill_queued_bytes == 60
    assert second.remote_fill_queued_bytes == 0

    engine._remote_fill_release_queue_capacity(first, 60)
    assert engine._remote_fill_acquire_queue_capacity(second, 50)
    engine._remote_fill_release_queue_capacity(second, 50)
    assert (
        engine.remote_fill_producer_metrics_snapshot()["gauges"]["inflight_bytes"] == 0
    )


def test_one_produced_batch_uses_one_bounded_executor_task_for_all_splits(
    caplog,
) -> None:
    class _RecordingExecutor:
        def __init__(self) -> None:
            self.functions = []

        def submit(self, function: Any) -> Future:
            self.functions.append(function)
            return Future()

    control_pages = tuple(
        ControlPage(
            canonical_key=f"chunk-{chunk}-group-{group}",
            kv_group=group,
            chunk_index=chunk,
            chunk_start=chunk * 1024,
            chunk_end=(chunk + 1) * 1024,
            valid_tokens=1024,
            destination_tp_rank=0,
            expected_bytes=5,
            layer_count=79,
            layout_tag="payload-v3",
        )
        for chunk in range(3)
        for group in (0, 1)
    )
    owner = object()
    batch = _DirectPageBatch(
        req_id="request",
        keys=[],
        ptrs=[],
        sizes=[[10], [20]],
        owners=(owner,),
        ready_event=object(),
        group_ends={0: 3072, 1: 3072},
    )
    executor = _RecordingExecutor()
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(
        remote_fill_max_inflight_bytes=1000,
        remote_fill_max_bytes_per_request=1000,
        remote_fill_max_inflight_windows_per_request=1,
    )
    engine._remote_fill_producer_executor = executor
    engine._remote_fill_producer_metrics = RemoteFillProducerMetrics()
    engine._remote_fill_control_pages = lambda _batch: control_pages
    engine._remote_fill_pages_per_window = lambda: 2
    source_plan_calls = []
    engine._remote_fill_source_plan = lambda queued_batch: (
        source_plan_calls.append(queued_batch) or object()
    )
    state = _DirectStoreRequestState(remote_fill_handoff=_handoff())

    engine._schedule_remote_fill_batch(state, batch, 3072)

    assert len(executor.functions) == 1
    assert source_plan_calls == [batch]
    assert state.remote_fill_next_window_id == 3
    assert state.remote_fill_queued_windows == 1
    assert state.remote_fill_queued_bytes == 30

    # A logical batch over the configured producer cap is rejected before its
    # source plan or owner set is captured by another executor closure.
    engine._remote_fill_release_queue_capacity(state, 30)
    engine.config.remote_fill_max_bytes_per_request = 20
    rejected = _DirectStoreRequestState(remote_fill_handoff=_handoff())
    with caplog.at_level(logging.WARNING):
        engine._schedule_remote_fill_batch(rejected, batch, 3072)
    assert len(executor.functions) == 1
    assert source_plan_calls == [batch]
    assert rejected.remote_fill_disabled_reason == "producer_backpressure"
    assert '"code":"RF-P-004"' in caplog.text
    assert '"stage":"producer_admission"' in caplog.text


def test_direct_required_end_uses_authoritative_accepted_store_frontier() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    state = _DirectStoreRequestState(accepted_store_end=768)

    assert engine._direct_required_end(state, list(range(1024))) == 768

    with pytest.raises(RuntimeError, match="authoritative accepted end"):
        engine._direct_required_end(
            _DirectStoreRequestState(),
            list(range(1024)),
        )


def test_remote_fill_layout_is_bound_once_and_capability_is_request_scoped(
    monkeypatch,
) -> None:
    calls = {"layout_tag": 0, "layout": 0}
    layout = object()

    def payload_layout(*_args: Any) -> tuple[str, dict]:
        calls["layout_tag"] += 1
        return "payload-v3", {}

    def decoder_layout(*_args: Any, **_kwargs: Any) -> object:
        calls["layout"] += 1
        return layout

    monkeypatch.setattr(
        "lmcache_ascend.v1.cache_engine.mooncake_payload_layout",
        payload_layout,
    )
    monkeypatch.setattr(
        "lmcache_ascend.v1.cache_engine.build_decoder_layout",
        decoder_layout,
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine._engine_state_lock = RLock()
    engine.config = object()
    engine.metadata = object()
    engine.num_layers = 79

    assert engine._remote_fill_immutable_layout() == ("payload-v3", layout)
    assert engine._remote_fill_immutable_layout() == ("payload-v3", layout)
    assert _handoff().descriptor_verification_key == _SECRET
    assert _handoff(
        descriptor_verification_capability=(b"d" * 32).hex()
    ).descriptor_verification_key == b"d" * 32
    assert calls == {"layout_tag": 1, "layout": 1}


def test_direct_page_batch_retains_same_owner_and_all_producer_events() -> None:
    class _Key:
        kv_group = 0

        @staticmethod
        def to_string() -> str:
            return "page-key"

    class _Storage:
        def __init__(self) -> None:
            self.args = None
            self.future: Future = Future()

        def batched_put_external_pages(self, *args: Any) -> Future:
            self.args = args
            return self.future

    owner = object()
    event = object()
    second_event = object()
    batch = _DirectPageBatch(
        req_id="request",
        keys=[_Key()],
        ptrs=[[100]],
        sizes=[[64]],
        owners=(owner,),
        ready_event=event,
        group_ends={0: 1024},
        ranges=((0, 1024),),
        ready_events=(event, second_event),
    )
    storage = _Storage()
    engine = object.__new__(AscendLMCacheEngine)
    engine.storage_manager = storage
    engine._store_queue_maxsize = 0
    engine._direct_store_jobs = deque()
    engine._direct_store_states = {"request": _DirectStoreRequestState()}
    engine._pending_store_reqs = {}
    engine._store_cv = Condition()
    engine._direct_completed_futures = set()

    persistent_future = engine._submit_direct_page_batch(batch)
    direct_plan = engine._remote_fill_source_plan(batch)

    assert persistent_future is storage.future
    assert storage.args[3] is batch.owners
    assert storage.args[4] is event
    assert direct_plan.owners is batch.owners
    assert direct_plan.producer_events == (event, second_event)
    storage.future.set_result(None)


def test_persistent_and_direct_barriers_precede_finish_and_terminal_export() -> None:
    order: list[str] = []

    class _RecordingFuture:
        def __init__(self, name: str) -> None:
            self.name = name

        def result(self, timeout: float | None = None) -> None:
            del timeout
            order.append(self.name)

        @staticmethod
        def cancelled() -> bool:
            return False

        @staticmethod
        def exception() -> None:
            return None

    class _Session:
        direct_viable = True

        def finish(self, **kwargs: Any) -> RemoteFillTerminalResult:
            assert kwargs["persistent_common_end"] == 1024
            order.append("finish")
            return RemoteFillTerminalResult(
                transfer_id="transfer-1",
                outcome="LOCAL_FULL",
                persistent_common_end=1024,
                required_store_end=1024,
            )

    state = _DirectStoreRequestState(
        futures=deque([_RecordingFuture("persistent")]),
        submitted_end={0: 1024, 1: 1024},
        committed_end={0: 0, 1: 0},
        remote_fill_handoff=_handoff(),
        remote_fill_session=_Session(),
        remote_fill_futures=deque([_RecordingFuture("direct")]),
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(blocking_timeout_secs=1, chunk_size=1024)
    engine._direct_store_states = {"request": state}
    engine._direct_retry_args = {}

    engine.wait_for_direct_stores(("request",))
    engine._finish_remote_fill("request", state, 1024)
    exported = engine.drain_remote_fill_terminal_results()

    assert order == ["persistent", "direct", "finish"]
    assert exported["request"]["outcome"] == "LOCAL_FULL"


def test_ambiguous_armed_window_blocks_finalize_and_release() -> None:
    client = _ScriptedClient(
        reserve_dispositions=(PageDisposition.ALLOCATED, PageDisposition.ALLOCATED),
        fail_arm=True,
    )
    session = _session(client)
    with pytest.raises(RemoteFillFatalError):
        session.transfer_window(
            window_id=0,
            source_generation=44,
            control_pages=_pages(),
            source_plan=_source_plan(),
            submitter=lambda **_kwargs: pytest.fail("native submit must not run"),
            activation_factory=lambda attempt: SimpleNamespace(attempt=attempt),
        )
    state = _DirectStoreRequestState(
        accepted_store_end=1024,
        submitted_end={0: 1024, 1: 1024},
        committed_end={0: 1024, 1: 1024},
        remote_fill_handoff=_handoff(),
        remote_fill_session=session,
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(save_unfull_chunk=True, chunk_size=1024)
    engine._direct_store_states = {"request": state}
    engine._live_source_builders = {}
    engine._completed_live_sources = {}

    with pytest.raises(RuntimeError, match="FATAL_RESTART"):
        engine._finalize_direct_store(
            "request", list(range(1024)), (0, 1), state, final=True
        )
    with pytest.raises(RemoteFillFatalError):
        engine.drop_direct_store_states(("request",))


@pytest.mark.parametrize(
    ("token_end", "slot_mapping_base", "chunk_size", "expected"),
    [
        (0, 0, 1024, False),
        (18878, 18877, 1024, False),
        (18878, 18432, 1024, True),
        (18878, 18000, 1024, True),
        (18878, 0, 0, False),
    ],
)
def test_remote_fill_requires_one_complete_source_page(
    token_end: int,
    slot_mapping_base: int,
    chunk_size: int,
    expected: bool,
) -> None:
    assert (
        AscendLMCacheEngine._remote_fill_has_addressable_source_page(
            token_end,
            slot_mapping_base,
            chunk_size,
        )
        is expected
    )
