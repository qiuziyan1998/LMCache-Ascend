# SPDX-License-Identifier: Apache-2.0
"""Producer admission and ordered remote-fill jobs, independent of the engine.

Owns executor, session cache, circuit state and queue accounting. Request fields
are borrowed during the R5b transition; persistent preparation remains external.
Caller schedules; per-request chained jobs mutate session outcomes; accounting
uses the queue condition and circuit lock. Close drains before closing sessions.
The fatal reporter is weak and invoked only on fatal paths, preserving immediate
worker unhealthy marking without retaining its engine.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import logging
from concurrent.futures import Future
import os
import secrets
import threading
import time
from typing import Any, Callable, Optional, TYPE_CHECKING
from weakref import WeakMethod

from lmcache.v1.remote_fill import (
    ControlPage,
    OperationKind,
    log_remote_fill_diagnostic,
)
from lmcache.v1.remote_fill.native import (
    DIRECT_PUSH_H0_QUALIFICATION_V1,
    NativeDirectPushActivation,
)
from lmcache.v1.serving_perf import serving_perf_enabled, serving_perf_log
from lmcache_ascend.v1.direct_store_plan import (
    DirectPageBatch,
    build_remote_fill_source_plan,
    select_remote_fill_batch_pages,
)
from lmcache_ascend.v1.remote_fill import (
    build_remote_fill_protocol_limits,
    remote_fill_token_hash_identity,
)
from lmcache_ascend.v1.remote_fill_producer import (
    RemoteFillFatalError,
    RemoteFillHandoff,
    RemoteFillNegotiationCache,
    RemoteFillProducerMetrics,
    RemoteFillProducerSession,
    RemoteFillStaticSpec,
    create_remote_fill_client,
    parse_remote_fill_handoff,
)

if TYPE_CHECKING:
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache_ascend.v1.remote_fill import RemoteFillDecoderLayout


# Preserve diagnostic logger identity across extraction.
logger = logging.getLogger("lmcache_ascend.v1.cache_engine")


@dataclass(frozen=True, slots=True)
class ProducerSessionContext:
    layout: RemoteFillDecoderLayout
    chunk_hash_type: type[int] | type[bytes]
    chunk_hash_bytes: int | None
    tp_size: int
    shared_group1: bool


class RemoteFillCoordinator:
    def __init__(
        self,
        *,
        config: LMCacheEngineConfig,
        tp_size: int,
        storage_manager: StorageManager,
        fatal_reporter: Callable[[tuple[str, ...]], None],
    ) -> None:
        # Borrow the validated worker configuration. Production treats it as
        # immutable after startup; do not shadow values or change cast timing.
        self.config = config
        self.tp_size = tp_size
        self.storage_manager = storage_manager
        self._fatal_reporter = WeakMethod(fatal_reporter)
        self._session_lock = threading.RLock()

    def configure(
        self,
        *,
        client_factory: Callable[..., Any] | None = None,
        direct_submitter: Callable[..., Future] | None = None,
        activation_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """Install the existing optional client/native test boundaries."""
        if client_factory is not None:
            self._client_factory = client_factory
        if direct_submitter is not None:
            self._direct_submitter = direct_submitter
        if activation_factory is not None:
            self._activation_factory = activation_factory

    def reject_source_batch(self, state: Any, *, req_id: str, error: Exception) -> None:
        """Record source-planning fallback before producer admission."""
        state.remote_fill_disabled_reason = type(error).__name__
        self._log_prearm_failure(
            state,
            req_id=req_id,
            stage="control_page_planning",
            reason=state.remote_fill_disabled_reason,
            error=error,
        )
        self._record_failure()

    def _latch_fatal(self, state) -> None:
        handoff = state.remote_fill_handoff
        if handoff is None:
            raise RuntimeError("remote-fill fatal state lacks a transfer identity")
        reporter = self._fatal_reporter()
        if reporter is not None:
            reporter((handoff.transfer_id,))

    def close(self, states) -> None:
        """Drain producer work and close lazily created control resources."""

        deadline = time.perf_counter() + (
            float(self.config.remote_fill_native_hard_timeout_ms) / 1000.0
        )
        for state in states.values():
            while state.remote_fill_futures:
                future = state.remote_fill_futures.popleft()
                try:
                    future.result(timeout=max(0.0, deadline - time.perf_counter()))
                except FutureTimeoutError as error:
                    self._latch_fatal(state)
                    raise RemoteFillFatalError(
                        "remote-fill producer shutdown exceeded its hard deadline"
                    ) from error
            if (
                state.remote_fill_terminal is not None
                and state.remote_fill_terminal.outcome == "FATAL_RESTART"
            ) or getattr(state.remote_fill_session, "fatal_restart_required", False):
                self._latch_fatal(state)
                raise RemoteFillFatalError(
                    "cannot close fatal remote-fill producer state"
                )
            if state.remote_fill_session is not None:
                state.remote_fill_session.close()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return enabled-path producer aggregates without request identities."""

        metrics = getattr(self, "_metrics", None)
        return {} if metrics is None else metrics.snapshot()

    def get_metrics(self) -> RemoteFillProducerMetrics:
        metrics = getattr(self, "_metrics", None)
        if metrics is None:
            metrics = RemoteFillProducerMetrics()
            self._metrics = metrics
        return metrics

    @staticmethod
    def _log_prearm_failure(
        state: Any,
        *,
        req_id: str,
        stage: str,
        reason: str,
        error: Optional[BaseException] = None,
    ) -> None:
        """Make a safe direct-fill fallback immediately operator-visible."""

        handoff = state.remote_fill_handoff
        log_remote_fill_diagnostic(
            logger,
            event="remote_fill_fallback",
            code="RF-P-004",
            stage=stage,
            action="PERSISTENT_ONLY",
            req_id=req_id,
            transfer_id=handoff.transfer_id if handoff is not None else None,
            reason=reason,
            error=error,
            severity="warning",
        )

    def _record_window_metrics(
        self,
        state: Any,
        result: Any,
    ) -> None:
        metrics = self.get_metrics()
        if metrics.timing_enabled:
            metrics.observe("reserve_seconds", float(result.reserve_seconds))
            metrics.observe("arm_seconds", float(result.arm_seconds))
        metrics.existing_pages(int(result.existing_pages))
        submitted_bytes = int(result.submitted_bytes)
        if submitted_bytes:
            if metrics.timing_enabled:
                metrics.observe(
                    "source_event_wait_seconds",
                    float(result.source_event_wait_seconds),
                )
                metrics.observe(
                    "source_registration_seconds",
                    float(result.source_registration_seconds),
                )
                metrics.observe(
                    "native_slot_wait_seconds",
                    float(result.native_slot_wait_seconds),
                )
                metrics.observe("native_seconds", float(result.native_seconds))
                metrics.observe("report_seconds", float(result.report_seconds))
            metrics.add_bytes("submitted_bytes", submitted_bytes)
            state.remote_fill_submitted_bytes += submitted_bytes
        if not result.direct_satisfied:
            allocation = result.reason in (
                "direct destination hole",
                "reservation rejected",
                "producer backpressure",
            )
            metrics.abandon(result.reason, allocation=allocation)
            if result.armed or result.reason == "native transfer failed":
                metrics.failure(result.reason)
            if serving_perf_enabled():
                serving_perf_log(
                    logger,
                    "remote_fill_direct_abandoned",
                    reason=result.reason,
                    armed=bool(result.armed),
                    fatal_restart_required=bool(result.fatal_restart_required),
                )
            fatal = bool(result.fatal_restart_required)
            handoff = state.remote_fill_handoff
            log_remote_fill_diagnostic(
                logger,
                event=(
                    "remote_fill_fatal_restart" if fatal else "remote_fill_fallback"
                ),
                code="RF-P-900" if fatal else "RF-P-004",
                stage="window_terminal",
                action=("PAIRED_RESTART_REQUIRED" if fatal else "PERSISTENT_ONLY"),
                transfer_id=(handoff.transfer_id if handoff is not None else None),
                reason=str(result.reason),
                memory_safety_uncertain=fatal,
                severity="critical" if fatal else "warning",
            )

    def prepare_request(
        self,
        req_id: str,
        request_configs: Optional[dict],
        state: Any,
    ) -> bool:
        if not bool(getattr(self.config, "enable_remote_lmcache_store", False)):
            return False
        if state.remote_fill_handoff is not None:
            return not state.remote_fill_disabled_reason
        try:
            handoff = parse_remote_fill_handoff(request_configs)
        except ValueError as error:
            state.remote_fill_disabled_reason = type(error).__name__
            log_remote_fill_diagnostic(
                logger,
                event="remote_fill_handoff_rejected",
                code="RF-P-001",
                stage="handoff_validation",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                reason="malformed handoff",
                error=error,
                severity="warning",
            )
            return False
        if handoff is None:
            state.remote_fill_disabled_reason = "missing_handoff"
            return False
        state.remote_fill_handoff = handoff
        metrics = self.get_metrics()
        if not state.remote_fill_metrics_started:
            metrics.start_attempt()
            state.remote_fill_metrics_started = True
        if not handoff.global_te_push:
            state.remote_fill_disabled_reason = "native_not_qualified"
            log_remote_fill_diagnostic(
                logger,
                event="remote_fill_handoff_rejected",
                code="RF-P-002",
                stage="handoff_validation",
                action="LEGACY_PATH",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        if handoff.destination_dp_rank >= handoff.destination_dp_size:
            state.remote_fill_disabled_reason = "invalid_dp_mapping"
            log_remote_fill_diagnostic(
                logger,
                event="remote_fill_handoff_rejected",
                code="RF-P-002",
                stage="handoff_validation",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        if handoff.destination_tp_size != self.tp_size:
            state.remote_fill_disabled_reason = "incompatible_tp_mapping"
            log_remote_fill_diagnostic(
                logger,
                event="remote_fill_handoff_rejected",
                code="RF-P-002",
                stage="handoff_validation",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        if not self._circuit_allows():
            state.remote_fill_disabled_reason = "circuit_open"
            log_remote_fill_diagnostic(
                logger,
                event="remote_fill_fallback",
                code="RF-P-003",
                stage="producer_admission",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        state.remote_fill_source_generation = secrets.randbits(63) or 1
        metrics.add_gauge("direct_viable", 1)
        state.remote_fill_viable_counted = True
        if serving_perf_enabled():
            serving_perf_log(
                logger,
                "remote_fill_producer_decision",
                req_id=req_id,
                enabled=True,
                destination_dp_rank=handoff.destination_dp_rank,
                destination_tp_size=handoff.destination_tp_size,
            )
        return True

    def _circuit_allows(self) -> bool:
        if not bool(self.config.remote_fill_circuit_breaker_enabled):
            return True
        lock = getattr(self, "_circuit_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._circuit_lock = lock
        with lock:
            now = time.monotonic()
            open_until = float(getattr(self, "_circuit_open_until", 0.0))
            if now < open_until:
                return False
            if open_until:
                self._circuit_open_until = 0.0
                self._circuit_failures = 0
                metrics = getattr(self, "_metrics", None)
                if metrics is not None:
                    metrics.set_gauge("circuit_breaker_state", 0)
            return True

    def _record_failure(self) -> None:
        if not bool(self.config.remote_fill_circuit_breaker_enabled):
            return
        lock = getattr(self, "_circuit_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._circuit_lock = lock
        with lock:
            failures = int(getattr(self, "_circuit_failures", 0)) + 1
            self._circuit_failures = failures
            if failures >= int(
                self.config.remote_fill_circuit_breaker_failure_threshold
            ):
                self._circuit_open_until = time.monotonic() + float(
                    self.config.remote_fill_circuit_breaker_cooldown_sec
                )
                self.get_metrics().set_gauge("circuit_breaker_state", 1)

    def record_success(self) -> None:
        lock = getattr(self, "_circuit_lock", None)
        if lock is None:
            return
        with lock:
            self._circuit_failures = 0
            self._circuit_open_until = 0.0
            self.get_metrics().set_gauge("circuit_breaker_state", 0)

    def _acquire_queue_capacity(
        self,
        state: Any,
        byte_count: int,
    ) -> bool:
        """Acquire bounded producer admission before retaining queued work."""

        if byte_count < 0 or byte_count > int(
            self.config.remote_fill_max_inflight_bytes
        ):
            return False
        condition = getattr(self, "_queue_condition", None)
        if condition is None:
            condition = threading.Condition()
            self._queue_condition = condition
        with condition:
            request_byte_limit = min(
                int(self.config.remote_fill_max_bytes_per_request),
                int(self.config.remote_fill_max_inflight_bytes)
                * int(self.config.remote_fill_max_inflight_windows_per_request),
            )
            # Producer source retention is independent of the decoder's
            # hidden LocalCPU reservation budget. Bound all requests by the
            # dedicated producer inflight-byte cap.
            global_byte_limit = int(self.config.remote_fill_max_inflight_bytes)
            if byte_count > request_byte_limit:
                return False
            global_bytes = int(getattr(self, "_queued_bytes", 0))
            if (
                state.remote_fill_queued_windows
                >= int(self.config.remote_fill_max_inflight_windows_per_request)
                or state.remote_fill_queued_bytes + byte_count > request_byte_limit
                or global_bytes + byte_count > global_byte_limit
            ):
                return False
            state.remote_fill_queued_windows += 1
            state.remote_fill_queued_bytes += byte_count
            metrics = self.get_metrics()
            if state.remote_fill_queued_windows == 1 and metrics.timing_enabled:
                state.remote_fill_oldest_enqueued_at = time.perf_counter()
            self._queued_bytes = global_bytes + byte_count
            metrics.add_gauge("inflight_windows", 1)
            metrics.add_gauge("inflight_bytes", byte_count)
            return True

    def _release_queue_capacity(
        self,
        state: Any,
        byte_count: int,
    ) -> None:
        condition = getattr(self, "_queue_condition", None)
        if condition is None:
            return
        with condition:
            state.remote_fill_queued_windows = max(
                0, state.remote_fill_queued_windows - 1
            )
            state.remote_fill_queued_bytes = max(
                0, state.remote_fill_queued_bytes - byte_count
            )
            if state.remote_fill_queued_windows == 0:
                state.remote_fill_oldest_enqueued_at = 0.0
            self._queued_bytes = max(
                0,
                int(getattr(self, "_queued_bytes", 0)) - byte_count,
            )
            metrics = self.get_metrics()
            metrics.add_gauge("inflight_windows", -1)
            metrics.add_gauge("inflight_bytes", -byte_count)

    def _static_spec(
        self,
        handoff: RemoteFillHandoff,
        context: ProducerSessionContext,
    ) -> RemoteFillStaticSpec:
        layout = context.layout
        configured_hash_algorithm = str(self.config.pre_caching_hash_algorithm)
        token_hash_algorithm = remote_fill_token_hash_identity(
            configured_hash_algorithm,
            context.chunk_hash_type,
            context.chunk_hash_bytes,
        )
        python_hash_seed = (
            os.environ.get("PYTHONHASHSEED", "")
            if configured_hash_algorithm == "builtin"
            else ""
        )
        if (
            token_hash_algorithm != handoff.token_hash_algorithm
            or python_hash_seed != handoff.python_hash_seed
        ):
            raise ValueError("remote-fill token hashing identity changed")
        return RemoteFillStaticSpec(
            cache_namespace_tag=layout.cache_namespace_tag,
            layout_tag=layout.layout_tag,
            model_artifact_id=layout.model_artifact_id,
            chunk_size=layout.chunk_size,
            model_layout="mla-dsa-layer-page-v3",
            group_dimensions=layout.group_dimensions,
            layer_count=layout.num_layers,
            save_only_first_rank=True,
            shared_group1=(context.shared_group1),
            tp_size=context.tp_size,
            dp_size=handoff.destination_dp_size,
            global_te_push=handoff.global_te_push,
            token_hash_algorithm=token_hash_algorithm,
            python_hash_seed=python_hash_seed,
        )

    def _create_session(
        self,
        req_id: str,
        state: Any,
        required_store_end_hint: int,
        context: ProducerSessionContext,
    ) -> RemoteFillProducerSession:
        handoff = state.remote_fill_handoff
        if handoff is None:
            raise RuntimeError("remote-fill handoff is unavailable")
        limits = build_remote_fill_protocol_limits(self.config)
        verification_key = handoff.descriptor_verification_key
        static_spec = self._static_spec(handoff, context)
        factory = getattr(self, "_client_factory", None)
        client = (
            factory(handoff, limits)
            if callable(factory)
            else create_remote_fill_client(
                handoff.control_endpoint,
                operation_timeouts_ms={
                    OperationKind.NEGOTIATE: int(
                        self.config.remote_fill_open_timeout_ms
                    ),
                    OperationKind.OPEN: int(self.config.remote_fill_open_timeout_ms),
                    OperationKind.RESERVE_WINDOW: int(
                        self.config.remote_fill_reserve_timeout_ms
                    ),
                    OperationKind.ARM_WINDOW: int(
                        self.config.remote_fill_arm_timeout_ms
                    ),
                    OperationKind.REPORT_TRANSFER_COMPLETE: int(
                        self.config.remote_fill_reserve_timeout_ms
                    ),
                    OperationKind.STATUS: int(
                        self.config.remote_fill_reserve_timeout_ms
                    ),
                    OperationKind.ABORT: int(self.config.remote_fill_arm_timeout_ms),
                    OperationKind.FINISH: int(
                        self.config.remote_fill_finish_timeout_ms
                    ),
                },
                limits=limits,
            )
        )
        planned_windows = (
            required_store_end_hint + int(self.config.remote_fill_window_tokens) - 1
        ) // int(self.config.remote_fill_window_tokens)
        negotiation_cache = getattr(self, "_negotiation_cache", None)
        if negotiation_cache is None:
            with self._session_lock:
                negotiation_cache = getattr(self, "_negotiation_cache", None)
                if negotiation_cache is None:
                    negotiation_cache = RemoteFillNegotiationCache()
                    self._negotiation_cache = negotiation_cache
        session = RemoteFillProducerSession(
            request_id=req_id,
            handoff=handoff,
            static_spec=static_spec,
            client=client,
            secret=verification_key,
            planned_window_count_hint=planned_windows,
            required_store_end_hint=required_store_end_hint,
            native_hard_timeout_seconds=(
                float(self.config.remote_fill_native_hard_timeout_ms) / 1000.0
            ),
            negotiation_cache=negotiation_cache,
        )
        if not state.remote_fill_active_counted:
            self.get_metrics().add_gauge("active_transactions", 1)
            state.remote_fill_active_counted = True
        return session

    def submit_probe(
        self,
        req_id: str,
        state: Any,
        pages: tuple[ControlPage, ...],
        required_store_end_hint: int,
        *,
        maximum: int,
        context: ProducerSessionContext,
    ) -> None:
        """Queue bounded, ordered cached-prefix probes before direct writes."""

        if not pages or state.remote_fill_handoff is None:
            return
        if maximum <= 0:
            state.remote_fill_disabled_reason = "invalid_control_page_limit"
            self.get_metrics().abandon("invalid control page limit", allocation=True)
            return
        executor = getattr(self, "_executor", None)
        if executor is None:
            try:
                executor = ThreadPoolExecutor(
                    max_workers=int(self.config.remote_fill_direct_worker_count),
                    thread_name_prefix="lmcache-remote-fill-producer",
                )
            except Exception as error:
                state.remote_fill_disabled_reason = type(error).__name__
                self._record_failure()
                return
            self._executor = executor
        control_windows = tuple(
            pages[offset : offset + maximum] for offset in range(0, len(pages), maximum)
        )
        metrics = self.get_metrics()
        queued_at = time.perf_counter() if metrics.timing_enabled else 0.0
        if not self._acquire_queue_capacity(state, 0):
            state.remote_fill_disabled_reason = "producer_backpressure"
            metrics.abandon("producer backpressure", allocation=True)
            return
        if metrics.timing_enabled:
            metrics.observe("queue_wait_seconds", time.perf_counter() - queued_at)
        first_window_id = state.remote_fill_next_window_id
        state.remote_fill_next_window_id += len(control_windows)
        previous = state.remote_fill_last_future

        def run() -> Any:
            try:
                if previous is not None:
                    previous.result()
                if state.remote_fill_session is None:
                    state.remote_fill_session = self._create_session(
                        req_id,
                        state,
                        required_store_end_hint,
                        context,
                    )
                session = state.remote_fill_session
                if not session.direct_viable:
                    return None
                results = []
                for window_offset, control_pages in enumerate(control_windows):
                    window_id = first_window_id + window_offset
                    result = session.probe_window(
                        window_id=window_id,
                        source_generation=state.remote_fill_source_generation,
                        control_pages=control_pages,
                    )
                    results.append(result)
                    if not result.direct_satisfied:
                        state.remote_fill_disabled_reason = result.reason
                        if result.reason != "cached-prefix hole":
                            self._record_failure()
                    self._record_window_metrics(state, result)
                    if serving_perf_enabled():
                        serving_perf_log(
                            logger,
                            "remote_fill_probe_complete",
                            req_id=session.request_id,
                            window_id=window_id,
                            page_count=len(control_pages),
                            direct_satisfied=result.direct_satisfied,
                            reason=result.reason,
                        )
                    if not result.direct_satisfied:
                        break
                return tuple(results)
            except RemoteFillFatalError:
                self._latch_fatal(state)
                raise
            except Exception as error:
                state.remote_fill_disabled_reason = type(error).__name__
                if state.remote_fill_session is not None:
                    state.remote_fill_session.direct_viable = False
                self._record_failure()
                return None
            finally:
                self._release_queue_capacity(state, 0)

        try:
            future = executor.submit(run)
        except Exception as error:
            self._release_queue_capacity(state, 0)
            state.remote_fill_disabled_reason = type(error).__name__
            self._record_failure()
            return
        state.remote_fill_futures.append(future)
        state.remote_fill_last_future = future

    def submit_batch(
        self,
        state: Any,
        batch: DirectPageBatch,
        required_store_end_hint: int,
        *,
        control_pages: tuple[ControlPage, ...],
        maximum: int,
        context: ProducerSessionContext,
    ) -> None:
        if maximum <= 0:
            state.remote_fill_disabled_reason = "invalid_control_page_limit"
            self._log_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="control_page_planning",
                reason=state.remote_fill_disabled_reason,
            )
            self.get_metrics().abandon("invalid control page limit", allocation=True)
            return
        control_windows = tuple(
            control_pages[offset : offset + maximum]
            for offset in range(0, len(control_pages), maximum)
        )
        # Every window plan retains the batch-wide owners/events, so its one
        # producer job must be charged for the complete retained source set.
        byte_count = sum(sum(page_sizes) for page_sizes in batch.sizes)
        request_oversized = byte_count > int(
            self.config.remote_fill_max_bytes_per_request
        )
        metrics = self.get_metrics()
        queued_at = time.perf_counter() if metrics.timing_enabled else 0.0
        if request_oversized or not self._acquire_queue_capacity(state, byte_count):
            state.remote_fill_disabled_reason = "producer_backpressure"
            self._log_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="producer_admission",
                reason=state.remote_fill_disabled_reason,
            )
            metrics = self.get_metrics()
            metrics.abandon("producer backpressure", allocation=True)
            if serving_perf_enabled():
                serving_perf_log(
                    logger,
                    "remote_fill_batch_skipped",
                    req_id=batch.req_id,
                    reason="producer_backpressure",
                    bytes=byte_count,
                )
            return
        if metrics.timing_enabled:
            metrics.observe("queue_wait_seconds", time.perf_counter() - queued_at)
        try:
            source_plans = tuple(
                build_remote_fill_source_plan(
                    batch
                    if len(control_windows) == 1
                    else select_remote_fill_batch_pages(batch, pages)
                )
                for pages in control_windows
            )
        except Exception as error:
            self._release_queue_capacity(state, byte_count)
            state.remote_fill_disabled_reason = type(error).__name__
            self._log_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="source_plan_validation",
                reason=state.remote_fill_disabled_reason,
                error=error,
            )
            self._record_failure()
            return
        if serving_perf_enabled():
            serving_perf_log(
                logger,
                "remote_fill_source_ready",
                req_id=batch.req_id,
                page_count=len(control_pages),
                bytes=byte_count,
            )
        executor = getattr(self, "_executor", None)
        if executor is None:
            try:
                executor = ThreadPoolExecutor(
                    max_workers=int(self.config.remote_fill_direct_worker_count),
                    thread_name_prefix="lmcache-remote-fill-producer",
                )
            except Exception as error:
                self._release_queue_capacity(state, byte_count)
                state.remote_fill_disabled_reason = type(error).__name__
                self._log_prearm_failure(
                    state,
                    req_id=batch.req_id,
                    stage="producer_executor_creation",
                    reason=state.remote_fill_disabled_reason,
                    error=error,
                )
                self._record_failure()
                return
            self._executor = executor
        first_window_id = state.remote_fill_next_window_id
        state.remote_fill_next_window_id += len(control_windows)
        source_generation = state.remote_fill_source_generation
        previous = state.remote_fill_last_future

        def run() -> Any:
            try:
                if previous is not None:
                    previous.result()
                if state.remote_fill_session is None:
                    state.remote_fill_session = self._create_session(
                        batch.req_id, state, required_store_end_hint, context
                    )
                session = state.remote_fill_session
                if not session.direct_viable:
                    return None
                submitter = getattr(
                    self,
                    "_direct_submitter",
                    self.storage_manager.submit_remote_fill_direct_push,
                )
                activation_factory = getattr(
                    self,
                    "_activation_factory",
                    lambda attempt_id: NativeDirectPushActivation(
                        h0_qualification=DIRECT_PUSH_H0_QUALIFICATION_V1,
                        native_transfer_attempt_id=attempt_id,
                        arm_acknowledged=True,
                    ),
                )
                results = []
                windows = zip(control_windows, source_plans, strict=True)
                for window_offset, (window_pages, source_plan) in enumerate(windows):
                    window_id = first_window_id + window_offset
                    chunk_start = min(page.chunk_start for page in window_pages)
                    chunk_end = max(page.chunk_end for page in window_pages)
                    window_tokens = chunk_end - chunk_start
                    result = session.transfer_window(
                        window_id=window_id,
                        source_generation=source_generation,
                        control_pages=window_pages,
                        source_plan=source_plan,
                        submitter=submitter,
                        activation_factory=activation_factory,
                        preparer=self.storage_manager.prepare_remote_fill_source,
                    )
                    results.append(result)
                    if not result.direct_satisfied:
                        state.remote_fill_disabled_reason = result.reason
                        if result.reason not in (
                            "cached-prefix hole",
                            "direct destination hole",
                        ):
                            self._record_failure()
                    self._record_window_metrics(state, result)
                    if serving_perf_enabled():
                        serving_perf_log(
                            logger,
                            "remote_fill_window_complete",
                            req_id=batch.req_id,
                            transfer_id=state.remote_fill_handoff.transfer_id,
                            window_id=window_id,
                            page_count=len(window_pages),
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            window_tokens=window_tokens,
                            full_window=(
                                window_tokens
                                == int(self.config.remote_fill_window_tokens)
                            ),
                            bytes=sum(page.expected_bytes for page in window_pages),
                            armed=result.armed,
                            direct_satisfied=result.direct_satisfied,
                            reason=result.reason,
                            reserve_ms=round(result.reserve_seconds * 1000, 3),
                            arm_ms=round(result.arm_seconds * 1000, 3),
                            source_event_wait_ms=round(
                                result.source_event_wait_seconds * 1000, 3
                            ),
                            source_fences_ready_monotonic_ms=round(
                                result.source_fences_ready_monotonic * 1000, 3
                            ),
                            source_registration_ms=round(
                                result.source_registration_seconds * 1000, 3
                            ),
                            native_slot_wait_ms=round(
                                result.native_slot_wait_seconds * 1000, 3
                            ),
                            native_ms=round(result.native_seconds * 1000, 3),
                            report_ms=round(result.report_seconds * 1000, 3),
                            native_started_monotonic_ms=round(
                                result.native_started_monotonic * 1000, 3
                            ),
                            native_ended_monotonic_ms=round(
                                result.native_ended_monotonic * 1000, 3
                            ),
                        )
                    if not result.direct_satisfied:
                        break
                return tuple(results)
            except Exception as error:
                if isinstance(error, RemoteFillFatalError):
                    self._latch_fatal(state)
                    metrics = self.get_metrics()
                    metrics.failure("fatal restart")
                    metrics.abandon("fatal restart")
                    if serving_perf_enabled():
                        serving_perf_log(
                            logger,
                            "remote_fill_fatal_restart",
                            req_id=batch.req_id,
                            phase="native_or_control_terminal",
                        )
                    raise
                state.remote_fill_disabled_reason = type(error).__name__
                if state.remote_fill_session is not None:
                    state.remote_fill_session.direct_viable = False
                self._log_prearm_failure(
                    state,
                    req_id=batch.req_id,
                    stage="producer_window_execution",
                    reason=state.remote_fill_disabled_reason,
                    error=error,
                )
                self._record_failure()
                metrics = self.get_metrics()
                metrics.failure("prearm failure")
                metrics.abandon("prearm failure")
                return None
            finally:
                self._release_queue_capacity(state, byte_count)

        try:
            future = executor.submit(run)
        except Exception as error:
            self._release_queue_capacity(state, byte_count)
            state.remote_fill_disabled_reason = type(error).__name__
            self._log_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="producer_executor_submission",
                reason=state.remote_fill_disabled_reason,
                error=error,
            )
            self._record_failure()
            return
        state.remote_fill_futures.append(future)
        state.remote_fill_last_future = future
