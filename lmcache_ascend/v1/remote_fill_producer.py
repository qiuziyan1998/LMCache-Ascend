# SPDX-License-Identifier: Apache-2.0
"""Prefiller-side control lifecycle for direct remote LMCache fill."""

# Standard
from collections.abc import Callable, Mapping
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4
import asyncio
import time

# Third Party
from lmcache.v1.remote_fill import (
    PROTOCOL_VERSION,
    AbortRequest,
    ArmWindowRequest,
    ControlPage,
    DestinationNativeState,
    NativeDirectPushPreSubmitError,
    NativeDirectPushResult,
    NativeDirectPushTerminalError,
    FinishRequest,
    NegotiateRequest,
    OpenRequest,
    OperationIdentity,
    OperationKind,
    PageDisposition,
    RemoteFillClient,
    RemoteFillResponse,
    ReportTransferCompleteRequest,
    ReplyLostError,
    ReserveWindowRequest,
    ResultCode,
    StatusRequest,
    TerminalOutcome,
    content_digest,
    destination_descriptor_digest,
    manifest_digest,
    transaction_manifest_digest,
    verify_descriptor,
)
from lmcache.v1.remote_fill import ProtocolLimits, RpcRemoteFillByteTransport
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
)
from lmcache.v1.remote_fill.native import DirectPushSourcePlan


REMOTE_FILL_REQUEST_CONFIG_KEY = "lmcache.remote_fill"
REMOTE_FILL_H0_QUALIFICATION_ENV = "LMCACHE_REMOTE_FILL_H0_QUALIFICATION"
_DESCRIPTOR_VERIFICATION_CAPABILITY_BYTES = 32


class DirectPushSubmitter(Protocol):
    """Callable that starts one already-armed native destination write."""

    def __call__(
        self,
        *,
        remote_session: str,
        source_plan: Any,
        destination_descriptors: tuple[Any, ...],
        activation: Any,
    ) -> Future:
        """Return a future for the terminal aggregate native result."""


@dataclass(frozen=True, slots=True)
class RemoteFillHandoff:
    """Pointer-free destination selection supplied by the proxy."""

    transfer_id: str
    request_attempt: int
    source_engine_id: str
    destination_engine_id: str
    destination_engine_epoch: int
    control_endpoint: str
    destination_dp_rank: int
    shared_cache_generation: int
    destination_tp_size: int
    destination_dp_size: int
    global_te_push: bool
    token_hash_algorithm: str
    python_hash_seed: str
    descriptor_verification_capability: str = field(repr=False)

    @property
    def descriptor_verification_key(self) -> bytes:
        """Return the validated decoder-incarnation descriptor HMAC key."""

        return bytes.fromhex(self.descriptor_verification_capability)


@dataclass(frozen=True, slots=True)
class RemoteFillStaticSpec:
    """Startup-computed layout values sent during negotiation."""

    cache_namespace_tag: str
    layout_tag: str
    model_artifact_id: str
    chunk_size: int
    model_layout: str
    group_dimensions: tuple[int, ...]
    layer_count: int
    save_only_first_rank: bool
    shared_group1: bool
    tp_size: int
    dp_size: int
    global_te_push: bool
    token_hash_algorithm: str
    python_hash_seed: str


class RemoteFillNegotiationCache:
    """Process-local cache for immutable decoder-epoch negotiation."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries <= 0:
            raise ValueError("negotiation cache bound must be positive")
        self._lock = Lock()
        self._max_entries = max_entries
        self._accepted: set[tuple[Any, ...]] = set()

    @staticmethod
    def key(
        handoff: RemoteFillHandoff,
        spec: RemoteFillStaticSpec,
    ) -> tuple[Any, ...]:
        return (
            handoff.control_endpoint,
            handoff.destination_engine_id,
            handoff.destination_engine_epoch,
            handoff.shared_cache_generation,
            handoff.destination_dp_rank,
            spec,
        )

    def contains(self, key: tuple[Any, ...]) -> bool:
        with self._lock:
            return key in self._accepted

    def record(self, key: tuple[Any, ...]) -> None:
        with self._lock:
            if key not in self._accepted and len(self._accepted) >= self._max_entries:
                self._accepted.pop()
            self._accepted.add(key)


@dataclass(frozen=True, slots=True)
class RemoteFillWindowResult:
    """Terminal producer knowledge for one bounded control window."""

    window_id: int
    direct_satisfied: bool
    armed: bool
    fatal_restart_required: bool = False
    reason: str = ""
    reserve_seconds: float = 0.0
    arm_seconds: float = 0.0
    source_event_wait_seconds: float = 0.0
    source_fences_ready_monotonic: float = 0.0
    source_registration_seconds: float = 0.0
    native_slot_wait_seconds: float = 0.0
    native_seconds: float = 0.0
    report_seconds: float = 0.0
    native_started_monotonic: float = 0.0
    native_ended_monotonic: float = 0.0
    submitted_bytes: int = 0
    existing_pages: int = 0


@dataclass(frozen=True, slots=True)
class RemoteFillTerminalResult:
    """Sanitized request-level outcome safe to return through the proxy."""

    transfer_id: str
    outcome: str
    persistent_common_end: int
    required_store_end: int

    def as_dict(self) -> dict[str, str | int]:
        """Return the pointer-free proxy handoff representation."""

        return {
            "transfer_id": self.transfer_id,
            "outcome": self.outcome,
            "persistent_common_end": self.persistent_common_end,
            "required_store_end": self.required_store_end,
        }


class RemoteFillFatalError(RuntimeError):
    """An armed operation no longer has a safely known terminal state."""


class RemoteFillProducerMetrics:
    """Enabled-only, low-cardinality producer aggregate metrics.

    The object is created lazily after a request has supplied a valid remote-
    fill handoff. It intentionally accepts only fixed metric names and bounded
    outcome/reason values; request IDs, transfer IDs, keys, and pointers never
    enter the aggregate.
    """

    _TIMERS = frozenset(
        {
            "queue_wait_seconds",
            "reserve_seconds",
            "arm_seconds",
            "source_event_wait_seconds",
            "source_registration_seconds",
            "native_slot_wait_seconds",
            "native_seconds",
            "report_seconds",
            "persistent_seconds",
            "finish_control_seconds",
            "final_wait_seconds",
        }
    )
    _GAUGES = frozenset(
        {
            "active_transactions",
            "direct_viable",
            "inflight_windows",
            "inflight_bytes",
            "circuit_breaker_state",
        }
    )
    _BYTE_KINDS = frozenset({"submitted_bytes", "published_bytes", "discarded_bytes"})
    _REASONS = frozenset(
        {
            "none",
            "open_rejected",
            "cached_prefix_hole",
            "direct_destination_hole",
            "probe_control_unavailable",
            "probe_result_invalid",
            "reserve_control_unavailable",
            "reservation_rejected",
            "reservation_result_invalid",
            "descriptor_validation_failed",
            "arm_rejected",
            "native_transfer_failed",
            "producer_backpressure",
            "invalid_control_page_limit",
            "incomplete_page_pair",
            "prearm_failure",
            "fatal_restart",
            "request_aborted",
            "other",
        }
    )
    _OUTCOMES = frozenset(
        {
            "LOCAL_FULL",
            "PERSISTENT_ONLY",
            "PERSISTENCE_FAILED",
            "FATAL_RESTART",
            "ABORTED",
        }
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._attempts: dict[tuple[str, str], int] = {}
        self._started_total = 0
        self._abandoned: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._allocation_rejections: dict[str, int] = {}
        self._timers: dict[str, list[float | int]] = {
            name: [0, 0.0, 0.0] for name in self._TIMERS
        }
        self._gauges = {name: 0 for name in self._GAUGES}
        self._bytes = {name: 0 for name in self._BYTE_KINDS}
        self._same_key_existing_total = 0
        self._fatal_restart_total = 0

    @classmethod
    def _reason(cls, reason: str) -> str:
        normalized = reason.strip().lower().replace(" ", "_")
        return normalized if normalized in cls._REASONS else "other"

    def start_attempt(self) -> None:
        """Count one producer attempt after RemoteFill negotiation succeeds."""
        with self._lock:
            self._started_total += 1

    def finish_attempt(self, outcome: str, reason: str = "none") -> None:
        """Count a terminal producer outcome using bounded label values."""
        bounded_outcome = outcome if outcome in self._OUTCOMES else "OTHER"
        bounded_reason = self._reason(reason)
        with self._lock:
            key = (bounded_outcome, bounded_reason)
            self._attempts[key] = self._attempts.get(key, 0) + 1
            if bounded_outcome == "FATAL_RESTART":
                self._fatal_restart_total += 1

    def abandon(self, reason: str, *, allocation: bool = False) -> None:
        """Count a direct-path abandonment and optional allocation rejection."""
        bounded = self._reason(reason)
        with self._lock:
            self._abandoned[bounded] = self._abandoned.get(bounded, 0) + 1
            if allocation:
                self._allocation_rejections[bounded] = (
                    self._allocation_rejections.get(bounded, 0) + 1
                )

    def failure(self, reason: str) -> None:
        """Count a transfer failure using a bounded reason label."""
        bounded = self._reason(reason)
        with self._lock:
            self._failures[bounded] = self._failures.get(bounded, 0) + 1

    def observe(self, name: str, seconds: float) -> None:
        """Record a nonnegative duration for a predefined timer."""
        if name not in self._TIMERS or seconds < 0:
            return
        with self._lock:
            values = self._timers[name]
            values[0] = int(values[0]) + 1
            values[1] = float(values[1]) + seconds
            values[2] = max(float(values[2]), seconds)

    def add_gauge(self, name: str, delta: int) -> None:
        """Adjust a predefined nonnegative gauge by ``delta``."""
        if name not in self._GAUGES:
            return
        with self._lock:
            self._gauges[name] = max(0, self._gauges[name] + delta)

    def set_gauge(self, name: str, value: int) -> None:
        """Set a predefined gauge to a nonnegative value."""
        if name not in self._GAUGES:
            return
        with self._lock:
            self._gauges[name] = max(0, value)

    def add_bytes(self, name: str, value: int) -> None:
        """Add a positive byte count to a predefined aggregate."""
        if name not in self._BYTE_KINDS or value <= 0:
            return
        with self._lock:
            self._bytes[name] += value

    def existing_pages(self, count: int) -> None:
        """Count compatible pages that already existed at the destination."""
        if count <= 0:
            return
        with self._lock:
            self._same_key_existing_total += count

    def snapshot(self) -> dict[str, Any]:
        """Return a stable pointer-free aggregate for logs/export adapters."""

        with self._lock:
            return {
                "started_total": self._started_total,
                "attempts_total": {
                    f"{outcome}:{reason}": count
                    for (outcome, reason), count in sorted(self._attempts.items())
                },
                "direct_abandoned_total": dict(sorted(self._abandoned.items())),
                "transfer_failure_total": dict(sorted(self._failures.items())),
                "allocation_rejection_total": dict(
                    sorted(self._allocation_rejections.items())
                ),
                "timers": {
                    name: {
                        "count": int(values[0]),
                        "total_seconds": float(values[1]),
                        "max_seconds": float(values[2]),
                    }
                    for name, values in sorted(self._timers.items())
                },
                "gauges": dict(sorted(self._gauges.items())),
                "bytes": dict(sorted(self._bytes.items())),
                "same_key_existing_total": self._same_key_existing_total,
                "fatal_restart_total": self._fatal_restart_total,
            }


def parse_remote_fill_handoff(
    request_configs: Mapping[str, Any] | None,
) -> RemoteFillHandoff | None:
    """Parse the proxy's nested remote-fill destination identity.

    Args:
        request_configs: Request-scoped LMCache configuration mapping.

    Returns:
        A validated handoff, or ``None`` when the request did not opt in.

    Raises:
        ValueError: If the nested handoff exists but is malformed.
    """

    if not isinstance(request_configs, Mapping):
        return None
    raw = request_configs.get(REMOTE_FILL_REQUEST_CONFIG_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.remote_fill must be a mapping")

    def required_text(name: str) -> str:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"lmcache.remote_fill.{name} must be nonempty")
        return value

    def required_int(name: str, *, positive: bool = False) -> int:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"lmcache.remote_fill.{name} must be an integer")
        if value < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"lmcache.remote_fill.{name} must be {qualifier}")
        return value

    global_te_push = raw.get("global_te_push")
    if not isinstance(global_te_push, bool):
        raise ValueError("lmcache.remote_fill.global_te_push must be a boolean")

    token_hash_algorithm = required_text("token_hash_algorithm")
    python_hash_seed = raw.get("python_hash_seed", "")
    if not isinstance(python_hash_seed, str):
        raise ValueError("lmcache.remote_fill.python_hash_seed must be a string")
    if token_hash_algorithm == "builtin" and not python_hash_seed:
        raise ValueError(
            "lmcache.remote_fill.python_hash_seed is required for builtin hashing"
        )
    verification_capability = required_text(
        "descriptor_verification_capability"
    )
    try:
        verification_key = bytes.fromhex(verification_capability)
    except ValueError as error:
        raise ValueError(
            "lmcache.remote_fill descriptor verification capability is invalid"
        ) from error
    if (
        len(verification_key) != _DESCRIPTOR_VERIFICATION_CAPABILITY_BYTES
        or verification_key.hex() != verification_capability
    ):
        raise ValueError(
            "lmcache.remote_fill descriptor verification capability is invalid"
        )

    return RemoteFillHandoff(
        transfer_id=required_text("transfer_id"),
        request_attempt=required_int("request_attempt"),
        source_engine_id=required_text("source_engine_id"),
        destination_engine_id=required_text("destination_engine_id"),
        destination_engine_epoch=required_int(
            "destination_engine_epoch", positive=True
        ),
        control_endpoint=required_text("control_endpoint"),
        destination_dp_rank=required_int("destination_dp_rank"),
        shared_cache_generation=required_int("shared_cache_generation"),
        destination_tp_size=required_int("destination_tp_size", positive=True),
        destination_dp_size=required_int("destination_dp_size", positive=True),
        global_te_push=global_te_push,
        token_hash_algorithm=token_hash_algorithm,
        python_hash_seed=python_hash_seed,
        descriptor_verification_capability=verification_capability,
    )


def create_remote_fill_client(
    endpoint: str,
    *,
    operation_timeouts_ms: Mapping[OperationKind, int],
    limits: ProtocolLimits,
) -> RemoteFillClient:
    """Create a one-rank client over LMCache's existing RPC transport.

    Args:
        endpoint: Decoder endpoint in ``protocol://address`` form.
        operation_timeouts_ms: Positive request/reply timeout per operation.
        limits: Shared protocol codec bounds.

    Returns:
        A remote-fill client targeting decoder TP0 only.

    Raises:
        ValueError: If the endpoint or timeout is invalid.
    """

    if "://" not in endpoint:
        raise ValueError("remote-fill endpoint must include its protocol")
    protocol, socket_path = endpoint.split("://", 1)
    timeouts = dict(operation_timeouts_ms)
    if not protocol or not socket_path or not timeouts or any(
        timeout <= 0 for timeout in timeouts.values()
    ):
        raise ValueError("remote-fill endpoint or timeout is invalid")
    transport = ZmqReqRepClientTransport(
        [SocketParams(socket_path, 0, protocol)],
        timeout_ms=max(timeouts.values()),
    )
    return RemoteFillClient(
        RpcRemoteFillByteTransport(transport, limits),
        operation_timeouts_ms=timeouts,
    )


class RemoteFillProducerSession:
    """Ordered producer control session for one request attempt."""

    def __init__(
        self,
        *,
        request_id: str,
        handoff: RemoteFillHandoff,
        static_spec: RemoteFillStaticSpec,
        client: RemoteFillClient,
        secret: bytes,
        planned_window_count_hint: int,
        required_store_end_hint: int,
        native_hard_timeout_seconds: float = 120.0,
        negotiation_cache: RemoteFillNegotiationCache | None = None,
    ) -> None:
        """Create a request-local session without issuing network traffic."""

        self.request_id = request_id
        self.handoff = handoff
        self.static_spec = static_spec
        self.client = client
        self.secret = secret
        self.planned_window_count_hint = max(0, planned_window_count_hint)
        self.required_store_end_hint = max(0, required_store_end_hint)
        if native_hard_timeout_seconds <= 0:
            raise ValueError("native hard timeout must be positive")
        self.native_hard_timeout_seconds = float(native_hard_timeout_seconds)
        self.negotiation_cache = negotiation_cache
        self.direct_viable = True
        self.opened = False
        self.closed = False
        self.fatal_restart_required = False
        self.prearm_control_unknown = False
        self.d_local_end_open = 0
        self.remote_session = ""
        self.window_manifests: list[tuple[int, str]] = []
        self.manifest_digest_seed = content_digest(
            (
                self.handoff.transfer_id,
                self.request_id,
                self.handoff.request_attempt,
            )
        )
        self._operation_sequence = 0
        self._terminal: RemoteFillTerminalResult | None = None

    def open(self) -> bool:
        """Negotiate static layout and open the destination transaction."""

        if self.opened:
            return True
        if self.closed or not self.direct_viable:
            return False
        if (
            self.static_spec.token_hash_algorithm != self.handoff.token_hash_algorithm
            or self.static_spec.python_hash_seed != self.handoff.python_hash_seed
        ):
            self.direct_viable = False
            return False
        negotiation_key = RemoteFillNegotiationCache.key(
            self.handoff, self.static_spec
        )
        negotiation_cached = bool(
            self.negotiation_cache is not None
            and self.negotiation_cache.contains(negotiation_key)
        )
        if not negotiation_cached:
            negotiate = NegotiateRequest(
                common=self._common(),
                cache_namespace_tag=self.static_spec.cache_namespace_tag,
                layout_tag=self.static_spec.layout_tag,
                model_artifact_id=self.static_spec.model_artifact_id,
                chunk_size=self.static_spec.chunk_size,
                model_layout=self.static_spec.model_layout,
                group_dimensions=self.static_spec.group_dimensions,
                layer_count=self.static_spec.layer_count,
                save_only_first_rank=self.static_spec.save_only_first_rank,
                shared_group1=self.static_spec.shared_group1,
                tp_size=self.static_spec.tp_size,
                dp_size=self.static_spec.dp_size,
                global_te_push=self.static_spec.global_te_push,
                token_hash_algorithm=self.static_spec.token_hash_algorithm,
                python_hash_seed=self.static_spec.python_hash_seed,
            )
            try:
                negotiated = self._execute(negotiate)
            except RemoteFillFatalError:
                raise
            except Exception:
                self.prearm_control_unknown = True
                self.direct_viable = False
                return False
            if not self._accepted(negotiated):
                self.direct_viable = False
                return False
            if self.negotiation_cache is not None:
                self.negotiation_cache.record(negotiation_key)
        try:
            opened = self._execute(
                OpenRequest(
                    common=self._common(),
                    request_id=self.request_id,
                    source_engine_id=self.handoff.source_engine_id,
                    destination_engine_id=self.handoff.destination_engine_id,
                    destination_dp_rank=self.handoff.destination_dp_rank,
                    required_store_end_hint=self.required_store_end_hint,
                    planned_window_count_hint=self.planned_window_count_hint,
                    cache_namespace_tag=self.static_spec.cache_namespace_tag,
                    layout_tag=self.static_spec.layout_tag,
                    model_artifact_id=self.static_spec.model_artifact_id,
                    manifest_digest_seed=self.manifest_digest_seed,
                )
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            self.prearm_control_unknown = True
            self.direct_viable = False
            return False
        if not self._accepted(opened):
            self.direct_viable = False
            return False
        if (
            opened.destination_engine_epoch != self.handoff.destination_engine_epoch
            or opened.shared_cache_generation != self.handoff.shared_cache_generation
            or not opened.remote_session
        ):
            self.direct_viable = False
            return False
        self.d_local_end_open = opened.d_local_end_open
        self.remote_session = opened.remote_session
        self.opened = True
        return True

    def probe_window(
        self,
        *,
        window_id: int,
        source_generation: int,
        control_pages: tuple[ControlPage, ...],
    ) -> RemoteFillWindowResult:
        """Register cached-prefix pages without allocating a destination.

        Probe-only manifests are required for exact final coverage. A missing
        decoder page is the first direct hole: it permanently disables later
        direct windows while leaving the persistent sink untouched.
        """

        if not self.direct_viable or not self.open():
            return self._abandoned_window(window_id, "open rejected")
        digest = manifest_digest(control_pages)
        reserve_started = time.perf_counter()
        try:
            reserved = self._execute(
                ReserveWindowRequest(
                    common=self._common(),
                    window_id=window_id,
                    source_generation=source_generation,
                    manifest_digest=digest,
                    control_pages=control_pages,
                    reserve_missing=False,
                )
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            reserve_seconds = time.perf_counter() - reserve_started
            self.prearm_control_unknown = True
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "probe control unavailable",
                reserve_seconds=reserve_seconds,
            )
        reserve_seconds = time.perf_counter() - reserve_started
        self.window_manifests.append((window_id, digest))
        try:
            dispositions = self._validate_page_results(control_pages, reserved)
        except ValueError:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "probe result invalid",
                reserve_seconds=reserve_seconds,
            )
        existing_pages = sum(item is PageDisposition.EXISTING for item in dispositions)
        if (
            reserved.code is not ResultCode.OK
            or reserved.descriptors
            or any(item is not PageDisposition.EXISTING for item in dispositions)
        ):
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "cached-prefix hole",
                reserve_seconds=reserve_seconds,
                existing_pages=existing_pages,
            )
        return RemoteFillWindowResult(
            window_id,
            True,
            False,
            reserve_seconds=reserve_seconds,
            existing_pages=existing_pages,
        )

    def transfer_window(
        self,
        *,
        window_id: int,
        source_generation: int,
        control_pages: tuple[ControlPage, ...],
        source_plan: Any,
        submitter: DirectPushSubmitter,
        activation_factory: Callable[[str], Any],
    ) -> RemoteFillWindowResult:
        """Reserve, arm, transfer, and report one authoritative page window."""

        if not self.direct_viable or not self.open():
            return self._abandoned_window(window_id, "open rejected")
        digest = manifest_digest(control_pages)
        reserve_started = time.perf_counter()
        try:
            reserved = self._execute(
                ReserveWindowRequest(
                    common=self._common(),
                    window_id=window_id,
                    source_generation=source_generation,
                    manifest_digest=digest,
                    control_pages=control_pages,
                    reserve_missing=True,
                )
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            reserve_seconds = time.perf_counter() - reserve_started
            self.prearm_control_unknown = True
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "reserve control unavailable",
                reserve_seconds=reserve_seconds,
            )
        reserve_seconds = time.perf_counter() - reserve_started
        self.window_manifests.append((window_id, digest))
        if reserved.code is not ResultCode.OK:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "reservation rejected",
                reserve_seconds=reserve_seconds,
            )
        descriptors = reserved.descriptors
        try:
            dispositions = self._validate_page_results(control_pages, reserved)
        except ValueError:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "reservation result invalid",
                reserve_seconds=reserve_seconds,
            )
        existing_pages = sum(item is PageDisposition.EXISTING for item in dispositions)
        if any(item is PageDisposition.MISSING for item in dispositions):
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "direct destination hole",
                reserve_seconds=reserve_seconds,
                existing_pages=existing_pages,
            )
        if not descriptors:
            return RemoteFillWindowResult(
                window_id,
                True,
                False,
                reserve_seconds=reserve_seconds,
                existing_pages=existing_pages,
            )
        try:
            self._validate_descriptors(control_pages, digest, window_id, reserved)
            descriptor_digest = destination_descriptor_digest(descriptors)
            if descriptor_digest != reserved.destination_descriptor_digest:
                raise ValueError("destination descriptor digest changed")
        except Exception:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "descriptor validation failed",
                reserve_seconds=reserve_seconds,
                existing_pages=existing_pages,
            )

        arm_started = time.perf_counter()
        try:
            armed = self._execute(
                ArmWindowRequest(
                    common=self._common(),
                    window_id=window_id,
                    native_transfer_attempt_id=(reserved.native_transfer_attempt_id),
                    manifest_digest=digest,
                    destination_descriptor_digest=descriptor_digest,
                )
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            try:
                armed = self._resolve_ambiguous_arm(
                    window_id,
                    reserved.native_transfer_attempt_id,
                )
            except Exception as status_error:
                self.fatal_restart_required = True
                raise RemoteFillFatalError(
                    "ARM_WINDOW acknowledgement is ambiguous and STATUS "
                    "could not prove the armed attempt"
                ) from status_error
        arm_seconds = time.perf_counter() - arm_started
        if armed.code is not ResultCode.OK:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "arm rejected",
                reserve_seconds=reserve_seconds,
                arm_seconds=arm_seconds,
                existing_pages=existing_pages,
            )

        attempt_id = reserved.native_transfer_attempt_id
        native_seconds = 0.0
        source_event_wait_seconds = 0.0
        source_fences_ready_monotonic = 0.0
        source_registration_seconds = 0.0
        native_slot_wait_seconds = 0.0
        native_started_monotonic = 0.0
        native_ended_monotonic = 0.0
        submitted_bytes = sum(
            descriptor.destination_length for descriptor in descriptors
        )
        native_wait_started = time.perf_counter()
        native_deadline = native_wait_started + self.native_hard_timeout_seconds
        native_result = None
        native_error: Exception | None = None
        try:
            source_by_identity = {
                (page.canonical_key, page.kv_group): page for page in source_plan.pages
            }
            selected_sources = tuple(
                source_by_identity[(descriptor.canonical_key, descriptor.kv_group)]
                for descriptor in descriptors
            )
            native_source_plan = DirectPushSourcePlan(
                pages=selected_sources,
                owners=source_plan.owners,
                producer_events=source_plan.producer_events,
            )
            future = submitter(
                remote_session=self.remote_session,
                source_plan=native_source_plan,
                destination_descriptors=descriptors,
                activation=activation_factory(attempt_id),
            )
            native_result = future.result(
                timeout=max(0.0, native_deadline - time.perf_counter())
            )
        except FutureTimeoutError as timeout_error:
            # Do not cancel the native future: it owns the source pages and
            # producer events until the paired process restart tears down the
            # now-ambiguous transfer.
            self.fatal_restart_required = True
            raise RemoteFillFatalError(
                "armed native transfer exceeded its hard terminal bound"
            ) from timeout_error
        except Exception as error:
            if hasattr(error, "terminal_future"):
                try:
                    native_result = self._wait_for_original_native_terminal(
                        error,
                        native_deadline,
                    )
                except FutureTimeoutError as timeout_error:
                    self.fatal_restart_required = True
                    raise RemoteFillFatalError(
                        "armed native transfer exceeded its hard terminal bound"
                    ) from timeout_error
                except (
                    NativeDirectPushPreSubmitError,
                    NativeDirectPushTerminalError,
                ) as terminal_error:
                    native_error = terminal_error
                except Exception as terminal_error:
                    self.fatal_restart_required = True
                    raise RemoteFillFatalError(
                        "original native transfer returned an unknown terminal state"
                    ) from terminal_error
            elif isinstance(
                error,
                (NativeDirectPushPreSubmitError, NativeDirectPushTerminalError),
            ):
                native_error = error
            else:
                self.fatal_restart_required = True
                raise RemoteFillFatalError(
                    "armed native transfer failed without terminal evidence"
                ) from error

        expected_bytes = sum(
            descriptor.destination_length for descriptor in descriptors
        )
        if native_error is None:
            if not isinstance(native_result, NativeDirectPushResult):
                self.fatal_restart_required = True
                raise RemoteFillFatalError(
                    "armed native transfer returned an invalid terminal result"
                )
            native_seconds = max(0.0, float(native_result.elapsed_ms) / 1000.0)
            source_event_wait_seconds = max(
                0.0, float(native_result.source_event_wait_ms) / 1000.0
            )
            source_fences_ready_monotonic = max(
                0.0, float(native_result.source_fences_ready_monotonic)
            )
            source_registration_seconds = max(
                0.0, float(native_result.source_registration_ms) / 1000.0
            )
            native_slot_wait_seconds = max(
                0.0, float(native_result.native_slot_wait_ms) / 1000.0
            )
            native_started_monotonic = max(
                0.0, float(native_result.native_started_monotonic)
            )
            native_ended_monotonic = max(
                native_started_monotonic,
                float(native_result.native_ended_monotonic),
            )
            return_code = int(native_result.return_code)
            completed_bytes = int(native_result.transferred_bytes)
            if (
                native_result.native_transfer_attempt_id != attempt_id
                or completed_bytes != expected_bytes
            ):
                return_code = -1
                completed_bytes = 0
        else:
            result = (
                native_error.result
                if isinstance(native_error, NativeDirectPushTerminalError)
                else None
            )
            return_code = int(getattr(result, "return_code", -1))
            completed_bytes = int(getattr(result, "transferred_bytes", 0))
            native_seconds = max(
                0.0, float(getattr(result, "elapsed_ms", 0.0)) / 1000.0
            )
            source_event_wait_seconds = max(
                0.0,
                float(getattr(result, "source_event_wait_ms", 0.0)) / 1000.0,
            )
            source_fences_ready_monotonic = max(
                0.0,
                float(getattr(result, "source_fences_ready_monotonic", 0.0)),
            )
            source_registration_seconds = max(
                0.0,
                float(getattr(result, "source_registration_ms", 0.0)) / 1000.0,
            )
            native_slot_wait_seconds = max(
                0.0,
                float(getattr(result, "native_slot_wait_ms", 0.0)) / 1000.0,
            )
            native_started_monotonic = max(
                0.0,
                float(getattr(result, "native_started_monotonic", 0.0)),
            )
            native_ended_monotonic = max(
                native_started_monotonic,
                float(getattr(result, "native_ended_monotonic", 0.0)),
            )

        report_succeeded: bool
        report_started = time.perf_counter()
        try:
            reported = self._execute(
                ReportTransferCompleteRequest(
                    common=self._common(),
                    window_id=window_id,
                    native_transfer_attempt_id=attempt_id,
                    native_return_code=return_code,
                    completed_bytes=completed_bytes,
                    manifest_digest=digest,
                )
            )
            report_succeeded = reported.code is ResultCode.OK
        except Exception:
            try:
                report_succeeded = self._resolve_ambiguous_report(
                    window_id=window_id,
                    native_transfer_attempt_id=attempt_id,
                    native_return_code=return_code,
                    completed_bytes=completed_bytes,
                    expected_bytes=expected_bytes,
                )
            except Exception as status_error:
                self.fatal_restart_required = True
                raise RemoteFillFatalError(
                    "terminal native report is ambiguous and STATUS could not "
                    "prove the reported result"
                ) from status_error
        report_seconds = time.perf_counter() - report_started
        if return_code != 0 or not report_succeeded:
            self.direct_viable = False
            return self._abandoned_window(
                window_id,
                "native transfer failed",
                True,
                reserve_seconds=reserve_seconds,
                arm_seconds=arm_seconds,
                source_event_wait_seconds=source_event_wait_seconds,
                source_fences_ready_monotonic=source_fences_ready_monotonic,
                source_registration_seconds=source_registration_seconds,
                native_slot_wait_seconds=native_slot_wait_seconds,
                native_seconds=native_seconds,
                report_seconds=report_seconds,
                native_started_monotonic=native_started_monotonic,
                native_ended_monotonic=native_ended_monotonic,
                submitted_bytes=submitted_bytes,
                existing_pages=existing_pages,
            )
        return RemoteFillWindowResult(
            window_id,
            True,
            True,
            reserve_seconds=reserve_seconds,
            arm_seconds=arm_seconds,
            source_event_wait_seconds=source_event_wait_seconds,
            source_fences_ready_monotonic=source_fences_ready_monotonic,
            source_registration_seconds=source_registration_seconds,
            native_slot_wait_seconds=native_slot_wait_seconds,
            native_seconds=native_seconds,
            report_seconds=report_seconds,
            native_started_monotonic=native_started_monotonic,
            native_ended_monotonic=native_ended_monotonic,
            submitted_bytes=submitted_bytes,
            existing_pages=existing_pages,
        )

    def _resolve_ambiguous_arm(
        self,
        window_id: int,
        native_transfer_attempt_id: str,
    ) -> RemoteFillResponse:
        """Resolve a lost ARM reply without ever resubmitting native work."""

        response = self._execute(
            StatusRequest(common=self._common(), window_id=window_id)
        )
        if response.code is not ResultCode.OK or len(response.windows) != 1:
            raise RuntimeError("STATUS did not identify the ambiguous ARM window")
        status = response.windows[0]
        if (
            status.window_id != window_id
            or status.native_transfer_attempt_id != native_transfer_attempt_id
            or status.native_state is not DestinationNativeState.ARMED
            or status.fatal_restart_required
            or response.fatal_restart_required
        ):
            raise RuntimeError("STATUS did not prove the exact armed attempt")
        return response

    @staticmethod
    def _wait_for_original_native_terminal(
        error: Exception,
        deadline: float,
    ) -> Any:
        """Await the already-submitted native call without resubmitting it."""

        terminal_future = getattr(error, "terminal_future", None)
        remaining = deadline - time.perf_counter()
        if terminal_future is None or remaining <= 0:
            raise FutureTimeoutError()
        if isinstance(terminal_future, Future):
            return terminal_future.result(timeout=remaining)
        done = getattr(terminal_future, "done", None)
        if callable(done) and done():
            return terminal_future.result()
        wait_for_terminal = getattr(error, "wait_for_terminal", None)
        get_loop = getattr(terminal_future, "get_loop", None)
        if not callable(wait_for_terminal) or not callable(get_loop):
            raise TypeError("native terminal future cannot be awaited safely")
        loop = get_loop()
        if not loop.is_running():
            raise RuntimeError("native terminal event loop is not running")
        bridged = asyncio.run_coroutine_threadsafe(wait_for_terminal(), loop)
        try:
            return bridged.result(timeout=remaining)
        except FutureTimeoutError:
            # Cancelling the bridge does not cancel the shielded native future.
            bridged.cancel()
            raise

    def _resolve_ambiguous_report(
        self,
        *,
        window_id: int,
        native_transfer_attempt_id: str,
        native_return_code: int,
        completed_bytes: int,
        expected_bytes: int,
    ) -> bool:
        """Resolve a lost completion reply from exact destination state."""

        response = self._execute(
            StatusRequest(common=self._common(), window_id=window_id)
        )
        if response.code is not ResultCode.OK or len(response.windows) != 1:
            raise RuntimeError("STATUS did not return the reported window")
        window = response.windows[0]
        if (
            window.window_id != window_id
            or window.native_transfer_attempt_id != native_transfer_attempt_id
            or window.fatal_restart_required
            or response.fatal_restart_required
        ):
            raise RuntimeError("STATUS returned a different native attempt")
        reported_success = (
            native_return_code == 0 and completed_bytes == expected_bytes
        )
        if (
            reported_success
            and window.native_state is DestinationNativeState.TERMINAL_SUCCESS
        ):
            return True
        if (
            not reported_success
            and window.native_state is DestinationNativeState.TERMINAL_FAILURE
        ):
            return False
        raise RuntimeError("STATUS does not match the reported terminal result")

    def finish(
        self,
        *,
        required_store_end: int,
        persistent_common_end: int,
        final_partial_valid_tokens: int = 0,
    ) -> RemoteFillTerminalResult:
        """Finalize after persistent and native work have reached terminal state."""

        if self._terminal is not None:
            return self._terminal
        if self.fatal_restart_required:
            outcome = TerminalOutcome.FATAL_RESTART.value
        elif persistent_common_end < required_store_end:
            outcome = TerminalOutcome.PERSISTENCE_FAILED.value
        elif not self.opened or self.prearm_control_unknown:
            if self.prearm_control_unknown:
                try:
                    self._execute(
                        AbortRequest(
                            common=self._common(),
                            reason="prefiller pre-arm control state is unknown",
                        )
                    )
                except Exception:
                    # No native operation was armed. Decoder reservations stay
                    # hidden and its bounded cleanup owns eventual reclamation.
                    pass
            outcome = TerminalOutcome.PERSISTENT_ONLY.value
        else:
            try:
                response = self._execute(
                    FinishRequest(
                        common=self._common(),
                        required_store_end=required_store_end,
                        persistent_common_end=persistent_common_end,
                        final_manifest_digest=transaction_manifest_digest(
                            self.manifest_digest_seed,
                            tuple(self.window_manifests),
                            required_store_end,
                            final_partial_valid_tokens,
                        ),
                        final_partial_valid_tokens=final_partial_valid_tokens,
                    )
                )
            except RemoteFillFatalError:
                raise
            except Exception:
                # FINISH may have committed before its reply was lost. Resolve
                # the transaction-level terminal state instead of discarding
                # valid decoder-local pages and under-reporting LOCAL_FULL.
                outcome = self._resolve_ambiguous_finish()
            else:
                if response.fatal_restart_required:
                    outcome = TerminalOutcome.FATAL_RESTART.value
                elif response.terminal_outcome is None:
                    raise RuntimeError(
                        "remote-fill FINISH returned no terminal outcome"
                    )
                else:
                    outcome = response.terminal_outcome.value
        self._terminal = RemoteFillTerminalResult(
            transfer_id=self.handoff.transfer_id,
            outcome=outcome,
            persistent_common_end=persistent_common_end,
            required_store_end=required_store_end,
        )
        self.closed = True
        return self._terminal

    def _resolve_ambiguous_finish(self) -> str:
        """Resolve a lost FINISH reply without resubmitting publication."""

        try:
            response = self._execute(
                StatusRequest(common=self._common(), window_id=-1)
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            self.direct_viable = False
            return TerminalOutcome.PERSISTENT_ONLY.value
        if response.fatal_restart_required or (
            response.terminal_outcome is TerminalOutcome.FATAL_RESTART
        ):
            self.fatal_restart_required = True
            raise RemoteFillFatalError(
                "decoder reported fatal state while resolving FINISH"
            )
        if any(
            window.native_state
            in (
                DestinationNativeState.ARMED,
                DestinationNativeState.FATAL_UNKNOWN,
            )
            for window in response.windows
        ):
            self.fatal_restart_required = True
            raise RemoteFillFatalError(
                "decoder still owns an ambiguous native attempt after FINISH"
            )
        if response.terminal_outcome in (
            TerminalOutcome.LOCAL_FULL,
            TerminalOutcome.PERSISTENT_ONLY,
            TerminalOutcome.PERSISTENCE_FAILED,
            TerminalOutcome.CANCELLED,
        ):
            return response.terminal_outcome.value
        # All native operations were terminal before FINISH. If publication
        # did not happen, explicitly abort the still-hidden transaction so it
        # cannot retain decoder pages indefinitely.
        try:
            self._execute(
                AbortRequest(
                    common=self._common(),
                    reason="FINISH terminal state could not be confirmed",
                )
            )
        except RemoteFillFatalError:
            raise
        except Exception:
            pass
        self.direct_viable = False
        return TerminalOutcome.PERSISTENT_ONLY.value

    def abort(self, reason: str) -> None:
        """Release a nonfatal hidden transaction that will not be finished."""

        if self._terminal is not None or self.closed:
            return
        if self.fatal_restart_required:
            raise RemoteFillFatalError(
                "an ambiguous armed transaction cannot be aborted safely"
            )
        if self.opened:
            response = self._execute(AbortRequest(common=self._common(), reason=reason))
            if response.code not in (
                ResultCode.OK,
                ResultCode.ACCEPTED,
                ResultCode.TERMINAL,
            ):
                raise RuntimeError("remote-fill ABORT was rejected")
        self.direct_viable = False
        self.closed = True

    def close(self) -> None:
        """Close client resources after every queued operation has drained."""

        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        self.closed = True

    def _common(self) -> OperationIdentity:
        self._operation_sequence += 1
        return OperationIdentity(
            protocol_version=PROTOCOL_VERSION,
            operation_id=uuid4().hex,
            operation_sequence=self._operation_sequence,
            payload_digest="",
            transfer_id=self.handoff.transfer_id,
            request_attempt=self.handoff.request_attempt,
            destination_engine_epoch=self.handoff.destination_engine_epoch,
            shared_cache_generation=self.handoff.shared_cache_generation,
        )

    def _execute(self, request: Any) -> RemoteFillResponse:
        """Retry one exact idempotent operation after a lost reply."""

        try:
            response = self.client.execute(request)
        except ReplyLostError:
            response = self.client.execute(request)
        expected_kind = {
            NegotiateRequest: OperationKind.NEGOTIATE,
            AbortRequest: OperationKind.ABORT,
            OpenRequest: OperationKind.OPEN,
            ReserveWindowRequest: OperationKind.RESERVE_WINDOW,
            ArmWindowRequest: OperationKind.ARM_WINDOW,
            ReportTransferCompleteRequest: OperationKind.REPORT_TRANSFER_COMPLETE,
            FinishRequest: OperationKind.FINISH,
            StatusRequest: OperationKind.STATUS,
        }.get(type(request))
        if (
            expected_kind is None
            or response.operation is not expected_kind
            or response.operation_id != request.common.operation_id
            or response.destination_engine_epoch
            != self.handoff.destination_engine_epoch
            or response.shared_cache_generation != self.handoff.shared_cache_generation
        ):
            raise ValueError("remote-fill response identity changed")
        if response.fatal_restart_required or (
            response.terminal_outcome is TerminalOutcome.FATAL_RESTART
        ):
            self.fatal_restart_required = True
            raise RemoteFillFatalError(
                "decoder reported an armed remote-fill transaction with an "
                "unknown terminal state"
            )
        return response

    def _validate_descriptors(
        self,
        pages: tuple[ControlPage, ...],
        digest: str,
        window_id: int,
        response: RemoteFillResponse,
    ) -> None:
        expected = {(page.canonical_key, page.kv_group): page for page in pages}
        seen: set[tuple[str, int]] = set()
        allocated = {
            (result.canonical_key, result.kv_group)
            for result in response.page_results
            if result.disposition is PageDisposition.ALLOCATED
        }
        for descriptor in response.descriptors:
            identity = (descriptor.canonical_key, descriptor.kv_group)
            page = expected.get(identity)
            if page is None or identity in seen:
                raise ValueError("destination descriptor coverage is invalid")
            seen.add(identity)
            if not verify_descriptor(self.secret, descriptor):
                raise ValueError("destination descriptor HMAC is invalid")
            if (
                descriptor.remote_session != self.remote_session
                or descriptor.transfer_id != self.handoff.transfer_id
                or descriptor.request_attempt != self.handoff.request_attempt
                or descriptor.native_transfer_attempt_id
                != response.native_transfer_attempt_id
                or descriptor.destination_engine_epoch
                != self.handoff.destination_engine_epoch
                or descriptor.shared_cache_generation
                != self.handoff.shared_cache_generation
                or descriptor.destination_dp_rank != self.handoff.destination_dp_rank
                or descriptor.destination_tp_rank != page.destination_tp_rank
                or descriptor.destination_length != page.expected_bytes
                or descriptor.manifest_digest != digest
                or descriptor.destination_ptr <= 0
                or descriptor.window_id != window_id
                or not descriptor.reservation_id
            ):
                raise ValueError("destination descriptor identity changed")
        if seen != allocated:
            raise ValueError("destination descriptor allocation set is invalid")

    @staticmethod
    def _validate_page_results(
        pages: tuple[ControlPage, ...],
        response: RemoteFillResponse,
    ) -> tuple[PageDisposition, ...]:
        if len(response.page_results) != len(pages):
            raise ValueError("destination page-result count is invalid")
        dispositions: list[PageDisposition] = []
        for page, result in zip(pages, response.page_results, strict=True):
            if (
                result.canonical_key != page.canonical_key
                or result.kv_group != page.kv_group
                or result.chunk_index != page.chunk_index
            ):
                raise ValueError("destination page-result identity changed")
            dispositions.append(result.disposition)
        return tuple(dispositions)

    @staticmethod
    def _accepted(response: RemoteFillResponse) -> bool:
        return response.code in (
            ResultCode.OK,
            ResultCode.ACCEPTED,
            ResultCode.ALREADY_EXISTS,
        )

    @staticmethod
    def _abandoned_window(
        window_id: int,
        reason: str,
        armed: bool = False,
        **metrics: Any,
    ) -> RemoteFillWindowResult:
        return RemoteFillWindowResult(
            window_id=window_id,
            direct_satisfied=False,
            armed=armed,
            reason=reason,
            **metrics,
        )
