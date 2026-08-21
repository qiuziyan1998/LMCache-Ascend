# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from bisect import bisect_right
from collections import deque
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from weakref import WeakSet
import json
import os
import queue
import secrets
import threading
import time
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    LayerCacheEngineKey,
    _lmcache_nvtx_annotate,
    convert_tokens_to_list,
)
from lmcache.v1.cache_engine import (
    _SHARED_SPARSE_DEFER_COMMIT,
    _SHARED_SPARSE_PREPARE_ONLY,
    LayerwiseStoreResult,
    LMCacheEngine,
)
from lmcache.v1.cold_start_perf import (
    cold_start_perf_enabled,
    cold_start_perf_log,
    cold_start_perf_now,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.sparse import PreparedSparseSource
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    LayerPageSource,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import (
    mooncake_layer_pages_enabled,
    mooncake_payload_layout,
    mooncake_page_layout_enabled,
)
from lmcache.v1.remote_fill import (
    ControlPage,
    OperationKind,
    ProtocolLimits,
    log_remote_fill_validation_failure,
)
from lmcache.v1.remote_fill.native import (
    DIRECT_PUSH_H0_QUALIFICATION_V1,
    DirectPushPageSource,
    DirectPushSourcePlan,
    NativeDirectPushActivation,
    NativeExternalPageTransferUnknownError,
)
from lmcache.v1.shared_cpu_cache import (
    SharedHandleBatch,
    SharedHandleEnvelope,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult
from lmcache.v1.token_database import (
    DSA_INDEX_CACHE_SCHEMA,
    DSA_INDEX_CACHE_SCHEMA_TAG,
    TokenDatabase,
)
import torch

# First Party
from lmcache_ascend.v1.content_diagnostics import (
    fingerprint_compact_group1,
    log_npu_content_diagnostic_event,
    npu_content_diagnostics_enabled,
)
from lmcache_ascend.v1.remote_fill import (
    RemoteFillDecoderLayout,
    DecoderRemoteFillRuntime,
    build_decoder_layout,
    create_decoder_remote_fill_runtime,
    remote_fill_h0_qualified,
)
from lmcache_ascend.v1.remote_fill_producer import (
    REMOTE_FILL_H0_QUALIFICATION_ENV,
    RemoteFillFatalError,
    RemoteFillHandoff,
    RemoteFillProducerMetrics,
    RemoteFillNegotiationCache,
    RemoteFillProducerSession,
    RemoteFillStaticSpec,
    RemoteFillTerminalResult,
    create_remote_fill_client,
    parse_remote_fill_handoff,
)

logger = init_logger(__name__)
REMOTE_FILL_DEFAULT_CONTROL_PORT = 19000


def _remote_fill_advertise_host_from_session(remote_session: str) -> str:
    """Extract the decoder host already advertised by its GlobalTE session."""

    value = str(remote_session).strip()
    if not value:
        raise ValueError("remote-fill GlobalTE session is empty")
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if not parsed.hostname:
        raise ValueError(
            "remote-fill cannot infer a routable control host from GlobalTE session"
        )
    return parsed.hostname


def _validate_remote_fill_control_port(
    base_port: int,
    dp_rank: int,
    dp_size: int,
    *,
    internal_api_enabled: bool,
    internal_api_port_start: int,
) -> int:
    """Validate the complete DP port range and return this rank's port."""

    control_port = int(base_port) + dp_rank
    last_control_port = int(base_port) + dp_size - 1
    if dp_size <= 0 or not 0 <= dp_rank < dp_size:
        raise ValueError("remote-fill decoder DP topology is invalid")
    if not 0 < int(base_port) < 65536 or last_control_port >= 65536:
        raise ValueError("remote-fill decoder DP control port is invalid")
    if internal_api_enabled:
        internal_start = int(internal_api_port_start)
        internal_end = internal_start + dp_size - 1
        if max(int(base_port), internal_start) <= min(
            last_control_port, internal_end
        ):
            raise ValueError(
                "remote-fill decoder control ports overlap the internal API "
                "port range"
            )
    return control_port

LOCAL_CPU_BACKEND_NAME = "LocalCPUBackend"
REMOTE_BACKEND_NAME = "RemoteBackend"
_SHARED_CPU_CHUNK_PLAN_KEY = "_shared_cpu_chunk_hash_plan"
_SHARED_CPU_COMPACT_COMMIT_MESSAGE = "compact shared batch committed"


def _shared_sparse_request(request):
    prepare_only = defer_commit = False
    if isinstance(request, dict):
        prepare_only = bool(request.get(_SHARED_SPARSE_PREPARE_ONLY, False))
        defer_commit = bool(request.get(_SHARED_SPARSE_DEFER_COMMIT, False))
        if prepare_only or defer_commit:
            request = dict(request)
            request.pop(_SHARED_SPARSE_PREPARE_ONLY, None)
            request.pop(_SHARED_SPARSE_DEFER_COMMIT, None)
        return (
            request,
            request.get("selected_token_ids"),
            request.get("token_start_index"),
            None,
            prepare_only,
            defer_commit,
        )
    if isinstance(request, tuple):
        if len(request) == 3:
            selected, start, target = request
        else:
            selected, start = request
            target = None
        return None, selected, start, target, False, False
    return None, request, 0, None, False, False


@dataclass(slots=True)
class _DirectStoreRequestState:
    futures: deque[Future] = field(default_factory=deque)
    pending_keys: set[str] = field(default_factory=set)
    submitted_end: dict[int, int] = field(default_factory=dict)
    committed_end: dict[int, int] = field(default_factory=dict)
    planned_end: int = 0
    planned_hash: int = 0
    accepted_store_end: Optional[int] = None
    submitted_jobs: int = 0
    submitted_pages: int = 0
    submitted_legacy_objects: int = 0
    submitted_bytes: int = 0
    fallback_reasons: dict[int, str] = field(default_factory=dict)
    source_ready_event: Any = None
    source_ready_event_source: str = "missing"
    source_ready_token_end: int = 0
    source_ready_events: tuple[Any, ...] = ()
    source_ready_events_token_end: int = 0
    finalized: bool = False
    remote_fill_handoff: Optional[RemoteFillHandoff] = None
    remote_fill_session: Optional[RemoteFillProducerSession] = None
    remote_fill_futures: deque[Future] = field(default_factory=deque)
    remote_fill_last_future: Optional[Future] = None
    remote_fill_next_window_id: int = 0
    remote_fill_source_generation: int = 0
    remote_fill_probe_end: int = 0
    remote_fill_queued_bytes: int = 0
    remote_fill_queued_windows: int = 0
    remote_fill_oldest_enqueued_at: float = 0.0
    remote_fill_backlog_logged: bool = False
    remote_fill_terminal: Optional[RemoteFillTerminalResult] = None
    remote_fill_disabled_reason: str = ""
    remote_fill_metrics_started: bool = False
    remote_fill_viable_counted: bool = False
    remote_fill_active_counted: bool = False
    remote_fill_submitted_bytes: int = 0
    remote_fill_persistent_started_at: float = 0.0


@dataclass(slots=True)
class _DirectPageBatch:
    req_id: str
    keys: List[CacheEngineKey]
    ptrs: List[List[int]]
    sizes: List[List[int]]
    owners: tuple[torch.Tensor, ...]
    ready_event: Any
    group_ends: dict[int, int]
    ranges: tuple[tuple[int, int], ...] = ()
    ready_events: tuple[Any, ...] = ()



def _mtp_dw_diag_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_MTP_DW_DIAG", "0") == "1"


def _mtp_dw_event(stage: str, **fields: Any) -> None:
    if not _mtp_dw_diag_enabled():
        return
    payload = {"schema": 1, "stage": stage, "owner": "lmcache_ascend_store"}
    payload.update(fields)
    logger.info("[MTP_DW] %s", json.dumps(payload, separators=(",", ":")))


def _dsa_debug_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dsa_debug_summary_enabled() -> bool:
    return os.environ.get(
        "VLLM_ASCEND_DSA_SHRINK_DEBUG_MODE", "fail_only"
    ).lower() in ("summary", "trace", "verbose", "all")


def _dsa_debug_limit() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG_LIMIT", "8")))
    except ValueError:
        return 8


def _dsa_debug_should_log(owner: Any, site: str) -> bool:
    if not _dsa_debug_enabled():
        return False
    if not _dsa_debug_summary_enabled():
        return False
    counts = getattr(owner, "_dsa_shrink_debug_counts", None)
    if counts is None:
        counts = {}
        owner._dsa_shrink_debug_counts = counts
    count = counts.get(site, 0)
    if count >= _dsa_debug_limit():
        return False
    counts[site] = count + 1
    return True


def _dsa_debug_shape(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    try:
        return len(value)
    except TypeError:
        return type(value).__name__


def _dsa_debug_sample(value: Any, limit: Optional[int] = None) -> Any:
    if value is None:
        return None
    limit = _dsa_debug_limit() if limit is None else limit
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            return value.detach().reshape(-1)[:limit].to(device="cpu").tolist()
        return list(value[:limit])
    except Exception as exc:
        return f"{type(value).__name__}:sample_failed:{exc}"


def _dsa_debug_minmax_count(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            flat = value.detach().reshape(-1)
            return (
                flat.min().to(device="cpu").item(),
                flat.max().to(device="cpu").item(),
                int(flat.numel()),
            )
        seq = list(value)
        if not seq:
            return None
        return (min(seq), max(seq), len(seq))
    except Exception as exc:
        return f"{type(value).__name__}:minmax_failed:{exc}"

class ThreadSafeEventList:
    """queue.Queue-backed, list-compatible thread-safe buffer for
    ``CacheStoreEvent`` objects.

    Upstream ``LMCacheEngine`` treats ``self.kv_events`` as a list that
    callers ``.append(...)`` to and ``get_kv_events`` snapshots-and-resets.
    When ``store_async`` is active, appends happen on the background
    worker thread while drains happen on the main thread - a data race
    on a plain ``list``.  This wrapper funnels all appends through a
    ``queue.Queue`` so we inherit its producer/consumer thread safety,
    while preserving ``.append(...)`` / truthiness semantics that
    upstream code paths rely on.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[CacheStoreEvent]" = queue.Queue()

    def append(self, event: CacheStoreEvent) -> None:
        self._q.put(event)

    def __bool__(self) -> bool:
        return not self._q.empty()

    def __len__(self) -> int:
        return self._q.qsize()

    def drain(self) -> List[CacheStoreEvent]:
        out: List[CacheStoreEvent] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class _SparseCacheAppend:
    """Rollback guard for append-only request cache mutations."""

    _LAYER_FIELDS = (
        "cached_keys",
        "cached_memory_objs",
        "cached_tensors",
        "cached_chunk_dev_ptrs",
        "cached_shared_handles",
    )

    def __init__(self, cache: dict[str, Any]) -> None:
        self._ranges = [
            (values, len(values))
            for name in ("cached_starts", "cached_ends")
            if (values := cache.get(name)) is not None
        ]
        self._layers = []
        for name in self._LAYER_FIELDS:
            values = cache.get(name)
            if values is not None:
                self._layers.append(
                    (values, len(values), [len(layer or []) for layer in values])
                )
        pointers = cache.get("cached_chunk_ptrs_npu")
        self._pointers = None if pointers is None else (pointers, list(pointers))
        self.committed = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        if self.committed:
            return
        for values, size in self._ranges:
            del values[size:]
        for values, outer_size, sizes in self._layers:
            for layer, size in zip(values[:outer_size], sizes, strict=True):
                if isinstance(layer, list):
                    del layer[size:]
            del values[outer_size:]
        if self._pointers is not None:
            values, snapshot = self._pointers
            values[:] = snapshot


class AscendLMCacheEngine(LMCacheEngine):
    """Ascend NPU variant of ``LMCacheEngine`` with an async store path."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
        collective_all_true_fn: Optional[Callable[[bool], bool]] = None,
    ):
        super().__init__(
            config,
            metadata,
            token_database,
            gpu_connector,
            broadcast_fn,
            broadcast_object_fn,
            collective_all_true_fn,
        )
        self.is_store_async = self.config.store_async
        self._pending_sync_store_futures: set[Future] = set()
        self._require_store_completion = False
        self._direct_store_enabled = bool(
            self.is_store_async
            and self.config.pd_role != "receiver"
            and (
                self.config.get_extra_config_value(
                    "mooncake_direct_npu_prefill_store", False
                )
                or bool(
                    self.config.enable_remote_lmcache_store
                    and remote_fill_h0_qualified()
                )
            )
            and self._is_passive() is False
            and mooncake_layer_pages_enabled(self.config)
            and not self.config.get_extra_config_value("save_chunk_meta", False)
            and self.config.get_extra_config_value("use_ascend_direct", False)
        )
        self._direct_store_states: dict[str, _DirectStoreRequestState] = {}
        self._direct_store_jobs: deque[Future] = deque()
        self._direct_retry_args: dict[Future, tuple[Any, ...]] = {}
        self._direct_completed_futures: WeakSet[Future] = WeakSet()
        self._live_source_builders: dict[str, dict[str, Any]] = {}
        self._completed_live_sources: dict[str, dict[str, Any]] = {}
        self._remote_fill_runtime: Optional[DecoderRemoteFillRuntime] = None
        self._remote_fill_decoder_initialized = False
        self._remote_fill_fatal_transfers: tuple[str, ...] = ()
        self._remote_fill_fatal_lock = threading.Lock()
        self._remote_fill_shared_group1_supported = metadata.world_size <= 1
        # Diagnostic tensor owners are retained only while the opt-in content
        # diagnostic is enabled.  Fingerprinting is deliberately deferred
        # until the adapter has fenced the producer event, so diagnostic host
        # readback cannot hide whether that event was initially incomplete.
        self._pending_live_source_diagnostics: dict[str, dict[str, Any]] = {}
        self._store_queue_maxsize = max(0, int(self.config.store_async_max_queue_size))
        if self.is_store_async:
            self._store_queue: Optional[queue.Queue] = None
            self._store_worker_thread: Optional[threading.Thread] = None
            self._store_lock = threading.Lock()
            self._store_cv = threading.Condition(self._store_lock)

            # req_id -> number of in-flight background stores.  Entry
            # removed when count hits 0.
            self._pending_store_reqs: Dict[str, int] = {}
            # req_ids whose generation finished while stores were still
            # draining.  Re-checked on each ``get_finished_stores`` call.
            self._deferred_finished_req_ids: set = set()
            # req_ids already reported as finished_sending, to prevent
            # duplicate reports after the scheduler frees blocks.
            self._reported_finished_store_ids: set = set()

        # Serialize between store and lookup.
        self._engine_state_lock = threading.RLock()

        self._device_id: Optional[int] = None

        if self.kv_events_enabled and self.is_store_async:
            self.kv_events = ThreadSafeEventList()

    def _ensure_store_worker(self) -> None:
        if self._store_queue is not None:
            return
        self._store_queue = queue.Queue(maxsize=self._store_queue_maxsize)
        self._store_worker_thread = threading.Thread(
            target=self._store_worker_loop,
            daemon=True,
            name="lmcache-ascend-store-worker",
        )
        self._store_worker_thread.start()

    def post_init(self, **kwargs) -> None:
        super().post_init(**kwargs)
        try:
            self._initialize_decoder_remote_fill()
        except Exception as error:
            log_remote_fill_validation_failure(
                logger,
                code="RF-D-000",
                stage="decoder_startup_validation",
                action="STARTUP_ABORT",
                reason=(
                    "RemoteFill was enabled but its decoder invariants were "
                    "not satisfied; the endpoint was not advertised"
                ),
                error=error,
                severity="critical",
            )
            raise
        if self.is_store_async and not self._direct_store_enabled:
            self._device_id = torch.npu.current_device()
            self._ensure_store_worker()
            queue_mode = "unbounded" if self._store_queue_maxsize == 0 else "bounded"
            logger.info(
                "Ascend async store queue initialized: mode=%s maxsize=%d",
                queue_mode,
                self._store_queue_maxsize,
            )
        elif self._direct_store_enabled:
            logger.info(
                "Direct NPU prefill store enabled: max_queue_size=%d",
                self._store_queue_maxsize,
            )

    def get_remote_fill_placement_info(self) -> dict[str, int | str | bool] | None:
        """Return pointer-free decoder placement or ``None`` when unavailable."""

        runtime = self._remote_fill_runtime
        if runtime is None or not runtime.available or not self.is_healthy():
            return None
        return runtime.placement.to_dict()

    def get_remote_fill_metrics(self) -> dict[str, int] | None:
        """Return fixed-cardinality decoder metrics when remote fill is active."""

        runtime = self._remote_fill_runtime
        return None if runtime is None else runtime.metrics_snapshot()

    def remote_fill_requires_paired_restart(self) -> bool:
        """Return whether an armed remote fill made this process unsafe."""

        return bool(self._remote_fill_fatal_transfers)

    def direct_prefill_store_enabled(self) -> bool:
        """Return whether this worker can use direct NPU page publication."""
        return self._direct_store_enabled

    def direct_prefill_plan_supported(
        self,
        group_caches: dict[int, list],
    ) -> bool:
        """Check the current direct layout before choosing the save path."""
        check = getattr(self.gpu_connector, "direct_page_layout_supported", None)
        if not callable(check):
            return False
        for group, caches in group_caches.items():
            if not check(caches, group):
                reason = getattr(
                    self.gpu_connector, "direct_page_plan_rejection", None
                )
                detail = (
                    reason(group) if callable(reason) else "unsupported_layout"
                )
                logged = getattr(self, "_direct_preflight_rejections", set())
                if (group, detail) not in logged:
                    logged.add((group, detail))
                    self._direct_preflight_rejections = logged
                    logger.warning(
                        "Direct NPU prefill layout preflight failed; using the "
                        "safe layerwise writer: kv_group=%d reason=%s",
                        group,
                        detail,
                    )
                return False
        return True

    def configure_remote_fill_producer(
        self,
        *,
        client_factory: Optional[Callable[..., Any]] = None,
        direct_submitter: Optional[Callable[..., Future]] = None,
        activation_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """Inject producer boundaries for hardware-independent tests.

        Args:
            client_factory: Optional ``(handoff, limits) -> RemoteFillClient``
                factory. Production uses the existing LMCache ZMQ transport.
            direct_submitter: Optional armed native submission callable.
                Production forwards to ``StorageManager``.
            activation_factory: Optional armed-attempt activation builder.
                Production requires the explicit H0 environment value.
        """

        if client_factory is not None:
            self._remote_fill_client_factory = client_factory
        if direct_submitter is not None:
            self._remote_fill_direct_submitter = direct_submitter
        if activation_factory is not None:
            self._remote_fill_activation_factory = activation_factory

    def close_remote_fill_producer(self) -> None:
        """Drain producer work and close lazily created control resources."""

        states = getattr(self, "_direct_store_states", {})
        deadline = time.perf_counter() + (
            float(self.config.remote_fill_native_hard_timeout_ms) / 1000.0
        )
        for state in states.values():
            while state.remote_fill_futures:
                future = state.remote_fill_futures.popleft()
                try:
                    future.result(timeout=max(0.0, deadline - time.perf_counter()))
                except FutureTimeoutError as error:
                    self._latch_remote_fill_producer_fatal(state)
                    raise RemoteFillFatalError(
                        "remote-fill producer shutdown exceeded its hard deadline"
                    ) from error
            if state.remote_fill_session is not None:
                state.remote_fill_session.close()
        executor = getattr(self, "_remote_fill_producer_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._remote_fill_producer_executor = None

    def drain_remote_fill_terminal_results(
        self,
    ) -> dict[str, dict[str, str | int]]:
        """Drain sanitized completed outcomes for scheduler/proxy handoff."""

        completed = getattr(self, "_completed_remote_fill_results", {})
        self._completed_remote_fill_results = {}
        return {
            req_id: terminal.as_dict()
            for req_id, terminal in completed.items()
        }

    def remote_fill_producer_metrics_snapshot(self) -> dict[str, Any]:
        """Return enabled-path producer aggregates without request identities."""

        metrics = getattr(self, "_remote_fill_producer_metrics", None)
        return {} if metrics is None else metrics.snapshot()

    def _get_remote_fill_producer_metrics(self) -> RemoteFillProducerMetrics:
        metrics = getattr(self, "_remote_fill_producer_metrics", None)
        if metrics is None:
            metrics = RemoteFillProducerMetrics()
            self._remote_fill_producer_metrics = metrics
        return metrics

    def _latch_remote_fill_producer_fatal(
        self,
        state: _DirectStoreRequestState,
    ) -> None:
        """Expose a producer-side armed ambiguity to the worker supervisor."""

        handoff = state.remote_fill_handoff
        if handoff is None:
            raise RuntimeError("remote-fill fatal state lacks a transfer identity")
        self._remote_fill_require_paired_restart((handoff.transfer_id,))

    @staticmethod
    def _log_remote_fill_prearm_failure(
        state: _DirectStoreRequestState,
        *,
        req_id: str,
        stage: str,
        reason: str,
        error: Optional[BaseException] = None,
    ) -> None:
        """Make a safe direct-fill fallback immediately operator-visible."""

        handoff = state.remote_fill_handoff
        log_remote_fill_validation_failure(
            logger,
            code="RF-P-004",
            stage=stage,
            action="PERSISTENT_ONLY",
            req_id=req_id,
            transfer_id=handoff.transfer_id if handoff is not None else None,
            reason=reason,
            error=error,
            severity="warning",
        )

    def _record_remote_fill_window_metrics(
        self,
        state: _DirectStoreRequestState,
        result: Any,
    ) -> None:
        metrics = self._get_remote_fill_producer_metrics()
        metrics.observe("reserve_seconds", float(result.reserve_seconds))
        metrics.observe("arm_seconds", float(result.arm_seconds))
        metrics.existing_pages(int(result.existing_pages))
        submitted_bytes = int(result.submitted_bytes)
        if submitted_bytes:
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
            cold_start_perf_log(
                logger,
                "remote_fill_direct_abandoned",
                reason=result.reason,
                armed=bool(result.armed),
                fatal_restart_required=bool(result.fatal_restart_required),
            )
            fatal = bool(result.fatal_restart_required)
            handoff = state.remote_fill_handoff
            log_remote_fill_validation_failure(
                logger,
                code="RF-P-900" if fatal else "RF-P-004",
                stage="window_terminal",
                action=(
                    "PAIRED_RESTART_REQUIRED" if fatal else "PERSISTENT_ONLY"
                ),
                transfer_id=(handoff.transfer_id if handoff is not None else None),
                reason=str(result.reason),
                memory_safety_uncertain=fatal,
                severity="critical" if fatal else "warning",
            )

    def _remote_fill_prepare_request(
        self,
        req_id: str,
        request_configs: Optional[dict],
        state: _DirectStoreRequestState,
    ) -> bool:
        if not bool(getattr(self.config, "enable_remote_lmcache_store", False)):
            return False
        if state.remote_fill_handoff is not None:
            return not state.remote_fill_disabled_reason
        try:
            handoff = parse_remote_fill_handoff(request_configs)
        except ValueError as error:
            state.remote_fill_disabled_reason = type(error).__name__
            log_remote_fill_validation_failure(
                logger,
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
        metrics = self._get_remote_fill_producer_metrics()
        if not state.remote_fill_metrics_started:
            metrics.start_attempt()
            state.remote_fill_metrics_started = True
        if (
            not handoff.global_te_push
            or os.environ.get(REMOTE_FILL_H0_QUALIFICATION_ENV, "")
            != DIRECT_PUSH_H0_QUALIFICATION_V1
        ):
            state.remote_fill_disabled_reason = "native_not_qualified"
            log_remote_fill_validation_failure(
                logger,
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
            log_remote_fill_validation_failure(
                logger,
                code="RF-P-002",
                stage="handoff_validation",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        if handoff.destination_tp_size != int(self.metadata.world_size):
            state.remote_fill_disabled_reason = "incompatible_tp_mapping"
            log_remote_fill_validation_failure(
                logger,
                code="RF-P-002",
                stage="handoff_validation",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=handoff.transfer_id,
                reason=state.remote_fill_disabled_reason,
                severity="warning",
            )
            return False
        if not self._remote_fill_circuit_allows():
            state.remote_fill_disabled_reason = "circuit_open"
            log_remote_fill_validation_failure(
                logger,
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
        cold_start_perf_log(
            logger,
            "remote_fill_producer_decision",
            req_id=req_id,
            enabled=True,
            destination_dp_rank=handoff.destination_dp_rank,
            destination_tp_size=handoff.destination_tp_size,
        )
        return True

    def _remote_fill_circuit_allows(self) -> bool:
        if not bool(self.config.remote_fill_circuit_breaker_enabled):
            return True
        lock = getattr(self, "_remote_fill_circuit_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._remote_fill_circuit_lock = lock
        with lock:
            now = time.monotonic()
            open_until = float(
                getattr(self, "_remote_fill_circuit_open_until", 0.0)
            )
            if now < open_until:
                return False
            if open_until:
                self._remote_fill_circuit_open_until = 0.0
                self._remote_fill_circuit_failures = 0
                metrics = getattr(self, "_remote_fill_producer_metrics", None)
                if metrics is not None:
                    metrics.set_gauge("circuit_breaker_state", 0)
            return True

    def _remote_fill_record_failure(self) -> None:
        if not bool(self.config.remote_fill_circuit_breaker_enabled):
            return
        lock = getattr(self, "_remote_fill_circuit_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._remote_fill_circuit_lock = lock
        with lock:
            failures = int(
                getattr(self, "_remote_fill_circuit_failures", 0)
            ) + 1
            self._remote_fill_circuit_failures = failures
            if failures >= int(
                self.config.remote_fill_circuit_breaker_failure_threshold
            ):
                self._remote_fill_circuit_open_until = (
                    time.monotonic()
                    + float(self.config.remote_fill_circuit_breaker_cooldown_sec)
                )
                self._get_remote_fill_producer_metrics().set_gauge(
                    "circuit_breaker_state", 1
                )

    def _remote_fill_record_success(self) -> None:
        lock = getattr(self, "_remote_fill_circuit_lock", None)
        if lock is None:
            return
        with lock:
            self._remote_fill_circuit_failures = 0
            self._remote_fill_circuit_open_until = 0.0
            self._get_remote_fill_producer_metrics().set_gauge(
                "circuit_breaker_state", 0
            )

    def _remote_fill_acquire_queue_capacity(
        self,
        state: _DirectStoreRequestState,
        byte_count: int,
    ) -> bool:
        """Acquire bounded producer admission before retaining queued work."""

        if byte_count < 0 or byte_count > int(
            self.config.remote_fill_max_inflight_bytes
        ):
            return False
        condition = getattr(self, "_remote_fill_queue_condition", None)
        if condition is None:
            condition = threading.Condition()
            self._remote_fill_queue_condition = condition
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
            global_bytes = int(getattr(self, "_remote_fill_queued_bytes", 0))
            if (
                state.remote_fill_queued_windows
                >= int(self.config.remote_fill_max_inflight_windows_per_request)
                or state.remote_fill_queued_bytes + byte_count > request_byte_limit
                or global_bytes + byte_count > global_byte_limit
            ):
                return False
            state.remote_fill_queued_windows += 1
            state.remote_fill_queued_bytes += byte_count
            if state.remote_fill_queued_windows == 1:
                state.remote_fill_oldest_enqueued_at = time.perf_counter()
            self._remote_fill_queued_bytes = global_bytes + byte_count
            metrics = self._get_remote_fill_producer_metrics()
            metrics.add_gauge("inflight_windows", 1)
            metrics.add_gauge("inflight_bytes", byte_count)
            return True

    def _remote_fill_release_queue_capacity(
        self,
        state: _DirectStoreRequestState,
        byte_count: int,
    ) -> None:
        condition = getattr(self, "_remote_fill_queue_condition", None)
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
            self._remote_fill_queued_bytes = max(
                0,
                int(getattr(self, "_remote_fill_queued_bytes", 0)) - byte_count,
            )
            metrics = self._get_remote_fill_producer_metrics()
            metrics.add_gauge("inflight_windows", -1)
            metrics.add_gauge("inflight_bytes", -byte_count)

    def _remote_fill_protocol_limits(self) -> ProtocolLimits:
        return ProtocolLimits(
            max_rpc_message_bytes=int(
                self.config.remote_fill_max_rpc_message_bytes
            ),
            max_control_pages_per_window=int(
                self.config.remote_fill_max_control_pages_per_window
            ),
            max_window_bytes=int(self.config.remote_fill_max_inflight_bytes),
            max_active_transactions=int(
                self.config.remote_fill_max_active_transactions
            ),
            max_inflight_windows_per_transaction=int(
                self.config.remote_fill_max_inflight_windows_per_request
            ),
            max_reserved_bytes=int(self.config.remote_fill_max_reserved_bytes),
            max_bytes_per_transaction=int(
                self.config.remote_fill_max_bytes_per_request
            ),
        )

    def _remote_fill_immutable_layout(self) -> tuple[str, RemoteFillDecoderLayout]:
        """Build the startup-immutable source layout once after opt-in."""

        cached = getattr(self, "_remote_fill_layout_cache", None)
        if cached is not None:
            return cached
        with self._engine_state_lock:
            cached = getattr(self, "_remote_fill_layout_cache", None)
            if cached is None:
                layout_tag, _ = mooncake_payload_layout(self.config, self.metadata)
                layout = build_decoder_layout(
                    self.config,
                    self.metadata,
                    layout_tag=layout_tag,
                    num_layers=int(self.num_layers),
                )
                cached = (layout_tag, layout)
                self._remote_fill_layout_cache = cached
            return cached

    def _remote_fill_static_spec(
        self,
        handoff: RemoteFillHandoff,
    ) -> RemoteFillStaticSpec:
        layout_tag, layout = self._remote_fill_immutable_layout()
        token_hash_algorithm = str(self.config.pre_caching_hash_algorithm)
        python_hash_seed = (
            os.environ.get("PYTHONHASHSEED", "")
            if token_hash_algorithm == "builtin"
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
            shared_group1=True,
            tp_size=int(self.metadata.world_size),
            dp_size=handoff.destination_dp_size,
            global_te_push=handoff.global_te_push,
            token_hash_algorithm=token_hash_algorithm,
            python_hash_seed=python_hash_seed,
        )

    def _remote_fill_create_session(
        self,
        req_id: str,
        state: _DirectStoreRequestState,
        required_store_end_hint: int,
    ) -> RemoteFillProducerSession:
        handoff = state.remote_fill_handoff
        if handoff is None:
            raise RuntimeError("remote-fill handoff is unavailable")
        limits = self._remote_fill_protocol_limits()
        verification_key = handoff.descriptor_verification_key
        static_spec = self._remote_fill_static_spec(handoff)
        factory = getattr(self, "_remote_fill_client_factory", None)
        client = (
            factory(handoff, limits)
            if callable(factory)
            else create_remote_fill_client(
                handoff.control_endpoint,
                operation_timeouts_ms={
                    OperationKind.NEGOTIATE: int(
                        self.config.remote_fill_open_timeout_ms
                    ),
                    OperationKind.OPEN: int(
                        self.config.remote_fill_open_timeout_ms
                    ),
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
                    OperationKind.ABORT: int(
                        self.config.remote_fill_arm_timeout_ms
                    ),
                    OperationKind.FINISH: int(
                        self.config.remote_fill_finish_timeout_ms
                    ),
                },
                limits=limits,
            )
        )
        planned_windows = (
            required_store_end_hint
            + int(self.config.remote_fill_window_tokens)
            - 1
        ) // int(self.config.remote_fill_window_tokens)
        negotiation_cache = getattr(self, "_remote_fill_negotiation_cache", None)
        if negotiation_cache is None:
            with self._engine_state_lock:
                negotiation_cache = getattr(
                    self, "_remote_fill_negotiation_cache", None
                )
                if negotiation_cache is None:
                    negotiation_cache = RemoteFillNegotiationCache()
                    self._remote_fill_negotiation_cache = negotiation_cache
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
            self._get_remote_fill_producer_metrics().add_gauge(
                "active_transactions", 1
            )
            state.remote_fill_active_counted = True
        return session

    def _remote_fill_control_pages(
        self,
        batch: _DirectPageBatch,
    ) -> tuple[ControlPage, ...]:
        if len(batch.keys) != len(batch.sizes) or len(batch.keys) != len(
            batch.ranges
        ):
            raise ValueError("remote-fill source page metadata is incomplete")
        layout_tag, layout = self._remote_fill_immutable_layout()
        pages = [
            ControlPage(
                canonical_key=key.to_string(),
                kv_group=int(key.kv_group),
                chunk_index=start // int(self.config.chunk_size),
                chunk_start=start,
                chunk_end=end,
                valid_tokens=end - start,
                destination_tp_rank=0,
                expected_bytes=sum(page_sizes),
                layer_count=int(self.num_layers),
                layout_tag=layout_tag,
            )
            for key, page_sizes, (start, end) in zip(
                batch.keys, batch.sizes, batch.ranges, strict=True
            )
        ]
        pages.sort(key=lambda page: (page.chunk_index, page.kv_group))
        for page in pages:
            expected = layout.group(page.kv_group).expected_bytes(
                page.valid_tokens, layout.num_layers
            )
            if page.expected_bytes != expected:
                raise ValueError("remote-fill source page byte layout changed")
        self._remote_fill_validate_page_pairs(tuple(pages))
        return tuple(pages)

    def _remote_fill_pages_per_window(self) -> int:
        """Return the bounded two-group control-page capacity of one window."""

        by_tokens = (
            int(self.config.remote_fill_window_tokens)
            // int(self.config.chunk_size)
            * 2
        )
        maximum = min(
            int(self.config.remote_fill_max_control_pages_per_window),
            by_tokens,
        )
        return maximum - maximum % 2

    @staticmethod
    def _remote_fill_select_batch_pages(
        batch: _DirectPageBatch,
        pages: tuple[ControlPage, ...],
    ) -> _DirectPageBatch:
        """Select one bounded control window from an existing source batch."""

        sources = {
            (key.to_string(), int(key.kv_group)): (
                key,
                page_ptrs,
                page_sizes,
                page_range,
            )
            for key, page_ptrs, page_sizes, page_range in zip(
                batch.keys,
                batch.ptrs,
                batch.sizes,
                batch.ranges,
                strict=True,
            )
        }
        if len(sources) != len(batch.keys):
            raise ValueError("remote-fill source page identity is duplicated")
        selected = [
            sources[(page.canonical_key, page.kv_group)] for page in pages
        ]
        group_ends: dict[int, int] = {}
        for page, (_, _, _, (_, end)) in zip(pages, selected, strict=True):
            group_ends[page.kv_group] = max(
                group_ends.get(page.kv_group, 0), end
            )
        return _DirectPageBatch(
            req_id=batch.req_id,
            keys=[item[0] for item in selected],
            ptrs=[item[1] for item in selected],
            sizes=[item[2] for item in selected],
            owners=batch.owners,
            ready_event=batch.ready_event,
            group_ends=group_ends,
            ranges=tuple(item[3] for item in selected),
            ready_events=batch.ready_events,
        )

    @staticmethod
    def _remote_fill_validate_page_pairs(
        pages: tuple[ControlPage, ...],
    ) -> None:
        by_chunk: dict[int, set[int]] = {}
        for page in pages:
            groups = by_chunk.setdefault(page.chunk_index, set())
            if page.kv_group in groups:
                raise ValueError("remote-fill page identity is duplicated")
            groups.add(page.kv_group)
        if not by_chunk or any(groups != {0, 1} for groups in by_chunk.values()):
            raise ValueError("remote fill requires an exact two-group page pair")

    def _remote_fill_probe_control_pages(
        self,
        plans: dict[int, list[tuple[int, int, CacheEngineKey]]],
        probe_start: int,
        probe_end: int,
    ) -> tuple[ControlPage, ...]:
        """Build pointer-free cached-prefix manifests from canonical plans."""

        _, layout = self._remote_fill_immutable_layout()
        pages: list[ControlPage] = []
        for group in (0, 1):
            for start, end, key in plans.get(group, ()):
                if end <= probe_start or end > probe_end:
                    continue
                valid_tokens = end - start
                pages.append(
                    ControlPage(
                        canonical_key=key.to_string(),
                        kv_group=group,
                        chunk_index=start // layout.chunk_size,
                        chunk_start=start,
                        chunk_end=end,
                        valid_tokens=valid_tokens,
                        destination_tp_rank=0,
                        expected_bytes=layout.group(group).expected_bytes(
                            valid_tokens, layout.num_layers
                        ),
                        layer_count=layout.num_layers,
                        layout_tag=layout.layout_tag,
                    )
                )
        pages.sort(key=lambda page: (page.chunk_index, page.kv_group))
        result = tuple(pages)
        if result:
            self._remote_fill_validate_page_pairs(result)
        return result

    def _remote_fill_prefix_plans(
        self,
        tokens: list[int],
        request_configs: Optional[dict],
        prefix_end: int,
    ) -> dict[int, list[tuple[int, int, CacheEngineKey]]]:
        """Rebuild canonical keys when a full hit has no suffix plan."""

        prefix_tokens = tokens[:prefix_end]
        latent = list(
            self.token_database.process_tokens(
                tokens=prefix_tokens,
                request_configs=request_configs,
                kv_group=0,
            )
        )
        hashes = [int(key.chunk_hash) for _, _, key in latent]
        offsets = [end - start for start, end, _ in latent]
        indexer = list(
            self.token_database.process_tokens(
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
                kv_group=1,
            )
        )
        if len(indexer) != len(latent) or any(
            (left[0], left[1], left[2].chunk_hash)
            != (right[0], right[1], right[2].chunk_hash)
            for left, right in zip(latent, indexer, strict=True)
        ):
            raise RuntimeError(
                "Remote-fill KV groups produced different prefix plans"
            )
        return {0: latent, 1: indexer}

    def _schedule_remote_fill_probe_pages(
        self,
        req_id: str,
        state: _DirectStoreRequestState,
        pages: tuple[ControlPage, ...],
        required_store_end_hint: int,
    ) -> None:
        """Queue bounded, ordered cached-prefix probes before direct writes."""

        if not pages or state.remote_fill_handoff is None:
            return
        maximum = self._remote_fill_pages_per_window()
        if maximum <= 0:
            state.remote_fill_disabled_reason = "invalid_control_page_limit"
            self._get_remote_fill_producer_metrics().abandon(
                "invalid control page limit", allocation=True
            )
            return
        executor = getattr(self, "_remote_fill_producer_executor", None)
        if executor is None:
            try:
                executor = ThreadPoolExecutor(
                    max_workers=int(self.config.remote_fill_direct_worker_count),
                    thread_name_prefix="lmcache-remote-fill-producer",
                )
            except Exception as error:
                state.remote_fill_disabled_reason = type(error).__name__
                self._remote_fill_record_failure()
                return
            self._remote_fill_producer_executor = executor
        control_windows = tuple(
            pages[offset : offset + maximum]
            for offset in range(0, len(pages), maximum)
        )
        metrics = self._get_remote_fill_producer_metrics()
        queued_at = time.perf_counter()
        if not self._remote_fill_acquire_queue_capacity(state, 0):
            state.remote_fill_disabled_reason = "producer_backpressure"
            metrics.abandon("producer backpressure", allocation=True)
            return
        metrics.observe("queue_wait_seconds", time.perf_counter() - queued_at)
        first_window_id = state.remote_fill_next_window_id
        state.remote_fill_next_window_id += len(control_windows)
        previous = state.remote_fill_last_future

        def run() -> Any:
            try:
                if previous is not None:
                    previous.result()
                if state.remote_fill_session is None:
                    state.remote_fill_session = self._remote_fill_create_session(
                        req_id,
                        state,
                        required_store_end_hint,
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
                            self._remote_fill_record_failure()
                    self._record_remote_fill_window_metrics(state, result)
                    cold_start_perf_log(
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
                self._latch_remote_fill_producer_fatal(state)
                raise
            except Exception as error:
                state.remote_fill_disabled_reason = type(error).__name__
                if state.remote_fill_session is not None:
                    state.remote_fill_session.direct_viable = False
                self._remote_fill_record_failure()
                return None
            finally:
                self._remote_fill_release_queue_capacity(state, 0)

        try:
            future = executor.submit(run)
        except Exception as error:
            self._remote_fill_release_queue_capacity(state, 0)
            state.remote_fill_disabled_reason = type(error).__name__
            self._remote_fill_record_failure()
            return
        state.remote_fill_futures.append(future)
        state.remote_fill_last_future = future

    @staticmethod
    def _remote_fill_source_plan(batch: _DirectPageBatch) -> DirectPushSourcePlan:
        pages = tuple(
            DirectPushPageSource(
                canonical_key=key.to_string(),
                kv_group=int(key.kv_group),
                source_ptrs=tuple(page_ptrs),
                source_lengths=tuple(page_sizes),
            )
            for key, page_ptrs, page_sizes in zip(
                batch.keys, batch.ptrs, batch.sizes, strict=True
            )
        )
        if not batch.ready_events:
            raise ValueError("remote fill requires complete producer fences")
        return DirectPushSourcePlan(
            pages=pages,
            owners=batch.owners,
            producer_events=batch.ready_events,
        )

    def _schedule_remote_fill_batch(
        self,
        state: _DirectStoreRequestState,
        batch: _DirectPageBatch,
        required_store_end_hint: int,
    ) -> None:
        if state.remote_fill_handoff is None or state.remote_fill_disabled_reason:
            return
        try:
            control_pages = self._remote_fill_control_pages(batch)
        except Exception as error:
            state.remote_fill_disabled_reason = type(error).__name__
            self._log_remote_fill_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="control_page_planning",
                reason=state.remote_fill_disabled_reason,
                error=error,
            )
            self._remote_fill_record_failure()
            return
        maximum = self._remote_fill_pages_per_window()
        if maximum <= 0:
            state.remote_fill_disabled_reason = "invalid_control_page_limit"
            self._log_remote_fill_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="control_page_planning",
                reason=state.remote_fill_disabled_reason,
            )
            self._get_remote_fill_producer_metrics().abandon(
                "invalid control page limit", allocation=True
            )
            return
        control_windows = tuple(
            control_pages[offset : offset + maximum]
            for offset in range(0, len(control_pages), maximum)
        )
        byte_count = sum(sum(page_sizes) for page_sizes in batch.sizes)
        queued_at = time.perf_counter()
        if not self._remote_fill_acquire_queue_capacity(state, byte_count):
            state.remote_fill_disabled_reason = "producer_backpressure"
            self._log_remote_fill_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="producer_admission",
                reason=state.remote_fill_disabled_reason,
            )
            metrics = self._get_remote_fill_producer_metrics()
            metrics.abandon("producer backpressure", allocation=True)
            cold_start_perf_log(
                logger,
                "remote_fill_batch_skipped",
                req_id=batch.req_id,
                reason="producer_backpressure",
                bytes=byte_count,
            )
            return
        self._get_remote_fill_producer_metrics().observe(
            "queue_wait_seconds", time.perf_counter() - queued_at
        )
        try:
            source_plan = self._remote_fill_source_plan(batch)
        except Exception as error:
            self._remote_fill_release_queue_capacity(state, byte_count)
            state.remote_fill_disabled_reason = type(error).__name__
            self._log_remote_fill_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="source_plan_validation",
                reason=state.remote_fill_disabled_reason,
                error=error,
            )
            self._remote_fill_record_failure()
            return
        cold_start_perf_log(
            logger,
            "remote_fill_source_ready",
            req_id=batch.req_id,
            page_count=len(control_pages),
            bytes=byte_count,
        )
        executor = getattr(self, "_remote_fill_producer_executor", None)
        if executor is None:
            try:
                executor = ThreadPoolExecutor(
                    max_workers=int(self.config.remote_fill_direct_worker_count),
                    thread_name_prefix="lmcache-remote-fill-producer",
                )
            except Exception as error:
                self._remote_fill_release_queue_capacity(state, byte_count)
                state.remote_fill_disabled_reason = type(error).__name__
                self._log_remote_fill_prearm_failure(
                    state,
                    req_id=batch.req_id,
                    stage="producer_executor_creation",
                    reason=state.remote_fill_disabled_reason,
                    error=error,
                )
                self._remote_fill_record_failure()
                return
            self._remote_fill_producer_executor = executor
        first_window_id = state.remote_fill_next_window_id
        state.remote_fill_next_window_id += len(control_windows)
        source_generation = state.remote_fill_source_generation
        previous = state.remote_fill_last_future

        def run() -> Any:
            try:
                if previous is not None:
                    previous.result()
                if state.remote_fill_session is None:
                    state.remote_fill_session = self._remote_fill_create_session(
                        batch.req_id, state, required_store_end_hint
                    )
                session = state.remote_fill_session
                if not session.direct_viable:
                    return None
                submitter = getattr(
                    self,
                    "_remote_fill_direct_submitter",
                    self.storage_manager.submit_remote_fill_direct_push,
                )
                activation_factory = getattr(
                    self,
                    "_remote_fill_activation_factory",
                    lambda attempt_id: NativeDirectPushActivation(
                        h0_qualification=os.environ.get(
                            REMOTE_FILL_H0_QUALIFICATION_ENV, ""
                        ),
                        native_transfer_attempt_id=attempt_id,
                        arm_acknowledged=True,
                    ),
                )
                results = []
                for window_offset, window_pages in enumerate(control_windows):
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
                            self._remote_fill_record_failure()
                    self._record_remote_fill_window_metrics(state, result)
                    cold_start_perf_log(
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
                        bytes=sum(
                            page.expected_bytes for page in window_pages
                        ),
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
                    self._latch_remote_fill_producer_fatal(state)
                    metrics = self._get_remote_fill_producer_metrics()
                    metrics.failure("fatal restart")
                    metrics.abandon("fatal restart")
                    cold_start_perf_log(
                        logger,
                        "remote_fill_fatal_restart",
                        req_id=batch.req_id,
                        phase="native_or_control_terminal",
                    )
                    raise
                state.remote_fill_disabled_reason = type(error).__name__
                if state.remote_fill_session is not None:
                    state.remote_fill_session.direct_viable = False
                self._log_remote_fill_prearm_failure(
                    state,
                    req_id=batch.req_id,
                    stage="producer_window_execution",
                    reason=state.remote_fill_disabled_reason,
                    error=error,
                )
                self._remote_fill_record_failure()
                metrics = self._get_remote_fill_producer_metrics()
                metrics.failure("prearm failure")
                metrics.abandon("prearm failure")
                return None
            finally:
                self._remote_fill_release_queue_capacity(state, byte_count)

        try:
            future = executor.submit(run)
        except Exception as error:
            self._remote_fill_release_queue_capacity(state, byte_count)
            state.remote_fill_disabled_reason = type(error).__name__
            self._log_remote_fill_prearm_failure(
                state,
                req_id=batch.req_id,
                stage="producer_executor_submission",
                reason=state.remote_fill_disabled_reason,
                error=error,
            )
            self._remote_fill_record_failure()
            return
        state.remote_fill_futures.append(future)
        state.remote_fill_last_future = future

    def _wait_remote_fill_windows(self, state: _DirectStoreRequestState) -> None:
        try:
            while state.remote_fill_futures:
                state.remote_fill_futures.popleft().result()
        except RemoteFillFatalError:
            self._latch_remote_fill_producer_fatal(state)
            raise

    def _finish_remote_fill(
        self,
        req_id: str,
        state: _DirectStoreRequestState,
        required_store_end: int,
    ) -> None:
        if state.remote_fill_handoff is None:
            return
        if state.remote_fill_terminal is not None:
            return
        self._wait_remote_fill_windows(state)
        persistent_common_end = min(
            state.committed_end.get(group, 0) for group in (0, 1)
        )
        cold_start_perf_log(
            logger,
            "remote_fill_persistent_complete",
            req_id=req_id,
            transfer_id=state.remote_fill_handoff.transfer_id,
            persistent_common_end=persistent_common_end,
            required_store_end=required_store_end,
        )
        finish_control_seconds = 0.0
        try:
            if state.remote_fill_session is None:
                terminal = RemoteFillTerminalResult(
                    transfer_id=state.remote_fill_handoff.transfer_id,
                    outcome="PERSISTENT_ONLY",
                    persistent_common_end=persistent_common_end,
                    required_store_end=required_store_end,
                )
            else:
                finish_started = time.perf_counter()
                try:
                    terminal = state.remote_fill_session.finish(
                        required_store_end=required_store_end,
                        persistent_common_end=persistent_common_end,
                        final_partial_valid_tokens=(
                            required_store_end % int(self.config.chunk_size)
                        ),
                    )
                finally:
                    finish_control_seconds = time.perf_counter() - finish_started
                    self._get_remote_fill_producer_metrics().observe(
                        "finish_control_seconds",
                        finish_control_seconds,
                    )
        except RemoteFillFatalError:
            self._latch_remote_fill_producer_fatal(state)
            raise
        if terminal.outcome == "FATAL_RESTART":
            self._latch_remote_fill_producer_fatal(state)
        state.remote_fill_terminal = terminal
        metrics = self._get_remote_fill_producer_metrics()
        if state.remote_fill_persistent_started_at:
            metrics.observe(
                "persistent_seconds",
                time.perf_counter() - state.remote_fill_persistent_started_at,
            )
        metrics.finish_attempt(
            terminal.outcome,
            state.remote_fill_disabled_reason or "none",
        )
        if state.remote_fill_viable_counted:
            metrics.add_gauge("direct_viable", -1)
            state.remote_fill_viable_counted = False
        if state.remote_fill_active_counted:
            metrics.add_gauge("active_transactions", -1)
            state.remote_fill_active_counted = False
        if terminal.outcome == "LOCAL_FULL":
            metrics.add_bytes(
                "published_bytes", state.remote_fill_submitted_bytes
            )
        else:
            metrics.add_bytes(
                "discarded_bytes", state.remote_fill_submitted_bytes
            )
        if terminal.outcome == "LOCAL_FULL":
            self._remote_fill_record_success()
        elif terminal.outcome == "PERSISTENT_ONLY":
            log_remote_fill_validation_failure(
                logger,
                code="RF-P-004",
                stage="producer_terminal",
                action="PERSISTENT_ONLY",
                req_id=req_id,
                transfer_id=state.remote_fill_handoff.transfer_id,
                reason=state.remote_fill_disabled_reason or terminal.outcome,
                severity="warning",
            )
        cold_start_perf_log(
            logger,
            "remote_fill_producer_terminal",
            req_id=req_id,
            transfer_id=state.remote_fill_handoff.transfer_id,
            outcome=terminal.outcome,
            persistent_common_end=persistent_common_end,
            required_store_end=required_store_end,
            finish_control_ms=round(finish_control_seconds * 1000, 3),
        )
        completed = getattr(self, "_completed_remote_fill_results", None)
        if completed is None:
            completed = {}
            self._completed_remote_fill_results = completed
        completed[req_id] = terminal
        cold_start_perf_log(
            logger,
            "remote_fill_producer_metrics_snapshot",
            metrics=metrics.snapshot(),
        )

    def _wait_direct_backpressure(self) -> None:
        limit = self._store_queue_maxsize or 2
        while len(self._direct_store_jobs) >= limit:
            future = self._direct_store_jobs[0]
            _, pending = wait(
                (future,), timeout=self.config.blocking_timeout_secs
            )
            if pending:
                raise TimeoutError("Timed out waiting for direct-store queue space")
            self._direct_store_jobs.popleft()

    def _direct_store_done(
        self,
        req_id: str,
        key_strings: set[str],
        future: Future,
    ) -> None:
        with self._store_cv:
            if future in self._direct_completed_futures:
                return
            state = self._direct_store_states.get(req_id)
            try:
                succeeded = not future.cancelled() and future.exception() is None
            except Exception:
                succeeded = False
            if state is not None:
                if succeeded:
                    state.pending_keys.difference_update(key_strings)
            if not succeeded:
                return
            self._direct_completed_futures.add(future)
            count = self._pending_store_reqs.get(req_id, 1) - 1
            if count <= 0:
                self._pending_store_reqs.pop(req_id, None)
            else:
                self._pending_store_reqs[req_id] = count
            self._store_cv.notify_all()

    def _submit_direct_page_batch(self, batch: _DirectPageBatch) -> Future:
        assert self.storage_manager is not None
        self._wait_direct_backpressure()
        started = cold_start_perf_now() if cold_start_perf_enabled() else None
        state = self._direct_store_states.setdefault(
            batch.req_id, _DirectStoreRequestState()
        )
        if (
            state.remote_fill_metrics_started
            and not state.remote_fill_persistent_started_at
        ):
            state.remote_fill_persistent_started_at = time.perf_counter()
        key_strings = {key.to_string() for key in batch.keys}
        state.pending_keys.update(key_strings)
        with self._store_cv:
            self._pending_store_reqs[batch.req_id] = (
                self._pending_store_reqs.get(batch.req_id, 0) + 1
            )
        try:
            future = self.storage_manager.batched_put_external_pages(
                batch.keys,
                batch.ptrs,
                batch.sizes,
                batch.owners,
                batch.ready_events,
                batch.req_id,
            )
        except Exception:
            state.pending_keys.difference_update(key_strings)
            with self._store_cv:
                count = self._pending_store_reqs.get(batch.req_id, 1) - 1
                if count <= 0:
                    self._pending_store_reqs.pop(batch.req_id, None)
                else:
                    self._pending_store_reqs[batch.req_id] = count
                self._store_cv.notify_all()
            raise
        state.futures.append(future)
        self._direct_store_jobs.append(future)
        legacy_objects = sum(hasattr(key, "layer_id") for key in batch.keys)
        state.submitted_jobs += 1
        state.submitted_pages += len(batch.keys) - legacy_objects
        state.submitted_legacy_objects += legacy_objects
        state.submitted_bytes += sum(map(sum, batch.sizes))
        if started is not None:
            cold_start_perf_log(
                logger,
                "direct_npu_submit",
                started=started,
                req_id=batch.req_id,
                pages=len(batch.keys) - legacy_objects,
                legacy_objects=legacy_objects,
                format=(
                    "legacy_tail" if legacy_objects else "page"
                ),
                queue_depth=len(self._direct_store_jobs),
            )
        return future

    def _store_direct_cpu_group(
        self,
        req_id: str,
        tokens: list[int],
        kvcaches: list,
        slot_mapping: torch.Tensor,
        kv_group: int,
        start: int,
        request_configs: Optional[dict],
        slot_mapping_base: int = 0,
    ) -> int:
        if start >= len(tokens):
            return start
        if start < slot_mapping_base:
            raise RuntimeError(
                "Direct prefill CPU fallback cannot address an uncommitted "
                "prefix: "
                f"req_id={req_id}, kv_group={kv_group}, committed={start}, "
                f"slot_mapping_base={slot_mapping_base}"
            )
        mask = torch.zeros(len(tokens), dtype=torch.bool)
        mask[start:] = True
        storer = self.store_layer(
            tokens,
            mask=mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=start,
            sync=True,
            req_id=req_id,
            kv_group=kv_group,
            request_configs=request_configs,
            require_completion=True,
            all_layers_ready=True,
            slot_mapping_base=slot_mapping_base,
        )
        self._require_store_completion = True
        try:
            result = None
            while True:
                try:
                    value = next(storer)
                    if value is not None:
                        result = value
                except StopIteration:
                    break
            self.wait_for_pending_sync_stores()
        finally:
            self._require_store_completion = False
        return int(getattr(result, "committed_end", start) or start)

    def _retry_direct_cpu(
        self,
        req_id: str,
        tokens: list[int],
        group_caches: dict[int, list],
        slot_mappings: dict[int, torch.Tensor],
        request_configs: Optional[dict],
        state: _DirectStoreRequestState,
        slot_mapping_base: int = 0,
    ) -> None:
        for group, kvcaches in group_caches.items():
            committed = self._store_direct_cpu_group(
                req_id,
                tokens,
                kvcaches,
                slot_mappings[group],
                group,
                state.committed_end.get(group, 0),
                request_configs,
                slot_mapping_base,
            )
            state.committed_end[group] = committed
            state.submitted_end[group] = committed

    def _direct_required_end(
        self,
        state: _DirectStoreRequestState,
        tokens: list[int],
    ) -> int:
        del tokens
        if state.accepted_store_end is None:
            raise RuntimeError("direct store has no authoritative accepted end")
        return state.accepted_store_end

    def _finalize_direct_store(
        self,
        req_id: str,
        tokens: list[int],
        groups: Iterable[int],
        state: _DirectStoreRequestState,
        final: bool,
    ) -> None:
        if not final:
            return
        if state.finalized:
            return
        required_end = self._direct_required_end(state, tokens)
        invalid = {
            group: state.committed_end.get(group, 0)
            for group in groups
            if state.committed_end.get(group, 0) != required_end
        }
        if invalid:
            raise RuntimeError(
                "Direct prefill store published an invalid final frontier: "
                f"req_id={req_id} committed={invalid} required={required_end}"
            )
        self._finish_remote_fill(req_id, state, required_end)
        if (
            state.remote_fill_terminal is not None
            and state.remote_fill_terminal.outcome
            in ("PERSISTENCE_FAILED", "FATAL_RESTART")
        ):
            raise RuntimeError(
                "Remote-fill finalization rejected request forwarding: "
                f"outcome={state.remote_fill_terminal.outcome}"
            )
        state.finalized = True
        logger.info(
            "[req_id=%s] Direct prefill store complete for %d tokens: "
            "submitted pages=%d, legacy objects=%d, size=%.4f GB, jobs=%d, "
            "committed=%s",
            req_id,
            required_end,
            state.submitted_pages,
            state.submitted_legacy_objects,
            state.submitted_bytes / 1024**3,
            state.submitted_jobs,
            dict(sorted(state.committed_end.items())),
        )

    def _direct_suffix_plans(
        self,
        state: _DirectStoreRequestState,
        tokens: list[int],
        groups: Iterable[int],
        request_configs: Optional[dict],
    ) -> dict[int, list[tuple[int, int, CacheEngineKey]]]:
        """Hash only complete chunks added since the previous forward."""
        chunk_size = int(self.config.chunk_size)
        full_end = len(tokens) // chunk_size * chunk_size
        full_tokens = tokens if full_end == len(tokens) else tokens[:full_end]
        groups = tuple(groups)
        if all(
            state.submitted_end.get(group, 0) >= full_end for group in groups
        ):
            return {group: [] for group in groups}
        base = (
            state.planned_end
            if state.planned_end <= full_end
            and all(
                state.submitted_end.get(group, 0) >= state.planned_end
                for group in groups
            )
            else 0
        )
        if base == full_end:
            plan0 = []
        elif base:
            plan0 = list(
                self.token_database.process_tokens_from_prefix(
                    full_tokens,
                    prefix_token_count=base,
                    prefix_hash=state.planned_hash,
                    request_configs=request_configs,
                    kv_group=0,
                )
            )
        else:
            base = 0
            plan0 = list(
                self.token_database.process_tokens(
                    tokens=full_tokens,
                    request_configs=request_configs,
                    kv_group=0,
                )
            )
        hashes = [int(key.chunk_hash) for _, _, key in plan0]
        offsets = [end - start for start, end, _ in plan0]
        plans = {0: plan0}
        for group in groups:
            if group == 0:
                continue
            if not plan0:
                plans[group] = []
                continue
            plans[group] = [
                (start + base, end + base, key)
                for start, end, key in self.token_database.process_tokens(
                    hashes=hashes,
                    offsets=offsets,
                    request_configs=request_configs,
                    kv_group=group,
                )
            ]
            if len(plans[group]) != len(plan0) or any(
                (left[0], left[1], left[2].chunk_hash)
                != (right[0], right[1], right[2].chunk_hash)
                for left, right in zip(plan0, plans[group], strict=True)
            ):
                raise RuntimeError(
                    "Direct prefill KV groups produced different chunk plans: "
                    f"kv_group={group}"
                )
        if plan0:
            state.planned_end = plan0[-1][1]
            state.planned_hash = int(plan0[-1][2].chunk_hash)
        return plans

    def adopt_completed_layerwise_store(
        self, result: LayerwiseStoreResult
    ) -> None:
        """Reuse a completed layerwise store as direct-store progress."""
        if result.committed_end <= 0:
            return
        state = self._direct_store_states.setdefault(
            result.request_id, _DirectStoreRequestState()
        )
        group = int(result.kv_group)
        latest_full = None
        if result.keys:
            chunk_size = int(self.config.chunk_size)
            for start, end, key in zip(
                result.starts, result.ends, result.keys[0], strict=True
            ):
                if end - start == chunk_size:
                    latest_full = (end, int(key.chunk_hash))
            if latest_full is not None:
                end, chunk_hash = latest_full
                if end == state.planned_end and chunk_hash != state.planned_hash:
                    raise RuntimeError(
                        "Completed latent/index stores disagree on their hash "
                        "frontier"
                    )
                if end > state.planned_end:
                    state.planned_end = end
                    state.planned_hash = chunk_hash
        state.submitted_end[group] = max(
            state.submitted_end.get(group, 0), result.committed_end
        )
        state.committed_end[group] = max(
            state.committed_end.get(group, 0), result.committed_end
        )

    def _track_direct_batch(
        self,
        batch: _DirectPageBatch,
        retry: tuple[Any, ...],
    ) -> Future:
        future = self._submit_direct_page_batch(batch)
        state = self._direct_store_states[batch.req_id]
        for group, end in batch.group_ends.items():
            state.submitted_end[group] = max(state.submitted_end.get(group, 0), end)
        key_strings = {key.to_string() for key in batch.keys}
        self._direct_retry_args[future] = (*retry, key_strings, batch.group_ends)
        future.add_done_callback(
            lambda done,
            req_id=batch.req_id,
            keys=key_strings: self._direct_store_done(req_id, keys, done)
        )
        return future

    @staticmethod
    def _remember_direct_source_readiness(
        state: _DirectStoreRequestState,
        token_end: int,
        event: Any,
        event_source: str,
        events: tuple[Any, ...] = (),
    ) -> None:
        """Retain the producer dependency needed by a deferred final tail."""
        if event is not None and token_end >= state.source_ready_token_end:
            state.source_ready_event = event
            state.source_ready_event_source = (
                event_source
                if event_source and event_source != "missing"
                else "caller_supplied"
            )
            state.source_ready_token_end = token_end
        if events and token_end >= state.source_ready_events_token_end:
            unique: dict[int, Any] = {}
            for producer_event in events:
                if producer_event is not None:
                    unique.setdefault(id(producer_event), producer_event)
            state.source_ready_events = tuple(unique.values())
            state.source_ready_events_token_end = token_end

    @staticmethod
    def _direct_source_ready_events(
        state: _DirectStoreRequestState,
        token_end: int,
    ) -> tuple[Any, ...]:
        """Return the complete direct-push fence set through ``token_end``."""

        if state.source_ready_events_token_end < token_end:
            return ()
        return state.source_ready_events

    @staticmethod
    def _direct_source_ready_event(
        state: _DirectStoreRequestState,
        token_end: int,
        *,
        require_producer_event: bool = False,
    ) -> tuple[Any, str]:
        """Return an event that covers every source byte through ``token_end``."""
        if (
            state.source_ready_event is not None
            and state.source_ready_token_end >= token_end
        ):
            return state.source_ready_event, state.source_ready_event_source
        if require_producer_event:
            raise RuntimeError(
                "Direct DSA NPU store has no attention producer event "
                f"covering token_end={token_end}; refusing an unsafe "
                "current-stream fallback"
            )

        event = torch.npu.Event()
        event.record()
        state.source_ready_event = event
        state.source_ready_event_source = (
            "engine_current_stream_fallback_non_dsa"
        )
        state.source_ready_token_end = token_end
        return event, state.source_ready_event_source

    @staticmethod
    def _remote_fill_has_addressable_source_page(
        token_end: int,
        slot_mapping_base: int,
        chunk_size: int,
    ) -> bool:
        """Whether P owns every row of at least one final payload page."""

        if token_end <= 0 or chunk_size <= 0:
            return False
        last_page_start = (token_end - 1) // chunk_size * chunk_size
        return last_page_start >= slot_mapping_base

    def begin_live_source_descriptor(
        self, req_id: str, groups: tuple[int, ...] = (0, 1)
    ) -> None:
        self._live_source_builders.setdefault(
            req_id,
            {
                "segments": [],
                "compact_layers": None,
                "compact_runs": [],
                "compact_owners": None,
                "latent_layers": None,
                "latent_pages": [],
                "ends": {},
                "groups": groups,
                "invalid": False,
                "planned_end": 0,
                "planned_hash": 0,
            },
        )

    def discard_live_source_descriptor(self, req_id: str) -> None:
        """Discard only live-P2P state while retaining persistent-store state."""
        self._live_source_builders.pop(req_id, None)
        self._completed_live_sources.pop(req_id, None)
        pending_diagnostics = getattr(
            self, "_pending_live_source_diagnostics", None
        )
        if pending_diagnostics is not None:
            pending_diagnostics.pop(req_id, None)

    def _record_live_source_layout(
        self,
        req_id: str,
        kv_group: int,
        start: int,
        end: int,
        layers: list[dict[str, int]],
        runs: list[dict[str, int]],
    ) -> None:
        builder = self._live_source_builders.get(req_id)
        if builder is None or kv_group not in builder["groups"]:
            return
        if builder["segments"]:
            builder["invalid"] = True
            return
        coverage_end = int(builder["ends"].get(kv_group, 0))
        if start != coverage_end or end <= start:
            builder["invalid"] = True
            return
        if builder["compact_layers"] not in (None, layers):
            builder["invalid"] = True
            return
        if (
            not runs
            or runs[0]["logical_token_start"] != start
            or sum(run["token_count"] for run in runs) != end - start
            or any(
                left["logical_token_start"] + left["token_count"]
                != right["logical_token_start"]
                for left, right in zip(runs, runs[1:], strict=False)
            )
        ):
            builder["invalid"] = True
            return
        builder["compact_layers"] = layers
        builder["compact_runs"].extend(runs)
        builder["ends"][kv_group] = end

    def _record_live_latent_source_layout(
        self,
        req_id: str,
        start: int,
        end: int,
        layers: list[dict[str, int]],
        pages: list[dict[str, Any]],
    ) -> None:
        builder = self._live_source_builders.get(req_id)
        if builder is None or 0 not in builder["groups"]:
            return
        coverage_end = int(builder["ends"].get(0, 0))
        if start != coverage_end or end <= start:
            builder["invalid"] = True
            return
        if builder["latent_layers"] not in (None, layers):
            builder["invalid"] = True
            return
        logical = start
        for page in pages:
            page_start = int(page.get("logical_token_start", -1))
            page_tokens = int(page.get("token_count", 0))
            runs = page.get("runs", ())
            if page_start != logical or page_tokens <= 0 or not runs:
                builder["invalid"] = True
                return
            run_logical = page_start
            for run in runs:
                run_start = int(run.get("logical_token_start", -1))
                run_tokens = int(run.get("token_count", 0))
                if run_start != run_logical or run_tokens <= 0:
                    builder["invalid"] = True
                    return
                run_logical += run_tokens
            if run_logical != page_start + page_tokens:
                builder["invalid"] = True
                return
            logical += page_tokens
        if logical != end:
            builder["invalid"] = True
            return
        builder["latent_layers"] = layers
        builder["latent_pages"].extend(pages)
        builder["ends"][0] = end

    @staticmethod
    def _disable_live_latent_group(builder: dict[str, Any]) -> None:
        """Keep a valid group-1 descriptor when optional group 0 cannot plan."""
        builder["groups"] = tuple(
            group for group in builder["groups"] if group != 0
        )
        builder["ends"].pop(0, None)
        builder["latent_layers"] = None
        builder["latent_pages"].clear()
        builder["invalid"] = False

    def _record_live_source_pages(
        self,
        req_id: str,
        kv_group: int,
        ranges: list[tuple[int, int]],
        ptrs: List[List[int]],
        sizes: List[List[int]],
        owners: tuple[torch.Tensor, ...],
    ) -> None:
        builder = self._live_source_builders.get(req_id)
        if builder is None or kv_group not in builder["groups"]:
            return
        if (
            kv_group == 1 and builder["compact_layers"] is not None
        ) or (
            kv_group == 0 and builder["latent_layers"] is not None
        ):
            # Rank 0's persistent direct-store path observes the same source
            # pages after compact live capture. The compact descriptor already
            # owns complete logical coverage, so this duplicate representation
            # must not invalidate it.
            return
        coverage_end = int(builder["ends"].get(kv_group, 0))
        if len(ranges) != len(ptrs) or len(ptrs) != len(sizes):
            builder["invalid"] = True
            return
        owner_ranges = sorted(
            (
                int(owner.data_ptr()),
                int(owner.numel() * owner.element_size()),
                index,
            )
            for index, owner in enumerate(owners)
        )
        starts = [base for base, _, _ in owner_ranges]
        prefix_ends: list[int] = []
        prefix_owners: list[tuple[int, int]] = []
        for base, capacity, index in owner_ranges:
            owner_end = base + capacity
            if not prefix_ends or owner_end > prefix_ends[-1]:
                prefix_owners.append((index, base))
                prefix_ends.append(owner_end)
            else:
                prefix_owners.append(prefix_owners[-1])
                prefix_ends.append(prefix_ends[-1])
        for (start, page_end), page_ptrs, page_sizes in zip(
            ranges, ptrs, sizes, strict=True
        ):
            if page_end <= coverage_end:
                continue
            if start != coverage_end:
                builder["invalid"] = True
                return
            for ptr, size in zip(page_ptrs, page_sizes, strict=True):
                owner_pos = bisect_right(starts, ptr) - 1
                if owner_pos < 0 or ptr + size > prefix_ends[owner_pos]:
                    builder["invalid"] = True
                    return
                index, base = prefix_owners[owner_pos]
                builder["segments"].append(
                    {
                        "group_id": kv_group,
                        "source_buffer_index": index,
                        "source_buffer_base": base,
                        "source_offset": ptr - base,
                        "length": size,
                    }
                )
            coverage_end = page_end
        builder["ends"][kv_group] = coverage_end

    def finalize_live_source_descriptor(
        self, req_id: str, token_count: int, tp_rank: int, dp_rank: int
    ) -> bool:
        builder = self._live_source_builders.pop(req_id, None)
        if builder is None or builder["invalid"]:
            cold_start_perf_log(
                logger,
                "live_source_finalize_detail",
                req_id=req_id,
                token_count=token_count,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                finalized=False,
                reason=("missing_builder" if builder is None else "invalid_builder"),
            )
            return False
        groups = builder["groups"]
        if any(builder["ends"].get(group, 0) != token_count for group in groups):
            cold_start_perf_log(
                logger,
                "live_source_finalize_detail",
                req_id=req_id,
                token_count=token_count,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                finalized=False,
                reason="incomplete_coverage",
                group_ends=builder["ends"],
            )
            return False
        totals = [0, 0]
        compact_layers = builder["compact_layers"]
        latent_layers = builder["latent_layers"]
        if latent_layers is not None:
            totals[0] = sum(
                int(page["token_count"])
                * sum(layer["token_bytes"] for layer in latent_layers)
                for page in builder["latent_pages"]
            )
        if compact_layers is not None:
            totals[1] = token_count * sum(
                layer["token_bytes"] for layer in compact_layers
            )
        for segment in builder["segments"]:
            totals[segment["group_id"]] += segment["length"]
        if any(not totals[group] for group in groups):
            cold_start_perf_log(
                logger,
                "live_source_finalize_detail",
                req_id=req_id,
                token_count=token_count,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                finalized=False,
                reason="empty_group",
                group_byte_totals=totals,
            )
            return False
        descriptor = {
            "group_byte_totals": totals,
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
        }
        if compact_layers is None and latent_layers is None:
            descriptor["segments"] = builder["segments"]
        else:
            if compact_layers is not None:
                descriptor["format"] = "layer_slot_runs_v1"
                descriptor["compact_layout"] = {
                    "group_id": 1,
                    "token_count": token_count,
                    "layers": compact_layers,
                    "runs": builder["compact_runs"],
                }
            if latent_layers is not None:
                # Keep the established group-1 carrier format.  The latent
                # layout is an optional extension: older peers ignore it and
                # can still execute group-1 P2P. Keep the base totals valid
                # for that established schema and carry the group-0 total in
                # its own extension field; a new peer validates and promotes
                # it before constructing a hybrid plan.
                descriptor["format"] = "layer_slot_runs_v1"
                latent_total = totals[0]
                descriptor["group_byte_totals"][0] = 0
                descriptor["latent_group_byte_total"] = latent_total
                descriptor["latent_layout"] = {
                    "group_id": 0,
                    "token_count": token_count,
                    "layers": latent_layers,
                    "pages": builder["latent_pages"],
                }
        self._completed_live_sources[req_id] = descriptor
        if (
            compact_layers is not None
            and npu_content_diagnostics_enabled()
            and builder.get("compact_owners")
        ):
            pending_diagnostics = getattr(
                self, "_pending_live_source_diagnostics", None
            )
            if pending_diagnostics is None:
                pending_diagnostics = {}
                self._pending_live_source_diagnostics = pending_diagnostics
            pending_diagnostics[req_id] = {
                "owners": builder["compact_owners"],
                "layers": compact_layers,
                "runs": builder["compact_runs"],
                "token_count": token_count,
                "chunk_size": int(self.config.chunk_size),
                "tp_rank": tp_rank,
                "dp_rank": dp_rank,
            }
        cold_start_perf_log(
            logger,
            "live_source_finalize_detail",
            req_id=req_id,
            token_count=token_count,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            finalized=True,
            segments=len(builder["segments"]),
            compact_layers=len(compact_layers or ()),
            compact_runs=len(builder["compact_runs"]),
            latent_layers=len(latent_layers or ()),
            latent_pages=len(builder["latent_pages"]),
            group_byte_totals=totals,
        )
        return True

    def finalize_live_source_readiness(self, req_ids: Iterable[str]) -> None:
        """Fingerprint live Group-1 sources after their producer fence.

        The adapter calls this only after synchronizing the final NPU producer
        event.  Keeping the device-to-host readback here preserves the causal
        evidence captured by the adapter's pre-readback event query.
        """
        pending_diagnostics = getattr(
            self, "_pending_live_source_diagnostics", {}
        )
        for req_id in req_ids:
            pending = pending_diagnostics.pop(req_id, None)
            if pending is None:
                continue
            descriptor = self._completed_live_sources.get(req_id)
            if descriptor is None:
                continue
            fingerprint = fingerprint_compact_group1(
                event="group1_source_post_fence_fingerprint",
                req_id=req_id,
                owners=pending["owners"],
                layers=pending["layers"],
                runs=pending["runs"],
                token_count=pending["token_count"],
                chunk_size=pending["chunk_size"],
                tp_rank=pending["tp_rank"],
                dp_rank=pending["dp_rank"],
            )
            if fingerprint is not None:
                descriptor["content_diagnostics"] = fingerprint

    def drain_live_source_descriptors(self) -> dict[str, dict[str, Any]]:
        pending_diagnostics = getattr(
            self, "_pending_live_source_diagnostics", {}
        )
        unfinished = self._completed_live_sources.keys() & (
            pending_diagnostics.keys()
        )
        if unfinished:
            raise RuntimeError(
                "Live source descriptors reached publication before their "
                f"post-fence diagnostics completed: {sorted(unfinished)}"
            )
        descriptors = self._completed_live_sources
        self._completed_live_sources = {}
        return descriptors

    def capture_live_source_step(
        self,
        req_id: str,
        tokens: list[int],
        group_caches: dict[int, list],
        slot_mappings: dict[int, torch.Tensor],
        request_configs: Optional[dict],
        slot_mapping_base: int,
        final: bool,
    ) -> None:
        if req_id in self._completed_live_sources:
            return
        planner = getattr(self.gpu_connector, "plan_compact_page_layout", None)
        latent_planner = getattr(
            self.gpu_connector, "plan_compact_latent_page_layout", None
        )
        legacy_planner = getattr(
            self.gpu_connector, "plan_direct_page_sources", None
        )
        if (
            not callable(planner)
            and not callable(legacy_planner)
        ) or set(group_caches) != {0, 1}:
            self._live_source_builders.pop(req_id, None)
            return
        try:
            self.begin_live_source_descriptor(req_id, (1,))
            builder = self._live_source_builders[req_id]
            target_end = (
                len(tokens)
                if final
                else len(tokens) // self.config.chunk_size * self.config.chunk_size
            )
            planned_end = int(builder["planned_end"])
            if planned_end == target_end:
                return
            if not slot_mapping_base <= planned_end < target_end:
                raise ValueError(
                    "live source token coverage is not available in this step"
                )
            target_tokens = tokens[:target_end]
            plan0 = list(
                self.token_database.process_tokens_from_prefix(
                    target_tokens,
                    prefix_token_count=planned_end,
                    prefix_hash=int(builder["planned_hash"]),
                    request_configs=request_configs,
                    kv_group=0,
                )
                if planned_end
                else self.token_database.process_tokens(
                    tokens=target_tokens,
                    request_configs=request_configs,
                    kv_group=0,
                )
            )
            if (
                not plan0
                or plan0[0][0] != planned_end
                or plan0[-1][1] != target_end
            ):
                raise ValueError("live source token plan is not contiguous")
            hashes = [int(key.chunk_hash) for _, _, key in plan0]
            offsets = [end - start for start, end, _ in plan0]
            group1_keys = list(
                self.token_database.process_tokens(
                    hashes=hashes,
                    offsets=offsets,
                    request_configs=request_configs,
                    kv_group=1,
                )
            )
            plans = {1: [
                (latent[0], latent[1], indexer[2])
                for latent, indexer in zip(plan0, group1_keys, strict=True)
            ]}
            if 0 in builder["groups"]:
                plans[0] = plan0
            for group, chunks in plans.items():
                if not chunks:
                    continue
                starts = [start for start, _, _ in chunks]
                ends = [end for _, end, _ in chunks]
                if group == 0:
                    if builder["segments"]:
                        # The v1 wire format supports either legacy segments
                        # or compact layouts, not a mixture. Preserve the
                        # already valid legacy group-1 source instead of
                        # adding a latent layout that would make the entire
                        # descriptor fail decoder validation.
                        self._disable_live_latent_group(builder)
                        continue
                    try:
                        if not callable(latent_planner):
                            raise ValueError(
                                "compact latent source layout is unsupported"
                            )
                        latent_planned = latent_planner(
                            group_caches[group],
                            slot_mappings[group],
                            starts,
                            ends,
                            slot_mapping_base=slot_mapping_base,
                        )
                        if latent_planned is None:
                            raise ValueError(
                                "compact latent source layout is unsupported"
                            )
                        layers, pages, _owners = latent_planned
                        self._record_live_latent_source_layout(
                            req_id, starts[0], ends[-1], layers, pages
                        )
                        if builder["invalid"]:
                            raise ValueError(
                                "compact latent source coverage is invalid"
                            )
                    except Exception as error:
                        self._disable_live_latent_group(builder)
                        logger.warning(
                            "Live latent source disabled for request %s; "
                            "preserving group-1 P2P: %s",
                            req_id,
                            error,
                        )
                    continue
                planned = (
                    planner(
                        group_caches[group],
                        slot_mappings[group],
                        starts,
                        ends,
                        group,
                        slot_mapping_base=slot_mapping_base,
                    )
                    if callable(planner)
                    else None
                )
                if planned is None and callable(legacy_planner):
                    planned = legacy_planner(
                        group_caches[group],
                        slot_mappings[group],
                        starts,
                        ends,
                        group,
                        slot_mapping_base=slot_mapping_base,
                    )
                    if planned is not None:
                        ptrs, sizes, owners = planned
                        self._record_live_source_pages(
                            req_id,
                            group,
                            list(zip(starts, ends, strict=True)),
                            ptrs,
                            sizes,
                            tuple(owners),
                        )
                        if builder["invalid"]:
                            raise ValueError(
                                "live group-1 source coverage is invalid"
                            )
                        continue
                if planned is None:
                    raise ValueError("direct page source layout is unsupported")
                layers, runs, owners = planned
                self._record_live_source_layout(
                    req_id, group, starts[0], ends[-1], layers, runs
                )
                if npu_content_diagnostics_enabled():
                    existing_owners = builder.get("compact_owners")
                    owner_identity = tuple(
                        int(owner.data_ptr()) for owner in owners
                    )
                    if existing_owners is not None and tuple(
                        int(owner.data_ptr()) for owner in existing_owners
                    ) != owner_identity:
                        raise ValueError(
                            "live group-1 diagnostic owners changed across steps"
                        )
                    builder["compact_owners"] = tuple(owners)
                if builder["invalid"]:
                    raise ValueError(
                        "live group-1 source coverage is invalid"
                    )
            builder["planned_end"] = target_end
            builder["planned_hash"] = int(plan0[-1][2].chunk_hash)
        except Exception as error:
            self._live_source_builders.pop(req_id, None)
            logger.warning(
                "Live source descriptor disabled for request %s: %s",
                req_id,
                error,
            )
            cold_start_perf_log(
                logger,
                "live_source_capture_failure",
                req_id=req_id,
                token_count=len(tokens),
                groups=sorted(group_caches),
                slot_mapping_base=slot_mapping_base,
                final=final,
                reason=type(error).__name__,
                detail=str(error),
            )

    def _submit_direct_tail(
        self,
        req_id: str,
        tokens: list[int],
        group_caches: dict[int, list],
        slot_mappings: dict[int, torch.Tensor],
        request_configs: Optional[dict],
        state: _DirectStoreRequestState,
        planner: Callable[..., Any],
        slot_mapping_base: int = 0,
    ) -> bool:
        """Publish the unaligned suffix as one exact-size page per KV group."""
        started = cold_start_perf_now() if cold_start_perf_enabled() else None
        start = state.planned_end
        if start >= len(tokens):
            return True
        remote_fill = self._remote_fill_prepare_request(
            req_id, request_configs, state
        )
        if remote_fill and set(group_caches) != {0, 1}:
            state.remote_fill_disabled_reason = "incomplete_groups"
            remote_fill = False
        remote_ready_events = self._direct_source_ready_events(
            state, len(tokens)
        )
        if remote_fill and not remote_ready_events:
            state.remote_fill_disabled_reason = "incomplete_producer_fence"
            self._get_remote_fill_producer_metrics().abandon(
                "incomplete producer fence"
            )
            remote_fill = False
        if not remote_fill:
            group_caches = {
                group: caches
                for group, caches in group_caches.items()
                if state.submitted_end.get(group, 0) < len(tokens)
            }
        if not group_caches:
            return True
        plan0 = list(
            self.token_database.process_tokens_from_prefix(
                tokens,
                prefix_token_count=start,
                prefix_hash=state.planned_hash,
                request_configs=request_configs,
                kv_group=0,
            )
            if start
            else self.token_database.process_tokens(
                tokens=tokens, request_configs=request_configs, kv_group=0
            )
        )
        tail = [item for item in plan0 if item[0] == start and item[1] == len(tokens)]
        if len(tail) != 1:
            return False
        _, end, latent_key = tail[0]
        keys: List[CacheEngineKey] = []
        ptrs: List[List[int]] = []
        sizes: List[List[int]] = []
        owners: dict[int, torch.Tensor] = {}
        group_ends: dict[int, int] = {}
        remote_keys: List[CacheEngineKey] = []
        remote_ptrs: List[List[int]] = []
        remote_sizes: List[List[int]] = []
        remote_owners: dict[int, torch.Tensor] = {}
        remote_group_ends: dict[int, int] = {}
        remote_probe_plans: dict[
            int, list[tuple[int, int, CacheEngineKey]]
        ] = {0: [], 1: []}
        max_buffers = int(
            self.config.get_extra_config_value(
                "mooncake_direct_max_buffers_per_page", 4096
            )
        )
        for group, kvcaches in group_caches.items():
            key = latent_key
            if group:
                key = next(
                    self.token_database.process_tokens(
                        hashes=[int(latent_key.chunk_hash)],
                        offsets=[end - start],
                        request_configs=request_configs,
                        kv_group=group,
                    )
                )[2]
            present = self.storage_manager.batched_external_pages_exist([key])
            if len(present) != 1:
                return False
            persistent_missing = not present[0]
            if not persistent_missing:
                state.submitted_end[group] = end
                if not state.futures:
                    state.committed_end[group] = end
                if not remote_fill:
                    continue
            if remote_fill and start < slot_mapping_base:
                if persistent_missing:
                    state.remote_fill_disabled_reason = (
                        "unaddressable_partial_source"
                    )
                    remote_fill = False
                else:
                    # A full prefix hit may recompute only the final token, so
                    # the already-committed partial page is not addressable in
                    # this forward. Register it as probe-only coverage; never
                    # build a native source from an unrelated slot window.
                    remote_probe_plans[group].append((start, end, key))
                continue
            planned = planner(
                kvcaches,
                slot_mappings[group],
                [start],
                [end],
                group,
                slot_mapping_base=slot_mapping_base,
            )
            if planned is None:
                if remote_fill:
                    state.remote_fill_disabled_reason = "tail_planner_fallback"
                    remote_fill = False
                if persistent_missing:
                    return False
                continue
            group_ptrs, group_sizes, group_owners = planned
            if (
                len(group_ptrs) != 1
                or len(group_sizes) != 1
                or any(len(item) > max_buffers for item in group_ptrs)
            ):
                if remote_fill:
                    state.remote_fill_disabled_reason = "tail_buffer_limit"
                    remote_fill = False
                if persistent_missing:
                    return False
                continue
            if remote_fill:
                remote_keys.append(key)
                remote_ptrs.extend(group_ptrs)
                remote_sizes.extend(group_sizes)
                remote_owners.update(
                    (id(owner), owner) for owner in group_owners
                )
                remote_group_ends[group] = end
            if persistent_missing:
                keys.append(key)
                self._record_live_source_pages(
                    req_id,
                    group,
                    [(start, end)],
                    group_ptrs,
                    group_sizes,
                    tuple(group_owners),
                )
                ptrs.extend(group_ptrs)
                sizes.extend(group_sizes)
                owners.update((id(owner), owner) for owner in group_owners)
                group_ends[group] = end
        if remote_fill and any(remote_probe_plans.values()):
            probe_pages = self._remote_fill_probe_control_pages(
                remote_probe_plans,
                start,
                end,
            )
            if len(probe_pages) == 2:
                self._schedule_remote_fill_probe_pages(
                    req_id,
                    state,
                    probe_pages,
                    len(tokens),
                )
            else:
                state.remote_fill_disabled_reason = "incomplete_partial_probe"
                remote_fill = False
        if not keys and not (remote_fill and remote_keys):
            return True
        if started is not None:
            cold_start_perf_log(
                logger,
                "direct_npu_plan",
                started=started,
                req_id=req_id,
                groups=sorted(group_ends),
                pages=len(keys),
                legacy_objects=0,
                buffers=sum(map(len, ptrs)),
                bytes=sum(map(sum, sizes)),
                token_count=len(tokens),
                slot_mapping_base=slot_mapping_base,
                format="partial_page",
            )
        ready_event, ready_event_source = self._direct_source_ready_event(
            state,
            len(tokens),
            require_producer_event=bool(
                getattr(self.config, "dsa_two_groups", False)
            ),
        )
        if (
            1 in (remote_group_ends if remote_fill else group_ends)
            and npu_content_diagnostics_enabled()
        ):
            log_npu_content_diagnostic_event(
                "group1_persistent_store_ready_dependency",
                req_id=req_id,
                token_count=len(tokens),
                event_source=ready_event_source,
                source_ready_token_end=state.source_ready_token_end,
                format="partial_page",
            )
        shared_sink_owners = (
            tuple(remote_owners.values())
            if remote_fill and remote_keys
            else tuple(owners.values())
        )
        if keys:
            batch = _DirectPageBatch(
                req_id,
                keys,
                ptrs,
                sizes,
                shared_sink_owners,
                ready_event,
                group_ends,
                tuple((start, end) for _ in keys),
            )
            self._track_direct_batch(
                batch,
                (
                    req_id,
                    tokens,
                    group_caches,
                    slot_mappings,
                    request_configs,
                    slot_mapping_base,
                ),
            )
        if remote_fill and remote_keys:
            if {int(key.kv_group) for key in remote_keys} == {0, 1}:
                remote_batch = _DirectPageBatch(
                    req_id,
                    remote_keys,
                    remote_ptrs,
                    remote_sizes,
                    shared_sink_owners,
                    ready_event,
                    remote_group_ends,
                    tuple((start, end) for _ in remote_keys),
                    remote_ready_events,
                )
                self._schedule_remote_fill_batch(
                    state, remote_batch, len(tokens)
                )
            else:
                state.remote_fill_disabled_reason = "incomplete_partial_pair"
        return True

    def store_direct_prefill(
        self,
        req_id: str,
        tokens: list[int],
        group_caches: dict[int, list],
        slot_mappings: dict[int, torch.Tensor],
        request_configs: Optional[dict] = None,
        final: bool = False,
        slot_mapping_base: int = 0,
        verified_prefix_end: int = 0,
        accepted_store_end: Optional[int] = None,
        source_ready_event: Any = None,
        source_ready_event_source: str = "missing",
        source_ready_events: tuple[Any, ...] = (),
    ) -> bool:
        """Publish newly completed pages and the final tail directly from NPU."""
        if not self._direct_store_enabled or not group_caches:
            return False
        if accepted_store_end is None:
            accepted_store_end = len(tokens)
        if not 0 <= accepted_store_end <= len(tokens):
            raise ValueError(
                "Direct prefill accepted store end is outside the token range: "
                f"accepted={accepted_store_end}, tokens={len(tokens)}"
            )
        tokens = tokens[:accepted_store_end]
        if not 0 <= slot_mapping_base <= len(tokens):
            raise ValueError(
                "Direct prefill slot-mapping base is outside the token range: "
                f"base={slot_mapping_base}, tokens={len(tokens)}"
            )
        if not 0 <= verified_prefix_end <= slot_mapping_base:
            raise ValueError(
                "Direct prefill verified prefix is outside the addressable "
                "prefix: "
                f"verified={verified_prefix_end}, base={slot_mapping_base}"
            )
        assert self.storage_manager is not None
        assert self.gpu_connector is not None
        state = self._direct_store_states.setdefault(req_id, _DirectStoreRequestState())
        if (
            state.accepted_store_end is not None
            and accepted_store_end < state.accepted_store_end
        ):
            raise RuntimeError(
                "Direct prefill accepted store frontier moved backwards: "
                f"req_id={req_id} previous={state.accepted_store_end} "
                f"accepted={accepted_store_end}"
            )
        state.accepted_store_end = accepted_store_end
        if final and state.finalized:
            return True
        self._remember_direct_source_readiness(
            state,
            len(tokens),
            source_ready_event,
            source_ready_event_source,
            source_ready_events,
        )
        remote_fill = self._remote_fill_prepare_request(
            req_id, request_configs, state
        )
        if remote_fill and set(group_caches) != {0, 1}:
            state.remote_fill_disabled_reason = "incomplete_groups"
            remote_fill = False
        previous_remote_work = bool(
            state.remote_fill_session is not None
            or state.remote_fill_futures
            or state.remote_fill_probe_end
        )
        if (
            remote_fill
            and not previous_remote_work
            and not self._remote_fill_has_addressable_source_page(
                len(tokens),
                slot_mapping_base,
                int(self.config.chunk_size),
            )
        ):
            state.remote_fill_disabled_reason = "no_addressable_source_page"
            self._get_remote_fill_producer_metrics().abandon(
                "no addressable source page"
            )
            cold_start_perf_log(
                logger,
                "remote_fill_direct_abandoned",
                req_id=req_id,
                reason="no_addressable_source_page",
                slot_mapping_base=slot_mapping_base,
                accepted_store_end=len(tokens),
            )
            remote_fill = False
        remote_ready_events = self._direct_source_ready_events(
            state, len(tokens)
        )
        if remote_fill and not remote_ready_events:
            state.remote_fill_disabled_reason = "incomplete_producer_fence"
            self._get_remote_fill_producer_metrics().abandon(
                "incomplete producer fence"
            )
            remote_fill = False
        for group in group_caches:
            if group not in state.submitted_end:
                state.submitted_end[group] = verified_prefix_end
                state.committed_end[group] = verified_prefix_end
        started = cold_start_perf_now() if cold_start_perf_enabled() else None
        cpu_fallback_s = 0.0
        plans = self._direct_suffix_plans(
            state, tokens, group_caches, request_configs
        )
        if remote_fill:
            probe_target = verified_prefix_end // int(self.config.chunk_size) * int(
                self.config.chunk_size
            )
            if state.remote_fill_probe_end < probe_target:
                probe_pages = self._remote_fill_probe_control_pages(
                    plans,
                    state.remote_fill_probe_end,
                    probe_target,
                )
                expected_pages = (
                    (probe_target - state.remote_fill_probe_end)
                    // int(self.config.chunk_size)
                    * 2
                )
                if len(probe_pages) != expected_pages:
                    prefix_plans = self._remote_fill_prefix_plans(
                        tokens,
                        request_configs,
                        probe_target,
                    )
                    if prefix_plans[0]:
                        state.planned_end = prefix_plans[0][-1][1]
                        state.planned_hash = int(
                            prefix_plans[0][-1][2].chunk_hash
                        )
                    probe_pages = self._remote_fill_probe_control_pages(
                        prefix_plans,
                        state.remote_fill_probe_end,
                        probe_target,
                    )
                if len(probe_pages) != expected_pages:
                    state.remote_fill_disabled_reason = "incomplete_probe_plan"
                    remote_fill = False
                else:
                    self._schedule_remote_fill_probe_pages(
                        req_id,
                        state,
                        probe_pages,
                        len(tokens),
                    )
                    state.remote_fill_probe_end = probe_target

        keys: List[CacheEngineKey] = []
        ptrs: List[List[int]] = []
        sizes: List[List[int]] = []
        ranges: list[tuple[int, int]] = []
        owners: dict[int, torch.Tensor] = {}
        group_ends: dict[int, int] = {}
        remote_keys: List[CacheEngineKey] = []
        remote_ptrs: List[List[int]] = []
        remote_sizes: List[List[int]] = []
        remote_ranges: list[tuple[int, int]] = []
        remote_owners: dict[int, torch.Tensor] = {}
        remote_group_ends: dict[int, int] = {}
        merged_runs = 0
        chunk_size = int(self.config.chunk_size)
        planner = getattr(self.gpu_connector, "plan_direct_page_sources", None)
        if not callable(planner):
            self.wait_for_direct_stores((req_id,))
            self._retry_direct_cpu(
                req_id,
                tokens,
                group_caches,
                slot_mappings,
                request_configs,
                state,
                slot_mapping_base,
            )
            self._finalize_direct_store(
                req_id, tokens, group_caches, state, final
            )
            return True

        candidates: dict[int, list[tuple[int, int, CacheEngineKey]]] = {}
        candidate_ends: dict[int, int] = {}
        candidate_refs: list[tuple[int, tuple[int, int, CacheEngineKey]]] = []
        for group in group_caches:
            submitted = state.submitted_end.get(group, 0)
            group_candidates = [
                (start, end, key)
                for start, end, key in plans[group]
                if end > submitted and end - start == chunk_size
            ]
            candidates[group] = group_candidates
            candidate_refs.extend((group, item) for item in group_candidates)
            if group_candidates:
                candidate_ends[group] = group_candidates[-1][1]

        try:
            exists = (
                self.storage_manager.batched_external_pages_exist(
                    [item[2] for _, item in candidate_refs]
                )
                if candidate_refs
                else []
            )
        except Exception:
            self.wait_for_direct_stores((req_id,))
            self._retry_direct_cpu(
                req_id,
                tokens,
                group_caches,
                slot_mappings,
                request_configs,
                state,
                slot_mapping_base,
            )
            self._finalize_direct_store(
                req_id, tokens, group_caches, state, final
            )
            return True
        if len(exists) != len(candidate_refs):
            raise RuntimeError("Direct page lookup returned an invalid result count")
        authoritative_candidates = {
            group: list(group_candidates)
            for group, group_candidates in candidates.items()
        }
        candidates = {group: [] for group in group_caches}
        missing_by_group: set[int] = set()
        for (group, item), present in zip(candidate_refs, exists, strict=True):
            if present:
                continue
            missing_by_group.add(group)
            candidates[group].append(item)
        unaddressable = {
            group: [(start, end) for start, end, _ in group_candidates]
            for group, group_candidates in candidates.items()
            if any(start < slot_mapping_base for start, _, _ in group_candidates)
        }
        if unaddressable:
            raise RuntimeError(
                "Direct prefill store found an uncommitted prefix outside its "
                "slot-mapping window: "
                f"req_id={req_id}, slot_mapping_base={slot_mapping_base}, "
                f"missing={unaddressable}"
            )
        for group in group_caches.keys() - missing_by_group:
            if group in candidate_ends:
                state.submitted_end[group] = max(
                    state.submitted_end.get(group, 0), candidate_ends[group]
                )
            if not state.futures:
                state.committed_end[group] = max(
                    state.committed_end.get(group, 0),
                    state.submitted_end.get(group, 0),
                )

        for group, kvcaches in group_caches.items():
            submitted = state.committed_end.get(group, 0)
            persistent_full = candidates[group]
            persistent_full = [
                item
                for item in persistent_full
                if item[2].to_string() not in state.pending_keys
            ]
            full = (
                authoritative_candidates[group]
                if remote_fill
                else persistent_full
            )
            if full:
                if remote_fill and any(
                    start < slot_mapping_base for start, _, _ in full
                ):
                    state.remote_fill_disabled_reason = "unaddressable_source"
                    remote_fill = False
                    full = persistent_full
                starts = [start for start, _, _ in full]
                ends = [end for _, end, _ in full]
                planned = planner(
                    kvcaches,
                    slot_mappings[group],
                    starts,
                    ends,
                    group,
                    slot_mapping_base=slot_mapping_base,
                )
                fallback_reason = None
                if planned is None:
                    rejection = getattr(
                        self.gpu_connector, "direct_page_plan_rejection", None
                    )
                    fallback_reason = (
                        rejection(group) or "unsupported_layout"
                        if callable(rejection)
                        else "unsupported_layout"
                    )
                else:
                    group_ptrs, group_sizes, group_owners = planned
                    max_buffers = int(
                        self.config.get_extra_config_value(
                            "mooncake_direct_max_buffers_per_page", 4096
                        )
                    )
                    if any(
                        len(page_ptrs) > max_buffers for page_ptrs in group_ptrs
                    ):
                        fallback_reason = "buffer_limit_exceeded"
                if fallback_reason is not None:
                    if remote_fill:
                        state.remote_fill_disabled_reason = fallback_reason
                        remote_fill = False
                    if not persistent_full:
                        continue
                    if state.fallback_reasons.get(group) != fallback_reason:
                        logger.warning(
                            "Direct NPU prefill store is using CPU fallback: "
                            "req_id=%s kv_group=%d reason=%s",
                            req_id,
                            group,
                            fallback_reason,
                        )
                        state.fallback_reasons[group] = fallback_reason
                    self.wait_for_direct_stores((req_id,))
                    fallback_started = (
                        cold_start_perf_now() if cold_start_perf_enabled() else None
                    )
                    committed = self._store_direct_cpu_group(
                        req_id,
                        tokens,
                        kvcaches,
                        slot_mappings[group],
                        group,
                        submitted,
                        request_configs,
                        slot_mapping_base,
                    )
                    state.committed_end[group] = committed
                    state.submitted_end[group] = committed
                    if fallback_started is not None:
                        fallback_elapsed = cold_start_perf_now() - fallback_started
                        cpu_fallback_s += fallback_elapsed
                        cold_start_perf_log(
                            logger,
                            "direct_npu_cpu_fallback",
                            started=fallback_started,
                            req_id=req_id,
                            kv_group=group,
                            reason=fallback_reason,
                            start=submitted,
                            end=committed,
                        )
                    continue
                assert planned is not None
                if len(group_ptrs) != len(full) or len(group_sizes) != len(full):
                    raise RuntimeError(
                        "Direct page planner returned an invalid page count"
                    )
                merged_runs += sum(
                    len(page_ptrs) // len(group_owners)
                    for page_ptrs in group_ptrs
                )
                persistent_key_strings = {
                    item[2].to_string() for item in persistent_full
                }
                persistent_ranges: list[tuple[int, int]] = []
                persistent_ptrs: List[List[int]] = []
                persistent_sizes: List[List[int]] = []
                for item, page_ptrs, page_sizes in zip(
                    full, group_ptrs, group_sizes, strict=True
                ):
                    start, end, key = item
                    if remote_fill:
                        remote_keys.append(key)
                        remote_ptrs.append(page_ptrs)
                        remote_sizes.append(page_sizes)
                        remote_ranges.append((start, end))
                    if key.to_string() in persistent_key_strings:
                        keys.append(key)
                        persistent_ptrs.append(page_ptrs)
                        persistent_sizes.append(page_sizes)
                        persistent_ranges.append((start, end))
                if persistent_ranges:
                    self._record_live_source_pages(
                        req_id,
                        group,
                        persistent_ranges,
                        persistent_ptrs,
                        persistent_sizes,
                        tuple(group_owners),
                    )
                    ptrs.extend(persistent_ptrs)
                    sizes.extend(persistent_sizes)
                    ranges.extend(persistent_ranges)
                    owners.update((id(owner), owner) for owner in group_owners)
                    # Existing pages between missing pages are part of the same
                    # contiguous committed prefix once this batch succeeds.
                    group_ends[group] = candidate_ends[group]
                if remote_fill:
                    remote_owners.update(
                        (id(owner), owner) for owner in group_owners
                    )
                    remote_group_ends[group] = full[-1][1]

        if started is not None:
            elapsed_s = cold_start_perf_now() - started
            plan_only_ms = max(elapsed_s - cpu_fallback_s, 0.0) * 1000
            cold_start_perf_log(
                logger,
                "direct_npu_plan",
                req_id=req_id,
                groups=sorted(group_caches),
                pages=len(keys),
                buffers=sum(map(len, ptrs)),
                merged_runs=merged_runs,
                bytes=sum(map(sum, sizes)),
                token_count=len(tokens),
                slot_mapping_base=slot_mapping_base,
                final=final,
                format="page",
                elapsed_ms=round(plan_only_ms, 3),
                total_ms=round(elapsed_s * 1000, 3),
                plan_only_ms=round(plan_only_ms, 3),
                cpu_fallback_ms=round(cpu_fallback_s * 1000, 3),
                submitted_ends=dict(state.submitted_end),
                committed_ends=dict(state.committed_end),
            )
        ready_event = None
        ready_event_source = "missing"
        if keys or (remote_fill and remote_keys):
            ready_event, ready_event_source = self._direct_source_ready_event(
                state,
                len(tokens),
                require_producer_event=bool(
                    getattr(self.config, "dsa_two_groups", False)
                ),
            )
            if (
                1 in (remote_group_ends if remote_fill else group_ends)
                and npu_content_diagnostics_enabled()
            ):
                log_npu_content_diagnostic_event(
                    "group1_persistent_store_ready_dependency",
                    req_id=req_id,
                    token_count=len(tokens),
                    event_source=ready_event_source,
                    source_ready_token_end=state.source_ready_token_end,
                    format="page",
                )
        shared_sink_owners = (
            tuple(remote_owners.values())
            if remote_fill and remote_keys
            else tuple(owners.values())
        )
        if keys:
            batch = _DirectPageBatch(
                req_id,
                keys,
                ptrs,
                sizes,
                shared_sink_owners,
                ready_event,
                group_ends,
                tuple(ranges),
            )
            try:
                self._track_direct_batch(
                    batch,
                    (
                        req_id,
                        tokens,
                        group_caches,
                        slot_mappings,
                        request_configs,
                        slot_mapping_base,
                    ),
                )
            except Exception as error:
                retry_started = cold_start_perf_now()
                self.wait_for_direct_stores((req_id,))
                self._retry_direct_cpu(
                    req_id,
                    tokens,
                    group_caches,
                    slot_mappings,
                    request_configs,
                    state,
                    slot_mapping_base,
                )
                if cold_start_perf_enabled():
                    cold_start_perf_log(
                        logger,
                        "direct_npu_cpu_retry",
                        started=retry_started,
                        req_id=req_id,
                        reason=type(error).__name__,
                        failed_pages=len(keys),
                    )
                self._finalize_direct_store(
                    req_id, tokens, group_caches, state, final
                )
                return True
        if remote_fill and remote_keys:
            if {int(key.kv_group) for key in remote_keys} != {0, 1}:
                state.remote_fill_disabled_reason = "incomplete_page_pair"
            else:
                remote_batch = _DirectPageBatch(
                    req_id,
                    remote_keys,
                    remote_ptrs,
                    remote_sizes,
                    shared_sink_owners,
                    ready_event,
                    remote_group_ends,
                    tuple(remote_ranges),
                    remote_ready_events,
                )
                self._schedule_remote_fill_batch(
                    state, remote_batch, len(tokens)
                )

        # Final publication is also the chunked-prefill correctness fence.
        if final:
            save_tail = bool(getattr(self.config, "save_unfull_chunk", True))
            required_end = self._direct_required_end(state, tokens)
            if save_tail and state.planned_end < len(tokens):
                try:
                    self._submit_direct_tail(
                        req_id,
                        tokens,
                        group_caches,
                        slot_mappings,
                        request_configs,
                        state,
                        planner,
                        slot_mapping_base,
                    )
                except Exception:
                    logger.warning(
                        "Direct NPU tail planning failed for %s; using CPU fallback",
                        req_id,
                        exc_info=True,
                    )
            with self._store_cv:
                self._pending_store_reqs[req_id] = (
                    self._pending_store_reqs.get(req_id, 0) + 1
                )
            try:
                self.wait_for_direct_stores((req_id,))
                repaired = False
                for group, kvcaches in group_caches.items():
                    committed = state.committed_end.get(group, 0)
                    if committed >= required_end:
                        state.committed_end.setdefault(group, required_end)
                        state.submitted_end.setdefault(group, required_end)
                        continue
                    repair_started = cold_start_perf_now()
                    repaired_end = self._store_direct_cpu_group(
                        req_id,
                        tokens,
                        kvcaches,
                        slot_mappings[group],
                        group,
                        committed,
                        request_configs,
                        slot_mapping_base,
                    )
                    state.committed_end[group] = repaired_end
                    state.submitted_end[group] = repaired_end
                    repaired = True
                    if cold_start_perf_enabled():
                        cold_start_perf_log(
                            logger,
                            "direct_npu_cpu_retry",
                            started=repair_started,
                            req_id=req_id,
                            kv_group=group,
                            reason=(
                                "incomplete_full_page_coverage"
                                if committed < state.planned_end
                                else "tail_planner_fallback"
                            ),
                            committed_end=committed,
                            required_end=required_end,
                        )
                if repaired:
                    logger.warning(
                        "Repaired incomplete direct prefill coverage for %s",
                        req_id,
                    )
                self._finalize_direct_store(
                    req_id, tokens, group_caches, state, final=True
                )
            finally:
                with self._store_cv:
                    count = self._pending_store_reqs.get(req_id, 1) - 1
                    if count <= 0:
                        self._pending_store_reqs.pop(req_id, None)
                    else:
                        self._pending_store_reqs[req_id] = count
                    self._store_cv.notify_all()
        return True

    def wait_for_direct_stores(self, req_ids: Iterable[str]) -> set[str]:
        """Fence request-owned direct puts before vLLM may release KV blocks."""
        waited: set[str] = set()
        for req_id in req_ids:
            state = self._direct_store_states.get(req_id)
            if state is None:
                continue
            remote_wait_started = (
                time.perf_counter() if state.remote_fill_metrics_started else None
            )
            started = cold_start_perf_now() if cold_start_perf_enabled() else None
            if (
                started is not None
                and state.remote_fill_metrics_started
                and not state.remote_fill_backlog_logged
            ):
                now = time.perf_counter()
                cold_start_perf_log(
                    logger,
                    "remote_fill_backlog_at_prefill_end",
                    req_id=req_id,
                    remaining_windows=state.remote_fill_queued_windows,
                    remaining_bytes=state.remote_fill_queued_bytes,
                    oldest_window_age_ms=round(
                        max(
                            now - state.remote_fill_oldest_enqueued_at,
                            0.0,
                        )
                        * 1000,
                        3,
                    )
                    if state.remote_fill_oldest_enqueued_at
                    else 0.0,
                )
                state.remote_fill_backlog_logged = True
            outstanding = len(state.futures)
            failed: list[tuple[Future, tuple[Any, ...], Exception]] = []
            completed: list[tuple[tuple[Any, ...], Optional[Exception]]] = []
            verified_ends = dict(state.committed_end)
            while state.futures:
                future = state.futures.popleft()
                error = None
                try:
                    try:
                        future.result(timeout=self.config.blocking_timeout_secs)
                    except FutureTimeoutError:
                        logger.warning(
                            "Direct NPU page store exceeded the blocking timeout "
                            "for %s; waiting for the uncancellable native transfer "
                            "before releasing KV blocks",
                            req_id,
                        )
                        future.result()
                except Exception as caught:
                    error = caught
                if isinstance(error, NativeExternalPageTransferUnknownError):
                    self._latch_remote_fill_producer_fatal(state)
                    raise RemoteFillFatalError(
                        "persistent external-page DMA exceeded its hard deadline"
                    ) from error
                retry = self._direct_retry_args.pop(future, None)
                if retry is None:
                    if error is not None:
                        raise error
                    continue
                completed.append((retry, error))
                if error is None:
                    self._direct_store_done(req_id, retry[-2], future)
                else:
                    failed.append((future, retry, error))

            if failed:
                retry_started = cold_start_perf_now()
                logger.warning(
                    "Direct NPU page store failed for %s; retrying through the "
                    "CPU layerwise path after draining %d direct job(s): %s",
                    req_id,
                    outstanding,
                    failed[0][2],
                )
                try:
                    # Replay submission order: successful jobs advance the
                    # verified frontier; only failed windows use CPU repair.
                    for retry, error in completed:
                        (
                            _,
                            tokens,
                            group_caches,
                            slot_mappings,
                            configs,
                            mapping_base,
                            _,
                            group_ends,
                        ) = retry
                        for group, required_end in group_ends.items():
                            start = verified_ends.get(group, 0)
                            if start < mapping_base:
                                raise RuntimeError(
                                    "Direct-store completion cannot bridge an "
                                    "unverified prefix gap: "
                                    f"req_id={req_id} group={group} "
                                    f"verified={start} "
                                    f"mapping_base={mapping_base}"
                                )
                            if error is None:
                                verified_ends[group] = max(start, required_end)
                                continue
                            if start >= required_end:
                                continue
                            committed = self._store_direct_cpu_group(
                                req_id,
                                tokens,
                                group_caches[group],
                                slot_mappings[group],
                                group,
                                start,
                                configs,
                                mapping_base,
                            )
                            if committed < required_end:
                                raise RuntimeError(
                                    "CPU retry did not restore direct-store coverage: "
                                    f"req_id={req_id} group={group} "
                                    f"committed={committed} required={required_end}"
                                )
                            verified_ends[group] = committed
                    # All later direct jobs have also completed, so repairing
                    # every failed window makes their submitted frontier safe.
                    for group, end in state.submitted_end.items():
                        state.committed_end[group] = max(
                            verified_ends.get(group, 0), end
                        )
                finally:
                    with self._store_cv:
                        for future, retry, _ in failed:
                            self._direct_completed_futures.add(future)
                            state.pending_keys.difference_update(retry[-2])
                            count = self._pending_store_reqs.get(req_id, 1) - 1
                            if count <= 0:
                                self._pending_store_reqs.pop(req_id, None)
                            else:
                                self._pending_store_reqs[req_id] = count
                        self._store_cv.notify_all()
                if cold_start_perf_enabled():
                    cold_start_perf_log(
                        logger,
                        "direct_npu_cpu_retry",
                        started=retry_started,
                        req_id=req_id,
                        reason=type(failed[0][2]).__name__,
                        failed_pages=sum(
                            len(getattr(error, "failed_pages", ()))
                            for _, _, error in failed
                        )
                        or "unknown",
                    )
            else:
                for group, end in state.submitted_end.items():
                    state.committed_end[group] = max(
                        state.committed_end.get(group, 0), end
                    )
            # The same source owners may also be retained by the optional
            # remote-fill sink. Drain it before vLLM can release NPU blocks.
            metrics = (
                self._get_remote_fill_producer_metrics()
                if remote_wait_started is not None
                else None
            )
            self._wait_remote_fill_windows(state)
            if metrics is not None:
                metrics.observe(
                    "final_wait_seconds",
                    time.perf_counter() - remote_wait_started,
                )
            waited.add(req_id)
            if started is not None:
                cold_start_perf_log(
                    logger,
                    "direct_npu_final_wait",
                    started=started,
                    req_id=req_id,
                    outstanding_jobs=outstanding,
                )
        return waited

    def drop_direct_store_states(self, req_ids: Iterable[str]) -> None:
        """Forget completed request bookkeeping after vLLM releases ownership."""
        for req_id in req_ids:
            state = self._direct_store_states.get(req_id)
            if (
                state is not None
                and not state.futures
                and not state.pending_keys
                and not state.remote_fill_futures
                and (
                    state.remote_fill_last_future is None
                    or state.remote_fill_last_future.done()
                )
            ):
                if state.remote_fill_session is not None:
                    if state.remote_fill_terminal is None:
                        try:
                            state.remote_fill_session.abort(
                                "prefiller released request before FINISH"
                            )
                            cold_start_perf_log(
                                logger,
                                "remote_fill_abort",
                                req_id=req_id,
                                reason="request_released_before_finish",
                            )
                        except RemoteFillFatalError:
                            self._latch_remote_fill_producer_fatal(state)
                            raise
                        except Exception:
                            logger.warning(
                                "Failed to release hidden remote-fill state for %s",
                                req_id,
                                exc_info=True,
                            )
                    state.remote_fill_session.close()
                    cold_start_perf_log(
                        logger,
                        "remote_fill_release",
                        req_id=req_id,
                        terminal=state.remote_fill_terminal is not None,
                    )
                if (
                    state.remote_fill_metrics_started
                    and state.remote_fill_terminal is None
                ):
                    metrics = self._get_remote_fill_producer_metrics()
                    metrics.finish_attempt("ABORTED", "request aborted")
                    metrics.abandon("request aborted")
                    if state.remote_fill_viable_counted:
                        metrics.add_gauge("direct_viable", -1)
                        state.remote_fill_viable_counted = False
                    if state.remote_fill_active_counted:
                        metrics.add_gauge("active_transactions", -1)
                        state.remote_fill_active_counted = False
                    metrics.add_bytes(
                        "discarded_bytes", state.remote_fill_submitted_bytes
                    )
                self._direct_store_states.pop(req_id, None)
            self._live_source_builders.pop(req_id, None)
            self._completed_live_sources.pop(req_id, None)
            pending_diagnostics = getattr(
                self, "_pending_live_source_diagnostics", None
            )
            if pending_diagnostics is not None:
                pending_diagnostics.pop(req_id, None)

    def direct_store_committed_ends(self, req_id: str) -> dict[int, int]:
        """Return a snapshot of successfully persisted group frontiers."""
        state = self._direct_store_states.get(req_id)
        return {} if state is None else dict(state.committed_end)

    def _store_worker_loop(self) -> None:
        if not self.is_store_async:
            return
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)
        while True:
            work = self._store_queue.get()
            if work is None:  # poison pill
                self._store_queue.task_done()
                break

            (
                req_id,
                tokens,
                hashes,
                offsets,
                mask,
                num_to_store_tokens,
                kwargs,
            ) = work
            try:
                self._run_store_pipeline(
                    req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
                )
            except Exception:
                logger.exception("Background store failed for req %s", req_id)
            finally:
                with self._store_lock:
                    cnt = self._pending_store_reqs.get(req_id, 1) - 1
                    if cnt <= 0:
                        self._pending_store_reqs.pop(req_id, None)
                    else:
                        self._pending_store_reqs[req_id] = cnt
                    logger.debug(
                        "Async store done for req %s; remaining=%d",
                        req_id,
                        max(cnt, 0),
                    )
                    self._store_cv.notify_all()
                self._store_queue.task_done()

    @torch.inference_mode()
    def _run_store_pipeline(
        self,
        req_id: str,
        tokens: Optional[Union[torch.Tensor, list]],
        hashes: Optional[List[int]],
        offsets: Optional[List[int]],
        mask: Optional[torch.Tensor],
        num_to_store_tokens: int,
        kwargs: dict,
    ) -> None:
        """Shared implementation for sync and async store.
        From upstream store function.
        """
        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        kv_group = kwargs.get("kv_group", 0)

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
                kv_group=kv_group,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes, kv_dtypes = self._metadata_shapes_dtypes_for_kv_group(
                    kv_group=kv_group,
                    num_tokens=num_tokens,
                )

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self._memory_format_for_kv_group(kv_group),
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        put_submitted = False
        try:
            with store_stats.profile_from_gpu():
                self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

            with self._engine_state_lock:
                with store_stats.profile_put():
                    transfer_spec = kwargs.get("transfer_spec", None)
                    # TODO: we implicitly rely on batched_put to call ref_count_down
                    # this management should be done in a cleaner way
                    required_futures = self.storage_manager.batched_put(
                        keys,
                        memory_objs,
                        transfer_spec=transfer_spec,
                        location=self.store_location,
                    )
                    self._track_sync_store_futures(required_futures)
                    put_submitted = True

                self.stats_monitor.on_store_finished(
                    store_stats,
                    tot_token_num,
                )
        except Exception:
            if not put_submitted:
                for mem_obj in memory_objs:
                    mem_obj.ref_count_down()
            raise

        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s kv_group=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            kv_group,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    def get_finished_stores(self, finished_req_ids: set) -> set:
        if not self.is_store_async:
            return None
        result: set = set()
        with self._store_lock:
            # Forget req_ids the scheduler no longer asks about.
            # This bounds the set to at most |finished_req_ids|.
            self._reported_finished_store_ids &= finished_req_ids

            for req_id in list(self._deferred_finished_req_ids):
                if req_id not in self._pending_store_reqs:
                    result.add(req_id)
                    self._deferred_finished_req_ids.discard(req_id)

            for req_id in finished_req_ids:
                if req_id in self._reported_finished_store_ids:
                    # Already reported - skip to avoid scheduler seeing
                    # a duplicate finished_sending for blocks it has
                    # already freed.
                    continue
                if req_id in self._pending_store_reqs:
                    self._deferred_finished_req_ids.add(req_id)
                else:
                    result.add(req_id)

            self._reported_finished_store_ids.update(result)
        return result

    def wait_for_pending_stores(self, req_ids: Iterable[str]) -> set[str]:
        """Wait until async stores for the given requests have drained.

        vLLM reports preempted request ids to workers before the next forward can
        overwrite their freed KV blocks.  If one of those ids still has a
        background store reading paged KV, drain it here to avoid store-after-free.
        """
        req_id_set = set(req_ids)
        direct_waited = self.wait_for_direct_stores(req_id_set)
        if not self.is_store_async:
            return set()

        if not req_id_set:
            return set()

        with self._store_cv:
            pending_at_start = {
                req_id for req_id in req_id_set if req_id in self._pending_store_reqs
            }
            if not pending_at_start:
                return direct_waited

            pending_counts = {
                req_id: self._pending_store_reqs[req_id] for req_id in pending_at_start
            }
            logger.info(
                "Waiting for pending async stores before preemption: "
                "req_ids=%s pending_counts=%s",
                sorted(pending_at_start),
                pending_counts,
            )
            start_time = time.monotonic()
            self._store_cv.wait_for(
                lambda: not any(
                    req_id in self._pending_store_reqs for req_id in req_id_set
                )
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Pending async stores drained before preemption: req_ids=%s "
            "elapsed=%.4f ms",
            sorted(pending_at_start),
            elapsed_ms,
        )
        return pending_at_start | direct_waited

    def _track_sync_store_futures(
        self,
        futures: Iterable[Future],
        *,
        require_completion: bool = False,
    ) -> None:
        """Track backend futures that must complete before save publication."""
        if (
            require_completion
            or not self.is_store_async
            or self._require_store_completion
        ):
            self._pending_sync_store_futures.update(futures)

    def wait_for_pending_sync_stores(self) -> None:
        """Wait for stores submitted by the current synchronous save step."""
        if not self._pending_sync_store_futures:
            return

        futures = self._pending_sync_store_futures
        done, pending = wait(
            futures, timeout=self.config.blocking_timeout_secs
        )
        self._pending_sync_store_futures = set(pending)

        for future in done:
            future.result()
        if pending:
            raise TimeoutError(
                f"Timed out waiting for {len(pending)} remote store operation(s)"
            )

    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and self.kv_events:
            return self.kv_events.drain()
        return []

    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        # Serialize against the store-worker thread's
        with self._engine_state_lock:
            return super().lookup(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                search_range=search_range,
                lookup_id=lookup_id,
                pin=pin,
                request_configs=request_configs,
            )

    def lookup_unpin(self, lookup_id: str) -> None:
        with self._engine_state_lock:
            super().lookup_unpin(lookup_id)

    def _maybe_unpin_retrieved_objs(
        self,
        mem_objs: List[MemoryObj],
        location: Optional[str],
    ) -> None:
        # LocalCPU hot-cache objects are unpinned by lookup_unpin() in
        # wait_for_save(). batched_get returns the same object that lookup
        # pinned; unpinning here as well drives pin_count negative (#2954).
        for mem_obj in mem_objs:
            if mem_obj.is_pinned and location != "LocalCPUBackend":
                mem_obj.unpin()

    def _append_layerwise_store_cache_chunks(
        self,
        *,
        keys: List[List[CacheEngineKey]],
        starts: List[int],
        ends: List[int],
        memory_objs: List[List[MemoryObj]],
        cached_keys: Optional[List],
        cached_starts: Optional[List[int]],
        cached_ends: Optional[List[int]],
        cached_memory_objs: Optional[List],
        cached_tensors: Optional[List],
        cache_chunk_indices: Optional[List[int]] = None,
    ) -> None:
        """Append cacheable store chunks; storage still receives the full batch."""
        num_layers = len(keys)
        chunk_indices = (
            list(range(len(starts)))
            if cache_chunk_indices is None
            else cache_chunk_indices
        )
        if any(index < 0 or index >= len(starts) for index in chunk_indices):
            raise ValueError(
                "Layerwise store cache chunk selection is out of range: "
                f"indices={chunk_indices}, chunk_count={len(starts)}"
            )
        selected_starts = [starts[index] for index in chunk_indices]
        selected_ends = [ends[index] for index in chunk_indices]
        if cached_starts is not None:
            cached_starts.extend(selected_starts)
        if cached_ends is not None:
            cached_ends.extend(selected_ends)
        if cached_keys is not None:
            if not cached_keys:
                cached_keys.extend([] for _ in range(num_layers))
            assert len(cached_keys) == num_layers, (
                f"cached_keys has {len(cached_keys)} layers, expected {num_layers}"
            )
            for layer_id, layer_keys in enumerate(keys):
                cached_keys[layer_id].extend(
                    layer_keys[index] for index in chunk_indices
                )
        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend([] for _ in range(num_layers))
            assert len(cached_memory_objs) == num_layers
            for layer_id, layer_objs in enumerate(memory_objs):
                cached_memory_objs[layer_id].extend(
                    layer_objs[index] for index in chunk_indices
                )
        if cached_tensors is not None and not cached_tensors:
            cached_tensors.extend([] for _ in range(num_layers))

    @staticmethod
    def _truncate_store_cache_for_full_chunk_successor(
        *,
        starts: List[int],
        ends: List[int],
        new_starts: List[int],
        new_ends: List[int],
        cached_keys: Optional[List],
        cached_memory_objs: Optional[List],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List],
        cached_chunk_ptrs_npu: Optional[List],
    ) -> Optional[int]:
        """Drop a cached partial tail before publishing its full successor."""
        replace_at: Optional[int] = None
        for new_start, new_end in zip(new_starts, new_ends, strict=False):
            for chunk_index, (old_start, old_end) in enumerate(
                zip(starts, ends, strict=False)
            ):
                if old_start == new_start and new_end > old_end:
                    replace_at = chunk_index
                    break
            if replace_at is not None:
                break
        if replace_at is None:
            return None

        del starts[replace_at:]
        del ends[replace_at:]
        for layer_cache in (
            cached_keys,
            cached_memory_objs,
            cached_tensors,
            cached_chunk_dev_ptrs,
        ):
            if layer_cache is None:
                continue
            for layer_values in layer_cache:
                if isinstance(layer_values, list):
                    del layer_values[replace_at:]

        if cached_chunk_ptrs_npu is not None:
            for layer_id, ptrs in enumerate(cached_chunk_ptrs_npu):
                if isinstance(ptrs, torch.Tensor):
                    cached_chunk_ptrs_npu[layer_id] = ptrs[:replace_at]
        return replace_at

    @staticmethod
    def _remove_pending_layerwise_store_objs(
        cached_memory_objs: Optional[List],
        pending_store_release: dict[int, MemoryObj],
    ) -> None:
        if cached_memory_objs is None or not pending_store_release:
            return
        pending_ids = set(pending_store_release)
        for layer_cache in cached_memory_objs:
            if not layer_cache:
                continue
            layer_cache[:] = [
                mem_obj
                for mem_obj in layer_cache
                if id(mem_obj) not in pending_ids
            ]

    @staticmethod
    def _layer_memory_tensor(
        memory_obj: MemoryObj, layer_id: int
    ) -> torch.Tensor:
        tensor = (
            memory_obj.layer_tensor(layer_id)
            if isinstance(memory_obj, LayerPageMemoryObj)
            else memory_obj.tensor
        )
        if tensor is None:
            raise ValueError(
                "Layerwise cache source has no tensor: "
                f"layer_id={layer_id}, type={type(memory_obj).__name__}"
            )
        return tensor

    def _append_layer_store_tensors(
        self,
        layer_id: int,
        memory_objs: List[List[MemoryObj]],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List] = None,
        cached_chunk_ptrs_npu: Optional[List] = None,
        cache_chunk_indices: Optional[List[int]] = None,
    ) -> None:
        layer_memory_objs = memory_objs[layer_id]
        if cache_chunk_indices is not None:
            layer_memory_objs = [
                layer_memory_objs[index] for index in cache_chunk_indices
            ]
        has_pages = any(
            isinstance(memory_obj, LayerPageMemoryObj)
            for memory_obj in layer_memory_objs
        )
        cache_tensors = cached_tensors is not None and (
            any(cached_tensors) or not has_pages
        )
        new_tensors = (
            [
                AscendLMCacheEngine._layer_memory_tensor(mem_obj, layer_id)
                for mem_obj in layer_memory_objs
            ]
            if cache_tensors or not has_pages
            else []
        )
        if layer_memory_objs and cached_chunk_dev_ptrs is not None:
            # Resolve direct-load pointer tables while the dense store is on
            # the cold path, before WorkerRetrieveState is sealed.
            append_ptrs_fn = getattr(
                self.gpu_connector,
                "append_sparse_chunk_ptr_cache_for_layer",
                None,
            )
            if append_ptrs_fn is not None:
                append_ptrs_fn(
                    layer_id,
                    (
                        layer_memory_objs
                        if has_pages and not cache_tensors
                        else new_tensors
                    ),
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                )

        if not cache_tensors:
            return
        assert cached_tensors is not None
        while len(cached_tensors) <= layer_id:
            cached_tensors.append([])
        cached_tensors[layer_id].extend(new_tensors)

    def _append_retrieve_layer_cache(
        self,
        layer_id: int,
        mem_objs_layer: List[MemoryObj],
        cached_memory_objs: Optional[List],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List],
        cached_chunk_ptrs_npu: Optional[List],
    ) -> None:
        """Retain storage-get results for later retrieves in the same request."""
        new_tensors: List[torch.Tensor] = []
        append_ptrs_fn = getattr(
            self.gpu_connector,
            "append_sparse_chunk_ptr_cache_for_layer",
            None,
        )
        pointer_first = (
            append_ptrs_fn is not None
            and cached_memory_objs is not None
            and cached_chunk_dev_ptrs is not None
            and cached_chunk_ptrs_npu is not None
            and not any(cached_tensors or ())
        )
        if pointer_first:
            assert append_ptrs_fn is not None
            if any(not mem_obj.is_valid() for mem_obj in mem_objs_layer):
                raise ValueError(
                    "Layerwise sparse retrieve resolved an invalid MemoryObj."
                )
            append_ptrs_fn(
                layer_id,
                mem_objs_layer,
                cached_chunk_dev_ptrs,
                cached_chunk_ptrs_npu,
            )
        else:
            new_tensors = [
                AscendLMCacheEngine._layer_memory_tensor(mem_obj, layer_id)
                for mem_obj in mem_objs_layer
            ]
            if (
                new_tensors
                and cached_chunk_dev_ptrs is not None
                and append_ptrs_fn is not None
            ):
                append_ptrs_fn(
                    layer_id,
                    new_tensors,
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                )

        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend(
                    [] for _ in range(self.num_layers)
                )
            cached_memory_objs[layer_id].extend(mem_objs_layer)
        if cached_tensors is not None:
            if not cached_tensors:
                cached_tensors.extend([] for _ in range(self.num_layers))
            cached_tensors[layer_id].extend(new_tensors)

    def _append_group_store_tensors(
        self,
        memory_objs: List[List[MemoryObj]],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List],
        cached_chunk_ptrs_npu: Optional[List],
        host_pointer_rows: List[List[int]],
        layer_chunk_ptrs_npu: torch.Tensor,
    ) -> None:
        """Publish one native group store's packed pointer metadata."""
        num_layers = len(memory_objs)
        if len(host_pointer_rows) != num_layers:
            raise ValueError("Dense group store pointer rows do not match layers.")
        if (
            layer_chunk_ptrs_npu.dim() != 2
            or layer_chunk_ptrs_npu.size(0) != num_layers
        ):
            raise ValueError(
                "Dense group store NPU pointer table must be [layers, chunks]."
            )
        has_pages = any(
            isinstance(memory_obj, LayerPageMemoryObj)
            for memory_obj in memory_objs[0]
        )
        cache_tensors = cached_tensors is not None and (
            any(cached_tensors) or not has_pages
        )
        if cache_tensors and not cached_tensors:
            cached_tensors.extend([] for _ in range(num_layers))
        if cached_chunk_dev_ptrs is not None and not cached_chunk_dev_ptrs:
            cached_chunk_dev_ptrs.extend([] for _ in range(num_layers))
        if cached_chunk_ptrs_npu is not None and not cached_chunk_ptrs_npu:
            cached_chunk_ptrs_npu.extend(None for _ in range(num_layers))

        for layer_id, layer_memory_objs in enumerate(memory_objs):
            host_row = host_pointer_rows[layer_id]
            if len(host_row) != len(layer_memory_objs):
                raise ValueError("Dense group store pointer count mismatch.")
            if cache_tensors:
                tensors = [
                    AscendLMCacheEngine._layer_memory_tensor(
                        memory_obj, layer_id
                    )
                    for memory_obj in layer_memory_objs
                ]
                assert cached_tensors is not None
                cached_tensors[layer_id].extend(tensors)
            if cached_chunk_dev_ptrs is not None:
                cached_chunk_dev_ptrs[layer_id].extend(host_row)
            if cached_chunk_ptrs_npu is not None:
                existing = cached_chunk_ptrs_npu[layer_id]
                new_row = layer_chunk_ptrs_npu[layer_id]
                cached_chunk_ptrs_npu[layer_id] = (
                    new_row
                    if existing is None
                    else torch.cat((existing, new_row), dim=0)
                )

    def _resolve_shared_rank0_layer_pages(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        page_chunks: int,
        base_page_keys: Optional[list[CacheEngineKey]] = None,
        exact_chunk_locations: Optional[list[str]] = None,
    ) -> tuple[list[list[MemoryObj]], int]:
        """Resolve exact-size full or partial layer pages with legacy fallback."""
        if not 0 < page_chunks <= len(keys_layer_major[0]):
            raise ValueError("Layer-page retrieval requires at least one chunk")
        page_keys = (
            list(base_page_keys[:page_chunks])
            if base_page_keys is not None
            else [
                key.without_layer()
                for key in keys_layer_major[0][:page_chunks]
                if isinstance(key, LayerCacheEngineKey)
            ]
        )
        if len(page_keys) != page_chunks:
            raise ValueError("Layer-page retrieval requires one base key per page")
        if exact_chunk_locations is not None:
            return self._resolve_shared_rank0_exact_layer_pages(
                req_id=req_id,
                phase=phase,
                kv_group=kv_group,
                keys_layer_major=keys_layer_major,
                page_chunks=page_chunks,
                base_page_keys=page_keys,
                chunk_locations=exact_chunk_locations,
            )

        local = self._shared_local_cpu_backend()
        assert self.storage_manager is not None
        pages, local_count = local.batched_get_layer_page_prefix(page_keys)
        owned: list[MemoryObj] = list(pages)
        pinned: list[MemoryObj] = []
        try:
            legacy_suffix = local_count < page_chunks and local.contains_all_exact(
                page_keys[local_count].split_layers(self.num_layers)
            )
            tail_start = local_count
            if local_count < page_chunks and not legacy_suffix:
                remote = self.storage_manager.storage_backends.get("RemoteBackend")
                contains = getattr(remote, "batched_contains_layer_pages", None)
                retrieve = getattr(remote, "batched_get_layer_pages", None)
                if not callable(contains) or not callable(retrieve):
                    raise RuntimeError("RemoteBackend does not support layer pages")
                remote_keys = page_keys[local_count:page_chunks]
                remote_count = contains(remote_keys)
                if not 0 <= remote_count <= len(remote_keys):
                    raise ValueError("Remote layer-page lookup returned invalid count")
                fetched = retrieve(remote_keys[:remote_count]) if remote_count else []
                if len(fetched) != remote_count:
                    raise ValueError("Remote layer-page result count is inconsistent")
                pages.extend(fetched)
                owned.extend(fetched)
                tail_start += remote_count

            legacy_page_layers = (
                [
                    list(layer)
                    for layer in zip(
                        *(
                            key.split_layers(self.num_layers)
                            for key in page_keys[tail_start:page_chunks]
                        ),
                        strict=True,
                    )
                ]
                if tail_start < page_chunks
                else [[] for _ in range(self.num_layers)]
            )
            tail_keys_layer_major = [
                legacy_page_layers[layer_id]
                + list(keys_layer_major[layer_id][page_chunks:])
                for layer_id in range(self.num_layers)
            ]
            tail = (
                self._resolve_shared_rank0_page_first_layers(
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    keys_layer_major=tail_keys_layer_major,
                )
                if tail_keys_layer_major[0]
                else [[] for _ in keys_layer_major]
            )
            tail_objects = [obj for layer in tail for obj in layer]
            owned.extend(tail_objects)
            pinned.extend(tail_objects)
            if len(pages) != tail_start:
                raise ValueError(
                    f"Layer-page result count {len(pages)} != {tail_start}"
                )
            invalid_page = False
            for chunk_index, page in enumerate(pages):
                token_count = int(getattr(page, "valid_tokens", 0))
                expected_shape, _, _ = self._expected_shared_cpu_chunk_metadata(
                    kv_group=kv_group,
                    num_tokens=token_count,
                )
                invalid_page |= (
                    page.num_layers != self.num_layers
                    or page.get_shape() != expected_shape
                )
            if invalid_page or not LayerPageMemoryObj.pin_many(pages):
                raise ValueError("Invalid or unpinnable layer-page object")
            pinned.extend(pages)
            for chunk_index, page in enumerate(pages):
                self._validate_rank0_shared_mem_obj(
                    page,
                    req_id=req_id,
                    phase=phase,
                    layer_id=0,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                )
            return [list(pages) + layer for layer in tail], tail_start
        except Exception:
            for obj in reversed(pinned):
                obj.unpin()
            self._release_shared_retrieve_objs(owned, unpin=False)
            raise

    def prefetch_shared_layer_pages(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        retrieve_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[str]:
        """Fetch shared Group-1 pages without collectives or NPU writes.

        Args:
            tokens: Token IDs whose Group-1 pages should be prefetched.
            mask: Optional token-selection mask.
            retrieve_kwargs: Mutable retrieve state to populate in place. This
                avoids regenerating metadata or masks during materialization.
            **kwargs: Layerwise retrieve state. The cache lists are updated in
                place only after complete page coverage has been validated.

        Returns:
            The resolved storage location, or ``None`` when this rank or
            configuration cannot use the shared page-prefetch path.

        Raises:
            ValueError: If retrieve metadata or returned page coverage is
                incomplete.
            RuntimeError: If the configured backend cannot retrieve pages.
        """
        if retrieve_kwargs is not None:
            if kwargs:
                raise ValueError(
                    "Pass Group-1 prefetch state either as retrieve_kwargs "
                    "or keyword fields, not both"
                )
            kwargs = retrieve_kwargs
        kv_group = int(kwargs.get("kv_group", 0) or 0)
        if (
            kv_group != 1
            or not self._should_use_shared_layerwise_retrieve(kv_group)
            or self._is_shared_retrieve_passive(kv_group)
            or not mooncake_layer_pages_enabled(self.config)
        ):
            return None
        self._ensure_layerwise_connector_layout(**kwargs)
        cached_keys = kwargs["cached_keys"]
        cached_starts = kwargs["cached_starts"]
        cached_ends = kwargs["cached_ends"]
        cached_memory_objs = kwargs["cached_memory_objs"]
        if self._has_retrieve_data_cache(
            kwargs.get("cached_tensors"),
            cached_memory_objs,
            self.num_layers,
        ):
            return kwargs.get("cached_retrieve_location")
        ret_mask = kwargs.get("ret_mask")
        if ret_mask is None or ret_mask.numel() != len(tokens):
            ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
            kwargs["ret_mask"] = ret_mask
        started = cold_start_perf_now()
        location, _, _, retrieve_keys = self._ensure_retrieve_chunk_metadata(
            tokens=tokens,
            mask=mask,
            request_configs=kwargs.get("request_configs"),
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=kwargs,
        )
        required_chunks = len(retrieve_keys[0]) if retrieve_keys else 0
        if (
            len(retrieve_keys) != self.num_layers
            or not required_chunks
            or any(
                len(layer) != required_chunks for layer in retrieve_keys
            )
        ):
            raise ValueError("Group-1 prefetch metadata is incomplete")
        memory_objs, layer_page_chunks = self._resolve_shared_rank0_layer_pages(
            req_id=kwargs.get("req_id", "unspecified"),
            phase="persistent_parallel_prefetch",
            kv_group=kv_group,
            keys_layer_major=retrieve_keys,
            page_chunks=required_chunks,
        )
        if len(memory_objs) != self.num_layers or any(
            len(layer) != required_chunks for layer in memory_objs
        ):
            unique = {
                id(obj): obj for layer in memory_objs for obj in layer
            }
            self._release_shared_retrieve_objs(
                list(unique.values()), unpin=True
            )
            raise ValueError("Group-1 prefetch page coverage is incomplete")
        try:
            cold_start_perf_log(
                logger,
                "group1_persistent_prefetch_complete",
                started=started,
                req_id=kwargs.get("req_id", "unspecified"),
                pages=required_chunks,
                layers=self.num_layers,
                location=location,
            )
            cached_memory_objs[:] = memory_objs
            # Preserve the physical page layout for the later materialization
            # pass. Without this marker the cached page is treated as one
            # independent object per layer, which both builds the wrong NPU
            # source view and can retain/pin the same page once per layer.
            kwargs["_cached_layer_page_chunks"] = layer_page_chunks
        except BaseException:
            unique = {id(obj): obj for layer in memory_objs for obj in layer}
            self._release_shared_retrieve_objs(
                list(unique.values()), unpin=True
            )
            raise
        return location

    def _try_direct_indexer_page_load(
        self,
        *,
        req_id: str,
        keys_layer_major: list[list[CacheEngineKey]],
        starts: list[int],
        ends: list[int],
        retrieve_kwargs: dict[str, Any],
    ) -> bool:
        """Load a cold group-1 page set into NPU, or return False for CPU fallback."""
        if not keys_layer_major or not keys_layer_major[0]:
            return False
        planner = getattr(self.gpu_connector, "plan_direct_page_destinations", None)
        record = getattr(
            self.gpu_connector, "record_dense_load_readiness", None
        )
        consume = getattr(
            self.gpu_connector, "consume_dense_load_readiness", None
        )
        kvcaches = retrieve_kwargs.get("kvcaches")
        slot_mapping = retrieve_kwargs.get("slot_mapping")
        direct_load_submitted = False
        if (
            not callable(planner)
            or not callable(record)
            or not callable(consume)
            or kvcaches is None
            or slot_mapping is None
        ):
            return False
        try:
            pages = [
                key.without_layer()
                for key in keys_layer_major[0]
                if isinstance(key, LayerCacheEngineKey)
            ]
            if len(pages) != len(keys_layer_major[0]):
                return False
            planned = planner(kvcaches, slot_mapping, starts, ends, 1)
            if planned is None:
                if cold_start_perf_enabled():
                    rejection = getattr(
                        self.gpu_connector,
                        "direct_page_plan_rejection",
                        None,
                    )
                    cold_start_perf_log(
                        logger,
                        "cold_compact_indexer_fallback",
                        req_id=req_id,
                        pages=len(pages),
                        fallback="cpu_staged",
                        reason=(
                            rejection(1) or "unsupported_layout"
                            if callable(rejection)
                            else "unsupported_layout"
                        ),
                    )
                return False
            ptrs, sizes, owners = planned
            if len(ptrs) != len(pages) or len(sizes) != len(pages):
                return False
            assert self.storage_manager is not None
            direct_load_submitted = True
            self.storage_manager.batched_get_external_pages(
                pages, ptrs, sizes, owners, req_id
            )
            if not retrieve_kwargs.get("_defer_direct_load_readiness", False):
                consume(record())
            return True
        except Exception as exc:
            if isinstance(exc, NativeExternalPageTransferUnknownError):
                self._remote_fill_require_paired_restart((req_id,))
                raise RemoteFillFatalError(
                    "decoder external-page DMA exceeded its hard deadline"
                ) from exc
            if direct_load_submitted:
                # A backend failure can be reported after some direct writes
                # have already been queued.  Fence those writes, while their
                # destination owners are still live, before CPU fallback is
                # allowed to target the same group-1 slots.
                consume(record())
            logger.warning(
                "Direct Mooncake-to-NPU indexer load failed; using CPU-staged path",
                exc_info=True,
            )
            if cold_start_perf_enabled():
                cold_start_perf_log(
                    logger,
                    "cold_compact_indexer_fallback",
                    req_id=req_id,
                    pages=len(keys_layer_major[0]),
                    fallback="cpu_staged",
                    reason=type(exc).__name__,
                )
            return False

    def _prepare_live_split_import(
        self,
        *,
        tokens: list[int],
        indexer_slots: torch.Tensor,
        indexer_kvcaches: list,
        latent_kvcaches: Optional[list] = None,
        request_configs: Optional[dict],
        tp_rank: int,
        dp_rank: int,
        handled_groups: tuple[int, ...] = (0, 1),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Allocate cold destinations and describe opaque live-transfer extents."""
        mask = torch.ones(len(tokens), dtype=torch.bool)
        chunks = list(
            self._dense_retrieve_token_results(
                tokens, mask, request_configs, 0, {}
            )
        )
        if not chunks:
            raise ValueError("Live split import has no token pages")
        page_keys = [
            key.without_layer() if isinstance(key, LayerCacheEngineKey) else key
            for _, _, key in chunks
        ]
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("Live split import contains duplicate page keys")
        starts = [start for start, _, _ in chunks]
        ends = [end for _, end, _ in chunks]
        valid_tokens = [end - start for start, end, _ in chunks]
        pages: list[LayerPageMemoryObj] = []
        if 0 in handled_groups:
            if self._is_passive():
                raise RuntimeError(
                    "Passive ranks cannot allocate live group-0 destinations"
                )
            token_width_planner = getattr(
                self.gpu_connector, "direct_page_token_widths", None
            )
            latent_token_bytes = (
                token_width_planner(latent_kvcaches, 0)
                if latent_kvcaches and callable(token_width_planner)
                else None
            )
            if not latent_token_bytes:
                raise ValueError("Live split latent destination layout is unsupported")
            self._ensure_layerwise_connector_layout(
                kvcaches=latent_kvcaches,
                kv_group=0,
            )
            shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
                kv_group=0, num_tokens=self.config.chunk_size
            )
            local = self._shared_local_cpu_backend()
            allocated = local.batched_allocate_layer_pages(
                [shape], [dtype], len(chunks), self.num_layers, fmt,
                valid_tokens=valid_tokens,
                full_tokens=self.config.chunk_size,
                busy_loop=False,
            )
            if allocated is None:
                raise MemoryError("Live split shared-CPU page allocation failed")
            try:
                pinned = LayerPageMemoryObj.pin_many(allocated)
            except BaseException:
                self._release_shared_retrieve_objs(
                    list(allocated), unpin=False
                )
                raise
            if not pinned:
                self._release_shared_retrieve_objs(
                    list(allocated), unpin=False
                )
                raise MemoryError("Live split shared-CPU page pin failed")
            pages = allocated
        compact_destination = 1 in handled_groups
        destination_planner = getattr(
            self.gpu_connector,
            "plan_compact_page_layout" if compact_destination
            else "plan_direct_page_destinations",
            None,
        )
        if not callable(destination_planner) and compact_destination:
            destination_planner = getattr(
                self.gpu_connector, "plan_direct_page_destinations", None
            )
            compact_destination = False
        if not callable(destination_planner):
            self._release_shared_retrieve_objs(list(pages), unpin=True)
            raise RuntimeError("Live split destination planner is unavailable")
        try:
            destination1 = (
                destination_planner(
                    indexer_kvcaches, indexer_slots, starts, ends, 1
                ) if 1 in handled_groups else ([], [], ())
            )
            if destination1 is None:
                raise ValueError("Live split destination layout is unsupported")
            if compact_destination:
                dst1_layers, dst1_runs, dst1_owners = destination1
            else:
                dst1_ptrs, dst1_sizes, dst1_owners = destination1
                dst1_layers = dst1_runs = []

            segments: list[dict[str, Any]] = []
            latent_pages: list[dict[str, int]] = []
            totals = [0, 0]
            if 0 in handled_groups:
                for page, start, valid in zip(
                    pages, starts, valid_tokens, strict=True
                ):
                    size = page.get_size()
                    latent_pages.append(
                        {
                            "logical_token_start": start,
                            "destination_address": page.data_ptr,
                            "length": size,
                            "valid_tokens": valid,
                        }
                    )
                    totals[0] += size
            if 1 in handled_groups:
                if compact_destination:
                    totals[1] = len(tokens) * sum(
                        layer["token_bytes"] for layer in dst1_layers
                    )
                else:
                    for dest_ptrs, dest_sizes in zip(
                        dst1_ptrs, dst1_sizes, strict=True
                    ):
                        for dst_ptr, size in zip(
                            dest_ptrs, dest_sizes, strict=True
                        ):
                            segments.append({
                                "group_id": 1,
                                "destination_address": dst_ptr,
                                "length": size,
                                "destination_kind": "npu",
                            })
                            totals[1] += size
            plan = {
                "segments": segments,
                "group_byte_totals": tuple(totals),
                "tp_rank": tp_rank,
                "dp_rank": dp_rank,
            }
            if compact_destination:
                plan["format"] = (
                    "hybrid_compact_v1" if latent_pages else "layer_slot_runs_v1"
                )
                plan["compact_layout"] = {
                    "group_id": 1,
                    "token_count": len(tokens),
                    "layers": dst1_layers,
                    "runs": dst1_runs,
                }
            if latent_pages:
                plan["latent_pages"] = latent_pages
                plan["latent_token_bytes"] = list(latent_token_bytes)
            context = {
                "pages": pages,
                "keys": page_keys,
                "starts": starts,
                "ends": ends,
                "destination_owners": dst1_owners,
            }
            return plan, context
        except Exception:
            self._release_shared_retrieve_objs(list(pages), unpin=True)
            raise

    def admit_live_split_pages(self, context: dict[str, Any]) -> None:
        """Admit completed group-0 pages for the existing shared bootstrap.

        The LocalCPU cache takes its own references synchronously.  The live
        import's temporary pins and allocator references are then released so
        the ordinary rank-0 bootstrap can acquire the pages and publish the
        existing shared-handle collective without a second ownership model.
        """
        pages = context["pages"]
        if not pages:
            raise RuntimeError("Live split import has no latent pages to admit")
        context["pages"] = []
        local = self._shared_local_cpu_backend()
        try:
            local.batched_submit_layer_pages(context["keys"], pages)
            compatible = getattr(
                local, "contains_compatible_layer_pages_exact", None
            )
            if not callable(compatible) or not compatible(
                context["keys"], pages
            ):
                raise RuntimeError(
                    "Live split pages lost admission to incompatible cache entries"
                )
        finally:
            # Existing entries may win the atomic cache admission race. The
            # cache owns a reference for every newly admitted page; duplicate
            # imports retain no cache reference and are released here as well.
            self._release_shared_retrieve_objs(list(pages), unpin=True)

    def _release_live_split_import(self, context: dict[str, Any]) -> None:
        """Release an unacknowledged live-import allocation."""
        pages = context["pages"]
        context["pages"] = []
        self._release_shared_retrieve_objs(pages, unpin=True)

    def _append_retrieve_group_cache(
        self,
        mem_objs_by_layer: List[Union[List[MemoryObj], LayerPageSource]],
        cached_memory_objs: Optional[List],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List],
        cached_chunk_ptrs_npu: Optional[List],
        prepared_chunk_dev_ptrs: Optional[List] = None,
        prepared_chunk_ptrs_npu: Optional[List] = None,
    ) -> None:
        """Publish an all-layer retained source after one pointer-table copy."""
        group_append = getattr(
            self.gpu_connector, "append_sparse_chunk_ptr_cache_for_layers", None
        )
        retained_group = (
            callable(group_append)
            and cached_memory_objs is not None
            and cached_chunk_dev_ptrs is not None
            and cached_chunk_ptrs_npu is not None
        )
        page_sources = any(
            isinstance(source, LayerPageSource) for source in mem_objs_by_layer
        )
        if page_sources and (
            not callable(group_append) or cached_chunk_dev_ptrs is None
        ):
            raise ValueError(
                "Layer-page sparse sources require all-layer pointer append."
            )
        pointer_first = retained_group and not any(cached_tensors or ())
        if retained_group:
            layer_counts = (
                len(cached_memory_objs),
                len(cached_chunk_dev_ptrs),
                len(cached_chunk_ptrs_npu),
            )
            has_prefix_data = (
                any(bool(layer) for layer in cached_memory_objs)
                or any(bool(layer) for layer in cached_chunk_dev_ptrs)
                or any(
                    row is not None and int(row.numel())
                    for row in cached_chunk_ptrs_npu
                )
                or any(cached_tensors or ())
            )
            if has_prefix_data:
                if any(count != self.num_layers for count in layer_counts):
                    raise ValueError(
                        "Sparse group pointer prefix layer coverage mismatch: "
                        f"owners/host/NPU={layer_counts}, "
                        f"expected={self.num_layers}."
                    )
                for layer_id in range(self.num_layers):
                    owner_count = len(cached_memory_objs[layer_id])
                    host_count = len(cached_chunk_dev_ptrs[layer_id])
                    row = cached_chunk_ptrs_npu[layer_id]
                    npu_count = 0 if row is None else int(row.numel())
                    if owner_count != host_count or host_count != npu_count:
                        raise ValueError(
                            "Sparse group pointer prefix coverage mismatch: "
                            f"layer={layer_id}, owners={owner_count}, "
                            f"host_ptrs={host_count}, npu_ptrs={npu_count}."
                        )
        owners_by_layer: List[List[MemoryObj]] = []
        tensors_by_layer: List[List[torch.Tensor]] = []
        for layer_id, source in enumerate(mem_objs_by_layer):
            mem_objs_layer = (
                [*source.pages, *source.suffix]
                if isinstance(source, LayerPageSource)
                else source
            )
            if any(not mem_obj.is_valid() for mem_obj in mem_objs_layer):
                raise ValueError(
                    f"Layerwise sparse retrieve resolved an invalid layer {layer_id}."
                )
            owners_by_layer.append(mem_objs_layer)
            tensors = []
            if not pointer_first:
                tensors = [
                    AscendLMCacheEngine._layer_memory_tensor(mem_obj, layer_id)
                    for mem_obj in mem_objs_layer
                ]
            tensors_by_layer.append(tensors)

        if not callable(group_append) or cached_chunk_dev_ptrs is None:
            for layer_id, mem_objs_layer in enumerate(owners_by_layer):
                self._append_retrieve_layer_cache(
                    layer_id,
                    mem_objs_layer,
                    cached_memory_objs,
                    cached_tensors,
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                )
            return
        if prepared_chunk_dev_ptrs is None:
            group_append(
                mem_objs_by_layer if pointer_first else tensors_by_layer,
                cached_chunk_dev_ptrs,
                cached_chunk_ptrs_npu,
            )
        else:
            if any((cached_chunk_dev_ptrs, cached_chunk_ptrs_npu)):
                raise ValueError(
                    "Prepared sparse pointer rows require an empty cache suffix."
                )
            expected = [len(owners) for owners in owners_by_layer]
            if [len(row) for row in prepared_chunk_dev_ptrs] != expected:
                raise ValueError("Prepared sparse host pointer coverage mismatch.")
            if prepared_chunk_ptrs_npu is None or [
                0 if row is None else int(row.numel())
                for row in prepared_chunk_ptrs_npu
            ] != expected:
                raise ValueError("Prepared sparse NPU pointer coverage mismatch.")
            cached_chunk_dev_ptrs.extend(prepared_chunk_dev_ptrs)
            assert cached_chunk_ptrs_npu is not None
            cached_chunk_ptrs_npu.extend(prepared_chunk_ptrs_npu)
        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend([] for _ in range(self.num_layers))
            for layer_id, mem_objs_layer in enumerate(owners_by_layer):
                cached_memory_objs[layer_id].extend(mem_objs_layer)
        if cached_tensors is not None and not pointer_first:
            if not cached_tensors:
                cached_tensors.extend([] for _ in range(self.num_layers))
            for layer_id, tensors in enumerate(tensors_by_layer):
                cached_tensors[layer_id].extend(tensors)

    def _append_shared_handle_cache(
        self,
        layer_id: int,
        handles: List,
        cached_shared_handles: Optional[List],
        chunk_index_base: int = 0,
    ) -> None:
        """Append one validated suffix to a layer's shared handle cache."""
        if cached_shared_handles is None:
            return
        if not cached_shared_handles:
            cached_shared_handles.extend([] for _ in range(self.num_layers))
        while len(cached_shared_handles) <= layer_id:
            cached_shared_handles.append([])
        layer_handles = cached_shared_handles[layer_id]
        if len(layer_handles) != chunk_index_base:
            raise ValueError(
                "Sparse shared handle cache is not append-aligned: "
                f"layer_id={layer_id}, cached={len(layer_handles)}, "
                f"append_at={chunk_index_base}"
            )
        layer_handles.extend(handles)

    def _fence_shared_cpu_store_publication(self) -> None:
        fence = getattr(
            self.gpu_connector,
            "synchronize_shared_cpu_store_publication",
            None,
        )
        if callable(fence):
            fence()

    def _is_shared_retrieve_passive(self, kv_group: int) -> bool:
        """Return whether this rank is passive for a shared retrieve group."""
        if kv_group == 1:
            is_indexer_passive = getattr(self, "_is_indexer_passive", None)
            if callable(is_indexer_passive):
                return bool(is_indexer_passive())
        return bool(self._is_passive())

    def _cold_retrieve_collective_mode(
        self, kv_group: int, direct_external_pages: bool
    ) -> tuple[bool, bool]:
        """Resolve shared collective participation for a cold group load."""
        shared = self._should_use_shared_layerwise_retrieve(kv_group)
        if direct_external_pages:
            return False, False
        return shared, shared and self._is_shared_retrieve_passive(kv_group)

    def _use_sampled_worker_retrieve(self, kv_group: int) -> bool:
        """Use LocalCPU-prefix/remote-suffix resolution on the active rank."""
        config = getattr(self, "config", None)
        return bool(
            getattr(config, "experimental_sampled_layerwise_lookup", False)
            and self._should_use_shared_layerwise_retrieve(kv_group)
            and not self._is_shared_retrieve_passive(kv_group)
        )

    def _build_retrieve_metadata_extension(
        self,
        *,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor],
        request_configs: Optional[dict],
        kv_group: int,
        cached_keys: List,
        cached_starts: List[int],
        cached_ends: List[int],
        cached_metadata_token_ids: Optional[list[int]],
    ) -> Optional[tuple[List[int], List[int], List[List[CacheEngineKey]]]]:
        """Hash only a chunk-aligned suffix after validating the cached prefix."""
        chunk_size = int(getattr(self.token_database, "chunk_size", 0) or 0)
        if (
            chunk_size <= 0
            or not cached_keys
            or len(cached_keys) != self.num_layers
            or not cached_starts
            or len(cached_starts) != len(cached_ends)
            or any(len(layer) != len(cached_starts) for layer in cached_keys)
        ):
            return None
        expected_start = 0
        for start, end in zip(cached_starts, cached_ends, strict=True):
            if int(start) != expected_start or int(end) - int(start) != chunk_size:
                return None
            expected_start = int(end)
        if len(tokens) <= expected_start or len(tokens) % chunk_size != 0:
            return None
        if (
            cached_metadata_token_ids is None
            or len(cached_metadata_token_ids) != expected_start
        ):
            return None
        token_prefix = tokens[:expected_start]
        if isinstance(token_prefix, torch.Tensor):
            token_prefix = token_prefix.detach().to(device="cpu").tolist()
        if token_prefix != cached_metadata_token_ids:
            return None
        if mask is not None and (
            not isinstance(mask, torch.Tensor)
            or mask.ndim != 1
            or mask.dtype != torch.bool
            or int(mask.numel()) != len(tokens)
            or not bool(mask.all().item())
        ):
            return None

        identity = cached_keys[0][0]
        if not isinstance(identity, CacheEngineKey):
            return None
        stable_identity = (
            identity.model_name,
            identity.world_size,
            identity.worker_id,
            identity.dtype,
            identity.tags,
        )
        for chunk_index in range(len(cached_starts)):
            chunk_hash = None
            for layer_id in range(self.num_layers):
                cached_key = cached_keys[layer_id][chunk_index]
                if (
                    not isinstance(cached_key, CacheEngineKey)
                    or getattr(cached_key, "layer_id", None) != layer_id
                    or getattr(cached_key, "kv_group", None) != kv_group
                    or (
                        cached_key.model_name,
                        cached_key.world_size,
                        cached_key.worker_id,
                        cached_key.dtype,
                        cached_key.tags,
                    )
                    != stable_identity
                ):
                    return None
                if chunk_hash is None:
                    chunk_hash = cached_key.chunk_hash
                elif cached_key.chunk_hash != chunk_hash:
                    return None

        anchors = [layer[-1] for layer in cached_keys]
        if len({anchor.chunk_hash for anchor in anchors}) != 1:
            return None
        starts: List[int] = []
        ends: List[int] = []
        chunk_keys: List[List[CacheEngineKey]] = []
        for start, end, key in self.token_database.process_tokens_from_prefix(
            tokens,
            prefix_token_count=expected_start,
            prefix_hash=anchors[0].chunk_hash,
            request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)
            layer_keys = key.split_layers(self.num_layers)
            if any(
                layer_key.layer_id != layer_id
                or layer_key.kv_group != kv_group
                or layer_key.model_name != anchors[layer_id].model_name
                or layer_key.world_size != anchors[layer_id].world_size
                or layer_key.worker_id != anchors[layer_id].worker_id
                or layer_key.dtype != anchors[layer_id].dtype
                or layer_key.tags != anchors[layer_id].tags
                for layer_id, layer_key in enumerate(layer_keys)
            ):
                return None
            starts.append(int(start))
            ends.append(int(end))
            chunk_keys.append(layer_keys)
        if not starts or starts[0] != expected_start or ends[-1] != len(tokens):
            return None
        return starts, ends, [
            list(layer) for layer in zip(*chunk_keys, strict=True)
        ]

    @staticmethod
    def _needs_retrieve_metadata_refresh(
        cached_keys: List,
        cached_starts: List[int],
        cached_ends: List[int],
        tokens: Union[torch.Tensor, list[int]],
    ) -> bool:
        if not cached_keys or not cached_starts or not cached_ends:
            return True
        return len(tokens) > cached_ends[-1]

    @staticmethod
    def _fill_retrieve_mask(
        ret_mask: torch.Tensor,
        starts: List[int],
        ends: List[int],
        *,
        clear: bool = True,
    ) -> None:
        if clear:
            ret_mask.zero_()
        if not starts or not ends:
            return
        if len(starts) == len(ends) and all(
            int(start) == int(previous_end)
            for start, previous_end in zip(starts[1:], ends[:-1], strict=False)
        ):
            ret_mask[int(starts[0]) : int(ends[-1])] = True
            return
        for start, end in zip(starts, ends, strict=False):
            ret_mask[int(start) : int(end)] = True

    @staticmethod
    def _cache_layer_has_entries(layer_cache: Any) -> bool:
        if layer_cache is None:
            return False
        try:
            return len(layer_cache) > 0
        except TypeError:
            return True

    @staticmethod
    def _has_retrieve_data_cache(
        cached_tensors: Optional[List],
        cached_memory_objs: Optional[List],
        num_layers: int,
    ) -> bool:
        return any(
            cache is not None
            and len(cache) == num_layers
            and any(
                AscendLMCacheEngine._cache_layer_has_entries(layer)
                for layer in cache
            )
            for cache in (cached_tensors, cached_memory_objs)
        )

    @staticmethod
    def _retrieve_data_cache_covers(
        cache_by_layer: Optional[List],
        num_layers: int,
        required_chunks: int,
    ) -> bool:
        if required_chunks <= 0:
            return False
        if cache_by_layer is None or len(cache_by_layer) != num_layers:
            return False
        for layer_id in range(num_layers):
            layer_cache = cache_by_layer[layer_id]
            if layer_cache is None or len(layer_cache) < required_chunks:
                return False
        return True

    @staticmethod
    def _uniform_layer_cache_chunks(
        cache_by_layer: Optional[List],
        num_layers: int,
        name: str,
    ) -> int:
        if not cache_by_layer:
            return 0
        if len(cache_by_layer) != num_layers:
            raise ValueError(
                f"Sparse retrieve {name} has {len(cache_by_layer)} layers, "
                f"expected {num_layers}."
            )
        counts = {len(layer or []) for layer in cache_by_layer}
        if len(counts) != 1:
            raise ValueError(
                f"Sparse retrieve {name} has inconsistent per-layer chunk "
                f"counts: {sorted(counts)}"
            )
        return counts.pop()

    def _cached_sparse_prefix_chunks(
        self,
        cached_tensors: Optional[List],
        cached_memory_objs: Optional[List],
    ) -> int:
        counts = {
            count
            for count in (
                self._uniform_layer_cache_chunks(
                    cached_tensors,
                    self.num_layers,
                    "tensor cache",
                ),
                self._uniform_layer_cache_chunks(
                    cached_memory_objs,
                    self.num_layers,
                    "MemoryObj cache",
                ),
            )
            if count
        }
        if len(counts) > 1:
            raise ValueError(
                "Sparse retrieve tensor and MemoryObj caches cover different "
                f"prefix lengths: {sorted(counts)}"
            )
        return counts.pop() if counts else 0

    def _ensure_layerwise_connector_layout(self, **kwargs) -> None:
        """Initialize connector KV layout before allocating layerwise chunks.

        gpu_connector.get_shape(num_tokens) depends on the MLA/DSA layout
        being detected (kv_format, kv_lora_rank, qk_rope_head_dim, etc.),
        which happens in _lazy_initialize_buffer. If store_layer is called
        before that lazy init has run, get_shape returns a wrong shape and
        the allocated MemoryObj chunks have the wrong size -- the store
        kernel then writes KV into a malformed buffer, corrupting the cache.
        This ensures the layout is initialized before any chunk allocation.
        """
        init_kvcaches = getattr(
            self.gpu_connector, "initialize_kvcaches_ptr", None
        )
        lazy_init_with_staging = getattr(
            self.gpu_connector, "_lazy_initialize_buffer_with_staging", None
        )
        lazy_init = getattr(
            self.gpu_connector, "_lazy_initialize_buffer", None
        )
        if not callable(init_kvcaches) or (
            not callable(lazy_init_with_staging) and not callable(lazy_init)
        ):
            return
        if "kvcaches" in kwargs:
            init_kvcaches(**kwargs)
        kvcaches = getattr(self.gpu_connector, "kvcaches", None)
        if kvcaches is not None:
            kv_group = kwargs.get("kv_group", 0)
            if callable(lazy_init_with_staging):
                lazy_init_with_staging(
                    kvcaches,
                    kv_group=kv_group,
                    init_staging=False,
                )
                return
            # The layerwise NPU connector detects format per kv_group; other
            # connectors (e.g. the blending buffer connector) may not accept
            # the kwarg, so fall back to the positional call for them.
            try:
                lazy_init(kvcaches, kv_group=kv_group, init_staging=False)
            except TypeError:
                try:
                    lazy_init(kvcaches, kv_group=kv_group)
                except TypeError:
                    lazy_init(kvcaches)

    def _expected_shared_cpu_chunk_metadata(
        self,
        *,
        kv_group: int,
        num_tokens: int,
    ):
        """Use Ascend's group-aware connector shape for shared view checks.

        DSA index chunks are allocated with ``gpu_connector.get_shape(...,
        kv_group=1)``.  Once vLLM has registered both cache groups, its
        ``KVLayerGroupsManager`` is the authoritative startup description.
        Prefer it here because the NPU connector's per-group layout is lazy:
        querying ``get_shape`` before the first model input can otherwise
        mirror the wrong active group and reject a valid decoder layout.

        Generic LMCache metadata can still describe only the latent group in
        older startup/passive paths, so retain the connector fallback.
        """
        layer_groups = getattr(
            self.metadata.kv_layer_groups_manager,
            "kv_layer_groups",
            (),
        )
        if (
            self.metadata.use_mla
            and getattr(self, "dsa_two_groups", False)
            and 0 <= kv_group < len(layer_groups)
        ):
            layer_group = layer_groups[kv_group]
            return (
                torch.Size([num_tokens * layer_group.hidden_dim_size]),
                layer_group.dtype,
                self._memory_format_for_kv_group(kv_group),
            )

        get_shape = getattr(self.gpu_connector, "get_shape", None)
        shape = None
        if callable(get_shape):
            try:
                shape = torch.Size(get_shape(num_tokens, kv_group=kv_group))
            except TypeError:
                if kv_group == 0:
                    shape = torch.Size(get_shape(num_tokens))
        if shape is not None:
            return (
                shape,
                self._shared_cpu_dtype_for_kv_group(kv_group),
                self._memory_format_for_kv_group(kv_group),
            )
        return super()._expected_shared_cpu_chunk_metadata(
            kv_group=kv_group,
            num_tokens=num_tokens,
        )

    def _resolve_local_cpu_retrieve_location(
        self,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """Prefer LocalCPUBackend for in-process tensor cache hits."""
        if (
            self.storage_manager is not None
            and LOCAL_CPU_BACKEND_NAME in self.storage_manager.storage_backends
        ):
            return LOCAL_CPU_BACKEND_NAME
        return fallback

    def _layerwise_chunk_location(
        self,
        keys_multi_layer: List[CacheEngineKey],
    ) -> Optional[str]:
        """Return storage location only if every layer key for a chunk exists."""
        location: Optional[str] = None
        for layer_key in keys_multi_layer:
            current_location = self.storage_manager.contains(
                layer_key, self.retrieve_locations
            )
            if current_location is None:
                return None
            if location is None:
                location = current_location
            elif location != current_location:
                return None
        return location

    def _cached_layerwise_location_or_raise(
        self,
        cached_keys: List,
        cached_starts: List[int],
        cached_ends: List[int],
        retrieve_kwargs: Optional[dict],
    ) -> Optional[str]:
        if not cached_keys or len(cached_keys) < self.num_layers:
            return None
        if not cached_starts or not cached_ends or not cached_keys[0]:
            return None

        chunk_count = min(
            len(cached_starts),
            len(cached_ends),
            *(len(cached_keys[layer_id]) for layer_id in range(self.num_layers)),
        )
        if chunk_count <= 0:
            return None

        kwargs = retrieve_kwargs or {}
        req_id = self._get_req_id(kwargs)
        kv_group = int(kwargs.get("kv_group", 0) or 0)
        location: Optional[str] = None
        for chunk_index in range(chunk_count):
            keys_multi_layer = [
                cached_keys[layer_id][chunk_index]
                for layer_id in range(self.num_layers)
            ]
            current_location = self._layerwise_chunk_location_if_fully_stored(
                keys_multi_layer,
                req_id=req_id,
                kv_group=kv_group,
                start=int(cached_starts[chunk_index]),
                end=int(cached_ends[chunk_index]),
            )
            if current_location is None:
                raise ValueError(
                    "Layerwise sparse retrieve metadata is warm but the "
                    "cached chunk is no longer complete in storage: "
                    f"req_id={req_id}, kv_group={kv_group}, "
                    f"chunk_index={chunk_index}, "
                    f"start={cached_starts[chunk_index]}, "
                    f"end={cached_ends[chunk_index]}"
                )
            if location is None:
                location = current_location
            elif location != current_location:
                raise ValueError(
                    "Layerwise sparse retrieve metadata is warm but cached "
                    "chunks span multiple storage locations, which non-shared "
                    "layerwise retrieve cannot consume safely: "
                    f"req_id={req_id}, kv_group={kv_group}, "
                    f"first_location={location}, "
                    f"current_location={current_location}, "
                    f"chunk_index={chunk_index}"
                )
        return location

    def _ensure_retrieve_chunk_metadata(
        self,
        *,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor],
        request_configs: Optional[dict],
        cached_keys: List,
        cached_starts: List[int],
        cached_ends: List[int],
        ret_mask: torch.Tensor,
        retrieve_kwargs: Optional[dict] = None,
    ) -> tuple[
        Optional[str],
        List[int],
        List[int],
        List[List[CacheEngineKey]],
    ]:
        """Resolve retrieve chunk metadata once, then reuse the request cache."""
        if self._needs_retrieve_metadata_refresh(
            cached_keys, cached_starts, cached_ends, tokens
        ):
            if retrieve_kwargs is not None:
                retrieve_kwargs.pop("_retrieve_metadata_warm", None)

            kv_group = 0
            if retrieve_kwargs is not None:
                kv_group = int(retrieve_kwargs.get("kv_group", 0) or 0)
            shared_rank0_retrieve = (
                retrieve_kwargs is not None
                and self._should_use_shared_layerwise_retrieve(kv_group)
                and not self._is_shared_retrieve_passive(kv_group)
            )
            sampled_worker_retrieve = self._use_sampled_worker_retrieve(kv_group)

            location: Optional[str] = None
            new_starts: List[int] = []
            new_ends: List[int] = []
            keys: List[List[CacheEngineKey]] = []
            preflight_state = (
                retrieve_kwargs.get("shared_cpu_request_preflight_state")
                if retrieve_kwargs is not None
                else None
            )
            chunk_plan = (
                preflight_state.pop(
                    _SHARED_CPU_CHUNK_PLAN_KEY,
                    None,
                )
                if sampled_worker_retrieve
                and kv_group == 1
                and isinstance(preflight_state, dict)
                else None
            )
            metadata_extension = (
                self._build_retrieve_metadata_extension(
                    tokens=tokens,
                    mask=mask,
                    request_configs=request_configs,
                    kv_group=kv_group,
                    cached_keys=cached_keys,
                    cached_starts=cached_starts,
                    cached_ends=cached_ends,
                    cached_metadata_token_ids=(retrieve_kwargs or {}).get(
                        "cached_metadata_token_ids"
                    ),
                )
                if chunk_plan is None
                else None
            )
            chunk_index_base = 0
            if metadata_extension is not None:
                extension_starts, extension_ends, extension_keys = metadata_extension
                chunk_index_base = len(cached_starts)
                token_results = (
                    (start, end, extension_keys[layer_id][chunk_index])
                    for chunk_index, (start, end) in enumerate(
                        zip(extension_starts, extension_ends, strict=True)
                    )
                    for layer_id in (0,)
                )
            elif chunk_plan is not None:
                generated_keys = self.token_database.process_tokens(
                    hashes=[entry[2] for entry in chunk_plan],
                    offsets=[entry[1] - entry[0] for entry in chunk_plan],
                    request_configs=request_configs,
                    kv_group=kv_group,
                )
                token_results = (
                    (start, end, key)
                    for (start, end, _), (_, _, key) in zip(
                        chunk_plan, generated_keys, strict=True
                    )
                )
            else:
                token_results = self.token_database.process_tokens(
                    tokens=tokens,
                    mask=mask,
                    request_configs=request_configs,
                    kv_group=kv_group,
                )
            new_chunk_plan: Optional[list[tuple[int, int, Any]]] = (
                [
                    (
                        int(cached_starts[index]),
                        int(cached_ends[index]),
                        cached_keys[0][index].chunk_hash,
                    )
                    for index in range(len(cached_starts))
                ]
                if sampled_worker_retrieve
                and kv_group == 0
                and isinstance(preflight_state, dict)
                else None
            )
            for start, end, key in token_results:
                assert isinstance(key, CacheEngineKey)
                if new_chunk_plan is not None:
                    new_chunk_plan.append((start, end, key.chunk_hash))

                keys_multi_layer = key.split_layers(self.num_layers)
                if sampled_worker_retrieve:
                    current_location = (
                        REMOTE_BACKEND_NAME
                        if (retrieve_kwargs or {}).get("direct_external_pages")
                        else "mixed"
                    )
                elif shared_rank0_retrieve:
                    locations_multi_layer: list[str] = []
                    for layer_key in keys_multi_layer:
                        layer_location = self._find_shared_rank0_chunk_location(
                            layer_key
                        )
                        if layer_location is None:
                            locations_multi_layer = []
                            break
                        locations_multi_layer.append(layer_location)
                    current_location = (
                        locations_multi_layer[0]
                        if len(set(locations_multi_layer)) == 1
                        else "mixed"
                        if locations_multi_layer
                        else None
                    )
                else:
                    current_location = self._layerwise_chunk_location(
                        keys_multi_layer
                    )
                if current_location:
                    if location is None:
                        location = current_location
                    elif shared_rank0_retrieve:
                        location = (
                            location
                            if location == current_location
                            else "mixed"
                        )
                    else:
                        assert location == current_location, (
                            "All retrieved keys should be from the same location "
                            "when use layerwise retrieval."
                            "Please support multi-location retrieval in the future."
                        )
                else:
                    break

                new_starts.append(start)
                new_ends.append(end)
                keys.append(keys_multi_layer)

            if new_chunk_plan is not None:
                preflight_state[_SHARED_CPU_CHUNK_PLAN_KEY] = tuple(new_chunk_plan)

            new_keys = (
                [list(row) for row in zip(*keys, strict=False)] if keys else []
            )
            if not new_keys:
                if metadata_extension is not None:
                    self._fill_retrieve_mask(
                        ret_mask, cached_starts, cached_ends
                    )
                    if retrieve_kwargs is not None:
                        retrieve_kwargs["_retrieve_metadata_warm"] = True
                        retrieve_kwargs["_retrieve_metadata_mode"] = (
                            "incremental"
                        )
                        retrieve_kwargs[
                            "_retrieve_metadata_generated_chunks"
                        ] = 0
                    return location, cached_starts, cached_ends, cached_keys
                return location, new_starts, new_ends, new_keys

            if not cached_keys:
                cached_keys[:] = new_keys
                cached_starts[:] = new_starts
                cached_ends[:] = new_ends
            else:
                prev_chunks = len(cached_starts)
                for relative_chunk_idx, (start, end) in enumerate(
                    zip(new_starts, new_ends, strict=False)
                ):
                    chunk_idx = chunk_index_base + relative_chunk_idx
                    if chunk_idx < prev_chunks:
                        if (
                            cached_starts[chunk_idx] == start
                            and cached_ends[chunk_idx] == end
                        ):
                            if any(
                                chunk_idx >= len(cached_keys[layer_id])
                                or cached_keys[layer_id][chunk_idx]
                                != new_keys[layer_id][relative_chunk_idx]
                                for layer_id in range(self.num_layers)
                            ):
                                raise ValueError(
                                    "Sparse retrieve cached prefix key mismatch "
                                    f"at chunk {chunk_idx}."
                                )
                            continue
                        raise ValueError(
                            "Sparse retrieve cached prefix range changed at "
                            f"chunk {chunk_idx}: cached="
                            f"[{cached_starts[chunk_idx]}, "
                            f"{cached_ends[chunk_idx]}), current=[{start}, {end})."
                        )
                    cached_starts.append(start)
                    cached_ends.append(end)
                    for layer_id in range(self.num_layers):
                        cached_keys[layer_id].append(
                            new_keys[layer_id][relative_chunk_idx]
                        )

            self._fill_retrieve_mask(ret_mask, cached_starts, cached_ends)
            if retrieve_kwargs is not None:
                if location is not None:
                    retrieve_kwargs["cached_retrieve_location"] = location
                retrieve_kwargs["_retrieve_metadata_warm"] = True
                retrieve_kwargs["_retrieve_metadata_mode"] = (
                    "incremental" if metadata_extension is not None else "full"
                )
                retrieve_kwargs["_retrieve_metadata_generated_chunks"] = len(
                    new_starts
                )
            return location, cached_starts, cached_ends, cached_keys

        if retrieve_kwargs is not None and retrieve_kwargs.get(
            "_retrieve_metadata_warm"
        ):
            if retrieve_kwargs.get("_use_cached_retrieve"):
                self._fill_retrieve_mask(
                    ret_mask,
                    cached_starts,
                    cached_ends,
                )
                location = self._resolve_local_cpu_retrieve_location(
                    retrieve_kwargs.get("cached_retrieve_location"),
                )
                if retrieve_kwargs is not None and location is not None:
                    retrieve_kwargs["cached_retrieve_location"] = location
                return location, cached_starts, cached_ends, cached_keys

            self._fill_retrieve_mask(ret_mask, cached_starts, cached_ends)

            location = retrieve_kwargs.get("cached_retrieve_location")
            kv_group = int(retrieve_kwargs.get("kv_group", 0) or 0)
            shared_passive_retrieve = (
                self._should_use_shared_layerwise_retrieve(kv_group)
                and self._is_shared_retrieve_passive(kv_group)
            )
            shared_rank0_retrieve = (
                self._should_use_shared_layerwise_retrieve(kv_group)
                and not self._is_shared_retrieve_passive(kv_group)
            )
            if (
                cached_keys
                and cached_keys[0]
                and not shared_rank0_retrieve
                and not shared_passive_retrieve
            ):
                # Prefer the hottest tier (LocalCPUBackend is checked first).
                # After Mooncake write-back, stale cached_retrieve_location may
                # still point at RemoteBackend and force slow remote reads.
                preferred_location = self._cached_layerwise_location_or_raise(
                    cached_keys,
                    cached_starts,
                    cached_ends,
                    retrieve_kwargs,
                )
                if preferred_location is not None:
                    location = preferred_location
                    retrieve_kwargs["cached_retrieve_location"] = location
            return location, cached_starts, cached_ends, cached_keys

        self._fill_retrieve_mask(ret_mask, cached_starts, cached_ends)

        location = None
        if retrieve_kwargs is not None:
            location = retrieve_kwargs.get("cached_retrieve_location")
        kv_group = int((retrieve_kwargs or {}).get("kv_group", 0) or 0)
        shared_passive_retrieve = (
            retrieve_kwargs is not None
            and self._should_use_shared_layerwise_retrieve(kv_group)
            and self._is_shared_retrieve_passive(kv_group)
        )
        shared_rank0_retrieve = (
            retrieve_kwargs is not None
            and self._should_use_shared_layerwise_retrieve(kv_group)
            and not self._is_shared_retrieve_passive(kv_group)
        )
        if (
            location is None
            and cached_keys
            and cached_keys[0]
            and not shared_rank0_retrieve
            and not shared_passive_retrieve
        ):
            location = self._cached_layerwise_location_or_raise(
                cached_keys,
                cached_starts,
                cached_ends,
                retrieve_kwargs,
            )
            if retrieve_kwargs is not None and location is not None:
                retrieve_kwargs["cached_retrieve_location"] = location
        if retrieve_kwargs is not None:
            retrieve_kwargs["_retrieve_metadata_warm"] = True
        return location, cached_starts, cached_ends, cached_keys

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[LayerwiseStoreResult], None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None for each layer and a
            LayerwiseStoreResult after the final layer. In the first iteration,
            the generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends and yields the completed
            store output.
        """
        store_result = LayerwiseStoreResult(
            request_id=str(kwargs.get("req_id", "unspecified")),
            kv_group=int(kwargs.get("kv_group", 0) or 0),
        )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            for layer_id in range(self.num_layers):
                yield
            # Extra yield consumed by wait_for_save() after the last layer.
            yield store_result
            return

        # Passive rank guard: when save_only_first_rank is enabled, only rank 0
        # stores. Closes the known TODO - store_layer had no _is_passive() check,
        # causing duplicate stores on non-rank-0 workers under MLA + layerwise.
        if self._is_passive():
            logger.debug(
                "Passive rank (save_only_first_rank), skipping store_layer"
            )
            for layer_id in range(self.num_layers):
                yield
            # Extra yield consumed by wait_for_save() after the last layer.
            yield store_result
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)
        store_result.request_id = req_id

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            yield store_result
            return

        cached_keys = store_result.keys
        cached_starts = store_result.starts
        cached_ends = store_result.ends
        cached_memory_objs = store_result.memory_objs
        cached_tensors = store_result.tensors
        cached_chunk_dev_ptrs = store_result.chunk_dev_ptrs
        cached_chunk_ptrs_npu = store_result.chunk_ptrs

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        requested_end = 0
        store_complete = True
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # Ensure the connector's MLA/DSA layout is detected before allocating
        # chunks -- get_shape(num_tokens) below depends on kv_lora_rank etc.
        self._ensure_layerwise_connector_layout(**kwargs)

        prev_key = 0
        kv_group = kwargs.get("kv_group", 0)
        kv_dtype = self._shared_cpu_dtype_for_kv_group(kv_group)
        page_store = bool(
            mooncake_layer_pages_enabled(self.config)
            and self.storage_manager.supports_batched_put_layer_pages(
                self.store_location
            )
        )
        page_shape = (
            self.gpu_connector.get_shape(
                int(self.config.chunk_size), kv_group=kv_group
            )
            if page_store
            else None
        )
        local = self._shared_local_cpu_backend() if page_store else None
        memory_format = self._memory_format_for_kv_group(kv_group)
        force_store_wait = self.config.get_extra_config_value(
            "force_store_wait", False
        )
        pending_chunks = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)
            requested_end = end

            keys_multi_layer = key.split_layers(self.num_layers)
            if self._layerwise_chunk_fully_stored(
                keys_multi_layer,
                req_id=req_id,
                kv_group=kv_group,
                start=start,
                end=end,
            ):
                continue

            # Allocate the memory object
            num_tokens = end - start
            if num_tokens <= 0:
                logger.warning(
                    "Skipping zero-token layerwise store chunk for req_id=%s "
                    "(kv_group=%s, start=%d, end=%d)",
                    req_id,
                    kv_group,
                    start,
                    end,
                )
                continue
            pending_chunks.append((start, end, key, keys_multi_layer))

        page_batch = None
        if pending_chunks and page_store:
            assert local is not None and page_shape is not None
            page_batch = local.batched_allocate_layer_pages(
                [page_shape],
                [kv_dtype],
                len(pending_chunks),
                self.num_layers,
                memory_format,
                busy_loop=False,
                valid_tokens=[end - start for start, end, _, _ in pending_chunks],
                full_tokens=int(self.config.chunk_size),
                eviction=False,
            )
            if page_batch is not None and len(page_batch) != len(pending_chunks):
                for page in page_batch:
                    page.ref_count_down()
                raise RuntimeError("Layer-page allocation returned wrong page count")

        legacy_suffix = False
        for chunk_index, (start, end, key, keys_multi_layer) in enumerate(
            pending_chunks
        ):
            num_tokens = end - start
            kv_shape_single_layer = (
                page_shape
                if page_batch is not None
                else self.gpu_connector.get_shape(num_tokens, kv_group=kv_group)
            )
            memory_objs_multi_layer = (
                [page_batch[chunk_index]] * self.num_layers
                if page_batch is not None
                else None
            )
            if memory_objs_multi_layer is None and page_store and not legacy_suffix:
                assert local is not None and page_shape is not None
                page = local.batched_allocate_layer_pages(
                    [page_shape],
                    [kv_dtype],
                    1,
                    self.num_layers,
                    memory_format,
                    busy_loop=force_store_wait,
                    valid_tokens=num_tokens,
                    full_tokens=int(self.config.chunk_size),
                )
                if page is not None:
                    if len(page) != 1:
                        for item in page:
                            item.ref_count_down()
                        raise RuntimeError(
                            "Layer-page allocation returned wrong page count"
                        )
                    memory_objs_multi_layer = [page[0]] * self.num_layers
                else:
                    legacy_suffix = True
            if memory_objs_multi_layer is None:
                memory_objs_multi_layer = self.storage_manager.batched_allocate(
                    kv_shape_single_layer,
                    kv_dtype,
                    batch_size=self.num_layers,
                    fmt=memory_format,
                    busy_loop=force_store_wait,
                )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                store_complete = False
                break

            if len(memory_objs_multi_layer) != self.num_layers:
                logger.error(
                    "Layerwise store allocation returned wrong layer count: "
                    "req_id=%s kv_group=%s chunk=[%d,%d) shape=%s dtype=%s "
                    "fmt=%s got=%d expected=%d decode_window=%s",
                    req_id,
                    kv_group,
                    start,
                    end,
                    kv_shape_single_layer,
                    kv_dtype,
                    memory_format,
                    len(memory_objs_multi_layer),
                    self.num_layers,
                    kwargs.get("decode_window_save"),
                )
                raise RuntimeError(
                    "Layerwise store allocation layer count mismatch: "
                    f"got {len(memory_objs_multi_layer)}, "
                    f"expected {self.num_layers}"
                )

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if _mtp_dw_diag_enabled() and kwargs.get("decode_window_save"):
            diag_slots = kwargs.get("slot_mapping")
            if isinstance(diag_slots, torch.Tensor):
                diag_slots = diag_slots.detach().cpu().reshape(-1)
            else:
                diag_slots = torch.empty(0, dtype=torch.long)
            window_start = kwargs.get("decode_window_start")
            window_end = kwargs.get("decode_window_end")
            requested_tokens = (
                int(window_end) - int(window_start)
                if window_start is not None and window_end is not None
                else int(num_to_store_tokens)
            )
            _mtp_dw_event(
                "store",
                req=req_id,
                event="begin",
                frontier=len(tokens),
                window_start=window_start,
                window_end=window_end,
                kv_group=kv_group,
                requested_tokens=requested_tokens,
                actual_tokens=tot_token_num,
                chunk_starts=starts[:8],
                chunk_ends=ends[:8],
                slot_count=int(diag_slots.numel()),
                slot_min=(int(diag_slots.min()) if diag_slots.numel() else None),
                slot_max=(int(diag_slots.max()) if diag_slots.numel() else None),
                slot_sample=diag_slots[:8].tolist(),
            )
            if tot_token_num != requested_tokens:
                _mtp_dw_event(
                    "fail",
                    req=req_id,
                    frontier=len(tokens),
                    window_start=window_start,
                    window_end=window_end,
                    kv_group=kv_group,
                    invariant="store_actual_tokens",
                    requested_tokens=requested_tokens,
                    actual_tokens=tot_token_num,
                )

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]
            if len(memory_objs) != self.num_layers or len(keys) != self.num_layers:
                logger.error(
                    "Layerwise store transpose produced wrong layer count: "
                    "req_id=%s kv_group=%s memory_layers=%d key_layers=%d "
                    "expected=%d chunk_count=%d starts=%s ends=%s "
                    "decode_window=%s",
                    req_id,
                    kv_group,
                    len(memory_objs),
                    len(keys),
                    self.num_layers,
                    len(starts),
                    starts,
                    ends,
                    kwargs.get("decode_window_save"),
                )
                raise RuntimeError(
                    "Layerwise store transpose layer count mismatch: "
                    f"memory_layers={len(memory_objs)}, key_layers={len(keys)}, "
                    f"expected={self.num_layers}"
                )
            pending_store_release = {
                id(mem_obj): mem_obj
                for layer_objs in memory_objs
                for mem_obj in layer_objs
            }
            mem_obj_generator = None

            # Calculate total KV size for logging
            tot_kv_size = sum(
                obj.get_size() for obj in pending_store_release.values()
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            cache_chunk_indices: Optional[List[int]] = None
            if bool(kwargs.get("windowed_sparse_save", False)):
                # Partial tails are valid storage entries, but sparse direct
                # pointer tables assume fixed-size LMCache chunks. Keep tails
                # out of request-local pointer/cache publication until a later
                # full successor for the same aligned chunk is committed.
                chunk_size = int(self.config.chunk_size)
                cache_chunk_indices = [
                    index
                    for index, (start, end) in enumerate(
                        zip(starts, ends, strict=False)
                    )
                    if start % chunk_size == 0 and end - start == chunk_size
                ]

                selected_starts = [starts[index] for index in cache_chunk_indices]
                selected_ends = [ends[index] for index in cache_chunk_indices]
                self._truncate_store_cache_for_full_chunk_successor(
                    starts=cached_starts,
                    ends=cached_ends,
                    new_starts=selected_starts,
                    new_ends=selected_ends,
                    cached_keys=cached_keys,
                    cached_memory_objs=cached_memory_objs,
                    cached_tensors=cached_tensors,
                    cached_chunk_dev_ptrs=cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
                )

            self._append_layerwise_store_cache_chunks(
                keys=keys,
                starts=starts,
                ends=ends,
                memory_objs=memory_objs,
                cached_keys=cached_keys,
                cached_starts=cached_starts,
                cached_ends=cached_ends,
                cached_memory_objs=cached_memory_objs,
                cached_tensors=cached_tensors,
                cache_chunk_indices=cache_chunk_indices,
            )

            try:
                t_start = time.perf_counter()
                page_first_store = mooncake_page_layout_enabled(self.config)
                group_store = getattr(
                    self.gpu_connector, "batched_from_gpu_group", None
                )
                supports_group_store = getattr(
                    self.gpu_connector, "supports_batched_from_gpu_group", None
                )
                all_chunks_publishable = cache_chunk_indices is None or (
                    cache_chunk_indices == list(range(len(starts)))
                )
                use_group_store = (
                    bool(
                        kwargs.get("decode_window_save")
                        or kwargs.get("all_layers_ready")
                    )
                    and callable(group_store)
                    and callable(supports_group_store)
                    and supports_group_store(kv_group)
                    and all_chunks_publishable
                )
                if use_group_store:
                    for _ in range(self.num_layers):
                        yield
                    group_started = (
                        cold_start_perf_now()
                        if cold_start_perf_enabled()
                        else None
                    )
                    host_pointer_rows, layer_chunk_ptrs_npu = group_store(
                        memory_objs, starts, ends, **kwargs
                    )
                    if group_started is not None:
                        cold_start_perf_log(
                            logger,
                            "npu_group_submit_cpu",
                            started=group_started,
                            req_id=req_id,
                            kv_group=kv_group,
                            chunks=len(starts),
                            tokens=sum(
                                end - start
                                for start, end in zip(starts, ends, strict=True)
                            ),
                        )
                    self._append_group_store_tensors(
                        memory_objs,
                        cached_tensors,
                        cached_chunk_dev_ptrs,
                        cached_chunk_ptrs_npu,
                        host_pointer_rows,
                        layer_chunk_ptrs_npu,
                    )
                    if not page_first_store:
                        for layer_id in range(self.num_layers):
                            required_futures = self.storage_manager.batched_put(
                                keys[layer_id],
                                memory_objs[layer_id],
                                location=self.store_location,
                            )
                            self._track_sync_store_futures(
                                required_futures, require_completion=True
                            )
                            for mem_obj in memory_objs[layer_id]:
                                pending_store_release.pop(id(mem_obj), None)
                else:
                    mem_obj_generator = self.gpu_connector.batched_from_gpu(
                        memory_objs, starts, ends, **kwargs
                    )
                    next(mem_obj_generator)
                    for layer_id in range(self.num_layers):
                        yield
                        next(mem_obj_generator)
                        self._append_layer_store_tensors(
                            layer_id,
                            memory_objs,
                            cached_tensors,
                            cached_chunk_dev_ptrs,
                            cached_chunk_ptrs_npu,
                            cache_chunk_indices,
                        )
                        if page_first_store:
                            continue
                        required_futures = self.storage_manager.batched_put(
                            keys[layer_id],
                            memory_objs[layer_id],
                            location=self.store_location,
                        )
                        self._track_sync_store_futures(
                            required_futures, require_completion=True
                        )
                        for mem_obj in memory_objs[layer_id]:
                            pending_store_release.pop(id(mem_obj), None)

                if page_first_store:
                    if page_store:
                        page_indices = [
                            index
                            for index, obj in enumerate(memory_objs[0])
                            if isinstance(obj, LayerPageMemoryObj)
                        ]
                        legacy_indices = [
                            index
                            for index, obj in enumerate(memory_objs[0])
                            if not isinstance(obj, LayerPageMemoryObj)
                        ]
                        required_futures = []
                        submitted_objs = []
                        if page_indices:
                            layer_pages = [
                                memory_objs[0][index] for index in page_indices
                            ]
                            required_futures.extend(
                                self.storage_manager.batched_put_layer_pages(
                                    [
                                        keys[0][index].without_layer()
                                        for index in page_indices
                                    ],
                                    layer_pages,
                                    location=self.store_location,
                                    req_id=req_id,
                                )
                            )
                            submitted_objs.extend(layer_pages)
                            for page in layer_pages:
                                pending_store_release.pop(id(page), None)
                        if legacy_indices:
                            legacy_keys = [
                                layer_keys[index]
                                for layer_keys in keys
                                for index in legacy_indices
                            ]
                            legacy_objs = [
                                layer_objs[index]
                                for layer_objs in memory_objs
                                for index in legacy_indices
                            ]
                            required_futures.extend(
                                self.storage_manager.batched_put(
                                    legacy_keys,
                                    legacy_objs,
                                    location=self.store_location,
                                )
                            )
                            submitted_objs.extend(legacy_objs)
                    else:
                        flattened_keys = [
                            key for layer_keys in keys for key in layer_keys
                        ]
                        submitted_objs = [
                            obj for layer_objs in memory_objs for obj in layer_objs
                        ]
                        required_futures = self.storage_manager.batched_put(
                            flattened_keys,
                            submitted_objs,
                            location=self.store_location,
                        )
                    self._track_sync_store_futures(
                        required_futures, require_completion=True
                    )
                    for mem_obj in submitted_objs:
                        pending_store_release.pop(id(mem_obj), None)

                tot_time = time.perf_counter() - t_start
                logger.info(
                    "[req_id=%s kv_group=%s] Stored %d out of total %d tokens. "
                    "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                    req_id,
                    kv_group,
                    tot_token_num,
                    len(tokens),
                    tot_kv_size / 1024**3,
                    tot_time * 1000,
                    tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
                )
            finally:
                if mem_obj_generator is not None:
                    close_fn = getattr(mem_obj_generator, "close", None)
                    if close_fn is not None:
                        try:
                            close_fn()
                        except (GeneratorExit, RuntimeError, ValueError):
                            pass
                self._remove_pending_layerwise_store_objs(
                    cached_memory_objs,
                    pending_store_release,
                )
                for mem_obj in list(pending_store_release.values()):
                    if mem_obj.is_valid():
                        mem_obj.ref_count_down()
                pending_store_release.clear()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        if store_complete:
            store_result.committed_end = requested_end
        if _mtp_dw_diag_enabled() and kwargs.get("decode_window_save"):
            window_start = kwargs.get("decode_window_start")
            window_end = kwargs.get("decode_window_end")
            requested_tokens = (
                int(window_end) - int(window_start)
                if window_start is not None and window_end is not None
                else int(num_to_store_tokens)
            )
            _mtp_dw_event(
                "store",
                req=req_id,
                event="complete",
                frontier=len(tokens),
                window_start=window_start,
                window_end=window_end,
                kv_group=kv_group,
                requested_tokens=requested_tokens,
                actual_tokens=tot_token_num,
            )
        yield store_result

    def _retrieve_layer_head_token_wise_shared_passive(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor],
        ret_mask: torch.Tensor,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        assert self.gpu_connector is not None, (
            "gpu_connector is required for sparse shared retrieve"
        )
        if self.shared_cpu_cache_passive_allocator is None:
            raise ValueError(
                "Shared CPU cache passive allocator is not initialized for "
                "sparse decode. Startup preflight did not complete."
            )

        request_configs = kwargs.get("request_configs")
        kv_group = kwargs.get("kv_group", 0)
        req_id = kwargs.get("req_id", "unspecified")
        phase = kwargs.get("shared_cpu_phase", "sparse_decode_bootstrap")
        perf_enabled = cold_start_perf_enabled()
        request_ordinal = int(kwargs.get("shared_cpu_request_ordinal", 0))
        cached_keys = kwargs.get("cached_keys")
        cached_starts = kwargs.get("cached_starts")
        cached_ends = kwargs.get("cached_ends")
        cached_memory_objs = kwargs.get("cached_memory_objs")
        cached_tensors = kwargs.get("cached_tensors")
        cached_chunk_dev_ptrs = kwargs.get("cached_chunk_dev_ptrs")
        cached_chunk_ptrs_npu = kwargs.get("cached_chunk_ptrs_npu")
        cached_shared_handles = kwargs.get("cached_shared_handles")
        append = kwargs.get("_sparse_cache_append") or _SparseCacheAppend(kwargs)

        metadata_started = cold_start_perf_now() if perf_enabled else 0.0
        metadata_warm = bool(
            kwargs.get("_retrieve_metadata_warm")
            and cached_keys
            and cached_starts
            and cached_ends
            and int(cached_ends[-1]) == len(tokens)
        )
        if metadata_warm:
            starts = list(cached_starts)
            ends = list(cached_ends)
            keys_layer_major = cached_keys
        else:
            extension = self._build_retrieve_metadata_extension(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
                kv_group=kv_group,
                cached_keys=cached_keys,
                cached_starts=cached_starts,
                cached_ends=cached_ends,
                cached_metadata_token_ids=kwargs.get(
                    "cached_metadata_token_ids"
                ),
            )
            if extension is not None:
                suffix_starts, suffix_ends, suffix_keys = extension
                starts = list(cached_starts) + suffix_starts
                ends = list(cached_ends) + suffix_ends
                keys_layer_major = [
                    list(cached_keys[layer_id]) + suffix_keys[layer_id]
                    for layer_id in range(self.num_layers)
                ]
            else:
                starts, ends, keys = [], [], []
                for start, end, key in self.token_database.process_tokens(
                    tokens=tokens,
                    mask=mask,
                    request_configs=request_configs,
                    kv_group=kv_group,
                ):
                    assert isinstance(key, CacheEngineKey)
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.split_layers(self.num_layers))
                keys_layer_major = (
                    [list(row) for row in zip(*keys, strict=False)]
                    if keys
                    else []
                )
        required_chunks = len(starts)
        if perf_enabled:
            cold_start_perf_log(
                logger,
                "metadata_prepare",
                started=metadata_started,
                req_id=req_id,
                phase=phase,
                kv_group=kv_group,
                layers=self.num_layers,
                chunks=required_chunks,
                rank=self.metadata.worker_id,
                passive=True,
            )
        cached_prefix_chunks = self._cached_sparse_prefix_chunks(
            cached_tensors,
            cached_memory_objs,
        )
        if cached_prefix_chunks > required_chunks:
            raise ValueError(
                "Sparse passive cached prefix exceeds current metadata: "
                f"cached_chunks={cached_prefix_chunks}, "
                f"required_chunks={required_chunks}"
            )
        metadata_counts = {
            len(values)
            for values in (cached_starts, cached_ends)
            if values is not None
        }
        if cached_keys is not None:
            if cached_keys and len(cached_keys) != self.num_layers:
                raise ValueError("Sparse passive cached key metadata is incomplete.")
            metadata_counts.update(len(layer) for layer in cached_keys)
        if len(metadata_counts) > 1:
            raise ValueError("Sparse passive cached metadata has inconsistent lengths.")
        cached_metadata_chunks = metadata_counts.pop() if metadata_counts else 0
        if not cached_keys:
            cached_metadata_chunks = 0
        if not cached_starts or not cached_ends:
            cached_metadata_chunks = 0
        if not cached_prefix_chunks <= cached_metadata_chunks <= required_chunks:
            raise ValueError(
                "Sparse passive cached metadata does not cover the materialized "
                f"prefix: metadata={cached_metadata_chunks}, "
                f"cached={cached_prefix_chunks}, required={required_chunks}"
            )
        if not metadata_warm:
            for chunk_index in range(cached_metadata_chunks):
                if (
                    cached_starts[chunk_index] != starts[chunk_index]
                    or cached_ends[chunk_index] != ends[chunk_index]
                    or any(
                        cached_keys[layer_id][chunk_index]
                        != keys_layer_major[layer_id][chunk_index]
                        for layer_id in range(self.num_layers)
                    )
                ):
                    raise ValueError(
                        "Sparse passive cached prefix changed at chunk "
                        f"{chunk_index}."
                    )
        cached_handle_chunks = self._uniform_layer_cache_chunks(
            cached_shared_handles,
            self.num_layers,
            "passive shared handle cache",
        )
        if cached_handle_chunks != cached_prefix_chunks:
            raise ValueError(
                "Sparse passive cached handles do not cover the materialized "
                f"prefix: handles={cached_handle_chunks}, "
                f"cached_chunks={cached_prefix_chunks}"
            )
        missing_chunks = required_chunks - cached_prefix_chunks
        self._fill_retrieve_mask(
            ret_mask,
            starts[:cached_prefix_chunks],
            ends[:cached_prefix_chunks],
        )

        assert_layerwise_gpu_connector(self.gpu_connector)
        transfer_kwargs = kwargs
        transient_chunk_dev_ptrs = None
        transient_chunk_ptrs_npu = None
        if not cached_prefix_chunks and (
            cached_chunk_dev_ptrs is not None
            or cached_chunk_ptrs_npu is not None
        ):
            # The consumer needs transient pointers for the transfer, while
            # the cache engine publishes owners/host/NPU rows atomically.
            transfer_kwargs = dict(kwargs)
            if (
                cached_chunk_dev_ptrs is not None
                and cached_chunk_ptrs_npu is not None
            ):
                transient_chunk_dev_ptrs = []
                transient_chunk_ptrs_npu = []
            transfer_kwargs["cached_chunk_dev_ptrs"] = transient_chunk_dev_ptrs
            transfer_kwargs["cached_chunk_ptrs_npu"] = transient_chunk_ptrs_npu
        mem_obj_consumer = self.gpu_connector.batched_to_gpu_head_token_wise(
            **transfer_kwargs
        )
        next(mem_obj_consumer)

        to_release: list[MemoryObj] = []
        passive_views_handed_off = False
        metadata_appended = False
        compact_batch: Optional[SharedHandleBatch] = None
        compact_final_receive_attempted = False
        materialize_only = bool(kwargs.get("materialize_only", False))
        pending_materialized_layers: List[
            Union[List[MemoryObj], LayerPageSource]
        ] = []
        compact_pages = ()
        page_view_build_s = 0.0
        pointer_seal_s = 0.0
        legacy_tail_objects = 0
        prepared_compact_ptrs = False
        defer_finish = False

        try:
            for layer_id in range(self.num_layers):
                sparse_request = yield ret_mask
                (
                    sparse_payload,
                    selected_tokens,
                    token_start_index,
                    target_slot_mapping,
                    prepare_only,
                    defer_finish,
                ) = _shared_sparse_request(sparse_request)

                if not missing_chunks:
                    mem_objs_layer = []
                    source = mem_objs_layer
                else:
                    envelope = None
                    if compact_batch is None:
                        envelope = self._receive_shared_envelope()
                        self._validate_shared_layerwise_envelope(
                            envelope,
                            req_id=req_id,
                            phase=phase,
                            request_ordinal=request_ordinal,
                            layer_id=layer_id,
                            kv_group=kv_group,
                        )
                        if envelope.status != "ok":
                            raise ValueError(
                                "Sparse passive prefix extension requires present "
                                f"handles, received status={envelope.status!r}."
                            )
                        compact_batch = envelope.batch
                        if compact_batch is not None:
                            page_started = (
                                cold_start_perf_now() if perf_enabled else 0.0
                            )
                            compact_pages = self._make_passive_layer_page_views(
                                compact_batch,
                                starts=starts[cached_prefix_chunks:],
                                ends=ends[cached_prefix_chunks:],
                                keys_layer_major=[
                                    keys[cached_prefix_chunks:]
                                    for keys in keys_layer_major
                                ],
                                kv_group=kv_group,
                            )
                            if page_started:
                                page_view_build_s = (
                                    cold_start_perf_now() - page_started
                                )
                            to_release.extend(compact_pages)
                            prepare_page_ptrs = getattr(
                                self.gpu_connector,
                                "prepare_sparse_page_ptr_cache_for_layers",
                                None,
                            )
                            if (
                                len(compact_pages) == missing_chunks
                                and transient_chunk_dev_ptrs is not None
                                and transient_chunk_ptrs_npu is not None
                                and callable(prepare_page_ptrs)
                            ):
                                pointer_started = (
                                    cold_start_perf_now() if perf_enabled else 0.0
                                )
                                prepared_compact_ptrs = bool(
                                    prepare_page_ptrs(
                                        [
                                            LayerPageSource(compact_pages, index)
                                            for index in range(self.num_layers)
                                        ],
                                        transient_chunk_dev_ptrs,
                                        transient_chunk_ptrs_npu,
                                    )
                                )
                                if pointer_started:
                                    pointer_seal_s = (
                                        cold_start_perf_now() - pointer_started
                                    )
                        elif len(envelope.handles) != missing_chunks:
                            raise ValueError(
                                "Sparse shared CPU passive received inconsistent "
                                f"handle count at layer {layer_id}: "
                                f"{len(envelope.handles)} != {missing_chunks}"
                            )
                    prepare_started = (
                        cold_start_perf_now()
                        if perf_enabled
                        else 0.0
                    )
                    if not metadata_appended:
                        metadata_appended = True
                        self._fill_retrieve_mask(
                            ret_mask,
                            starts[cached_prefix_chunks:],
                            ends[cached_prefix_chunks:],
                            clear=False,
                        )
                        if cached_starts is not None:
                            cached_starts.extend(starts[cached_metadata_chunks:])
                        if cached_ends is not None:
                            cached_ends.extend(ends[cached_metadata_chunks:])
                        if cached_keys is not None:
                            if not cached_keys:
                                cached_keys.extend(
                                    [] for _ in range(self.num_layers)
                                )
                            for cache_layer_id in range(self.num_layers):
                                cached_keys[cache_layer_id].extend(
                                    keys_layer_major[cache_layer_id][
                                        cached_metadata_chunks:
                                    ]
                                )
                    mem_objs_layer: list[MemoryObj] = []
                    handles = (
                        [None] * missing_chunks
                        if compact_batch is not None
                        else envelope.handles
                    )
                    try:
                        for chunk_offset, handle in enumerate(
                            handles[len(compact_pages) :], len(compact_pages)
                        ):
                            chunk_index = cached_prefix_chunks + chunk_offset
                            expected_shape, expected_dtype, expected_fmt = (
                                self._expected_shared_cpu_chunk_metadata(
                                    kv_group=kv_group,
                                    num_tokens=int(
                                        ends[chunk_index] - starts[chunk_index]
                                    ),
                                )
                            )
                            if compact_batch is not None:
                                passive_allocator = (
                                    self.shared_cpu_cache_passive_allocator
                                )
                                mem_obj = passive_allocator.create_batch_view(
                                    compact_batch,
                                    layer_id=layer_id,
                                    chunk_index=chunk_offset,
                                    shape=expected_shape,
                                    dtype=expected_dtype,
                                    fmt=expected_fmt,
                                    cached_positions=range(
                                        int(starts[chunk_index]),
                                        int(ends[chunk_index]),
                                    ),
                                )
                            else:
                                mem_obj = (
                                    self.shared_cpu_cache_passive_allocator.create_view(
                                        handle,
                                        expected_request_id=req_id,
                                        expected_phase=phase,
                                        expected_layer_id=layer_id,
                                        expected_kv_group=kv_group,
                                        expected_chunk_index=chunk_index,
                                        expected_key=(
                                            keys_layer_major[layer_id][chunk_index]
                                        ),
                                        expected_shape=expected_shape,
                                        expected_dtype=expected_dtype,
                                        expected_fmt=expected_fmt,
                                        expected_cached_positions=range(
                                            int(starts[chunk_index]),
                                            int(ends[chunk_index]),
                                        ),
                                        expected_producer_rank=(
                                            self.metadata.first_rank
                                        ),
                                    )
                                )
                            mem_objs_layer.append(mem_obj)
                    except Exception:
                        for mem_obj in reversed(mem_objs_layer):
                            if mem_obj.is_valid():
                                mem_obj.ref_count_down()
                        raise
                    try:
                        source = (
                            LayerPageSource(
                                compact_pages,
                                layer_id,
                                tuple(mem_objs_layer),
                            )
                            if compact_pages
                            else mem_objs_layer
                        )
                        if materialize_only or compact_pages:
                            pending_materialized_layers.append(source)
                        else:
                            self._append_retrieve_layer_cache(
                                layer_id,
                                mem_objs_layer,
                                cached_memory_objs,
                                cached_tensors,
                                cached_chunk_dev_ptrs,
                                cached_chunk_ptrs_npu,
                            )
                    except Exception:
                        for mem_obj in mem_objs_layer:
                            if mem_obj.is_valid():
                                mem_obj.ref_count_down()
                        raise
                    to_release.extend(mem_objs_layer)
                    if compact_pages:
                        legacy_tail_objects += len(mem_objs_layer)
                    self._append_shared_handle_cache(
                        layer_id,
                        handles,
                        cached_shared_handles,
                        cached_prefix_chunks,
                    )
                    if perf_enabled:
                        cold_start_perf_log(
                            logger,
                            "passive_layer_prepare",
                            started=prepare_started,
                            req_id=req_id,
                            phase=phase,
                            kv_group=kv_group,
                            layer=layer_id,
                            objects=len(compact_pages) + len(mem_objs_layer),
                            rank=self.metadata.worker_id,
                        )

                if prepare_only:
                    sparse_request = yield ret_mask
                    (
                        sparse_payload,
                        selected_tokens,
                        token_start_index,
                        target_slot_mapping,
                        prepare_only,
                        defer_finish,
                    ) = _shared_sparse_request(sparse_request)
                    if prepare_only:
                        raise ValueError("Duplicate shared sparse prepare request")

                dispatch_started = (
                    cold_start_perf_now()
                    if perf_enabled
                    else 0.0
                )
                if sparse_payload is not None:
                    sparse_payload["memory_objs_layer"] = source
                    mem_obj_consumer.send(sparse_payload)
                else:
                    mem_obj_consumer.send(
                        (
                            source,
                            selected_tokens,
                            token_start_index,
                            target_slot_mapping,
                        )
                    )
                if perf_enabled:
                    source_chunks = len(compact_pages) + len(mem_objs_layer)
                    if (
                        not source_chunks
                        and cached_memory_objs is not None
                        and layer_id < len(cached_memory_objs)
                    ):
                        cached_layer = cached_memory_objs[layer_id]
                        source_chunks = (
                            len(cached_layer) if cached_layer is not None else 0
                        )
                    cold_start_perf_log(
                        logger,
                        "npu_layer_submit_cpu",
                        started=dispatch_started,
                        req_id=req_id,
                        phase=phase,
                        kv_group=kv_group,
                        layer=layer_id,
                        objects=len(compact_pages) + len(mem_objs_layer),
                        source_chunks=source_chunks,
                        rank=self.metadata.worker_id,
                        passive=True,
                    )
            next(mem_obj_consumer)
            if defer_finish:
                yield ret_mask
            if compact_batch is not None:
                compact_final_receive_attempted = True
                final_envelope = self._receive_shared_envelope()
                self._validate_shared_layerwise_envelope(
                    final_envelope,
                    req_id=req_id,
                    phase=phase,
                    request_ordinal=request_ordinal,
                    layer_id=self.num_layers,
                    kv_group=kv_group,
                )
                if (
                    final_envelope.status != "skipped"
                    or final_envelope.message
                    != _SHARED_CPU_COMPACT_COMMIT_MESSAGE
                ):
                    raise ValueError(
                        "Compact shared batch received an invalid final status: "
                        f"status={final_envelope.status!r}, "
                        f"message={final_envelope.message!r}"
                    )
            if pending_materialized_layers:
                pointer_started = (
                    cold_start_perf_now()
                    if perf_enabled and compact_pages
                    else 0.0
                )
                self._append_retrieve_group_cache(
                    pending_materialized_layers,
                    cached_memory_objs,
                    cached_tensors,
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                    transient_chunk_dev_ptrs if prepared_compact_ptrs else None,
                    transient_chunk_ptrs_npu if prepared_compact_ptrs else None,
                )
                if pointer_started:
                    pointer_seal_ms = round(
                        (
                            pointer_seal_s
                            + cold_start_perf_now()
                            - pointer_started
                        )
                        * 1000,
                        3,
                    )
                    cold_start_perf_log(
                        logger,
                        "passive_compact_materialize",
                        started=pointer_started,
                        req_id=req_id,
                        phase=phase,
                        kv_group=kv_group,
                        rank=self.metadata.worker_id,
                        physical_pages=len(compact_pages),
                        logical_entries=missing_chunks * self.num_layers,
                        legacy_tail_objects=legacy_tail_objects,
                        page_view_build_ms=round(page_view_build_s * 1000, 3),
                        pointer_seal_ms=pointer_seal_ms,
                    )
            passive_views_handed_off = (
                cached_memory_objs is not None and cached_keys is not None
            )
            append.commit()
            yield ret_mask
        finally:
            if compact_batch is not None and not compact_final_receive_attempted:
                compact_final_receive_attempted = True
                try:
                    self._receive_shared_envelope()
                except BaseException:
                    logger.exception(
                        "Failed to drain compact shared batch final sentinel "
                        "during passive cleanup"
                    )
            if not passive_views_handed_off:
                for mem_obj in to_release:
                    if mem_obj.is_valid():
                        mem_obj.ref_count_down()
            append.rollback()

    def retrieve_layer_head_token_wise(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Retrieve sparse KV layers through the existing generator protocol.

        A sealed ``prepared_sparse_source`` selects the metadata-free warm path.
        Requests without one retain the full storage/shared-cache bootstrap path.
        """
        if kwargs.get("prepared_sparse_source") is not None:
            yield from self._retrieve_prepared_sparse_layers(
                tokens,
                kwargs,
            )
            return
        yield from self._retrieve_layer_head_token_wise_bootstrap(
            tokens,
            mask,
            **kwargs,
        )

    def _retrieve_prepared_sparse_layers(
        self,
        tokens: Union[torch.Tensor, list[int]],
        retrieve_kwargs: dict[str, Any],
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Forward already-resolved source layers without cache-engine work."""
        prepared_source: PreparedSparseSource = retrieve_kwargs[
            "prepared_sparse_source"
        ]
        token_count = int(prepared_source.total_tokens)
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping sparse retrieve")
            yield torch.zeros(token_count, dtype=torch.bool)
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        ret_mask = retrieve_kwargs.get("ret_mask")
        if ret_mask is None:
            ret_mask = torch.zeros(token_count, dtype=torch.bool, device="cpu")

        consumer = self.gpu_connector.batched_to_gpu_head_token_wise(**retrieve_kwargs)
        next(consumer)
        try:
            for _ in prepared_source.layers:
                sparse_request = yield ret_mask
                consumer.send(sparse_request)
            next(consumer)
            yield ret_mask
        finally:
            consumer.close()

    def _retrieve_layer_head_token_wise_bootstrap(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        append = _SparseCacheAppend(kwargs)
        kwargs["_sparse_cache_append"] = append
        try:
            yield from self._retrieve_layer_head_token_wise_bootstrap_impl(
                tokens,
                mask,
                **kwargs,
            )
        finally:
            append.rollback()

    def _retrieve_layer_head_token_wise_bootstrap_impl(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask will be all True for
            sparse attention with chunk size 1

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        kv_group = kwargs.get("kv_group", 0)
        perf_enabled = cold_start_perf_enabled()
        kwargs.setdefault("shared_cpu_phase", "sparse_decode_bootstrap")
        direct_external_pages = bool(
            kv_group == 1 and kwargs.get("direct_external_pages", False)
        )
        shared_sparse_retrieve, shared_retrieve_passive = (
            self._cold_retrieve_collective_mode(kv_group, direct_external_pages)
        )
        if not (shared_sparse_retrieve and shared_retrieve_passive):
            assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )
        if shared_sparse_retrieve:
            self._ensure_layerwise_connector_layout(**kwargs)

        mem_obj_consumer = None
        num_tokens = len(tokens)
        metadata_warm = kwargs.get("_retrieve_metadata_warm", False)
        ret_mask = kwargs.get("ret_mask")
        if (
            ret_mask is None
            or ret_mask.numel() != num_tokens
            or ret_mask.device.type != "cpu"
        ):
            ret_mask = torch.zeros(num_tokens, dtype=torch.bool, device="cpu")
            kwargs["ret_mask"] = ret_mask
        elif not metadata_warm:
            ret_mask.zero_()

        initial_cached_memory_objs = kwargs.get("cached_memory_objs")
        initial_cached_tensors = kwargs.get("cached_tensors")
        has_shared_cached_retrieve = (
            shared_sparse_retrieve
            and self._has_retrieve_data_cache(
                initial_cached_tensors,
                initial_cached_memory_objs,
                self.num_layers,
            )
        )
        if has_shared_cached_retrieve:
            kwargs["_use_cached_retrieve"] = True

        if shared_sparse_retrieve and shared_retrieve_passive:
            passive_kwargs = dict(kwargs)
            passive_kwargs.pop("ret_mask", None)
            yield from self._retrieve_layer_head_token_wise_shared_passive(
                tokens,
                mask,
                ret_mask,
                **passive_kwargs,
            )
            return

        request_configs = kwargs.get("request_configs")

        cached_keys = kwargs.get("cached_keys")
        assert cached_keys is not None
        assert isinstance(cached_keys, list)

        cached_starts = kwargs.get("cached_starts")
        assert cached_starts is not None
        assert isinstance(cached_starts, list)

        cached_ends = kwargs.get("cached_ends")
        assert cached_ends is not None
        assert isinstance(cached_ends, list)

        cached_memory_objs = kwargs.get("cached_memory_objs")
        cached_tensors = kwargs.get("cached_tensors")
        cached_chunk_dev_ptrs = kwargs.get("cached_chunk_dev_ptrs")
        cached_chunk_ptrs_npu = kwargs.get("cached_chunk_ptrs_npu")
        cached_shared_handles = kwargs.get("cached_shared_handles")
        append: _SparseCacheAppend = kwargs["_sparse_cache_append"]
        cached_prefix_chunks = self._cached_sparse_prefix_chunks(
            cached_tensors,
            cached_memory_objs,
        )

        metadata_started = cold_start_perf_now() if perf_enabled else 0.0
        location, starts, ends, retrieve_keys = self._ensure_retrieve_chunk_metadata(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=kwargs,
        )
        kwargs.pop("_use_cached_retrieve", None)
        required_chunks = len(retrieve_keys[0]) if retrieve_keys else 0
        if perf_enabled:
            cold_start_perf_log(
                logger,
                "metadata_prepare",
                started=metadata_started,
                req_id=kwargs.get("req_id", "unspecified"),
                phase=kwargs.get(
                    "shared_cpu_phase",
                    "sparse_decode_bootstrap",
                ),
                kv_group=kv_group,
                layers=len(retrieve_keys),
                chunks=required_chunks,
                rank=self.metadata.worker_id,
                passive=False,
            )
        if cached_prefix_chunks > required_chunks:
            raise ValueError(
                "Sparse retrieve cached prefix exceeds current metadata: "
                f"cached_chunks={cached_prefix_chunks}, "
                f"required_chunks={required_chunks}"
            )
        cached_tensors_cover = self._retrieve_data_cache_covers(
            cached_tensors,
            self.num_layers,
            required_chunks,
        )
        cached_memory_objs_cover = self._retrieve_data_cache_covers(
            cached_memory_objs,
            self.num_layers,
            required_chunks,
        )
        use_cached_retrieve = cached_tensors_cover or cached_memory_objs_cover
        missing_keys = [
            layer_keys[cached_prefix_chunks:]
            for layer_keys in retrieve_keys
        ]
        direct_group1_pages = bool(
            direct_external_pages
            and not use_cached_retrieve
            and cached_prefix_chunks == 0
            and mooncake_layer_pages_enabled(self.config)
            and missing_keys
            and missing_keys[0]
            and all(len(layer) == len(missing_keys[0]) for layer in missing_keys)
        )
        direct_group1_load = False
        if direct_group1_pages:
            direct_group1_load = self._try_direct_indexer_page_load(
                req_id=kwargs.get("req_id", "unspecified"),
                keys_layer_major=missing_keys,
                starts=starts,
                ends=ends,
                retrieve_kwargs=kwargs,
            )

        cached_handle_chunks = 0
        if shared_sparse_retrieve:
            cached_handle_chunks = self._uniform_layer_cache_chunks(
                cached_shared_handles,
                self.num_layers,
                "shared handle cache",
            )
            if cached_handle_chunks > cached_prefix_chunks:
                raise ValueError(
                    "Sparse shared handle cache exceeds the materialized prefix: "
                    f"handles={cached_handle_chunks}, "
                    f"cached_chunks={cached_prefix_chunks}"
                )
            if (
                cached_prefix_chunks < required_chunks
                and cached_handle_chunks != cached_prefix_chunks
            ):
                raise ValueError(
                    "Sparse shared prefix extension requires complete cached "
                    f"handles: handles={cached_handle_chunks}, "
                    f"cached_chunks={cached_prefix_chunks}"
                )
        publish_shared_handles = (
            shared_sparse_retrieve
            and not shared_retrieve_passive
            and cached_handle_chunks < required_chunks
        )

        if use_cached_retrieve:
            location = self._resolve_local_cpu_retrieve_location(location)
            if location is not None:
                kwargs["cached_retrieve_location"] = location

        if not retrieve_keys:
            retrieve_keys = [[] for _ in range(self.num_layers)]
        elif len(retrieve_keys) < self.num_layers:
            retrieve_keys.extend(
                [] for _ in range(self.num_layers - len(retrieve_keys))
            )

        assert_layerwise_gpu_connector(self.gpu_connector)

        get_generator = None

        if cached_tensors_cover and cached_tensors is not None:
            kwargs["cached_tensors"] = cached_tensors
            if cached_chunk_dev_ptrs is not None:
                kwargs["cached_chunk_dev_ptrs"] = cached_chunk_dev_ptrs
            if cached_chunk_ptrs_npu is not None:
                kwargs["cached_chunk_ptrs_npu"] = cached_chunk_ptrs_npu

        cached_mem_layers = (
            cached_memory_objs
            if cached_memory_objs_cover
            and cached_memory_objs is not None
            and len(cached_memory_objs) == self.num_layers
            else None
        )
        shared_chunk_locations_layer_major: list[list[str]] = []
        pre_resolved_shared_mem_layers: Optional[List[List[MemoryObj]]] = None
        layer_page_chunks = 0
        prefetched_layer_page_cache = False
        cached_layer_page_chunks = kwargs.get("_cached_layer_page_chunks", 0)
        try:
            cached_layer_page_chunks = int(cached_layer_page_chunks or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cached layer-page chunk count must be an integer."
            ) from exc
        if cached_layer_page_chunks:
            if (
                cached_layer_page_chunks < 0
                or cached_layer_page_chunks > required_chunks
                or cached_mem_layers is None
                or not use_cached_retrieve
            ):
                raise ValueError(
                    "Cached layer-page layout does not cover the current "
                    "retrieve metadata: "
                    f"page_chunks={cached_layer_page_chunks}, "
                    f"required_chunks={required_chunks}, "
                    f"cached={cached_mem_layers is not None}."
                )
            layer_page_chunks = cached_layer_page_chunks
            prefetched_layer_page_cache = True
        local_prefix_layers: List[Optional[LocalCPUPrefixGetResult]] = []
        request_ordinal = int(kwargs.get("shared_cpu_request_ordinal", 0))
        preflight_error_envelope: Optional[SharedHandleEnvelope] = None
        preflight_error: Optional[BaseException] = None
        request_preflight_state = kwargs.get("shared_cpu_request_preflight_state")

        def release_pre_resolved_shared_mem_layers() -> None:
            nonlocal pre_resolved_shared_mem_layers
            if pre_resolved_shared_mem_layers is None:
                return
            unique = {
                id(obj): obj
                for layer in pre_resolved_shared_mem_layers
                for obj in layer
            }
            for mem_obj in reversed(unique.values()):
                try:
                    if getattr(mem_obj, "is_pinned", False):
                        mem_obj.unpin()
                    if getattr(mem_obj, "is_valid", lambda: True)():
                        mem_obj.ref_count_down()
                except Exception:
                    logger.exception(
                        "Failed to release pre-resolved shared sparse "
                        "MemoryObj after request-level preflight failure"
                    )
            pre_resolved_shared_mem_layers = None

        def release_local_prefix_layers() -> None:
            for local_prefix in reversed(local_prefix_layers):
                if local_prefix is None:
                    continue
                try:
                    local_prefix.release()
                except Exception:
                    logger.exception(
                        "Failed to release LocalCPU prefix after shared sparse "
                        "request preflight failure"
                    )
            local_prefix_layers.clear()

        def record_request_preflight_error() -> None:
            if (
                not isinstance(request_preflight_state, dict)
                or preflight_error_envelope is None
                or preflight_error is None
            ):
                return
            request_preflight_state.setdefault(
                "source_kv_group",
                kv_group,
            )
            request_preflight_state.setdefault(
                "source_layer_id",
                preflight_error_envelope.layer_id,
            )
            request_preflight_state.setdefault(
                "message",
                preflight_error_envelope.message,
            )
            request_preflight_state.setdefault(
                "details",
                preflight_error_envelope.error_details,
            )
            request_preflight_state.setdefault("error", preflight_error)

        def request_preflight_failed_elsewhere() -> bool:
            return (
                isinstance(request_preflight_state, dict)
                and "error" in request_preflight_state
                and request_preflight_state.get("source_kv_group") != kv_group
            )

        def make_request_preflight_error_envelope() -> SharedHandleEnvelope:
            assert isinstance(request_preflight_state, dict)
            source_kv_group = request_preflight_state.get("source_kv_group")
            message = (
                "Shared CPU sparse decode request preflight failed before "
                "any handle publication."
            )
            return self._shared_layerwise_error_envelope(
                req_id=kwargs.get("req_id", "unspecified"),
                phase=kwargs.get("shared_cpu_phase", "sparse_decode_bootstrap"),
                request_ordinal=request_ordinal,
                layer_id=0,
                kv_group=kv_group,
                message=message,
                details={
                    "source_kv_group": source_kv_group,
                    "source_layer_id": request_preflight_state.get(
                        "source_layer_id"
                    ),
                    "source_message": request_preflight_state.get("message"),
                    "source_details": request_preflight_state.get("details"),
                },
            )

        if direct_group1_pages and not direct_group1_load:
            (
                pre_resolved_shared_mem_layers,
                layer_page_chunks,
            ) = self._resolve_shared_rank0_layer_pages(
                req_id=kwargs.get("req_id", "unspecified"),
                phase=kwargs.get("shared_cpu_phase", "sparse_decode_bootstrap"),
                kv_group=kv_group,
                keys_layer_major=missing_keys,
                page_chunks=len(missing_keys[0]),
            )

        sampled_worker_retrieve = self._use_sampled_worker_retrieve(kv_group)
        remote_layers_per_batch = max(
            1,
            min(
                self.num_layers,
                int(
                    self._get_shared_config_value(
                        "shared_cpu_remote_layers_per_batch",
                        1,
                    )
                ),
            ),
        )
        if mooncake_page_layout_enabled(self.config):
            remote_layers_per_batch = self.num_layers
        if (
            shared_sparse_retrieve
            and not use_cached_retrieve
            and any(missing_keys)
        ):
            probe_started = (
                cold_start_perf_now() if perf_enabled else 0.0
            )
            missing_locations: list[tuple[int, int]] = []
            if sampled_worker_retrieve:
                local_cpu_backend = self._shared_local_cpu_backend()
                try:
                    batched_prefix_get = getattr(
                        local_cpu_backend,
                        "batched_get_prefixes_with_misses",
                        None,
                    )
                    if callable(batched_prefix_get):
                        local_prefix_layers.extend(batched_prefix_get(missing_keys))
                        if len(local_prefix_layers) != len(missing_keys):
                            raise ValueError(
                                "Shared CPU batched local-prefix lookup returned "
                                "an unexpected layer count: "
                                f"expected={len(missing_keys)}, "
                                f"got={len(local_prefix_layers)}"
                            )
                    else:
                        # Compatibility with LMCache versions that predate the
                        # cross-layer LocalCPU prefix API.
                        local_prefix_layers.extend(
                            local_cpu_backend.batched_get_prefix_with_misses(
                                layer_keys
                            )
                            for layer_keys in missing_keys
                        )
                    for local_prefix in local_prefix_layers:
                        layer_locations = [LOCAL_CPU_BACKEND_NAME] * len(
                            local_prefix.local_memory_objs
                        ) + ["RemoteBackend"] * len(local_prefix.remote_positions)
                        shared_chunk_locations_layer_major.append(layer_locations)
                    if mooncake_layer_pages_enabled(self.config):
                        page_chunks = len(missing_keys[0])
                        local_pages, _ = (
                            self.storage_manager.batched_contains_layer_pages(
                                missing_keys[0][:page_chunks],
                                [LOCAL_CPU_BACKEND_NAME],
                            )
                        )
                        for locations in shared_chunk_locations_layer_major:
                            locations[:local_pages] = [
                                LOCAL_CPU_BACKEND_NAME
                            ] * local_pages
                        if page_chunks < len(missing_keys[0]):
                            tail_keys = [
                                layer_keys[page_chunks]
                                for layer_keys in missing_keys
                            ]
                            local_tail, _ = self.storage_manager.batched_contains(
                                tail_keys, [LOCAL_CPU_BACKEND_NAME]
                            )
                            if local_tail == len(tail_keys):
                                for locations in (
                                    shared_chunk_locations_layer_major
                                ):
                                    locations[page_chunks] = (
                                        LOCAL_CPU_BACKEND_NAME
                                    )
                except Exception:
                    release_local_prefix_layers()
                    raise
            else:
                for layer_id, layer_keys in enumerate(missing_keys):
                    layer_locations: list[str] = []
                    for chunk_index, key in enumerate(layer_keys):
                        key_location = self._find_shared_rank0_chunk_location(key)
                        if key_location is None:
                            missing_locations.append((layer_id, chunk_index))
                            key_location = ""
                        layer_locations.append(key_location)
                    shared_chunk_locations_layer_major.append(layer_locations)
            if perf_enabled:
                local_chunks = sum(
                    location == LOCAL_CPU_BACKEND_NAME
                    for locations in shared_chunk_locations_layer_major
                    for location in locations
                )
                remote_chunks = sum(
                    location == "RemoteBackend"
                    for locations in shared_chunk_locations_layer_major
                    for location in locations
                )
                unresolved_chunks = sum(
                    not location
                    for locations in shared_chunk_locations_layer_major
                    for location in locations
                )
                cold_start_perf_log(
                    logger,
                    "location_probe",
                    started=probe_started,
                    req_id=kwargs.get("req_id", "unspecified"),
                    phase=kwargs.get(
                        "shared_cpu_phase",
                        "sparse_decode_bootstrap",
                    ),
                    kv_group=kv_group,
                    layers=len(missing_keys),
                    chunks=sum(len(layer) for layer in missing_keys),
                    missing=unresolved_chunks,
                    local_chunks=local_chunks,
                    remote_chunks=remote_chunks,
                    unresolved_chunks=unresolved_chunks,
                    sampled=sampled_worker_retrieve,
                )
            page_first_locations = (
                self._shared_page_first_common_prefix_plan(
                    shared_chunk_locations_layer_major
                )
                if mooncake_page_layout_enabled(self.config)
                else None
            )
            if page_first_locations is not None:
                shared_chunk_locations_layer_major = page_first_locations
            if missing_locations:
                message = (
                    "Shared CPU sparse decode missing required chunks before "
                    "rank0 materialization."
                )
                preflight_error_envelope = self._shared_layerwise_error_envelope(
                    req_id=kwargs.get("req_id", "unspecified"),
                    phase=kwargs.get("shared_cpu_phase", "sparse_decode_bootstrap"),
                    request_ordinal=request_ordinal,
                    layer_id=0,
                    kv_group=kv_group,
                    message=message,
                    details={
                        "missing_locations": missing_locations,
                        "failed_layer_id": missing_locations[0][0],
                    },
                )
                preflight_error = ValueError(
                    f"{message} missing={missing_locations}"
                )
                record_request_preflight_error()
            else:
                try:
                    capacity_started = (
                        cold_start_perf_now()
                        if perf_enabled
                        else 0.0
                    )
                    capacity_details = self._shared_cpu_runtime_capacity_details(
                        req_id=kwargs.get("req_id", "unspecified"),
                        phase=kwargs.get(
                            "shared_cpu_phase", "sparse_decode_bootstrap"
                        ),
                        kv_group=kv_group,
                        keys_layer_major=missing_keys,
                        chunk_locations_layer_major=(
                            shared_chunk_locations_layer_major
                        ),
                        token_count=len(tokens),
                        chunk_token_lengths=[
                            int(end - start)
                            for start, end in zip(
                                starts[cached_prefix_chunks:],
                                ends[cached_prefix_chunks:],
                                strict=False,
                            )
                        ],
                    )
                    if perf_enabled:
                        cold_start_perf_log(
                            logger,
                            "capacity_preflight",
                            started=capacity_started,
                            req_id=kwargs.get("req_id", "unspecified"),
                            phase=kwargs.get(
                                "shared_cpu_phase",
                                "sparse_decode_bootstrap",
                            ),
                            kv_group=kv_group,
                            fits=capacity_details.get("fits"),
                            missing_chunks=capacity_details.get(
                                "missing_chunk_count"
                            ),
                            required_bytes=capacity_details.get(
                                "required_bytes"
                            ),
                            available_bytes=capacity_details.get(
                                "available_after_eviction"
                            ),
                        )
                except Exception:
                    release_local_prefix_layers()
                    raise
                if not capacity_details.get("fits", False):
                    message = (
                        "Shared CPU sparse decode capacity check failed before "
                        "rank0 materialization."
                    )
                    preflight_error_envelope = self._shared_layerwise_error_envelope(
                        req_id=kwargs.get("req_id", "unspecified"),
                        phase=kwargs.get(
                            "shared_cpu_phase", "sparse_decode_bootstrap"
                        ),
                        request_ordinal=request_ordinal,
                        layer_id=0,
                        kv_group=kv_group,
                        message=message,
                        details=capacity_details,
                    )
                    preflight_error = ValueError(
                        f"{message} details={capacity_details}"
                    )
                    release_local_prefix_layers()
                    record_request_preflight_error()
                else:
                    pre_resolved_shared_mem_layers = []
                    try:
                        page_chunks = len(missing_keys[0])
                        if (
                            mooncake_layer_pages_enabled(self.config)
                            and page_chunks
                        ):
                            release_local_prefix_layers()
                            (
                                pre_resolved_shared_mem_layers,
                                layer_page_chunks,
                            ) = self._resolve_shared_rank0_layer_pages(
                                req_id=kwargs.get("req_id", "unspecified"),
                                phase=kwargs.get(
                                    "shared_cpu_phase", "sparse_decode_bootstrap"
                                ),
                                kv_group=kv_group,
                                keys_layer_major=missing_keys,
                                page_chunks=page_chunks,
                            )
                        elif page_first_locations is not None:
                            prefixes = None
                            if local_prefix_layers:
                                if any(
                                    prefix is None for prefix in local_prefix_layers
                                ):
                                    raise ValueError(
                                        "Page-first LocalCPU prefix is incomplete"
                                    )
                                prefixes = [
                                    prefix
                                    for prefix in local_prefix_layers
                                    if prefix is not None
                                ]
                            pre_resolved_shared_mem_layers = (
                                self._resolve_shared_rank0_page_first_layers(
                                    req_id=kwargs.get("req_id", "unspecified"),
                                    phase=kwargs.get(
                                        "shared_cpu_phase",
                                        "sparse_decode_bootstrap",
                                    ),
                                    kv_group=kv_group,
                                    keys_layer_major=missing_keys,
                                    local_prefix_layers=prefixes,
                                )
                            )
                            release_local_prefix_layers()
                        elif remote_layers_per_batch > 1 and all(
                            location == "RemoteBackend"
                            for layer in shared_chunk_locations_layer_major
                            for location in layer
                        ):
                            release_local_prefix_layers()
                            pre_resolved_shared_mem_layers = (
                                self._resolve_shared_rank0_remote_layers_windowed(
                                    req_id=kwargs.get("req_id", "unspecified"),
                                    phase=kwargs.get(
                                        "shared_cpu_phase",
                                        "sparse_decode_bootstrap",
                                    ),
                                    kv_group=kv_group,
                                    keys_layer_major=missing_keys,
                                    layers_per_batch=remote_layers_per_batch,
                                )
                            )
                        else:
                            for layer_id, layer_keys in enumerate(missing_keys):
                                if sampled_worker_retrieve:
                                    local_prefix = local_prefix_layers[layer_id]
                                    assert local_prefix is not None
                                    # The resolver consumes every retained local
                                    # reference, whether it succeeds or raises.
                                    local_prefix_layers[layer_id] = None
                                    mem_objs_layer = (
                                        self._resolve_shared_rank0_layer_mem_objs(
                                            req_id=kwargs.get(
                                                "req_id", "unspecified"
                                            ),
                                            phase=kwargs.get(
                                                "shared_cpu_phase",
                                                "sparse_decode_bootstrap",
                                            ),
                                            layer_id=layer_id,
                                            kv_group=kv_group,
                                            keys_layer=layer_keys,
                                            local_prefix=local_prefix,
                                        )
                                    )
                                else:
                                    mem_objs_layer = (
                                        self._resolve_shared_rank0_layer_mem_objs(
                                            req_id=kwargs.get(
                                                "req_id", "unspecified"
                                            ),
                                            phase=kwargs.get(
                                                "shared_cpu_phase",
                                                "sparse_decode_bootstrap",
                                            ),
                                            layer_id=layer_id,
                                            kv_group=kv_group,
                                            keys_layer=layer_keys,
                                            chunk_locations=(
                                                shared_chunk_locations_layer_major[
                                                    layer_id
                                                ]
                                            ),
                                        )
                                    )
                                pre_resolved_shared_mem_layers.append(
                                    mem_objs_layer
                                )
                        release_local_prefix_layers()
                    except Exception as exc:
                        failed_layer_id = len(pre_resolved_shared_mem_layers)
                        release_pre_resolved_shared_mem_layers()
                        release_local_prefix_layers()
                        message = (
                            "Shared CPU sparse decode rank0 materialization failed "
                            "before any handle publication."
                        )
                        preflight_error_envelope = (
                            self._shared_layerwise_error_envelope(
                                req_id=kwargs.get("req_id", "unspecified"),
                                phase=kwargs.get(
                                    "shared_cpu_phase",
                                    "sparse_decode_bootstrap",
                                ),
                                request_ordinal=request_ordinal,
                                layer_id=0,
                                kv_group=kv_group,
                                message=message,
                                details={
                                    "error": str(exc),
                                    "location": location,
                                    "failed_layer_id": failed_layer_id,
                                },
                            )
                        )
                        preflight_error = ValueError(
                            f"{message} error={exc}"
                        )
                        record_request_preflight_error()

        if (
            not use_cached_retrieve
            and not shared_sparse_retrieve
            and not direct_group1_load
            and pre_resolved_shared_mem_layers is None
        ):
            get_generator = self.storage_manager.layerwise_batched_get(
                missing_keys, location=location
            )

        if preflight_error_envelope is None and request_preflight_failed_elsewhere():
            release_pre_resolved_shared_mem_layers()
            release_local_prefix_layers()
            preflight_error_envelope = make_request_preflight_error_envelope()
            preflight_error = ValueError(
                "Shared CPU sparse decode request preflight failed before "
                "handle publication: "
                f"source_kv_group="
                f"{request_preflight_state.get('source_kv_group')}, "
                f"message={request_preflight_state.get('message')}"
            )

        group_cache_prepared = False
        compact_handle_batch: Optional[SharedHandleBatch] = None
        compact_publish_mem_objs: Optional[List[List[MemoryObj]]] = None
        if (
            preflight_error_envelope is None
            and pre_resolved_shared_mem_layers is not None
            and not use_cached_retrieve
            and cached_prefix_chunks < required_chunks
        ):
            try:
                page_sources = (
                    [
                        LayerPageSource(
                            tuple(
                                pre_resolved_shared_mem_layers[0][
                                    :layer_page_chunks
                                ]
                            ),
                            layer_id,
                            tuple(layer[layer_page_chunks:]),
                        )
                        for layer_id, layer in enumerate(
                            pre_resolved_shared_mem_layers
                        )
                    ]
                    if layer_page_chunks
                    else pre_resolved_shared_mem_layers
                )
                self._append_retrieve_group_cache(
                    page_sources,
                    cached_memory_objs,
                    cached_tensors,
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                )
                group_cache_prepared = True
                if publish_shared_handles:
                    compact_handle_batch = self._make_shared_handle_batch(
                        pre_resolved_shared_mem_layers,
                        missing_keys,
                    )
                    if compact_handle_batch is not None:
                        if cached_memory_objs is None:
                            raise ValueError(
                                "Compact shared publication requires retained "
                                "MemoryObjs."
                            )
                        compact_publish_mem_objs = []
                        for layer_id in range(self.num_layers):
                            publish_mem_objs = cached_memory_objs[layer_id][
                                cached_handle_chunks:
                            ]
                            if (
                                len(publish_mem_objs)
                                != compact_handle_batch.num_chunks
                            ):
                                raise ValueError(
                                    "Compact shared publication has incomplete "
                                    f"layer coverage: layer={layer_id}, "
                                    f"objects={len(publish_mem_objs)}, "
                                    f"chunks={compact_handle_batch.num_chunks}"
                                )
                            compact_publish_mem_objs.append(publish_mem_objs)
                            self._append_shared_handle_cache(
                                layer_id,
                                [None] * len(publish_mem_objs),
                                cached_shared_handles,
                                cached_handle_chunks,
                            )
                        self._fence_shared_cpu_store_publication()
            except Exception as exc:
                if not shared_sparse_retrieve:
                    release_pre_resolved_shared_mem_layers()
                    raise
                message = (
                    "Shared CPU sparse all-layer pointer preflight failed "
                    "before handle publication."
                )
                preflight_error_envelope = self._shared_layerwise_error_envelope(
                    req_id=kwargs.get("req_id", "unspecified"),
                    phase=kwargs.get(
                        "shared_cpu_phase", "sparse_decode_bootstrap"
                    ),
                    request_ordinal=request_ordinal,
                    layer_id=0,
                    kv_group=kv_group,
                    message=message,
                    details={"error": str(exc)},
                )
                preflight_error = ValueError(f"{message} error={exc}")
                record_request_preflight_error()

        if (
            preflight_error_envelope is None
            and prefetched_layer_page_cache
            and publish_shared_handles
        ):
            try:
                if cached_handle_chunks != 0:
                    raise ValueError(
                        "Prefetched layer pages require an empty shared-handle "
                        "cache before compact publication."
                    )
                assert cached_mem_layers is not None
                compact_handle_batch = self._make_shared_handle_batch(
                    cached_mem_layers,
                    retrieve_keys,
                )
                if compact_handle_batch is None:
                    raise ValueError(
                        "Prefetched layer pages cannot be represented by one "
                        "compact shared-handle batch."
                    )
                if (
                    len(compact_handle_batch.page_offsets)
                    != layer_page_chunks
                ):
                    raise ValueError(
                        "Prefetched layer-page marker does not match the "
                        "validated compact batch: "
                        f"marker={layer_page_chunks}, "
                        f"batch_pages={len(compact_handle_batch.page_offsets)}."
                    )
                if cached_memory_objs is None:
                    raise ValueError(
                        "Compact shared publication requires retained MemoryObjs."
                    )
                compact_publish_mem_objs = []
                for layer_id in range(self.num_layers):
                    publish_mem_objs = cached_memory_objs[layer_id]
                    if len(publish_mem_objs) != compact_handle_batch.num_chunks:
                        raise ValueError(
                            "Prefetched compact publication has incomplete "
                            f"layer coverage: layer={layer_id}, "
                            f"objects={len(publish_mem_objs)}, "
                            f"chunks={compact_handle_batch.num_chunks}"
                        )
                    compact_publish_mem_objs.append(publish_mem_objs)
                    self._append_shared_handle_cache(
                        layer_id,
                        [None] * len(publish_mem_objs),
                        cached_shared_handles,
                    )
                self._fence_shared_cpu_store_publication()
            except Exception as exc:
                message = (
                    "Prefetched shared CPU layer-page preflight failed before "
                    "handle publication."
                )
                preflight_error_envelope = self._shared_layerwise_error_envelope(
                    req_id=kwargs.get("req_id", "unspecified"),
                    phase=kwargs.get(
                        "shared_cpu_phase", "sparse_decode_bootstrap"
                    ),
                    request_ordinal=request_ordinal,
                    layer_id=0,
                    kv_group=kv_group,
                    message=message,
                    details={"error": str(exc)},
                )
                preflight_error = ValueError(f"{message} error={exc}")
                record_request_preflight_error()

        if preflight_error_envelope is not None:
            release_pre_resolved_shared_mem_layers()
            release_local_prefix_layers()
            # Compact preflight can append placeholder handle entries before a
            # later layer exposes incomplete coverage.  This branch executes
            # before the main generator try/finally below, so restore every
            # append-only cache to its entry snapshot explicitly.
            append.rollback()
            yield ret_mask
            self._broadcast_shared_envelope(preflight_error_envelope)
            assert preflight_error is not None
            raise preflight_error

        pending_pre_resolved_release: List[MemoryObj] = []
        if (
            pre_resolved_shared_mem_layers is not None
            and not use_cached_retrieve
        ):
            pending_pre_resolved_release = list(
                {
                    id(obj): obj
                    for layer in pre_resolved_shared_mem_layers
                    for obj in layer
                }.values()
            )

        def release_pending_pre_resolved() -> None:
            nonlocal pending_pre_resolved_release
            for mem_obj in pending_pre_resolved_release:
                if getattr(mem_obj, "is_pinned", False):
                    mem_obj.unpin()
                if mem_obj.is_valid():
                    mem_obj.ref_count_down()
            pending_pre_resolved_release = []

        published_cached_shared_mem_objs: List[MemoryObj] = []
        cached_publication_handed_off = False
        shared_store_publication_fenced = compact_handle_batch is not None
        compact_batch_published = False
        compact_final_status_attempted = False
        compact_final_status_sent = False
        existing_rank0_backing_ids = (
            self.shared_cpu_rank0_request_object_ids(
                kwargs.get("req_id"),
                int(kwargs.get("kv_group", 0) or 0),
            )
            if publish_shared_handles and cached_mem_layers is not None
            else set()
        )

        def claim_cached_shared_mem_objs_for_publication(
            mem_objs_layer: List[MemoryObj],
        ) -> None:
            """Take the request pin/ref needed by rank0 handle publication."""
            for mem_obj in mem_objs_layer:
                if id(mem_obj) in existing_rank0_backing_ids:
                    continue
                try:
                    mem_obj.ref_count_up()
                except Exception as exc:
                    raise ValueError(
                        "Shared CPU sparse decode cached MemoryObj cannot be "
                        "retained before handle publication."
                    ) from exc
                try:
                    pinned = mem_obj.pin()
                    if pinned is False:
                        raise ValueError("MemoryObj.pin() returned False")
                except Exception:
                    try:
                        if getattr(mem_obj, "is_valid", lambda: True)():
                            mem_obj.ref_count_down()
                    except Exception:
                        logger.exception(
                            "Failed to release cached shared MemoryObj after "
                            "publication pin failure"
                        )
                    raise
                published_cached_shared_mem_objs.append(mem_obj)

        def release_unowned_cached_publication_objs() -> None:
            if cached_publication_handed_off:
                return
            for mem_obj in reversed(published_cached_shared_mem_objs):
                try:
                    if getattr(mem_obj, "is_pinned", False):
                        mem_obj.unpin()
                    if getattr(mem_obj, "is_valid", lambda: True)():
                        mem_obj.ref_count_down()
                except Exception:
                    logger.exception(
                        "Failed to release cached shared MemoryObj after "
                        "handle publication failure"
                    )
            published_cached_shared_mem_objs.clear()

        sparse_memory_objs_notified = False
        backend_fetched_objs: List[MemoryObj] = []

        def ensure_mem_obj_consumer():
            nonlocal mem_obj_consumer, sparse_memory_objs_notified
            if mem_obj_consumer is not None:
                return mem_obj_consumer
            mem_obj_consumer = (
                self.gpu_connector.batched_to_gpu_head_token_wise(**kwargs)
            )
            notify_fn = getattr(
                self.gpu_connector,
                "notify_sparse_memory_objs_updated",
                None,
            )
            if (
                notify_fn is not None
                and not use_cached_retrieve
                and not sparse_memory_objs_notified
            ):
                notify_fn()
                sparse_memory_objs_notified = True
            next(mem_obj_consumer)
            return mem_obj_consumer

        def release_failed_backend_results() -> None:
            """Fence failed NPU submissions before dropping backend owners."""
            if not backend_fetched_objs:
                if mem_obj_consumer is not None:
                    try:
                        mem_obj_consumer.close()
                    except Exception:
                        logger.exception(
                            "Failed to close sparse bootstrap consumer after "
                            "retrieval failure"
                        )
                return
            if mem_obj_consumer is not None:
                synchronize = getattr(
                    self.gpu_connector,
                    "synchronize_dense_load_stream",
                    None,
                )
                try:
                    fenced = False
                    if callable(synchronize):
                        synchronize()
                        fenced = True
                    else:
                        load_stream = getattr(
                            self.gpu_connector, "load_stream", None
                        )
                        stream_synchronize = getattr(
                            load_stream, "synchronize", None
                        )
                        if callable(stream_synchronize):
                            stream_synchronize()
                            fenced = True
                    if not fenced:
                        logger.critical(
                            "Aborted sparse prefix load has no NPU stream "
                            "fence API; retaining backend MemoryObjs"
                        )
                        return
                except BaseException:
                    # Retaining the owners is safer than freeing storage while
                    # an NPU DMA may still be reading it.
                    logger.critical(
                        "Failed to fence an aborted sparse prefix load; "
                        "retaining backend MemoryObjs",
                        exc_info=True,
                    )
                    return
                try:
                    mem_obj_consumer.close()
                except Exception:
                    logger.exception(
                        "Failed to close sparse bootstrap consumer after "
                        "retrieval failure"
                    )
            unique = {id(mem_obj): mem_obj for mem_obj in backend_fetched_objs}
            for mem_obj in reversed(unique.values()):
                try:
                    if getattr(mem_obj, "is_pinned", False):
                        mem_obj.unpin()
                    if mem_obj.is_valid():
                        mem_obj.ref_count_down()
                except Exception:
                    logger.exception(
                        "Failed to release sparse bootstrap backend object "
                        "after retrieval failure"
                    )
            backend_fetched_objs.clear()

        def attempt_compact_final_error(error: BaseException) -> None:
            nonlocal compact_final_status_attempted, compact_final_status_sent
            if not compact_batch_published or compact_final_status_attempted:
                return
            compact_final_status_attempted = True
            try:
                self._broadcast_shared_envelope(
                    self._shared_layerwise_error_envelope(
                        req_id=kwargs.get("req_id", "unspecified"),
                        phase=kwargs.get(
                            "shared_cpu_phase", "sparse_decode_bootstrap"
                        ),
                        request_ordinal=request_ordinal,
                        layer_id=self.num_layers,
                        kv_group=kv_group,
                        message="Compact shared batch failed before final commit.",
                        details={"error": str(error)},
                    )
                )
                compact_final_status_sent = True
            except BaseException:
                logger.exception(
                    "Failed to broadcast compact shared batch final error sentinel"
                )

        defer_finish = False
        try:
            for layer_id in range(self.num_layers):
                sparse_request = yield ret_mask
                (
                    sparse_payload,
                    selected_tokens,
                    token_start_index,
                    target_slot_mapping,
                    prepare_only,
                    defer_finish,
                ) = _shared_sparse_request(sparse_request)

                if (
                    shared_sparse_retrieve
                    and (not use_cached_retrieve or publish_shared_handles)
                    and request_preflight_failed_elsewhere()
                ):
                    release_pending_pre_resolved()
                    pre_resolved_shared_mem_layers = None
                    error_envelope = make_request_preflight_error_envelope()
                    if not compact_batch_published:
                        self._broadcast_shared_envelope(error_envelope)
                    raise ValueError(
                        "Shared CPU sparse decode request preflight failed "
                        "before handle publication: "
                        f"source_kv_group="
                        f"{request_preflight_state.get('source_kv_group')}, "
                        f"message={request_preflight_state.get('message')}"
                    )

                if direct_group1_load:
                    mem_objs_layer = []
                elif cached_mem_layers is not None:
                    mem_objs_layer = cached_mem_layers[layer_id]
                elif use_cached_retrieve:
                    mem_objs_layer = []
                else:
                    if pre_resolved_shared_mem_layers is not None:
                        mem_objs_layer = pre_resolved_shared_mem_layers[layer_id]
                    else:
                        assert get_generator is not None
                        task = next(get_generator)
                        assert task is not None
                        mem_objs_layer = task.result()
                        if mem_objs_layer is not None:
                            backend_fetched_objs.extend(mem_objs_layer)
                    expected_missing_chunks = (
                        required_chunks - cached_prefix_chunks
                    )
                    retrieved_chunks = (
                        len(mem_objs_layer)
                        if mem_objs_layer is not None
                        else 0
                    )
                    if retrieved_chunks != expected_missing_chunks:
                        # Metadata lookup already filled ret_mask, but a
                        # backend short read means those tokens are not safe
                        # to expose. Fail before submitting any stale/missing
                        # row to the model-visible cache. Pre-resolved shared
                        # objects are released by the generator's common
                        # finally path; direct backend results are owned here.
                        raise RuntimeError(
                            "Layerwise sparse retrieve returned incomplete "
                            "layer data; refusing the prefix success mask: "
                            f"req_id={kwargs.get('req_id', 'unspecified')}, "
                            f"kv_group={kv_group}, layer_id={layer_id}, "
                            f"expected_chunks={expected_missing_chunks}, "
                            f"retrieved_chunks={retrieved_chunks}"
                        )
                    if mem_objs_layer is not None and not group_cache_prepared:
                        if cached_prefix_chunks < required_chunks:
                            self._append_retrieve_layer_cache(
                                layer_id,
                                mem_objs_layer,
                                cached_memory_objs,
                                cached_tensors,
                                cached_chunk_dev_ptrs,
                                cached_chunk_ptrs_npu,
                            )
                    elif mem_objs_layer is None:
                        mem_objs_layer = []

                source = (
                    LayerPageSource(
                        tuple(mem_objs_layer[:layer_page_chunks]),
                        layer_id,
                        tuple(mem_objs_layer[layer_page_chunks:]),
                    )
                    if layer_page_chunks
                    else mem_objs_layer
                )

                if publish_shared_handles and (
                    compact_handle_batch is None or layer_id == 0
                ):
                    publish_mem_objs = (
                        compact_publish_mem_objs[layer_id]
                        if compact_publish_mem_objs is not None
                        else cached_memory_objs[layer_id][cached_handle_chunks:]
                        if cached_memory_objs is not None
                        and len(cached_memory_objs) > layer_id
                        else []
                    )
                    if required_chunks and not publish_mem_objs:
                        message = (
                            "Shared CPU sparse decode has no MemoryObjs to "
                            "publish for required chunks."
                        )
                        if not compact_batch_published:
                            self._broadcast_shared_envelope(
                                self._shared_layerwise_error_envelope(
                                    req_id=kwargs.get("req_id", "unspecified"),
                                    phase=kwargs.get(
                                        "shared_cpu_phase",
                                        "sparse_decode_bootstrap",
                                    ),
                                    request_ordinal=request_ordinal,
                                    layer_id=layer_id,
                                    kv_group=kv_group,
                                    message=message,
                                    details={"required_chunks": required_chunks},
                                )
                            )
                        raise ValueError(message)
                    try:
                        if not shared_store_publication_fenced:
                            # Store generators normally synchronize each layer
                            # before storage publication. Reassert that contract
                            # at the cross-rank publication boundary so handles
                            # can never outrun a direct-to-host store.
                            self._fence_shared_cpu_store_publication()
                            shared_store_publication_fenced = True
                        if (
                            cached_mem_layers is not None
                            and not prefetched_layer_page_cache
                        ):
                            claim_cached_shared_mem_objs_for_publication(
                                publish_mem_objs
                            )
                        handles = (
                            [None] * len(publish_mem_objs)
                            if compact_handle_batch is not None
                            else self._make_shared_handles_for_layer(
                                req_id=kwargs.get("req_id", "unspecified"),
                                phase=kwargs.get(
                                    "shared_cpu_phase", "sparse_decode_bootstrap"
                                ),
                                keys_layer=retrieve_keys[layer_id][
                                    cached_handle_chunks:
                                ],
                                mem_objs_layer=publish_mem_objs,
                                layer_id=layer_id,
                                kv_group=kv_group,
                                chunk_index_base=cached_handle_chunks,
                                validate_memory_objs=(
                                    pre_resolved_shared_mem_layers is None
                                ),
                            )
                        )
                        if compact_handle_batch is None:
                            self._append_shared_handle_cache(
                                layer_id,
                                handles,
                                cached_shared_handles,
                                cached_handle_chunks,
                            )
                        envelope = SharedHandleEnvelope(
                            request_id=kwargs.get("req_id", "unspecified"),
                            phase=kwargs.get(
                                "shared_cpu_phase", "sparse_decode_bootstrap"
                            ),
                            request_ordinal=request_ordinal,
                            layer_id=layer_id,
                            kv_group=kv_group,
                            status="ok" if handles else "skipped",
                            generation=self.shared_cpu_cache_generation,
                            handles=[] if compact_handle_batch is not None else handles,
                            message=(
                                None
                                if handles
                                else "no sparse decode shared CPU cache chunks selected"
                            ),
                            batch=(
                                compact_handle_batch
                                if layer_id == 0
                                else None
                            ),
                        )
                    except Exception as exc:
                        message = (
                            "Shared CPU sparse decode rank0 handle publication "
                            "failed."
                        )
                        if not compact_batch_published:
                            try:
                                self._broadcast_shared_envelope(
                                    self._shared_layerwise_error_envelope(
                                        req_id=kwargs.get("req_id", "unspecified"),
                                        phase=kwargs.get(
                                            "shared_cpu_phase",
                                            "sparse_decode_bootstrap",
                                        ),
                                        request_ordinal=request_ordinal,
                                        layer_id=layer_id,
                                        kv_group=kv_group,
                                        message=message,
                                        details={
                                            "error": str(exc),
                                            "required_chunks": required_chunks,
                                        },
                                    )
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to broadcast shared CPU sparse decode "
                                    "handle-publication error envelope"
                                )
                        raise
                    if compact_handle_batch is None or layer_id == 0:
                        self._broadcast_shared_envelope(envelope)
                        if compact_handle_batch is not None:
                            compact_batch_published = True

                if prepare_only:
                    sparse_request = yield ret_mask
                    (
                        sparse_payload,
                        selected_tokens,
                        token_start_index,
                        target_slot_mapping,
                        prepare_only,
                        defer_finish,
                    ) = _shared_sparse_request(sparse_request)
                    if prepare_only:
                        raise ValueError("Duplicate shared sparse prepare request")

                dispatch_started = (
                    cold_start_perf_now()
                    if perf_enabled
                    else 0.0
                )
                if direct_group1_load:
                    pass
                elif sparse_payload is not None:
                    sparse_payload["memory_objs_layer"] = source
                    ensure_mem_obj_consumer().send(sparse_payload)
                else:
                    ensure_mem_obj_consumer().send(
                        (
                            source,
                            selected_tokens,
                            token_start_index,
                            target_slot_mapping,
                        )
                    )
                if perf_enabled:
                    source_chunks = len(mem_objs_layer)
                    if (
                        not source_chunks
                        and cached_tensors is not None
                        and layer_id < len(cached_tensors)
                    ):
                        cached_layer = cached_tensors[layer_id]
                        source_chunks = (
                            len(cached_layer) if cached_layer is not None else 0
                        )
                    cold_start_perf_log(
                        logger,
                        "npu_layer_submit_cpu",
                        started=dispatch_started,
                        req_id=kwargs.get("req_id", "unspecified"),
                        phase=kwargs.get(
                            "shared_cpu_phase",
                            "sparse_decode_bootstrap",
                        ),
                        kv_group=kv_group,
                        layer=layer_id,
                        objects=len(mem_objs_layer),
                        source_chunks=source_chunks,
                        rank=self.metadata.worker_id,
                        passive=False,
                    )

            if mem_obj_consumer is not None:
                next(mem_obj_consumer)
            if defer_finish:
                yield ret_mask
            if compact_handle_batch is not None:
                compact_final_status_attempted = True
                self._broadcast_shared_envelope(
                    SharedHandleEnvelope(
                        request_id=kwargs.get("req_id", "unspecified"),
                        phase=kwargs.get(
                            "shared_cpu_phase", "sparse_decode_bootstrap"
                        ),
                        request_ordinal=request_ordinal,
                        layer_id=self.num_layers,
                        kv_group=kv_group,
                        status="skipped",
                        generation=self.shared_cpu_cache_generation,
                        handles=[],
                        message=_SHARED_CPU_COMPACT_COMMIT_MESSAGE,
                    )
                )
                compact_final_status_sent = True
            if compact_handle_batch is not None and not compact_final_status_sent:
                raise RuntimeError("Compact shared batch final status was not sent.")
            # The LMCache vLLM adapter records request state after this final
            # yield and drains sparse retrievers with close(), so ownership must
            # be handed off before yielding.
            cached_publication_handed_off = cached_memory_objs is not None
            if cached_publication_handed_off:
                pending_pre_resolved_release = []
            else:
                release_pending_pre_resolved()
            append.commit()

            yield ret_mask
        except Exception as exc:
            attempt_compact_final_error(exc)
            raise
        finally:
            attempt_compact_final_error(
                GeneratorExit("compact shared batch generator exited")
            )
            if not append.committed:
                release_failed_backend_results()
            release_unowned_cached_publication_objs()
            release_pending_pre_resolved()
            append.rollback()


    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug("rank=%s ignore store", self.metadata.worker_id)
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="npu"
            )

        # lmcache-ascend start ---------------------
        if not self.is_store_async:
            self._run_store_pipeline(
                req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
            )
        else:
            self._ensure_store_worker()
            with self._store_lock:
                self._pending_store_reqs[req_id] = (
                    self._pending_store_reqs.get(req_id, 0) + 1
                )
                pending = self._pending_store_reqs[req_id]

            enqueued = False
            try:
                self._store_queue.put(
                    (
                        req_id,
                        tokens,
                        hashes,
                        offsets,
                        mask,
                        num_to_store_tokens,
                        kwargs,
                    )
                )
                enqueued = True
            finally:
                if not enqueued:
                    with self._store_lock:
                        cnt = self._pending_store_reqs.get(req_id, 1) - 1
                        if cnt <= 0:
                            self._pending_store_reqs.pop(req_id, None)
                        else:
                            self._pending_store_reqs[req_id] = cnt
                        self._store_cv.notify_all()

            logger.debug(
                "Enqueued async store for req %s; pending=%d tokens=%d",
                req_id,
                pending,
                num_to_store_tokens,
            )
        # lmcache-ascend end ---------------------

    def _remote_fill_group1_startup_identity(
        self,
        layout: RemoteFillDecoderLayout,
        payload_descriptor: dict[str, object],
    ) -> dict[str, object]:
        """Return the rank-local Group-1 shared-page ABI identity."""

        shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
            kv_group=1,
            num_tokens=1,
        )
        group = layout.group(1)
        if (
            dtype != group.dtype
            or fmt != group.fmt
            or int(shape.numel()) != group.raw_token_dim
        ):
            raise ValueError(
                "remote-fill Group-1 shared-page metadata disagrees with "
                "the negotiated decoder layout: "
                f"shape={tuple(shape)}, elements={int(shape.numel())}, "
                f"dtype={dtype}, format={fmt.name}, "
                f"negotiated_elements={group.raw_token_dim}, "
                f"negotiated_dtype={group.dtype}, "
                f"negotiated_format={group.fmt.name}"
            )
        return {
            "model_layout": "mla-dsa-layer-page-v3",
            "layout_tag": layout.layout_tag,
            "cache_namespace_tag": layout.cache_namespace_tag,
            "model_artifact_id": layout.model_artifact_id,
            "payload_layout_version": int(payload_descriptor.get("version", 0)),
            "group1_payload_schema": str(
                payload_descriptor.get("group1_schema_version", "")
            ),
            "dsa_index_key_schema": DSA_INDEX_CACHE_SCHEMA,
            "num_layers": layout.num_layers,
            "valid_tokens": 1,
            "shape": [int(dimension) for dimension in shape],
            "dtype": str(dtype),
            "format": fmt.name,
            "logical_bytes": group.expected_bytes(1, layout.num_layers),
            "shared_cache_generation": int(self.shared_cpu_cache_generation),
        }

    def _validate_remote_fill_decoder_layout(
        self,
        layout: RemoteFillDecoderLayout,
    ) -> None:
        """Match both negotiated groups to the model-visible cache layout."""

        layer_groups = tuple(
            getattr(
                getattr(self.metadata, "kv_layer_groups_manager", None),
                "kv_layer_groups",
                (),
            )
        )
        if layer_groups and len(layer_groups) != 2:
            raise ValueError(
                "remote-fill requires exactly two registered decoder cache "
                f"groups, found {len(layer_groups)}"
            )

        for kv_group in (0, 1):
            shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
                kv_group=kv_group,
                num_tokens=1,
            )
            group = layout.group(kv_group)
            registered_layers = (
                layer_groups[kv_group].num_layers if layer_groups else None
            )
            if (
                registered_layers is not None
                and registered_layers != layout.num_layers
            ):
                raise ValueError(
                    "remote-fill requires identical cache-bearing layer "
                    "coverage for both groups: "
                    f"kv_group={kv_group}, "
                    f"registered_layers={registered_layers}, "
                    f"negotiated_layers={layout.num_layers}"
                )
            if (
                dtype != group.dtype
                or fmt != group.fmt
                or int(shape.numel()) != group.raw_token_dim
            ):
                raise ValueError(
                    "remote-fill negotiated group layout disagrees with "
                    "decoder cache metadata: "
                    f"kv_group={kv_group}, "
                    f"decoder_shape={tuple(shape)}, "
                    f"decoder_elements_per_token={int(shape.numel())}, "
                    f"decoder_dtype={dtype}, decoder_format={fmt.name}, "
                    f"negotiated_elements_per_token={group.raw_token_dim}, "
                    f"negotiated_dtype={group.dtype}, "
                    f"negotiated_format={group.fmt.name}"
                )

    def _preflight_remote_fill_shared_group1(
        self,
        layout: RemoteFillDecoderLayout,
        payload_descriptor: dict[str, object],
    ) -> None:
        """Prove every TP rank can map one real Group-1 layer page.

        This collective is startup-only. Rank 0 allocates and pins a hidden
        one-token page, broadcasts its compact shared-slab descriptor, and
        every passive rank creates and validates its ordinary passive view.
        Every rank then participates in one CPU-group boolean consensus. The
        remote-fill service is advertised only after every rank succeeds.
        """

        if self.metadata.world_size <= 1:
            self._remote_fill_shared_group1_supported = True
            return
        first_rank = int(self.metadata.first_rank)
        worker_id = int(self.metadata.worker_id)
        ranks = tuple(
            range(first_rank, first_rank + int(self.metadata.world_size))
        )
        if worker_id not in ranks:
            raise ValueError("remote-fill worker rank is outside its TP group")

        rank0_page: Optional[LayerPageMemoryObj] = None
        rank0_pinned = False
        payload: Optional[dict[str, object]] = None
        if self.metadata.is_first_rank():
            try:
                local_backend = self._shared_local_cpu_backend()
                group = layout.group(1)
                pages = local_backend.batched_allocate_layer_pages(
                    [group.full_shape(layout.chunk_size)],
                    [group.dtype],
                    1,
                    layout.num_layers,
                    group.fmt,
                    busy_loop=False,
                    valid_tokens=[1],
                    full_tokens=layout.chunk_size,
                    eviction=False,
                )
                if pages is None or len(pages) != 1:
                    if pages:
                        for page in pages:
                            page.ref_count_down()
                    raise MemoryError(
                        "remote-fill Group-1 startup page allocation failed"
                    )
                rank0_page = pages[0]
                if not LayerPageMemoryObj.pin_many([rank0_page]):
                    raise MemoryError("remote-fill Group-1 startup page pin failed")
                rank0_pinned = True
                if (
                    rank0_page.num_layers != layout.num_layers
                    or rank0_page.valid_tokens != 1
                    or rank0_page.get_size()
                    != group.expected_bytes(1, layout.num_layers)
                    or rank0_page.get_dtype() != group.dtype
                    or rank0_page.get_memory_format() != group.fmt
                ):
                    raise ValueError(
                        "remote-fill Group-1 startup allocation layout mismatch"
                    )
                self._validate_rank0_shared_mem_obj(
                    rank0_page,
                    req_id="remote-fill-startup",
                    phase="group1-capability",
                    layer_id=0,
                    kv_group=1,
                    chunk_index=0,
                )
                startup_key = CacheEngineKey(
                    layout.model_name,
                    layout.key_world_size,
                    0,
                    0,
                    group.dtype,
                    {
                        "lmcache.tag.payload_v3": layout.layout_tag,
                        DSA_INDEX_CACHE_SCHEMA_TAG: DSA_INDEX_CACHE_SCHEMA,
                    },
                    kv_group=1,
                )
                keys = [[startup_key] for _ in range(layout.num_layers)]
                batch = self._make_shared_handle_batch(
                    [[rank0_page] for _ in range(layout.num_layers)],
                    keys,
                )
                if batch is None:
                    raise ValueError(
                        "remote-fill Group-1 startup page cannot be shared"
                    )
                payload = {
                    "status": "ok",
                    "identity": self._remote_fill_group1_startup_identity(
                        layout,
                        payload_descriptor,
                    ),
                    "key": startup_key.to_string(),
                    "batch": batch.to_dict(),
                }
            except Exception as exc:
                payload = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }

        passive_views: list[MemoryObj] = []
        try:
            received = self.broadcast_object_fn(payload, first_rank)
            if self.metadata.is_first_rank():
                received = payload
            local_ack: dict[str, object] = {
                "rank": worker_id,
                "status": "error",
                "message": "startup payload was not validated",
            }
            try:
                if not isinstance(received, dict):
                    raise ValueError("startup payload is not a dictionary")
                if received.get("status") != "ok":
                    raise ValueError(
                        "rank0 Group-1 capability setup failed: "
                        f"{received.get('message')}"
                    )
                expected_identity = self._remote_fill_group1_startup_identity(
                    layout,
                    payload_descriptor,
                )
                if received.get("identity") != expected_identity:
                    raise ValueError(
                        "remote-fill Group-1 startup ABI identity mismatch"
                    )
                key_raw = received.get("key")
                batch_raw = received.get("batch")
                if not isinstance(key_raw, str) or not isinstance(batch_raw, dict):
                    raise ValueError("startup shared-page descriptor is incomplete")
                startup_key = CacheEngineKey.from_string(key_raw)
                if (
                    startup_key.kv_group != 1
                    or startup_key.worker_id != 0
                    or startup_key.model_name != layout.model_name
                    or startup_key.world_size != layout.key_world_size
                    or startup_key.dtype != layout.group(1).dtype
                ):
                    raise ValueError("startup shared-page key identity is invalid")
                batch = SharedHandleBatch.from_dict(batch_raw)
                if self.metadata.is_first_rank():
                    if rank0_page is None:
                        raise ValueError("rank0 startup page owner is missing")
                    if (
                        batch.num_layers != layout.num_layers
                        or batch.num_chunks != 1
                        or batch.page_offsets
                        != [int(rank0_page.metadata.address)]
                        or batch.physical_sizes != [int(rank0_page.layer_size)]
                        or batch.page_physical_sizes
                        != [int(rank0_page.metadata.phy_size)]
                    ):
                        raise ValueError("rank0 startup shared descriptor is invalid")
                else:
                    views = self._make_passive_layer_page_views(
                        batch,
                        starts=[0],
                        ends=[1],
                        keys_layer_major=[
                            [startup_key] for _ in range(layout.num_layers)
                        ],
                        kv_group=1,
                    )
                    passive_views.extend(views)
                    if len(views) != 1:
                        raise ValueError(
                            "passive Group-1 startup mapping returned wrong count"
                        )
                    view = views[0]
                    shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
                        kv_group=1,
                        num_tokens=1,
                    )
                    if (
                        view.num_layers != layout.num_layers
                        or view.valid_tokens != 1
                        or view.get_shape() != shape
                        or view.get_dtype() != dtype
                        or view.get_memory_format() != fmt
                        or view.get_size()
                        != layout.group(1).expected_bytes(1, layout.num_layers)
                        or int(view.metadata.address) != batch.page_offsets[0]
                        or int(view.metadata.phy_size)
                        != batch.page_physical_sizes[0]
                    ):
                        raise ValueError(
                            "passive Group-1 startup page validation failed"
                        )
                local_ack = {"rank": worker_id, "status": "ok", "message": ""}
            except Exception as exc:
                local_ack = {
                    "rank": worker_id,
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }

            collective = getattr(self, "collective_all_true_fn", None)
            if not callable(collective):
                raise ValueError(
                    "remote-fill Group-1 startup lacks TP consensus callback"
                )
            all_ranks_ready = bool(collective(local_ack["status"] == "ok"))
            if not all_ranks_ready:
                local_message = str(local_ack.get("message") or "")
                detail = (
                    f"rank={worker_id} error={local_message}"
                    if local_ack["status"] != "ok"
                    else "another TP rank rejected the shared page"
                )
                raise ValueError(
                    "remote-fill Group-1 shared startup capability failed: "
                    + detail
                )
            self._remote_fill_shared_group1_supported = True
        finally:
            self._release_shared_retrieve_objs(passive_views, unpin=False)
            if rank0_page is not None:
                if rank0_pinned and rank0_page.is_pinned:
                    rank0_page.unpin()
                if rank0_page.is_valid():
                    rank0_page.ref_count_down()

    def _initialize_decoder_remote_fill(self) -> None:
        if self._remote_fill_decoder_initialized:
            return
        if not bool(self.config.enable_remote_lmcache_store):
            return
        if self.metadata.role == "scheduler" or self.config.pd_role == "sender":
            return
        if not remote_fill_h0_qualified():
            # Enabling the prototype configuration before the hardware gate is
            # qualified must leave the deployed legacy P/D path untouched.
            self._remote_fill_decoder_initialized = True
            logger.info(
                "Remote-fill decoder service remains disabled until Gate H0 "
                "is explicitly qualified"
            )
            return
        configured_control_host = self.config.remote_fill_control_host
        configured_control_port = self.config.remote_fill_control_port_start
        is_decoder = self.config.pd_role == "receiver" or bool(
            configured_control_host or configured_control_port is not None
        )
        if not is_decoder:
            return
        control_host = str(configured_control_host or "0.0.0.0").strip()
        base_port = int(
            configured_control_port or REMOTE_FILL_DEFAULT_CONTROL_PORT
        )
        if not (self.save_only_first_rank and self.save_indexer_only_first_rank):
            raise ValueError(
                "remote-fill decoder requires TP0 ownership for both DSA groups"
            )
        tp_shared = self.metadata.world_size > 1
        if tp_shared and not (
            self.enable_shared_cpu_cache and self.shared_cpu_cache_strict
        ):
            raise ValueError(
                "remote-fill TP>1 decoder requires strict shared LocalCPU storage"
            )
        if tp_shared and self.shared_cpu_cache_generation <= 0:
            raise ValueError(
                "remote-fill TP>1 decoder requires a completed shared-cache preflight"
            )
        if not control_host:
            raise ValueError("remote-fill decoder control bind host is empty")
        layout_tag, payload_descriptor = mooncake_payload_layout(
            self.config,
            self.metadata,
        )
        layout = build_decoder_layout(
            self.config,
            self.metadata,
            layout_tag=layout_tag,
            num_layers=int(self.num_layers),
        )
        self._validate_remote_fill_decoder_layout(layout)
        if tp_shared:
            self._preflight_remote_fill_shared_group1(
                layout,
                payload_descriptor,
            )
        if not self.metadata.is_first_rank():
            if self.shared_cpu_cache_mapping is None:
                raise ValueError(
                    "remote-fill passive rank lacks the shared-cache mapping"
                )
            self._remote_fill_decoder_initialized = True
            return
        if self.storage_manager is None:
            raise ValueError("remote-fill decoder requires StorageManager on TP0")
        extra = self.metadata.kv_connector_extra_config or {}
        dp_rank = int(extra.get("lmcache_remote_fill_destination_dp_rank", 0))
        dp_size = int(extra.get("lmcache_remote_fill_destination_dp_size", 1))
        control_port = _validate_remote_fill_control_port(
            int(base_port),
            dp_rank,
            dp_size,
            internal_api_enabled=bool(
                getattr(self.config, "internal_api_server_enabled", False)
            ),
            internal_api_port_start=int(
                getattr(self.config, "internal_api_server_port_start", 6999)
            ),
        )
        native_session = self.storage_manager.get_remote_fill_destination_session()
        if not native_session:
            raise ValueError(
                "remote-fill decoder requires a registered GlobalTE session"
            )
        control_advertise_host = str(
            getattr(self.config, "remote_fill_control_advertise_host", None)
            or _remote_fill_advertise_host_from_session(native_session)
        ).strip()
        local_backend = self._shared_local_cpu_backend()
        if not callable(getattr(local_backend, "get_allocator_capacity_bytes", None)):
            raise ValueError(
                "remote-fill decoder requires constant-time LocalCPU capacity"
            )
        if not self._remote_fill_shared_group1_supported:
            raise ValueError("remote-fill Group-1 shared capability is unavailable")
        epoch = secrets.randbits(63) or 1
        runtime = create_decoder_remote_fill_runtime(
            config=self.config,
            metadata=self.metadata,
            local_backend=local_backend,
            layout=layout,
            destination_engine_epoch=epoch,
            destination_dp_rank=dp_rank,
            destination_dp_size=dp_size,
            destination_remote_session=native_session,
            control_host=control_host,
            control_advertise_host=control_advertise_host,
            control_port=control_port,
            shared_cache_generation=int(self.shared_cpu_cache_generation),
            capacity_available=self._remote_fill_capacity_available,
            fatal_restart=self._remote_fill_require_paired_restart,
            global_te_push=True,
            save_only_first_rank=self.save_only_first_rank,
            save_indexer_only_first_rank=self.save_indexer_only_first_rank,
            shared_cpu_cache_strict=self.shared_cpu_cache_strict,
        )
        try:
            runtime.start()
            if not runtime.available:
                raise RuntimeError(
                    "remote-fill decoder control service exited during startup"
                )
        except Exception:
            try:
                runtime.close()
            except Exception:
                logger.exception(
                    "Failed to close remote-fill runtime after startup failure"
                )
            raise
        self._remote_fill_runtime = runtime
        self._remote_fill_decoder_initialized = True
        logger.info(
            "Remote-fill decoder service started: feature_enabled=%s "
            "transport_qualified=%s cache_namespace=%s model_artifact_id=%s "
            "control_endpoint=%s destination_engine_epoch=%d "
            "destination_dp_rank=%d shared_cache_generation=%d "
            "group0_dim=%d group1_dim=%d tp_size=%d dp_size=%d "
            "source_hash_identity=%s:%s",
            True,
            True,
            layout.cache_namespace_tag,
            layout.model_artifact_id,
            runtime.placement.control_endpoint,
            runtime.placement.destination_engine_epoch,
            dp_rank,
            self.shared_cpu_cache_generation,
            layout.groups[0].raw_token_dim,
            layout.groups[1].raw_token_dim,
            self.metadata.world_size,
            dp_size,
            runtime.placement.token_hash_algorithm,
            runtime.placement.python_hash_seed,
        )

    def _remote_fill_capacity_available(self, requested_bytes: int) -> bool:
        if requested_bytes < 0:
            return False
        try:
            free_bytes, heap_bytes = (
                self._shared_local_cpu_backend().get_allocator_capacity_bytes()
            )
        except (NotImplementedError, RuntimeError):
            return False
        reserve = max(
            int(self.config.remote_fill_min_free_bytes),
            int(heap_bytes * float(self.config.remote_fill_min_free_ratio)),
        )
        return free_bytes - requested_bytes >= reserve

    def _remote_fill_require_paired_restart(
        self,
        transfer_ids: tuple[str, ...],
    ) -> None:
        normalized = tuple(sorted({item for item in transfer_ids if item}))
        if not normalized:
            return
        lock = getattr(self, "_remote_fill_fatal_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._remote_fill_fatal_lock = lock
        with lock:
            previous = self._remote_fill_fatal_transfers
            self._remote_fill_fatal_transfers = tuple(
                sorted(set(previous).union(normalized))
            )
            first_report = not previous
        if first_report:
            self.mark_init_failed(
                "remote-fill armed transfer requires paired P+D restart"
            )
        logger.critical(
            "Remote-fill process requires paired restart: transfer_count=%d",
            len(self._remote_fill_fatal_transfers),
        )
        logger.critical(
            '[LMCACHE_REMOTE_FILL] {"schema":1,'
            '"event":"remote_fill_fatal_restart","count":%d}',
            len(self._remote_fill_fatal_transfers),
        )
        log_remote_fill_validation_failure(
            logger,
            code="RF-D-900",
            stage="armed_native_lifecycle",
            action="PAIRED_RESTART_REQUIRED",
            transfer_id=normalized[0] if len(normalized) == 1 else None,
            reason=(
                "destination memory may still be targeted; allocator teardown "
                "is intentionally blocked"
            ),
            memory_safety_uncertain=True,
            severity="critical",
        )

    def close(self) -> None:
        """Stop the bg worker gracefully, then close the base engine."""
        runtime = self._remote_fill_runtime
        if runtime is not None:
            runtime.close()
        if self._remote_fill_fatal_transfers:
            # An unknown native writer may still target registered P or D
            # storage. Deliberately retain runtime and allocator ownership
            # until the supervisor terminates the paired deployment.
            raise RuntimeError(
                "remote-fill shutdown requires paired P+D restart; "
                "native-owned memory was deliberately retained"
            )
        self._remote_fill_runtime = None
        self.close_remote_fill_producer()
        self.wait_for_direct_stores(tuple(self._direct_store_states))
        self._direct_store_states.clear()
        self._live_source_builders.clear()
        self._completed_live_sources.clear()
        pending_diagnostics = getattr(
            self, "_pending_live_source_diagnostics", None
        )
        if pending_diagnostics is not None:
            pending_diagnostics.clear()
        # Push poison pill first so any in-flight work drains before
        # ``storage_manager.close()`` runs inside ``super().close()``.
        if self._store_queue is not None:
            try:
                # Bounded queues can be full here; avoid blocking forever
                # while still preferring graceful drain.
                while True:
                    try:
                        self._store_queue.put_nowait(None)
                        break
                    except queue.Full:
                        if (
                            self._store_worker_thread is None
                            or not self._store_worker_thread.is_alive()
                        ):
                            logger.warning(
                                "Ascend store queue is full and worker is not alive; "
                                "cannot enqueue shutdown signal cleanly."
                            )
                            break
                        time.sleep(0.01)
                if self._store_worker_thread is not None:
                    self._store_worker_thread.join(timeout=10)
                    if self._store_worker_thread.is_alive():
                        logger.warning(
                            "Ascend store worker did not stop within 10s; "
                            "proceeding with engine shutdown."
                        )
            except Exception:
                logger.exception("Error stopping Ascend store worker")

        super().close()
