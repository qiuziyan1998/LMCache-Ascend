# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from bisect import bisect_right
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError, wait
from dataclasses import dataclass, field
from weakref import WeakSet
import json
import os
import queue
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
    mooncake_page_layout_enabled,
)
from lmcache.v1.shared_cpu_cache import (
    SharedHandleBatch,
    SharedHandleEnvelope,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult
from lmcache.v1.token_database import TokenDatabase
import torch

logger = init_logger(__name__)

LOCAL_CPU_BACKEND_NAME = "LocalCPUBackend"
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
    submitted_jobs: int = 0
    submitted_pages: int = 0
    submitted_legacy_objects: int = 0
    submitted_bytes: int = 0
    finalized: bool = False


@dataclass(slots=True)
class _DirectPageBatch:
    req_id: str
    keys: List[CacheEngineKey]
    ptrs: List[List[int]]
    sizes: List[List[int]]
    owners: tuple[torch.Tensor, ...]
    ready_event: Any
    group_ends: dict[int, int]



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
    ):
        super().__init__(
            config,
            metadata,
            token_database,
            gpu_connector,
            broadcast_fn,
            broadcast_object_fn,
        )
        self.is_store_async = self.config.store_async
        self._pending_sync_store_futures: set[Future] = set()
        self._require_store_completion = False
        self._direct_store_enabled = bool(
            self.is_store_async
            and self.config.get_extra_config_value(
                "mooncake_direct_npu_prefill_store", False
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

    def direct_prefill_store_enabled(self) -> bool:
        """Return whether this worker can use direct NPU page publication."""
        return self._direct_store_enabled

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
                batch.ready_event,
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
        return (
            len(tokens)
            if bool(getattr(self.config, "save_unfull_chunk", True))
            else state.planned_end
        )

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
        base = (
            state.planned_end
            if state.planned_end <= full_end
            and all(
                state.submitted_end.get(group, 0) >= state.planned_end
                for group in groups
            )
            else 0
        )
        if base:
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

    def begin_live_source_descriptor(self, req_id: str) -> None:
        self._live_source_builders.setdefault(
            req_id,
            {
                "segments": [],
                "ends": {},
                "invalid": False,
                "planned_end": 0,
                "planned_hash": 0,
            },
        )

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
        if builder is None:
            return
        coverage_end = int(builder["ends"].get(kv_group, 0))
        if len(ranges) != len(ptrs) or len(ptrs) != len(sizes):
            builder["invalid"] = True
            return
        owner_ranges = sorted(
            (
                int(owner.untyped_storage().data_ptr()),
                int(owner.untyped_storage().nbytes()),
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
    ) -> None:
        builder = self._live_source_builders.pop(req_id, None)
        if builder is None or builder["invalid"]:
            return
        if any(builder["ends"].get(group, 0) != token_count for group in (0, 1)):
            return
        totals = [0, 0]
        for segment in builder["segments"]:
            totals[segment["group_id"]] += segment["length"]
        if not all(totals):
            return
        self._completed_live_sources[req_id] = {
            "segments": builder["segments"],
            "group_byte_totals": totals,
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
        }

    def drain_live_source_descriptors(self) -> dict[str, dict[str, Any]]:
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
        planner = getattr(self.gpu_connector, "plan_direct_page_sources", None)
        if not callable(planner) or set(group_caches) != {0, 1}:
            self._live_source_builders.pop(req_id, None)
            return
        try:
            self.begin_live_source_descriptor(req_id)
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
            plans = {
                0: plan0,
                1: [
                    (latent[0], latent[1], indexer[2])
                    for latent, indexer in zip(plan0, group1_keys, strict=True)
                ],
            }
            for group, chunks in plans.items():
                if not chunks:
                    continue
                starts = [start for start, _, _ in chunks]
                ends = [end for _, end, _ in chunks]
                planned = planner(
                    group_caches[group],
                    slot_mappings[group],
                    starts,
                    ends,
                    group,
                    slot_mapping_base=slot_mapping_base,
                )
                if planned is None:
                    raise ValueError("direct page source layout is unsupported")
                ptrs, sizes, owners = planned
                self._record_live_source_pages(
                    req_id,
                    group,
                    list(zip(starts, ends, strict=True)),
                    ptrs,
                    sizes,
                    tuple(owners),
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
            if present[0]:
                state.submitted_end[group] = end
                if not state.futures:
                    state.committed_end[group] = end
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
                return False
            group_ptrs, group_sizes, group_owners = planned
            if (
                len(group_ptrs) != 1
                or len(group_sizes) != 1
                or any(len(item) > max_buffers for item in group_ptrs)
            ):
                return False
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
        if not keys:
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
        ready_event = torch.npu.Event()
        ready_event.record()
        batch = _DirectPageBatch(
            req_id,
            keys,
            ptrs,
            sizes,
            tuple(owners.values()),
            ready_event,
            group_ends,
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
    ) -> bool:
        """Publish newly completed pages and the final tail directly from NPU."""
        if not self._direct_store_enabled or not group_caches:
            return False
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
        if final and state.finalized:
            return True
        for group in group_caches:
            if group not in state.submitted_end:
                state.submitted_end[group] = verified_prefix_end
                state.committed_end[group] = verified_prefix_end
        started = cold_start_perf_now() if cold_start_perf_enabled() else None
        plans = self._direct_suffix_plans(
            state, tokens, group_caches, request_configs
        )

        keys: List[CacheEngineKey] = []
        ptrs: List[List[int]] = []
        sizes: List[List[int]] = []
        owners: dict[int, torch.Tensor] = {}
        group_ends: dict[int, int] = {}
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
            full = candidates[group]
            full = [
                item
                for item in full
                if item[2].to_string() not in state.pending_keys
            ]
            if full:
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
                if planned is None:
                    self.wait_for_direct_stores((req_id,))
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
                    continue
                group_ptrs, group_sizes, group_owners = planned
                max_buffers = int(
                    self.config.get_extra_config_value(
                        "mooncake_direct_max_buffers_per_page", 4096
                    )
                )
                if any(len(page_ptrs) > max_buffers for page_ptrs in group_ptrs):
                    self.wait_for_direct_stores((req_id,))
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
                    continue
                merged_runs += sum(
                    len(page_ptrs) // len(group_owners)
                    for page_ptrs in group_ptrs
                )
                self._record_live_source_pages(
                    req_id,
                    group,
                    list(zip(starts, ends, strict=True)),
                    group_ptrs,
                    group_sizes,
                    tuple(group_owners),
                )
                keys.extend(key for _, _, key in full)
                ptrs.extend(group_ptrs)
                sizes.extend(group_sizes)
                owners.update((id(owner), owner) for owner in group_owners)
                # Existing pages between missing pages are part of the same
                # contiguous committed prefix once this batch succeeds.
                group_ends[group] = candidate_ends[group]

        if started is not None:
            cold_start_perf_log(
                logger,
                "direct_npu_plan",
                started=started,
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
                submitted_ends=dict(state.submitted_end),
                committed_ends=dict(state.committed_end),
            )
        if keys:
            ready_event = torch.npu.Event()
            ready_event.record()
            batch = _DirectPageBatch(
                req_id,
                keys,
                ptrs,
                sizes,
                tuple(owners.values()),
                ready_event,
                group_ends,
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
            started = cold_start_perf_now() if cold_start_perf_enabled() else None
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
                            "for %s; waiting for the uncancellable native read "
                            "before releasing KV blocks",
                            req_id,
                        )
                        future.result()
                except Exception as caught:
                    error = caught
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
            if state is not None and not state.futures and not state.pending_keys:
                self._direct_store_states.pop(req_id, None)
            self._live_source_builders.pop(req_id, None)
            self._completed_live_sources.pop(req_id, None)

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
        new_tensors: List[torch.Tensor] = [
            mem_obj.tensor
            for mem_obj in layer_memory_objs
            if mem_obj.tensor is not None
        ]
        if new_tensors and cached_chunk_dev_ptrs is not None:
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
                    new_tensors,
                    cached_chunk_dev_ptrs,
                    cached_chunk_ptrs_npu,
                )

        if cached_tensors is None:
            return
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
            for chunk_index, mem_obj in enumerate(mem_objs_layer):
                tensor = mem_obj.tensor
                if tensor is None:
                    raise ValueError(
                        "Layerwise sparse retrieve resolved a chunk without a "
                        "tensor; refusing to shift the direct-load pointer order: "
                        f"layer_id={layer_id}, chunk_index={chunk_index}"
                    )
                new_tensors.append(tensor)
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
        if cached_tensors is not None and not cached_tensors:
            cached_tensors.extend([] for _ in range(num_layers))
        if cached_chunk_dev_ptrs is not None and not cached_chunk_dev_ptrs:
            cached_chunk_dev_ptrs.extend([] for _ in range(num_layers))
        if cached_chunk_ptrs_npu is not None and not cached_chunk_ptrs_npu:
            cached_chunk_ptrs_npu.extend(None for _ in range(num_layers))

        for layer_id, layer_memory_objs in enumerate(memory_objs):
            tensors = []
            for chunk_index, memory_obj in enumerate(layer_memory_objs):
                tensor = memory_obj.tensor
                if tensor is None:
                    raise ValueError(
                        "Dense group store cannot publish a MemoryObj without "
                        f"a tensor at layer={layer_id}, chunk={chunk_index}."
                    )
                tensors.append(tensor)
            host_row = host_pointer_rows[layer_id]
            if len(host_row) != len(tensors):
                raise ValueError("Dense group store pointer count mismatch.")
            if cached_tensors is not None:
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
    ) -> tuple[list[list[MemoryObj]], int]:
        """Resolve exact-size full or partial layer pages with legacy fallback."""
        if not 0 < page_chunks <= len(keys_layer_major[0]):
            raise ValueError("Layer-page retrieval requires at least one chunk")
        page_keys = [
            key
            for key in keys_layer_major[0][:page_chunks]
            if isinstance(key, LayerCacheEngineKey)
        ]
        if len(page_keys) != page_chunks:
            raise ValueError("Layer-page retrieval requires layer cache keys")

        local = self._shared_local_cpu_backend()
        assert self.storage_manager is not None
        pages, local_count = local.batched_get_layer_page_prefix(
            [key.without_layer() for key in page_keys]
        )
        owned: list[MemoryObj] = list(pages)
        pinned: list[MemoryObj] = []
        try:
            legacy_suffix = local_count < page_chunks and local.contains_any_exact(
                [layer[local_count] for layer in keys_layer_major]
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

            tail = (
                self._resolve_shared_rank0_page_first_layers(
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    keys_layer_major=[
                        layer[tail_start:] for layer in keys_layer_major
                    ],
                )
                if tail_start < len(keys_layer_major[0])
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
        page_keys = [key.without_layer() for _, _, key in chunks]
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("Live split import contains duplicate page keys")
        starts = [start for start, _, _ in chunks]
        ends = [end for _, end, _ in chunks]
        valid_tokens = [end - start for start, end, _ in chunks]
        pages: list[LayerPageMemoryObj] = []
        if 0 in handled_groups:
            shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
                kv_group=0, num_tokens=self.config.chunk_size
            )
            local = self._shared_local_cpu_backend()
            allocated = local.batched_allocate_layer_pages(
                [shape], [dtype], len(chunks), self.num_layers, fmt,
                valid_tokens=valid_tokens,
                full_tokens=self.config.chunk_size,
            )
            if allocated is None or not LayerPageMemoryObj.pin_many(allocated):
                raise MemoryError("Live split shared-CPU page allocation failed")
            pages = allocated
        destination_planner = getattr(
            self.gpu_connector, "plan_direct_page_destinations", None
        )
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
            dst1_ptrs, dst1_sizes, dst1_owners = destination1

            segments: list[dict[str, Any]] = []
            totals = [0, 0]
            for page in pages:
                size = page.get_size()
                segments.append({
                    "group_id": 0,
                    "destination_address": page.data_ptr,
                    "length": size,
                    "destination_kind": "cpu",
                })
                totals[0] += size
            for dest_ptrs, dest_sizes in zip(
                dst1_ptrs, dst1_sizes, strict=True
            ):
                for dst_ptr, size in zip(dest_ptrs, dest_sizes, strict=True):
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

    def _commit_live_split_import(self, context: dict[str, Any]) -> dict[str, Any]:
        """Publish and return request-owned group-0 sparse source metadata."""
        pages = context["pages"]
        if not pages:
            raise RuntimeError("Live split import has no latent pages to commit")
        self._shared_local_cpu_backend().batched_submit_layer_pages(
            context["keys"], pages
        )
        cache: dict[str, Any] = {
            "cached_keys": [
                [key.get_layer(layer_id) for key in context["keys"]]
                for layer_id in range(self.num_layers)
            ],
            "cached_starts": list(context["starts"]),
            "cached_ends": list(context["ends"]),
            "cached_memory_objs": [],
            "cached_tensors": [],
            "cached_chunk_dev_ptrs": [],
            "cached_chunk_ptrs_npu": [],
            "cached_shared_handles": [],
        }
        self._append_retrieve_group_cache(
            [
                LayerPageSource(tuple(pages), layer_id)
                for layer_id in range(self.num_layers)
            ],
            cache["cached_memory_objs"],
            cache["cached_tensors"],
            cache["cached_chunk_dev_ptrs"],
            cache["cached_chunk_ptrs_npu"],
        )
        context["pages"] = []
        return cache

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
                for chunk_index, mem_obj in enumerate(mem_objs_layer):
                    tensor = (
                        mem_obj.layer_tensor(source.layer_id)
                        if isinstance(source, LayerPageSource)
                        and chunk_index < len(source.pages)
                        else mem_obj.tensor
                    )
                    if tensor is None:
                        raise ValueError(
                            "Layerwise sparse retrieve resolved a chunk without "
                            f"a tensor: layer_id={layer_id}, chunk={chunk_index}"
                        )
                    tensors.append(tensor)
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
        group_append(
            mem_objs_by_layer if pointer_first else tensors_by_layer,
            cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu,
        )
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
        kv_group=1)``.  Generic LMCache metadata can still describe only the
        latent group in some startup/passive paths, so using it first makes
        passive ranks reject valid index handles as latent-shaped objects.
        """
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
                    current_location = "mixed"
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
            kv_shape_single_layer = self.gpu_connector.get_shape(
                num_tokens, kv_group=kv_group
            )
            memory_format = self._memory_format_for_kv_group(kv_group)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=memory_format,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
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
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
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
                    bool(kwargs.get("decode_window_save"))
                    and callable(group_store)
                    and callable(supports_group_store)
                    and supports_group_store(kv_group)
                    and all_chunks_publishable
                )
                if use_group_store:
                    for _ in range(self.num_layers):
                        yield
                    host_pointer_rows, layer_chunk_ptrs_npu = group_store(
                        memory_objs, starts, ends, **kwargs
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
                    flattened_keys = [key for layer_keys in keys for key in layer_keys]
                    flattened_memory_objs = [
                        obj for layer_objs in memory_objs for obj in layer_objs
                    ]
                    required_futures = self.storage_manager.batched_put(
                        flattened_keys,
                        flattened_memory_objs,
                        location=self.store_location,
                    )
                    self._track_sync_store_futures(
                        required_futures, require_completion=True
                    )
                    for mem_obj in flattened_memory_objs:
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
        if not cached_prefix_chunks and (
            cached_chunk_dev_ptrs is not None
            or cached_chunk_ptrs_npu is not None
        ):
            # The consumer needs transient pointers for the transfer, while
            # the cache engine publishes owners/host/NPU rows atomically.
            transfer_kwargs = dict(kwargs)
            transfer_kwargs["cached_chunk_dev_ptrs"] = None
            transfer_kwargs["cached_chunk_ptrs_npu"] = None
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
        legacy_tail_objects = 0
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
                )
                if pointer_started:
                    pointer_seal_ms = round(
                        (cold_start_perf_now() - pointer_started) * 1000, 3
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
        direct_group1_load = False
        if (
            kv_group == 1
            and not shared_sparse_retrieve
            and not use_cached_retrieve
            and cached_prefix_chunks == 0
            and mooncake_layer_pages_enabled(self.config)
            and missing_keys
            and all(len(layer) == len(missing_keys[0]) for layer in missing_keys)
        ):
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
        if (
            not use_cached_retrieve
            and not shared_sparse_retrieve
            and not direct_group1_load
        ):
            get_generator = self.storage_manager.layerwise_batched_get(
                missing_keys, location=location
            )

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

        if preflight_error_envelope is not None:
            release_pre_resolved_shared_mem_layers()
            release_local_prefix_layers()
            yield ret_mask
            self._broadcast_shared_envelope(preflight_error_envelope)
            assert preflight_error is not None
            raise preflight_error

        pending_pre_resolved_release: List[MemoryObj] = []
        if (
            shared_sparse_retrieve
            and pre_resolved_shared_mem_layers is not None
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
                    if shared_sparse_retrieve:
                        assert pre_resolved_shared_mem_layers is not None
                        mem_objs_layer = pre_resolved_shared_mem_layers[layer_id]
                    else:
                        assert get_generator is not None
                        task = next(get_generator)
                        assert task is not None
                        mem_objs_layer = task.result()
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
                        if cached_mem_layers is not None:
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

    def close(self) -> None:
        """Stop the bg worker gracefully, then close the base engine."""
        self.wait_for_direct_stores(tuple(self._direct_store_states))
        self._direct_store_states.clear()
        self._live_source_builders.clear()
        self._completed_live_sources.clear()
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
