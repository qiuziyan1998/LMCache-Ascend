# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
import hashlib
import json
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
import os
import queue
import threading
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    convert_tokens_to_list,
)
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import TokenDatabase
import torch

logger = init_logger(__name__)

LOCAL_CPU_BACKEND_NAME = "LocalCPUBackend"

LOCAL_CPU_BACKEND_NAME = "LocalCPUBackend"


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


def _dsa_validate_kv_enabled() -> bool:
    return os.environ.get("LMCACHE_ASCEND_DSA_VALIDATE_KV", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dsa_validate_dump_dir() -> Optional[str]:
    value = os.environ.get("LMCACHE_ASCEND_DSA_VALIDATE_DUMP_DIR", "").strip()
    return value or None


def _dsa_validate_dump_tensors() -> bool:
    return os.environ.get("LMCACHE_ASCEND_DSA_VALIDATE_DUMP_TENSORS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dsa_validate_max_records() -> int:
    try:
        return max(
            1,
            int(os.environ.get("LMCACHE_ASCEND_DSA_VALIDATE_MAX_RECORDS", "200000")),
        )
    except ValueError:
        return 200000


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
        setattr(owner, "_dsa_shrink_debug_counts", counts)
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
    worker thread while drains happen on the main thread — a data race
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
        self._dsa_validate_kv_digests: Dict[str, Dict[str, Any]] = {}
        self._dsa_validate_kv_order: List[str] = []

        if self.kv_events_enabled and self.is_store_async:
            self.kv_events = ThreadSafeEventList()

    def _dsa_validate_key(self, layer_id: int, key: Any) -> str:
        return f"layer={layer_id}|key={repr(key)}"

    def _dsa_validate_tensor_digest(self, tensor: torch.Tensor) -> Dict[str, Any]:
        cpu_tensor = tensor.detach()
        if cpu_tensor.device.type != "cpu":
            cpu_tensor = cpu_tensor.to(device="cpu")
        cpu_tensor = cpu_tensor.contiguous()
        raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
        return {
            "shape": tuple(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "numel": int(cpu_tensor.numel()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sample": cpu_tensor.reshape(-1)[:8].tolist()
            if cpu_tensor.numel() > 0
            else [],
        }

    def _dsa_validate_event(self, event: str, payload: Dict[str, Any]) -> None:
        dump_dir = _dsa_validate_dump_dir()
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(
                dump_dir,
                f"dsa_kv_validate_worker_{self.metadata.worker_id}.jsonl",
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": event, **payload}, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write DSA KV validation event")

    def _dsa_validate_tensor_dump(
        self,
        *,
        req_id: str,
        layer_id: int,
        key: Any,
        tensor: torch.Tensor,
        phase: str,
    ) -> None:
        if not _dsa_validate_dump_tensors():
            return
        dump_dir = _dsa_validate_dump_dir()
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            key_hash = hashlib.sha1(
                self._dsa_validate_key(layer_id, key).encode("utf-8")
            ).hexdigest()
            path = os.path.join(
                dump_dir,
                (
                    f"dsa_kv_{phase}_worker_{self.metadata.worker_id}_"
                    f"layer_{layer_id}_{key_hash}.pt"
                ),
            )
            cpu_tensor = tensor.detach()
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.to(device="cpu")
            torch.save(
                {
                    "req_id": req_id,
                    "layer_id": layer_id,
                    "key": repr(key),
                    "tensor": cpu_tensor.contiguous(),
                },
                path,
            )
        except Exception:
            logger.exception("Failed to dump DSA KV validation tensor")

    def _dsa_record_store_digests(
        self,
        *,
        req_id: str,
        layer_id: int,
        keys: List[Any],
        memory_objs: List[MemoryObj],
    ) -> None:
        if not _dsa_validate_kv_enabled():
            return
        max_records = _dsa_validate_max_records()
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            tensor = memory_obj.tensor
            if tensor is None:
                self._dsa_validate_event(
                    "store_missing_tensor",
                    {
                        "req_id": req_id,
                        "layer_id": layer_id,
                        "key": repr(key),
                    },
                )
                logger.error(
                    "[DSA_SHRINK_VALIDATE_FAIL] store_missing_tensor "
                    "req=%s layer=%s key=%s",
                    req_id,
                    layer_id,
                    key,
                )
                continue
            try:
                digest = self._dsa_validate_tensor_digest(tensor)
            except Exception as exc:
                logger.exception(
                    "[DSA_SHRINK_VALIDATE_FAIL] store_digest_failed "
                    "req=%s layer=%s key=%s error=%s",
                    req_id,
                    layer_id,
                    key,
                    exc,
                )
                continue
            record_key = self._dsa_validate_key(layer_id, key)
            old = self._dsa_validate_kv_digests.get(record_key)
            payload = {
                "req_id": req_id,
                "layer_id": layer_id,
                "key": repr(key),
                "digest": digest,
                "old_digest": old,
            }
            if old is not None and old["digest"]["sha256"] != digest["sha256"]:
                self._dsa_validate_event("store_digest_changed", payload)
                logger.error(
                    "[DSA_SHRINK_VALIDATE_FAIL] store_digest_changed "
                    "req=%s layer=%s key=%s old_sha=%s new_sha=%s",
                    req_id,
                    layer_id,
                    key,
                    old["digest"]["sha256"],
                    digest["sha256"],
                )
            self._dsa_validate_kv_digests[record_key] = {
                "req_id": req_id,
                "layer_id": layer_id,
                "key": repr(key),
                "digest": digest,
            }
            self._dsa_validate_kv_order.append(record_key)
            while len(self._dsa_validate_kv_order) > max_records:
                stale = self._dsa_validate_kv_order.pop(0)
                self._dsa_validate_kv_digests.pop(stale, None)
            self._dsa_validate_event("store_digest", payload)
            self._dsa_validate_tensor_dump(
                req_id=req_id,
                layer_id=layer_id,
                key=key,
                tensor=tensor,
                phase="store",
            )

    def _dsa_check_retrieve_digests(
        self,
        *,
        req_id: str,
        layer_id: int,
        keys: List[Any],
        tensors: List[torch.Tensor],
    ) -> None:
        if not _dsa_validate_kv_enabled():
            return
        for key, tensor in zip(keys, tensors, strict=False):
            record_key = self._dsa_validate_key(layer_id, key)
            expected = self._dsa_validate_kv_digests.get(record_key)
            if expected is None:
                payload = {
                    "req_id": req_id,
                    "layer_id": layer_id,
                    "key": repr(key),
                }
                self._dsa_validate_event("retrieve_digest_missing", payload)
                logger.error(
                    "[DSA_SHRINK_VALIDATE_FAIL] retrieve_digest_missing "
                    "req=%s layer=%s key=%s",
                    req_id,
                    layer_id,
                    key,
                )
                continue
            try:
                actual = self._dsa_validate_tensor_digest(tensor)
            except Exception as exc:
                logger.exception(
                    "[DSA_SHRINK_VALIDATE_FAIL] retrieve_digest_failed "
                    "req=%s layer=%s key=%s error=%s",
                    req_id,
                    layer_id,
                    key,
                    exc,
                )
                continue
            if actual["sha256"] != expected["digest"]["sha256"]:
                payload = {
                    "req_id": req_id,
                    "layer_id": layer_id,
                    "key": repr(key),
                    "expected": expected,
                    "actual": actual,
                }
                self._dsa_validate_event("retrieve_digest_mismatch", payload)
                logger.error(
                    "[DSA_SHRINK_VALIDATE_FAIL] retrieve_digest_mismatch "
                    "req=%s layer=%s key=%s expected_sha=%s actual_sha=%s "
                    "expected_sample=%s actual_sample=%s",
                    req_id,
                    layer_id,
                    key,
                    expected["digest"]["sha256"],
                    actual["sha256"],
                    expected["digest"].get("sample"),
                    actual.get("sample"),
                )
                self._dsa_validate_tensor_dump(
                    req_id=req_id,
                    layer_id=layer_id,
                    key=key,
                    tensor=tensor,
                    phase="retrieve",
                )

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
        if self.is_store_async:
            self._device_id = torch.npu.current_device()
            self._ensure_store_worker()
            queue_mode = "unbounded" if self._store_queue_maxsize == 0 else "bounded"
            logger.info(
                "Ascend async store queue initialized: mode=%s maxsize=%d",
                queue_mode,
                self._store_queue_maxsize,
            )

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
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

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
                    self.storage_manager.batched_put(
                        keys,
                        memory_objs,
                        transfer_spec=transfer_spec,
                        location=self.store_location,
                    )
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
                    # Already reported — skip to avoid scheduler seeing
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
        if not self.is_store_async:
            return set()

        req_id_set = set(req_ids)
        if not req_id_set:
            return set()

        with self._store_cv:
            pending_at_start = {
                req_id for req_id in req_id_set if req_id in self._pending_store_reqs
            }
            if not pending_at_start:
                return set()

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
        return pending_at_start

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
    ) -> None:
        """Append one store_layer batch; never clear (chunked prefill calls many times)."""
        num_layers = len(keys)
        if cached_starts is not None:
            cached_starts.extend(starts)
        if cached_ends is not None:
            cached_ends.extend(ends)
        if cached_keys is not None:
            if not cached_keys:
                cached_keys.extend([] for _ in range(num_layers))
            assert len(cached_keys) == num_layers, (
                f"cached_keys has {len(cached_keys)} layers, expected {num_layers}"
            )
            for layer_id, layer_keys in enumerate(keys):
                cached_keys[layer_id].extend(layer_keys)
        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend([] for _ in range(num_layers))
            assert len(cached_memory_objs) == num_layers
            for layer_id, layer_objs in enumerate(memory_objs):
                cached_memory_objs[layer_id].extend(layer_objs)
        if cached_tensors is not None and not cached_tensors:
            cached_tensors.extend([] for _ in range(num_layers))

    def _append_layer_store_tensors(
        self,
        layer_id: int,
        memory_objs: List[List[MemoryObj]],
        cached_tensors: Optional[List],
    ) -> None:
        if cached_tensors is None:
            return
        while len(cached_tensors) <= layer_id:
            cached_tensors.append([])
        cached_tensors[layer_id].extend(
            mem_obj.tensor
            for mem_obj in memory_objs[layer_id]
            if mem_obj.tensor is not None
        )

    def _append_retrieve_layer_cache(
        self,
        layer_id: int,
        mem_objs_layer: List[MemoryObj],
        cached_memory_objs: Optional[List],
        cached_tensors: Optional[List],
        cached_chunk_dev_ptrs: Optional[List],
        cached_chunk_ptrs_npu: Optional[List],
    ) -> None:
        """Retain storage-get results in ReqMeta for later retrieve calls (same request)."""
        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend(
                    [] for _ in range(self.num_layers)
                )
            cached_memory_objs[layer_id].extend(mem_objs_layer)
        new_tensors: List[torch.Tensor] = []
        if cached_tensors is not None:
            if not cached_tensors:
                cached_tensors.extend([] for _ in range(self.num_layers))
            for mem_obj in mem_objs_layer:
                if mem_obj.tensor is not None:
                    cached_tensors[layer_id].append(mem_obj.tensor)
                    new_tensors.append(mem_obj.tensor)
        if new_tensors and cached_chunk_dev_ptrs is not None:
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
    def _has_retrieve_data_cache(
        cached_tensors: Optional[List],
        cached_memory_objs: Optional[List],
        num_layers: int,
    ) -> bool:
        if cached_tensors is not None and len(cached_tensors) == num_layers:
            return any(cached_tensors)
        if cached_memory_objs is not None and len(cached_memory_objs) == num_layers:
            return any(cached_memory_objs)
        return False

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
    def _min_layer_cache_chunks(
        cache_by_layer: Optional[List],
        num_layers: int,
    ) -> int:
        if cache_by_layer is None or len(cache_by_layer) != num_layers:
            return 0
        min_chunks: Optional[int] = None
        for layer_id in range(num_layers):
            layer_cache = cache_by_layer[layer_id]
            layer_chunks = 0 if layer_cache is None else len(layer_cache)
            min_chunks = (
                layer_chunks
                if min_chunks is None
                else min(min_chunks, layer_chunks)
            )
        return 0 if min_chunks is None else min_chunks

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
        lazy_init = getattr(
            self.gpu_connector, "_lazy_initialize_buffer", None
        )
        if not callable(init_kvcaches) or not callable(lazy_init):
            return
        if "kvcaches" in kwargs:
            init_kvcaches(**kwargs)
        kvcaches = getattr(self.gpu_connector, "kvcaches", None)
        if kvcaches is not None:
            kv_group = kwargs.get("kv_group", 0)
            # The layerwise NPU connector detects format per kv_group; other
            # connectors (e.g. the blending buffer connector) may not accept
            # the kwarg, so fall back to the positional call for them.
            try:
                lazy_init(kvcaches, kv_group=kv_group)
            except TypeError:
                lazy_init(kvcaches)

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
        """Resolve retrieve chunk metadata once per request; reuse ReqMeta cache after."""
        if self._needs_retrieve_metadata_refresh(
            cached_keys, cached_starts, cached_ends, tokens
        ):
            if retrieve_kwargs is not None:
                retrieve_kwargs.pop("_retrieve_metadata_warm", None)

            kv_group = 0
            if retrieve_kwargs is not None:
                kv_group = int(retrieve_kwargs.get("kv_group", 0) or 0)

            location: Optional[str] = None
            new_starts: List[int] = []
            new_ends: List[int] = []
            keys: List[List[CacheEngineKey]] = []

            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
                kv_group=kv_group,
            ):
                assert isinstance(key, CacheEngineKey)

                keys_multi_layer = key.split_layers(self.num_layers)
                if current_location := self._layerwise_chunk_location(
                    keys_multi_layer
                ):
                    if location is None:
                        location = current_location
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

            new_keys = (
                [list(row) for row in zip(*keys, strict=False)] if keys else []
            )
            if not new_keys:
                return location, new_starts, new_ends, new_keys

            if not cached_keys:
                cached_keys[:] = new_keys
                cached_starts[:] = new_starts
                cached_ends[:] = new_ends
            else:
                prev_chunks = len(cached_starts)
                for chunk_idx, (start, end) in enumerate(
                    zip(new_starts, new_ends, strict=False)
                ):
                    if chunk_idx < prev_chunks:
                        if (
                            cached_starts[chunk_idx] == start
                            and cached_ends[chunk_idx] == end
                        ):
                            continue
                    cached_starts.append(start)
                    cached_ends.append(end)
                    for layer_id in range(self.num_layers):
                        cached_keys[layer_id].append(new_keys[layer_id][chunk_idx])

            ret_mask.zero_()
            for start, end in zip(cached_starts, cached_ends, strict=False):
                ret_mask[start:end] = True
            if retrieve_kwargs is not None:
                if location is not None:
                    retrieve_kwargs["cached_retrieve_location"] = location
                retrieve_kwargs["_retrieve_metadata_warm"] = True
            return location, cached_starts, cached_ends, cached_keys

        if retrieve_kwargs is not None and retrieve_kwargs.get(
            "_retrieve_metadata_warm"
        ):
            if retrieve_kwargs.get("_use_cached_retrieve"):
                location = self._resolve_local_cpu_retrieve_location(
                    retrieve_kwargs.get("cached_retrieve_location"),
                )
                if retrieve_kwargs is not None and location is not None:
                    retrieve_kwargs["cached_retrieve_location"] = location
                return location, cached_starts, cached_ends, cached_keys

            location = retrieve_kwargs.get("cached_retrieve_location")
            if cached_keys and cached_keys[0]:
                # Prefer the hottest tier (LocalCPUBackend is checked first).
                # After Mooncake write-back, stale cached_retrieve_location may
                # still point at RemoteBackend and force slow remote reads.
                preferred_location = self.storage_manager.contains(
                    cached_keys[0][0], self.retrieve_locations
                )
                if preferred_location is not None:
                    location = preferred_location
                    retrieve_kwargs["cached_retrieve_location"] = location
            return location, cached_starts, cached_ends, cached_keys

        for start, end in zip(cached_starts, cached_ends, strict=False):
            ret_mask[start:end] = True

        location = None
        if retrieve_kwargs is not None:
            location = retrieve_kwargs.get("cached_retrieve_location")
        if location is None and cached_keys and cached_keys[0]:
            location = self.storage_manager.contains(
                cached_keys[0][0], self.retrieve_locations
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
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        # Passive rank guard: when save_only_first_rank is enabled, only rank 0
        # stores. Closes the known TODO — store_layer had no _is_passive() check,
        # causing duplicate stores on non-rank-0 workers under MLA + layerwise.
        if self._is_passive():
            logger.debug(
                "Passive rank (save_only_first_rank), skipping store_layer"
            )
            for layer_id in range(self.num_layers):
                yield
            # Extra yield consumed by wait_for_save() after the last layer.
            yield
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

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
            return

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

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # Ensure the connector's MLA/DSA layout is detected before allocating
        # chunks -- get_shape(num_tokens) below depends on kv_lora_rank etc.
        self._ensure_layerwise_connector_layout(**kwargs)

        prev_key = 0
        kv_group = kwargs.get("kv_group", 0)
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
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

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self._memory_format_for_kv_group(kv_group),
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

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

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

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
            )

            t_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self._append_layer_store_tensors(
                    layer_id, memory_objs, cached_tensors
                )
                self._dsa_record_store_digests(
                    req_id=req_id,
                    layer_id=layer_id,
                    keys=keys[layer_id],
                    memory_objs=memory_objs[layer_id],
                )
                self.storage_manager.batched_put(
                    keys[layer_id], memory_objs[layer_id], location=self.store_location
                )

            if kwargs.get("decode_window_save"):
                torch.npu.synchronize()

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
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    def retrieve_layer_head_token_wise(
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

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

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
            metadata_warm = False
        elif not metadata_warm:
            ret_mask.zero_()

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

        has_cached_retrieve_data = self._has_retrieve_data_cache(
            cached_tensors,
            cached_memory_objs,
            self.num_layers,
        )

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
        if _dsa_debug_should_log(self, "head_retrieve_metadata"):
            slot_mapping = kwargs.get("slot_mapping")
            logger.warning(
                "[DSA_SHRINK_CHECK] lmcache_ascend_head_retrieve "
                "req=%s num_tokens=%s mask_shape=%s mask_minmax_count=%s "
                "ret_mask_shape=%s ret_mask_minmax_count=%s slot_mapping_shape=%s "
                "slot_mapping_sample=%s slot_mapping_minmax_count=%s "
                "vllm_cached=%s lmcache_cached=%s metadata_warm=%s "
                "use_cached_retrieve=%s cached_tensors_layers=%s "
                "cached_mem_layers=%s cached_chunk_ptr_layers=%s "
                "retrieve_key_groups=%s first_layer_chunks=%s starts_sample=%s "
                "ends_sample=%s location=%s",
                kwargs.get("req_id"),
                num_tokens,
                _dsa_debug_shape(mask),
                _dsa_debug_minmax_count(mask),
                _dsa_debug_shape(ret_mask),
                _dsa_debug_minmax_count(ret_mask),
                _dsa_debug_shape(slot_mapping),
                _dsa_debug_sample(slot_mapping),
                _dsa_debug_minmax_count(slot_mapping),
                kwargs.get("vllm_cached_tokens"),
                kwargs.get("lmcache_cached_tokens"),
                metadata_warm,
                use_cached_retrieve,
                len(cached_tensors) if cached_tensors is not None else None,
                len(cached_memory_objs) if cached_memory_objs is not None else None,
                len(cached_chunk_ptrs_npu)
                if cached_chunk_ptrs_npu is not None else None,
                len(retrieve_keys) if retrieve_keys is not None else None,
                len(retrieve_keys[0]) if retrieve_keys else 0,
                _dsa_debug_sample(starts),
                _dsa_debug_sample(ends),
                location,
            )

        required_chunks = len(retrieve_keys[0]) if retrieve_keys else 0
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
        if has_cached_retrieve_data and required_chunks and not use_cached_retrieve:
            logger.warning(
                "Layerwise sparse retrieve cache is incomplete; falling back to "
                "storage get. required_chunks=%d cached_tensor_chunks=%d "
                "cached_memory_obj_chunks=%d",
                required_chunks,
                self._min_layer_cache_chunks(cached_tensors, self.num_layers),
                self._min_layer_cache_chunks(cached_memory_objs, self.num_layers),
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
        if not use_cached_retrieve:
            get_generator = self.storage_manager.layerwise_batched_get(
                retrieve_keys, location=location
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

        mem_obj_consumer = self.gpu_connector.batched_to_gpu_head_token_wise(**kwargs)
        notify_fn = getattr(self.gpu_connector, "notify_sparse_memory_objs_updated", None)
        if notify_fn is not None and not use_cached_retrieve:
            notify_fn()
        next(mem_obj_consumer)

        for layer_id in range(self.num_layers):
            sparse_request = yield ret_mask
            sparse_payload = None
            target_slot_mapping = None
            if isinstance(sparse_request, dict):
                sparse_payload = sparse_request
                selected_tokens = sparse_payload.get("selected_token_ids")
                token_start_index = sparse_payload.get("token_start_index")
            elif isinstance(sparse_request, tuple):
                if len(sparse_request) == 3:
                    (
                        selected_tokens,
                        token_start_index,
                        target_slot_mapping,
                    ) = sparse_request
                else:
                    selected_tokens, token_start_index = sparse_request
            else:
                selected_tokens = sparse_request
                token_start_index = 0

            if cached_mem_layers is not None:
                mem_objs_layer = cached_mem_layers[layer_id]
            elif use_cached_retrieve:
                mem_objs_layer = []
            else:
                assert get_generator is not None
                task = next(get_generator)
                assert task is not None
                mem_objs_layer = task.result()
                if mem_objs_layer is not None:
                    layer_cached_chunks = (
                        len(cached_tensors[layer_id])
                        if cached_tensors is not None
                        and len(cached_tensors) > layer_id
                        else 0
                    )
                    if layer_cached_chunks < len(retrieve_keys[0]):
                        self._append_retrieve_layer_cache(
                            layer_id,
                            mem_objs_layer,
                            cached_memory_objs,
                            cached_tensors,
                            cached_chunk_dev_ptrs,
                            cached_chunk_ptrs_npu,
                        )
                else:
                    mem_objs_layer = []

            if sparse_payload is not None:
                sparse_payload["memory_objs_layer"] = mem_objs_layer
                mem_obj_consumer.send(sparse_payload)
            else:
                mem_obj_consumer.send(
                    (
                        mem_objs_layer,
                        selected_tokens,
                        token_start_index,
                        target_slot_mapping,
                    )
                )

        next(mem_obj_consumer)

        yield ret_mask


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
