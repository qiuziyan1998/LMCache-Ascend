# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
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

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
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
                    fmt=self.fmt,
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
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
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

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Layerwise retrieve; skips post-sync unpin for LocalCPU (#2954)."""
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts: List[int] = []
        ends: List[int] = []
        keys: List[List[CacheEngineKey]] = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            if current_location := self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
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

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        to_count_down: List[MemoryObj] = []
        if keys:
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            for layer_id in range(self.num_layers):
                task = next(get_generator)
                assert task is not None

                if layer_id == 0:
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = task.result()
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

        else:
            for _layer_id in range(self.num_layers):
                yield None

        yield None

        if keys:
            next(mem_obj_consumer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()

            # LocalCPU hot-cache objects are unpinned by lookup_unpin() in
            # wait_for_save().  batched_get returns the same object that lookup
            # pinned; unpinning here as well drives pin_count negative (#2954).
            for mem_obj in to_count_down:
                if mem_obj.is_pinned and location != "LocalCPUBackend":
                    mem_obj.unpin()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield ret_mask

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
    ) -> None:
        """Retain storage-get results in ReqMeta for later retrieve calls (same request)."""
        if cached_memory_objs is not None:
            if not cached_memory_objs:
                cached_memory_objs.extend(
                    [] for _ in range(self.num_layers)
                )
            cached_memory_objs[layer_id].extend(mem_objs_layer)
        if cached_tensors is not None:
            if not cached_tensors:
                cached_tensors.extend([] for _ in range(self.num_layers))
            cached_tensors[layer_id].extend(
                mem_obj.tensor
                for mem_obj in mem_objs_layer
                if mem_obj.tensor is not None
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
        return (
            cached_tensors is not None
            and len(cached_tensors) == num_layers
        ) or (
            cached_memory_objs is not None
            and len(cached_memory_objs) == num_layers
        )

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
            location: Optional[str] = None
            new_starts: List[int] = []
            new_ends: List[int] = []
            keys: List[List[CacheEngineKey]] = []

            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)

                keys_multi_layer = key.split_layers(self.num_layers)
                if current_location := self.storage_manager.contains(
                    keys_multi_layer[0], self.retrieve_locations
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
            if retrieve_kwargs is not None and location is not None:
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

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
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
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
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
                self.storage_manager.batched_put(
                    keys[layer_id], memory_objs[layer_id], location=self.store_location
                )

            tot_time = time.perf_counter() - t_start
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
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
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

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

        if not retrieve_keys:
            # If no cache are found, we still need to yield to avoid `StopIteration`
            for layer_id in range(self.num_layers):
                yield None
            # synchronize the last layer
            if not mem_obj_consumer:
                mem_obj_consumer = (x for x in range(self.num_layers))  
            next(mem_obj_consumer)
            yield ret_mask
            return

        assert_layerwise_gpu_connector(self.gpu_connector)

        use_cached_retrieve = self._has_retrieve_data_cache(
            cached_tensors,
            cached_memory_objs,
            self.num_layers,
        )

        get_generator = None
        if not use_cached_retrieve:
            get_generator = self.storage_manager.layerwise_batched_get(
                retrieve_keys, location=location
            )

        if use_cached_retrieve and cached_tensors is not None:
            kwargs["cached_tensors"] = cached_tensors

        cached_mem_layers = (
            cached_memory_objs
            if use_cached_retrieve
            and cached_memory_objs is not None
            and len(cached_memory_objs) == self.num_layers
            else None
        )

        mem_obj_consumer = self.gpu_connector.batched_to_gpu_head_token_wise(**kwargs)
        next(mem_obj_consumer)

        for layer_id in range(self.num_layers):
            selected_tokens, token_start_index = yield ret_mask

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
                        )
                else:
                    mem_objs_layer = []

            mem_obj_consumer.send(
                (mem_objs_layer, selected_tokens, token_start_index)
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
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
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
