# SPDX-License-Identifier: Apache-2.0
"""Benchmark sparse decode cold bootstrap metadata and page-first retrieval.

This benchmark drives ``AscendLMCacheEngine.retrieve_layer_head_token_wise``
through the same cold generator protocol used by the vLLM adapter:

1. build token/chunk/layer metadata;
2. probe the shared LocalCPU cache;
3. run the shared-slab capacity preflight;
4. retrieve all missing layers from a synthetic Mooncake page store;
5. install pointer tables and publish shared handles;
6. seal a prepared source and execute one metadata-free warm retrieve.

It also runs latent and DSA-index bootstrap as a paired request, comparing
independent token hashing with request-local reuse of the latent chunk plan.

``--cold-compact-ab`` executes both production generator protocols. The
disabled case densely consumes both KV groups. The enabled case matches
``_run_dsa_cold_compact_load``: materialize latent, densely consume the indexer,
then synchronize and resume. H2D, per-layer consumer, and barrier delays are
synthetic and configurable; the outer protocol wall is measured directly.
Model compute and compute/H2D overlap are intentionally outside this LMCache
cold-bootstrap benchmark.

The remote store is synthetic so CPU metadata can be measured independently
from a particular Mooncake deployment. By default, allocation, ``MemoryObj``
construction, and LRU insertion use the production LMCache implementations;
``--object-mode synthetic`` retains the lightweight metadata-only model. Page
accounting matches ``mooncake_page_first_multi_buffer=true``: full token
chunks use one multi-buffer Mooncake page, while an unfull tail chunk falls
back to one legacy file per layer. Every layer remains a separate LMCache
``MemoryObj`` and destination buffer.
"""

# Standard
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, cast
import json
import math
import statistics
import sys
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_page_key
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
    LocalCPUPrefixGetResult,
)
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


GB = 1_000_000_000
_RESOLVER_EMITTED_STAGES = (
    "windows",
    "remote_get",
    "results",
    "classification",
    "pinning",
    "scatter",
    "rollback",
)
_RESOLVER_STAGE_FIELDS = {
    stage: f"resolver_{stage}_s"
    for stage in (*_RESOLVER_EMITTED_STAGES, "validation")
}
_RESOLVER_CPU_STAGES = (
    "windows",
    "results",
    "classification",
    "pinning",
    "validation",
    "scatter",
)


@dataclass
class SyntheticTokenDatabaseConfig:
    """Version-independent config subset required by ``ChunkedTokenDatabase``.

    The benchmark can run against several LMCache revisions. Constructing a
    complete config through ``LMCacheEngineConfig.from_legacy`` makes it depend
    on every legacy config field having a matching dataclass field, even though
    the synthetic token database only consumes the fields below.
    """

    chunk_size: int
    pre_caching_hash_algorithm: str = "builtin"
    save_unfull_chunk: bool = True
    dsa_two_groups: bool = True
    remote_url: Optional[str] = None
    enable_pd: bool = False
    extra_config: dict[str, Any] = field(
        default_factory=lambda: {"save_only_first_rank": True}
    )

    def get_extra_config_value(
        self,
        key: str,
        default_value: Any = None,
    ) -> Any:
        return self.extra_config.get(key, default_value)


@dataclass
class StageStats:
    """Stage timings and operation counts collected from one bootstrap."""

    metadata_s: float = 0.0
    token_process_s: float = 0.0
    page_key_s: float = 0.0
    local_probe_s: float = 0.0
    capacity_s: float = 0.0
    resolver_s: float = 0.0
    resolver_cpu_s: float = 0.0
    resolver_windows_s: float = 0.0
    resolver_remote_get_s: float = 0.0
    resolver_results_s: float = 0.0
    resolver_classification_s: float = 0.0
    resolver_pinning_s: float = 0.0
    resolver_validation_s: float = 0.0
    resolver_scatter_s: float = 0.0
    resolver_rollback_s: float = 0.0
    resolver_unattributed_s: float = 0.0
    prime_other_s: float = 0.0
    remote_s: float = 0.0
    remote_layout_s: float = 0.0
    remote_lookup_s: float = 0.0
    remote_buffer_s: float = 0.0
    remote_wrapper_s: float = 0.0
    remote_pointer_s: float = 0.0
    remote_transfer_s: float = 0.0
    remote_scatter_s: float = 0.0
    remote_writeback_s: float = 0.0
    pointer_s: float = 0.0
    handle_s: float = 0.0
    cache_append_s: float = 0.0
    cache_append_cpu_s: float = 0.0
    handle_cache_s: float = 0.0
    broadcast_s: float = 0.0
    fence_s: float = 0.0
    layer_other_s: float = 0.0
    cold_prime_s: float = 0.0
    cold_layers_s: float = 0.0
    cold_close_s: float = 0.0
    cold_total_s: float = 0.0
    warm_total_s: float = 0.0
    device_consumer_s: float = 0.0
    device_h2d_s: float = 0.0
    device_consumer_layers: int = 0
    device_bytes: int = 0
    local_probe_calls: int = 0
    local_key_probes: int = 0
    remote_calls: int = 0
    remote_lookup_calls: int = 0
    remote_transfer_calls: int = 0
    remote_logical_keys: int = 0
    mooncake_page_objects: int = 0
    lmcache_memory_objects: int = 0
    pointer_objects: int = 0
    physical_pages: int = 0
    logical_entries: int = 0
    legacy_tail_objects: int = 0
    handle_objects: int = 0
    broadcast_envelopes: int = 0
    transferred_bytes: int = 0
    hash_plan_reuses: int = 0


@dataclass
class BenchmarkResult:
    """Median result for one benchmark case."""

    name: str
    prefix_mode: str
    chunk_plan_mode: str
    object_mode: str
    kv_group: int
    num_layers: int
    num_chunks: int
    bytes_per_token: int
    repeats: int
    median: StageStats
    samples: list[StageStats] = field(default_factory=list)


@dataclass
class ColdCompactVariant:
    """Exclusive cold-start critical-path stages for one flag setting."""

    flag: str
    protocol: list[str]
    consumer_modes: dict[int, str]
    lookup_metadata_s: float
    mooncake_s: float
    mooncake_allocation_s: float
    mooncake_transfer_s: float
    mooncake_other_s: float
    resolver_cpu_s: float
    pointer_handle_s: float
    bootstrap_other_s: float
    device_bytes: float
    device_h2d_s: float
    device_consumer_s: float
    synchronization_barrier_s: float
    critical_path_s: float


@dataclass
class ColdCompactABReport:
    """Filtered A/B report for ``LMCACHE_ENABLE_DSA_COLD_COMPACT_LOAD``."""

    schema: int
    model: str
    parameters: dict[str, Any]
    variants: dict[str, ColdCompactVariant]
    comparison: dict[str, float]
    samples: list[dict[str, Any]]


def _record_resolver_stage(
    stats: StageStats,
    stage: str,
    elapsed_s: float,
) -> None:
    try:
        field_name = _RESOLVER_STAGE_FIELDS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown resolver timing stage: {stage}") from exc
    setattr(stats, field_name, getattr(stats, field_name) + elapsed_s)


def _resolver_cpu_attributed_s(stats: StageStats) -> float:
    return sum(
        getattr(stats, _RESOLVER_STAGE_FIELDS[stage])
        for stage in _RESOLVER_CPU_STAGES
    )


@dataclass
class ColdCompactProtocolSample:
    """One executable two-group protocol sample."""

    enabled: bool
    layer_merged: bool
    wall_s: float
    synchronization_barrier_s: float
    groups: dict[int, StageStats]


class SyntheticMemoryObj:
    """Small stand-in preserving bootstrap pin/ref/tensor semantics."""

    __slots__ = (
        "is_pinned",
        "logical_bytes",
        "ref_count",
        "tensor",
    )

    def __init__(self, logical_bytes: int, shared_tensor: torch.Tensor) -> None:
        self.logical_bytes = logical_bytes
        self.tensor = shared_tensor
        self.ref_count = 1
        self.is_pinned = False

    def pin(self) -> bool:
        self.is_pinned = True
        return True

    def unpin(self) -> bool:
        self.is_pinned = False
        return True

    def ref_count_up(self) -> None:
        if not self.is_valid():
            raise RuntimeError("cannot retain a released synthetic MemoryObj")
        self.ref_count += 1

    def ref_count_down(self) -> None:
        if self.ref_count <= 0:
            raise RuntimeError("synthetic MemoryObj reference underflow")
        self.ref_count -= 1

    def is_valid(self) -> bool:
        return self.ref_count > 0

    def get_physical_size(self) -> int:
        return self.logical_bytes

    def get_size(self) -> int:
        return self.logical_bytes

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()


class SyntheticAddressManager:
    """Capacity-only allocator view used by the real preflight method."""

    def __init__(self, free_bytes: int) -> None:
        self.free_bytes = free_bytes

    def get_free_size(self) -> int:
        return self.free_bytes


class SyntheticBuffer:
    """Avoid allocating a slab merely to expose its logical byte count."""

    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


class BenchmarkLocalCPUBackend:
    """LocalCPU hot-cache surface with measurable prefix lookup modes."""

    def __init__(
        self,
        *,
        stats: StageStats,
        slab_bytes: int,
        prefix_mode: str,
        object_mode: str,
        chunk_size: int,
    ) -> None:
        self.stats = stats
        self.prefix_mode = prefix_mode
        self.object_mode = object_mode
        self.use_hot = True
        self.cpu_lock = threading.RLock()
        if object_mode == "production":
            PinMonitor.GetOrCreate(
                SimpleNamespace(pin_check_interval_sec=60, pin_timeout_sec=18000)
            )
        self.cache_policy = (
            get_cache_policy("LRU")
            if object_mode == "production"
            else SimpleNamespace(
                update_on_put=lambda _key: None,
                update_on_put_many=lambda _keys: None,
            )
        )
        self.hot_cache = (
            self.cache_policy.init_mutable_mapping()
            if object_mode == "production"
            else {}
        )
        self.batched_msg_sender = None
        self.metadata = SimpleNamespace(chunk_size=chunk_size)
        root_allocator = (
            TensorMemoryAllocator(torch.empty(slab_bytes, dtype=torch.uint8))
            if object_mode == "production"
            else SimpleNamespace(
                buffer=SyntheticBuffer(slab_bytes),
                address_manager=SyntheticAddressManager(slab_bytes),
                align_bytes=4096,
            )
        )
        if object_mode == "production":
            root_allocator.shm_name = "/lmcache-benchmark"
            root_allocator.pin_allocator = root_allocator
        self.memory_allocator = root_allocator
        if prefix_mode == "per-layer":
            # The production code deliberately supports an older LMCache that
            # does not expose the cross-layer method.
            self.batched_get_prefixes_with_misses = None

    allocate = LocalCPUBackend.allocate
    batched_allocate = LocalCPUBackend.batched_allocate
    batched_allocate_layer_pages = LocalCPUBackend.batched_allocate_layer_pages
    batched_submit_layer_pages = LocalCPUBackend.batched_submit_layer_pages
    batched_get_layer_page_prefix = LocalCPUBackend.batched_get_layer_page_prefix
    contains_any_exact = LocalCPUBackend.contains_any_exact

    def close(self) -> None:
        """Release production benchmark objects without per-object free calls."""
        if self.object_mode != "production":
            return
        objects = list(self.hot_cache.values())
        for obj in objects:
            while obj.is_pinned:
                obj.unpin()
        self.memory_allocator.batched_free(objects)
        self.hot_cache.clear()

    def _record_probe(self, layers: Iterable[list[CacheEngineKey]]) -> None:
        layer_list = list(layers)
        self.stats.local_probe_calls += 1
        # A cold prefix stops at the first miss. A warm prefix examines all
        # present keys. Count the exact probes performed by this synthetic
        # cache rather than layer_count * chunk_count.
        for keys in layer_list:
            for key in keys:
                self.stats.local_key_probes += 1
                if key not in self.hot_cache:
                    break

    def batched_get_prefix_with_misses(
        self,
        keys: list[CacheEngineKey],
    ) -> LocalCPUPrefixGetResult:
        started = time.perf_counter()
        self._record_probe([keys])
        try:
            return LocalCPUBackend.batched_get_prefix_with_misses(self, keys)
        finally:
            self.stats.local_probe_s += time.perf_counter() - started

    def batched_get_prefixes_with_misses(
        self,
        keys_layer_major: list[list[CacheEngineKey]],
    ) -> list[LocalCPUPrefixGetResult]:
        started = time.perf_counter()
        self._record_probe(keys_layer_major)
        try:
            return LocalCPUBackend.batched_get_prefixes_with_misses(
                self,
                keys_layer_major,
            )
        finally:
            self.stats.local_probe_s += time.perf_counter() - started


class SyntheticPageFirstStorageManager:
    """Synthetic synchronous Mooncake backend with page-first accounting."""

    def __init__(
        self,
        *,
        stats: StageStats,
        local_backend: BenchmarkLocalCPUBackend,
        num_layers: int,
        num_tokens: int,
        chunk_size: int,
        bytes_per_token: int,
        remote_gbps: float,
        rpc_overhead_us: float,
        object_mode: str,
    ) -> None:
        self.stats = stats
        self.local_backend = local_backend
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.bytes_per_token = bytes_per_token
        self.remote_gbps = remote_gbps
        self.rpc_overhead_us = rpc_overhead_us
        self.object_mode = object_mode
        self.storage_backends = {"RemoteBackend": self}
        self.allocator_connector = None
        self.shared_tensor = (
            torch.empty(chunk_size * bytes_per_token, dtype=torch.uint8)
            if object_mode == "synthetic"
            else None
        )
        if object_mode == "production":
            connector = object.__new__(MooncakestoreConnector)
            connector.local_cpu_backend = local_backend
            connector._dsa_raw_token_dims = {
                0: bytes_per_token // 2,
                1: bytes_per_token // 2,
            }
            connector.meta_shapes = [torch.Size([chunk_size * bytes_per_token // 2])]
            connector.meta_dtypes = [torch.bfloat16]
            connector.meta_fmt = MemoryFormat.KV_MLA_LATENT_FMT
            connector.single_token_size = bytes_per_token
            self.allocator_connector = connector

    def _tokens_for_chunk(self, chunk_index: int, num_chunks: int) -> int:
        if chunk_index + 1 < num_chunks:
            return self.chunk_size
        return self.num_tokens - self.chunk_size * (num_chunks - 1)

    def _allocate_production(
        self,
        keys: list[CacheEngineKey],
    ) -> list[Any]:
        assert self.allocator_connector is not None
        allocated, _metadata, _mode = (
            self.allocator_connector._allocate_zero_copy_buffers(keys)
        )
        if any(obj is None for obj in allocated):
            raise MemoryError("production zero-copy allocation returned None")
        return cast(list[Any], allocated)

    def batched_get(
        self,
        keys: list[CacheEngineKey],
        location: Optional[str] = None,
    ) -> list[Any]:
        if location != "RemoteBackend":
            raise ValueError(f"unexpected synthetic remote location: {location}")
        started = time.perf_counter()
        self.stats.remote_calls += 1
        self.stats.remote_logical_keys += len(keys)

        page_key_started = time.perf_counter()
        page_keys_by_identity: dict[tuple[Any, ...], str] = {}
        page_ids: list[str] = []
        for key in keys:
            identity = (
                key.model_name,
                key.world_size,
                key.worker_id,
                key.chunk_hash,
                key.dtype,
                key.tags,
                key.kv_group,
            )
            page_id = page_keys_by_identity.get(identity)
            if page_id is None:
                page_id = mooncake_page_key(key, self.num_layers)
                page_keys_by_identity[identity] = page_id
            page_ids.append(page_id)
        self.stats.page_key_s += time.perf_counter() - page_key_started

        layout_started = time.perf_counter()
        indices_by_page: dict[str, list[int]] = {}
        for index, page_id in enumerate(page_ids):
            indices_by_page.setdefault(page_id, []).append(index)

        ordered_pages = list(indices_by_page.items())
        expected_layers = list(range(self.num_layers))
        for page_id, indices in ordered_pages:
            layer_ids = sorted(int(keys[index].layer_id) for index in indices)
            if layer_ids != expected_layers:
                raise ValueError(
                    "Synthetic page-first Mooncake request is incomplete: "
                    f"page={page_id}, layers={layer_ids}, "
                    f"expected={expected_layers}. The benchmark refuses to "
                    "model incomplete pages as independent files."
                )

        num_chunks = len(ordered_pages)
        bytes_by_hash = {
            page_id: self._tokens_for_chunk(chunk_index, num_chunks)
            * self.bytes_per_token
            for chunk_index, (page_id, _indices) in enumerate(ordered_pages)
        }
        full_bytes = self.chunk_size * self.bytes_per_token
        page_groups = [
            group for group in ordered_pages if bytes_by_hash[group[0]] == full_bytes
        ]
        legacy_indices = [
            index
            for page_id, indices in ordered_pages
            if bytes_by_hash[page_id] != full_bytes
            for index in indices
        ]
        transferred_bytes = sum(bytes_by_hash[page_id] for page_id in page_ids)
        self.stats.mooncake_page_objects += len(page_groups) + len(legacy_indices)
        self.stats.lmcache_memory_objects += len(keys)
        self.stats.logical_entries += len(keys)
        self.stats.legacy_tail_objects += len(legacy_indices)
        self.stats.transferred_bytes += transferred_bytes
        self.stats.remote_layout_s += time.perf_counter() - layout_started

        lookup_started = time.perf_counter()
        lookup_calls = int(bool(ordered_pages))
        self.stats.remote_lookup_calls += lookup_calls
        if lookup_calls and self.rpc_overhead_us > 0:
            time.sleep(self.rpc_overhead_us / 1_000_000)
        self.stats.remote_lookup_s += time.perf_counter() - lookup_started

        results: list[Optional[Any]] = [None] * len(keys)

        def allocate(indices: list[int]) -> list[Any]:
            buffer_started = time.perf_counter()
            if self.object_mode == "production":
                objects = self._allocate_production([keys[index] for index in indices])
            else:
                assert self.shared_tensor is not None
                allocations = [
                    (
                        bytes_by_hash[page_ids[index]],
                        self.shared_tensor[: bytes_by_hash[page_ids[index]]],
                    )
                    for index in indices
                ]
                objects = []
            self.stats.remote_buffer_s += time.perf_counter() - buffer_started

            wrapper_started = time.perf_counter()
            if self.object_mode == "synthetic":
                objects = [
                    SyntheticMemoryObj(logical_bytes, tensor)
                    for logical_bytes, tensor in allocations
                ]
            self.stats.remote_wrapper_s += time.perf_counter() - wrapper_started
            return objects

        def transfer(byte_count: int) -> None:
            transfer_started = time.perf_counter()
            self.stats.remote_transfer_calls += 1
            delay_s = self.rpc_overhead_us / 1_000_000
            if self.remote_gbps > 0:
                delay_s += byte_count / (self.remote_gbps * GB)
            if delay_s > 0:
                time.sleep(delay_s)
            self.stats.remote_transfer_s += time.perf_counter() - transfer_started

        if page_groups:
            page_indices = [
                index for _page_id, indices in page_groups for index in indices
            ]
            page_objects = allocate(page_indices)

            pointer_started = time.perf_counter()
            buffer_ptrs: list[list[int]] = []
            buffer_sizes: list[list[int]] = []
            offset = 0
            for _page_id, indices in page_groups:
                group_objects = page_objects[offset : offset + len(indices)]
                buffer_ptrs.append([obj.data_ptr for obj in group_objects])
                buffer_sizes.append([obj.get_size() for obj in group_objects])
                offset += len(indices)
            assert len(buffer_ptrs) == len(buffer_sizes) == len(page_groups)
            self.stats.remote_pointer_s += time.perf_counter() - pointer_started

            transfer(sum(bytes_by_hash[page_ids[index]] for index in page_indices))

            scatter_started = time.perf_counter()
            for index, obj in zip(page_indices, page_objects, strict=True):
                results[index] = obj
            self.stats.remote_scatter_s += time.perf_counter() - scatter_started

        if legacy_indices:
            legacy_objects = allocate(legacy_indices)

            pointer_started = time.perf_counter()
            legacy_ptrs = [obj.data_ptr for obj in legacy_objects]
            legacy_sizes = [obj.get_size() for obj in legacy_objects]
            assert len(legacy_ptrs) == len(legacy_sizes) == len(legacy_indices)
            self.stats.remote_pointer_s += time.perf_counter() - pointer_started

            transfer(
                sum(bytes_by_hash[page_ids[index]] for index in legacy_indices)
            )

            scatter_started = time.perf_counter()
            for index, obj in zip(legacy_indices, legacy_objects, strict=True):
                if self.object_mode == "production":
                    obj = (
                        MooncakestoreConnector._reshape_partial_chunk_with_token_size(
                            obj,
                            bytes_by_hash[page_ids[index]],
                            self.bytes_per_token,
                        )
                    )
                results[index] = obj
            self.stats.remote_scatter_s += time.perf_counter() - scatter_started

        writeback_started = time.perf_counter()
        LocalCPUBackend.batched_submit_put_task(
            self.local_backend,
            keys,
            cast(list[Any], results),
        )
        self.stats.remote_writeback_s += time.perf_counter() - writeback_started

        self.stats.remote_s += time.perf_counter() - started
        return cast(list[Any], results)

    def batched_contains_layer_pages(self, keys: list[CacheEngineKey]) -> int:
        self.stats.remote_lookup_calls += int(bool(keys))
        return len(keys)

    def batched_get_layer_pages(self, keys: list[CacheEngineKey]) -> list[Any]:
        """Exercise the production merged-page allocator without an NPU."""
        if self.object_mode != "production":
            raise RuntimeError("layer-merged matrix requires --object-mode production")
        assert self.allocator_connector is not None
        started = time.perf_counter()
        shapes, dtypes, fmt, _ = self.allocator_connector._metadata_for_raw_key(
            keys[0]
        )
        pages = self.local_backend.batched_allocate_layer_pages(
            shapes,
            dtypes,
            len(keys),
            self.num_layers,
            fmt,
        )
        if pages is None:
            return []
        full_bytes = self.chunk_size * self.bytes_per_token * self.num_layers
        transfer_started = time.perf_counter()
        if self.remote_gbps:
            time.sleep(len(keys) * full_bytes / (self.remote_gbps * GB))
        self.stats.remote_transfer_s += time.perf_counter() - transfer_started
        self.local_backend.batched_submit_layer_pages(
            [key.without_layer() for key in keys], pages
        )
        self.stats.remote_calls += 1
        self.stats.remote_transfer_calls += int(bool(keys))
        self.stats.physical_pages += len(pages)
        self.stats.logical_entries += len(pages) * self.num_layers
        self.stats.transferred_bytes += len(pages) * full_bytes
        self.stats.remote_s += time.perf_counter() - started
        return pages

    def contains(
        self,
        _key: CacheEngineKey,
        _locations: Optional[list[str]] = None,
    ) -> str:
        """Report the synthetic dataset's always-present remote objects."""
        return "RemoteBackend"

    def close(self) -> None:
        """Release production allocations owned by this benchmark run."""
        self.local_backend.close()


class SyntheticGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    """Connector surface needed by cold and prepared sparse generators."""

    def __init__(
        self,
        stats: StageStats,
        num_layers: int,
        num_tokens: int = 0,
        bytes_per_token: int = 0,
        h2d_gbps: float = 22.0,
        consumer_layer_ms: float = 0.0,
    ) -> None:
        # Deliberately skip the production connector initializer: it creates
        # device streams and buffers that a metadata-only benchmark must not
        # require. Inheriting the concrete layerwise type keeps this test
        # double compatible with LMCache's nominal connector assertion.
        self.stats = stats
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.bytes_per_token = bytes_per_token
        self.h2d_gbps = h2d_gbps
        self.consumer_layer_ms = consumer_layer_ms
        self.supports_layer_page_source = True

    def append_sparse_chunk_ptr_cache_for_layer(
        self,
        layer_id: int,
        new_sources: list[Any],
        cached_chunk_dev_ptrs: list[list[int]],
        cached_chunk_ptrs_npu: list[Optional[torch.Tensor]],
    ) -> None:
        started = time.perf_counter()
        while len(cached_chunk_dev_ptrs) < self.num_layers:
            cached_chunk_dev_ptrs.append([])
        while len(cached_chunk_ptrs_npu) < self.num_layers:
            cached_chunk_ptrs_npu.append(None)
        new_ptrs = [
            int(ptr() if callable(ptr) else ptr)
            for source in new_sources
            for ptr in (source.data_ptr,)
        ]
        cached_chunk_dev_ptrs[layer_id].extend(new_ptrs)
        pointer_tensor = torch.tensor(new_ptrs, dtype=torch.int64)
        existing = cached_chunk_ptrs_npu[layer_id]
        cached_chunk_ptrs_npu[layer_id] = (
            pointer_tensor
            if existing is None
            else torch.cat((existing, pointer_tensor))
        )
        self.stats.pointer_objects += len(new_ptrs)
        self.stats.pointer_s += time.perf_counter() - started

    def batched_to_gpu_head_token_wise(self, **_kwargs):
        for _layer_id in range(self.num_layers):
            yield
        yield
        yield

    def _consume(self, tokens: int, layer_ms: float) -> None:
        started = time.perf_counter()
        byte_count = tokens * self.bytes_per_token
        h2d_s = byte_count / (self.h2d_gbps * GB)
        if h2d_s + layer_ms / 1000:
            time.sleep(h2d_s + layer_ms / 1000)
        self.stats.device_h2d_s += h2d_s
        self.stats.device_consumer_s += time.perf_counter() - started
        self.stats.device_consumer_layers += 1
        self.stats.device_bytes += byte_count

    def batched_to_gpu(self, _starts, _ends, **_kwargs):
        """Model one dense H2D consumer submission per layer."""
        memory_objs = yield
        while memory_objs is not None:
            self._consume(self.num_tokens, self.consumer_layer_ms)
            if self.stats.device_consumer_layers > self.num_layers:
                raise AssertionError("dense consumer received too many layers")
            memory_objs = yield
        yield

    def notify_sparse_memory_objs_updated(self) -> None:
        return

    def synchronize_shared_cpu_store_publication(self) -> None:
        return


def _timed_method(
    owner: Any,
    name: str,
    stats: StageStats,
    field_name: str,
) -> None:
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            setattr(
                stats,
                field_name,
                getattr(stats, field_name) + time.perf_counter() - started,
            )

    setattr(owner, name, wrapper)


def _make_token_database(
    *,
    chunk_size: int,
    num_layers: int,
) -> tuple[ChunkedTokenDatabase, LMCacheMetadata]:
    config = SyntheticTokenDatabaseConfig(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="/workspace/models/GLM-5.1-w4a8",
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 1, chunk_size, 1, 576),
        use_mla=True,
        chunk_size=chunk_size,
    )
    return ChunkedTokenDatabase(config, metadata), metadata


def _make_engine(
    args: Namespace,
    stats: StageStats,
    *,
    kv_group: int,
    prefix_mode: str,
) -> tuple[
    AscendLMCacheEngine,
    SyntheticPageFirstStorageManager,
    dict[str, list],
]:
    token_database, metadata = _make_token_database(
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
    )
    process_tokens = token_database.process_tokens

    def measured_process_tokens(
        *process_args: Any,
        **process_kwargs: Any,
    ) -> Any:
        iterator = iter(process_tokens(*process_args, **process_kwargs))
        reused_hashes = process_kwargs.get("hashes") is not None
        if reused_hashes:
            stats.hash_plan_reuses += 1
        while True:
            started = time.perf_counter()
            try:
                result = next(iterator)
            except StopIteration:
                stats.token_process_s += time.perf_counter() - started
                return
            stats.token_process_s += time.perf_counter() - started
            yield result

    token_database.process_tokens = measured_process_tokens
    bytes_per_token = (
        args.latent_bytes_per_token if kv_group == 0 else args.indexer_bytes_per_token
    )
    num_objects = math.ceil(args.num_tokens / args.chunk_size) * args.num_layers
    object_bytes = args.chunk_size * bytes_per_token
    slab_bytes = math.ceil(object_bytes / 4096) * 4096 * num_objects
    local_backend = BenchmarkLocalCPUBackend(
        stats=stats,
        slab_bytes=slab_bytes,
        prefix_mode=prefix_mode,
        object_mode=args.object_mode,
        chunk_size=args.chunk_size,
    )
    storage_manager = SyntheticPageFirstStorageManager(
        stats=stats,
        local_backend=local_backend,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        bytes_per_token=bytes_per_token,
        remote_gbps=args.remote_gbps,
        rpc_overhead_us=args.rpc_overhead_us,
        object_mode=args.object_mode,
    )
    connector = SyntheticGPUConnector(
        stats,
        args.num_layers,
        args.num_tokens,
        bytes_per_token,
        getattr(args, "h2d_gbps", 22.0),
        getattr(args, "dense_consumer_layer_ms", 0.0),
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = args.num_layers
    engine.use_layerwise = True
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_strict = True
    engine.save_only_first_rank = True
    engine.save_indexer_only_first_rank = True
    engine.dsa_two_groups = True
    engine.shared_cpu_cache_name = "/lmcache-benchmark"
    engine.shared_cpu_cache_generation = 1
    engine.metadata = metadata
    engine.token_database = token_database
    engine.storage_manager = storage_manager
    engine.retrieve_locations = ["RemoteBackend"]
    engine.gpu_connector = connector
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _tokens: 0,
        on_retrieve_finished=lambda _request, _tokens: None,
    )
    engine._shared_cpu_request_leases = {}
    engine._shared_page_first_location_plan = (
        lambda keys: ["RemoteBackend"] * len(keys)
    )
    engine.config = SimpleNamespace(
        chunk_size=args.chunk_size,
        enable_shared_cpu_cache=True,
        experimental_sampled_layerwise_lookup=True,
        remote_url="mooncakestore://benchmark/",
        shared_cpu_remote_layers_per_batch=1,
        dsa_two_groups=True,
        use_layerwise=True,
        extra_config={
            "mooncake_page_first_multi_buffer": True,
            "mooncake_layer_merged_page_objects": bool(
                getattr(args, "_layer_merged", False)
            ),
            "save_only_first_rank": True,
        },
    )
    engine.is_healthy = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._shared_local_cpu_backend = lambda: local_backend
    if args.object_mode == "synthetic":
        engine._is_rank0_shared_mem_obj = lambda _obj: True
        engine._validate_rank0_shared_mem_obj = lambda _obj, **_kwargs: None
    engine._shared_cpu_estimated_physical_chunk_bytes = (
        lambda _kv_group, num_tokens=None: (
            (
                (num_tokens or args.chunk_size) * bytes_per_token
                + 4095
            )
            // 4096
            * 4096
        )
    )
    engine.shared_cpu_rank0_request_object_ids = lambda _req_id, _kv_group: set()

    def record_resolver_stage(stage: str, elapsed_s: float) -> None:
        _record_resolver_stage(stats, stage, elapsed_s)

    engine._shared_rank0_resolver_timing_hook = record_resolver_stage

    def make_handles(**kwargs):
        started = time.perf_counter()
        handles = [
            (
                kwargs["layer_id"],
                kwargs["chunk_index_base"] + index,
                id(memory_obj),
            )
            for index, memory_obj in enumerate(kwargs["mem_objs_layer"])
        ]
        stats.handle_objects += len(handles)
        stats.handle_s += time.perf_counter() - started
        return handles

    engine._make_shared_handles_for_layer = make_handles

    def broadcast(_envelope):
        started = time.perf_counter()
        stats.broadcast_envelopes += 1
        stats.broadcast_s += time.perf_counter() - started

    engine._broadcast_shared_envelope = broadcast

    _timed_method(engine, "_ensure_retrieve_chunk_metadata", stats, "metadata_s")
    _timed_method(engine, "_shared_cpu_runtime_capacity_details", stats, "capacity_s")
    _timed_method(
        engine,
        "_resolve_shared_rank0_remote_layers_windowed",
        stats,
        "resolver_s",
    )
    _timed_method(engine, "_append_retrieve_layer_cache", stats, "cache_append_s")
    _timed_method(engine, "_append_shared_handle_cache", stats, "handle_cache_s")
    _timed_method(engine, "_fence_shared_cpu_store_publication", stats, "fence_s")

    caches: dict[str, list] = {
        "cached_keys": [],
        "cached_starts": [],
        "cached_ends": [],
        "cached_memory_objs": [],
        "cached_tensors": [],
        "cached_chunk_dev_ptrs": [],
        "cached_chunk_ptrs_npu": [],
        "cached_shared_handles": [],
    }
    return engine, storage_manager, caches


def _drive_retriever(
    retriever,
    *,
    num_layers: int,
    num_tokens: int,
) -> tuple[float, float, float, float]:
    started = time.perf_counter()
    first_mask = next(retriever)
    prime_s = time.perf_counter() - started
    if first_mask is None or int(first_mask.sum().item()) != num_tokens:
        raise AssertionError("cold bootstrap did not retrieve the complete token mask")
    layers_started = time.perf_counter()
    selected = torch.arange(min(num_tokens, 2048), dtype=torch.long)
    for _ in range(num_layers):
        retriever.send((selected, 0))
    layers_s = time.perf_counter() - layers_started
    close_started = time.perf_counter()
    retriever.close()
    close_s = time.perf_counter() - close_started
    return prime_s, time.perf_counter() - started, layers_s, close_s


def run_once(
    args: Namespace,
    *,
    kv_group: int,
    prefix_mode: str,
    tokens: Optional[list[int]] = None,
    shared_preflight_state: Optional[dict[str, Any]] = None,
) -> StageStats:
    """Execute one cold bootstrap followed by one prepared warm retrieve."""
    stats = StageStats()
    engine, storage_manager, caches = _make_engine(
        args,
        stats,
        kv_group=kv_group,
        prefix_mode=prefix_mode,
    )
    if tokens is None:
        tokens = list(range(args.num_tokens))
    ret_mask = torch.zeros(args.num_tokens, dtype=torch.bool)
    cold = engine.retrieve_layer_head_token_wise(
        tokens,
        ret_mask=ret_mask,
        kv_group=kv_group,
        req_id=f"cold-g{kv_group}-{prefix_mode}",
        shared_cpu_request_preflight_state=shared_preflight_state,
        **caches,
    )
    (
        stats.cold_prime_s,
        stats.cold_total_s,
        stats.cold_layers_s,
        stats.cold_close_s,
    ) = _drive_retriever(cold, num_layers=args.num_layers, num_tokens=args.num_tokens)
    tolerance_s = 1e-6
    if stats.remote_s > stats.resolver_s + tolerance_s:
        raise AssertionError("remote timing exceeds its enclosing resolver timing")
    if stats.pointer_s > stats.cache_append_s + tolerance_s:
        raise AssertionError("pointer timing exceeds its enclosing cache append")
    layer_stages_s = sum(
        (
            stats.cache_append_s,
            stats.handle_s,
            stats.handle_cache_s,
            stats.broadcast_s,
            stats.fence_s,
        )
    )
    if layer_stages_s > stats.cold_layers_s + tolerance_s:
        raise AssertionError("layer stage timings exceed the measured layer loop")
    if (
        stats.cold_prime_s + stats.cold_layers_s + stats.cold_close_s
        > stats.cold_total_s + tolerance_s
    ):
        raise AssertionError("cold sub-stage timings exceed total cold time")
    stats.resolver_cpu_s = max(0.0, stats.resolver_s - stats.remote_s)
    resolver_attributed_s = _resolver_cpu_attributed_s(stats)
    if resolver_attributed_s > stats.resolver_cpu_s + tolerance_s:
        raise AssertionError(
            "resolver sub-stage timings exceed resolver CPU time"
        )
    stats.resolver_unattributed_s = max(
        0.0,
        stats.resolver_cpu_s - resolver_attributed_s,
    )
    stats.cache_append_cpu_s = max(
        0.0, stats.cache_append_s - stats.pointer_s
    )
    stats.prime_other_s = max(
        0.0,
        stats.cold_prime_s
        - stats.metadata_s
        - stats.local_probe_s
        - stats.capacity_s
        - stats.resolver_s,
    )
    stats.layer_other_s = max(0.0, stats.cold_layers_s - layer_stages_s)

    num_chunks = math.ceil(args.num_tokens / args.chunk_size)
    chunk_counts = [args.chunk_size] * num_chunks
    chunk_counts[-1] = args.num_tokens - args.chunk_size * (num_chunks - 1)
    prepared = build_prepared_sparse_source(
        caches["cached_tensors"],
        caches["cached_chunk_ptrs_npu"],
        num_layers=args.num_layers,
        total_tokens=args.num_tokens,
        chunk_token_counts=chunk_counts,
        cached_memory_objs=caches["cached_memory_objs"],
    )
    if prepared is None:
        raise AssertionError("cold bootstrap did not produce a prepared source")

    remote_calls_before_warm = storage_manager.stats.remote_calls
    warm = engine.retrieve_layer_head_token_wise(
        tokens,
        ret_mask=ret_mask,
        kv_group=kv_group,
        req_id=f"warm-g{kv_group}-{prefix_mode}",
        prepared_sparse_source=prepared,
    )
    _prime_s, stats.warm_total_s, _layers_s, _close_s = _drive_retriever(
        warm,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
    )
    if storage_manager.stats.remote_calls != remote_calls_before_warm:
        raise AssertionError(
            "prepared warm retrieve unexpectedly accessed remote storage"
        )

    expected_chunks = math.ceil(args.num_tokens / args.chunk_size)
    expected_memory_objects = expected_chunks * args.num_layers
    full_pages, tail_tokens = divmod(args.num_tokens, args.chunk_size)
    expected_files = full_pages + (args.num_layers if tail_tokens else 0)
    if stats.remote_calls != 1:
        raise AssertionError(
            "page-first cold bootstrap used "
            f"{stats.remote_calls} remote calls, expected 1"
        )
    expected_transfer_calls = int(full_pages > 0) + int(tail_tokens > 0)
    if stats.remote_lookup_calls != 1:
        raise AssertionError(
            f"Mooncake page lookup count mismatch: {stats.remote_lookup_calls} != 1"
        )
    if stats.remote_transfer_calls != expected_transfer_calls:
        raise AssertionError(
            "Mooncake transfer call count mismatch: "
            f"{stats.remote_transfer_calls} != {expected_transfer_calls}"
        )
    if stats.mooncake_page_objects != expected_files:
        raise AssertionError(
            "page-first Mooncake object count mismatch: "
            f"{stats.mooncake_page_objects} != {expected_files}"
        )
    if stats.lmcache_memory_objects != expected_memory_objects:
        raise AssertionError(
            "LMCache MemoryObj count mismatch: "
            f"{stats.lmcache_memory_objects} != {expected_memory_objects}"
        )
    if stats.pointer_objects != expected_memory_objects:
        raise AssertionError(
            f"pointer count mismatch: {stats.pointer_objects} "
            f"!= {expected_memory_objects}"
        )
    if args.object_mode == "production":
        memory_objs = [
            obj for layer in caches["cached_memory_objs"] for obj in layer
        ]
        if any(caches["cached_tensors"]) or any(
            not obj.is_pinned for obj in memory_objs
        ):
            raise AssertionError(
                "pointer-first bootstrap must retain pinned MemoryObjs without "
                "constructing per-chunk tensor views"
            )
    if stats.handle_objects != expected_memory_objects:
        raise AssertionError(
            f"handle count mismatch: {stats.handle_objects} "
            f"!= {expected_memory_objects}"
        )
    if stats.broadcast_envelopes != args.num_layers:
        raise AssertionError(
            f"envelope count mismatch: {stats.broadcast_envelopes} "
            f"!= {args.num_layers}"
        )
    storage_manager.close()
    return stats


def _median_stats(samples: list[StageStats]) -> StageStats:
    values: dict[str, Any] = {}
    for field_name in StageStats.__dataclass_fields__:
        field_values = [getattr(sample, field_name) for sample in samples]
        values[field_name] = statistics.median(field_values)
    return StageStats(**values)


def _run_pair_once(
    args: Namespace,
    *,
    prefix_mode: str,
    chunk_plan_mode: str,
) -> dict[int, StageStats]:
    tokens = list(range(args.num_tokens))
    shared_state: Optional[dict[str, Any]] = (
        {} if chunk_plan_mode == "reuse" else None
    )
    return {
        kv_group: run_once(
            args,
            kv_group=kv_group,
            prefix_mode=prefix_mode,
            tokens=tokens,
            shared_preflight_state=shared_state,
        )
        for kv_group in args.kv_groups
    }


def run_pair_cases(
    args: Namespace,
    *,
    prefix_mode: str,
) -> list[BenchmarkResult]:
    """Interleave plan modes to avoid order-dependent timing bias."""
    samples_by_mode = {mode: [] for mode in args.chunk_plan_modes}
    for repeat in range(args.warmup + args.repeats):
        modes = args.chunk_plan_modes[:: 1 if repeat % 2 == 0 else -1]
        for chunk_plan_mode in modes:
            sample = _run_pair_once(
                args,
                prefix_mode=prefix_mode,
                chunk_plan_mode=chunk_plan_mode,
            )
            if repeat >= args.warmup:
                samples_by_mode[chunk_plan_mode].append(sample)

    results = []
    for chunk_plan_mode, pair_samples in samples_by_mode.items():
        for kv_group in args.kv_groups:
            samples = [sample[kv_group] for sample in pair_samples]
            bytes_per_token = (
                args.latent_bytes_per_token
                if kv_group == 0
                else args.indexer_bytes_per_token
            )
            results.append(
                BenchmarkResult(
                    name=(
                        f"g{kv_group}_{prefix_mode.replace('-', '_')}_"
                        f"{chunk_plan_mode}"
                    ),
                    prefix_mode=prefix_mode,
                    chunk_plan_mode=chunk_plan_mode,
                    object_mode=args.object_mode,
                    kv_group=kv_group,
                    num_layers=args.num_layers,
                    num_chunks=math.ceil(args.num_tokens / args.chunk_size),
                    bytes_per_token=bytes_per_token,
                    repeats=args.repeats,
                    median=_median_stats(samples),
                    samples=samples,
                )
            )
    return results


def _run_cold_compact_protocol_once(
    args: Namespace,
    *,
    enabled: bool,
    layer_merged: bool = False,
) -> ColdCompactProtocolSample:
    """Execute the two production-ordered generator protocols once."""
    tokens = list(range(args.num_tokens))
    args._layer_merged = layer_merged
    token_mask = torch.ones(args.num_tokens, dtype=torch.bool)
    request_id = f"ab-{'on' if enabled else 'off'}"
    shared_state: dict[str, Any] = {}
    engines: dict[int, AscendLMCacheEngine] = {}
    stores: list[SyntheticPageFirstStorageManager] = []
    retrievers = []
    caches_by_group: dict[int, dict[str, list]] = {}
    stats_by_group: dict[int, StageStats] = {}

    def close_resources(suppress: bool) -> None:
        error = None
        for resource in reversed(retrievers):
            try:
                resource.close()
            except BaseException as exc:
                error = error or exc
        for engine in engines.values():
            try:
                engine.release_shared_cpu_sparse_request(request_id)
            except BaseException as exc:
                error = error or exc
        for store in reversed(stores):
            try:
                store.close()
            except BaseException as exc:
                error = error or exc
        if error is not None and not suppress:
            raise error

    try:
        for group in (0, 1):
            stats = StageStats()
            engine, storage, caches = _make_engine(
                args,
                stats,
                kv_group=group,
                prefix_mode=args.ab_prefix_mode,
            )
            engines[group] = engine
            stores.append(storage)
            caches_by_group[group] = caches
            stats_by_group[group] = stats
    except BaseException:
        close_resources(True)
        raise

    def dense_retriever(group: int):
        options = {}
        if not enabled:
            options = dict(caches_by_group[group])
            options["_retain_shared_dense_cache"] = True
        return engines[group].retrieve_layer(
            tokens,
            token_mask,
            kv_group=group,
            req_id=request_id,
            slot_mapping=torch.arange(args.num_tokens, dtype=torch.long),
            sync=True,
            shared_cpu_request_preflight_state=(
                shared_state if not enabled else None
            ),
            **options,
        )

    def sparse_retriever(group: int, *, materialize_only: bool = False):
        options = dict(caches_by_group[group])
        if materialize_only:
            options["materialize_only"] = True
        return engines[group].retrieve_layer_head_token_wise(
            tokens,
            ret_mask=torch.zeros(args.num_tokens, dtype=torch.bool),
            kv_group=group,
            req_id=request_id,
            shared_cpu_request_preflight_state=shared_state,
            **options,
        )

    synchronization_barrier_s = 0.0
    started = time.perf_counter()
    try:
        if enabled:
            latent = sparse_retriever(0, materialize_only=True)
            indexer = dense_retriever(1)
            retrievers.extend((latent, indexer))
            latent_result = next(latent)
            next(indexer)
            next(indexer)
            indexer_result = None
            for _layer_id in range(args.num_layers):
                latent_result = latent.send(None)
                indexer_result = next(indexer)
            sync_delay_s = args.cold_compact_sync_ms / 1000
            barrier_delay_s = args.cold_compact_barrier_ms / 1000
        else:
            latent = dense_retriever(0)
            indexer = dense_retriever(1)
            retrievers.extend((latent, indexer))
            next(latent)
            next(indexer)
            next(latent)
            next(indexer)
            latent_result = indexer_result = None
            for _layer_id in range(args.num_layers):
                latent_result = next(latent)
                indexer_result = next(indexer)
            sync_delay_s = args.normal_sync_ms / 1000
            barrier_delay_s = 0.0
        while retrievers:
            retrievers.pop().close()
        sync_started = time.perf_counter()
        if sync_delay_s:
            time.sleep(sync_delay_s)
        synchronization_barrier_s = time.perf_counter() - sync_started
        if any(
            result is None or int(result.sum().item()) != args.num_tokens
            for result in (latent_result, indexer_result)
        ):
            raise AssertionError("cold-start protocol retrieved an incomplete mask")
        if not enabled and any(
            len(caches_by_group[group]["cached_memory_objs"])
            != args.num_layers
            or any(
                not layer
                for layer in caches_by_group[group]["cached_memory_objs"]
            )
            for group in (0, 1)
        ):
            raise AssertionError("dense prefix did not retain request-owned cache")
        if enabled:
            chunks = math.ceil(args.num_tokens / args.chunk_size)
            chunk_counts = [args.chunk_size] * chunks
            chunk_counts[-1] = args.num_tokens - args.chunk_size * (chunks - 1)
            prepared = build_prepared_sparse_source(
                caches_by_group[0]["cached_tensors"],
                caches_by_group[0]["cached_chunk_ptrs_npu"],
                num_layers=args.num_layers,
                total_tokens=args.num_tokens,
                chunk_token_counts=chunk_counts,
                cached_memory_objs=caches_by_group[0]["cached_memory_objs"],
            )
            if prepared is None:
                raise AssertionError("cold compact did not seal the latent source")
        barrier_started = time.perf_counter()
        if barrier_delay_s:
            time.sleep(barrier_delay_s)
        synchronization_barrier_s += time.perf_counter() - barrier_started
        wall_s = time.perf_counter() - started
    finally:
        close_resources(sys.exc_info()[0] is not None)
    for stats in stats_by_group.values():
        stats.resolver_cpu_s = max(0.0, stats.resolver_s - stats.remote_s)
    return ColdCompactProtocolSample(
        enabled=enabled,
        layer_merged=layer_merged,
        wall_s=wall_s,
        synchronization_barrier_s=synchronization_barrier_s,
        groups=stats_by_group,
    )


def run_cold_compact_ab_cases(
    args: Namespace,
) -> dict[str, list[ColdCompactProtocolSample]]:
    """Interleave explicit flag-off and flag-on generator protocols."""
    samples: dict[str, list[ColdCompactProtocolSample]] = {
        "flag_unset": [],
        "flag_true": [],
    }
    for repeat in range(args.warmup + args.repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for enabled in order:
            result = _run_cold_compact_protocol_once(args, enabled=enabled)
            if repeat >= args.warmup:
                samples["flag_true" if enabled else "flag_unset"].append(result)
    return samples


def _cold_compact_variant(
    sample: ColdCompactProtocolSample,
) -> ColdCompactVariant:
    """Attribute one measured outer protocol wall without nested double counts."""
    enabled = sample.enabled
    stats = list(sample.groups.values())
    lookup_metadata_s = sum(
        max(item.metadata_s, item.token_process_s)
        + item.local_probe_s
        + item.capacity_s
        for item in stats
    )
    mooncake_s = sum(item.remote_s for item in stats)
    mooncake_allocation_s = sum(
        item.remote_buffer_s + item.remote_wrapper_s for item in stats
    )
    mooncake_transfer_s = sum(item.remote_transfer_s for item in stats)
    mooncake_other_s = max(
        0.0, mooncake_s - mooncake_allocation_s - mooncake_transfer_s
    )
    resolver_cpu_s = sum(item.resolver_cpu_s for item in stats)
    pointer_handle_s = sum(item.pointer_s + item.handle_s for item in stats)
    device_consumer_s = sum(item.device_consumer_s for item in stats)
    device_h2d_s = sum(item.device_h2d_s for item in stats)
    device_bytes = sum(item.device_bytes for item in stats)
    attributed_s = sum(
        (
            lookup_metadata_s,
            mooncake_s,
            resolver_cpu_s,
            pointer_handle_s,
            device_consumer_s,
            sample.synchronization_barrier_s,
        )
    )
    bootstrap_other_s = max(0.0, sample.wall_s - attributed_s)
    return ColdCompactVariant(
        flag="true" if enabled else "unset",
        protocol=(
            [
                "materialize_latent",
                "dense_consume_indexer",
                "synchronize_and_resume",
            ]
            if enabled
            else ["dense_consume_latent", "dense_consume_indexer", "resume"]
        ),
        consumer_modes=(
            {0: "materialize_only", 1: "dense"}
            if enabled
            else {0: "dense", 1: "dense"}
        ),
        lookup_metadata_s=lookup_metadata_s,
        mooncake_s=mooncake_s,
        mooncake_allocation_s=mooncake_allocation_s,
        mooncake_transfer_s=mooncake_transfer_s,
        mooncake_other_s=mooncake_other_s,
        resolver_cpu_s=resolver_cpu_s,
        pointer_handle_s=pointer_handle_s,
        bootstrap_other_s=bootstrap_other_s,
        device_bytes=device_bytes,
        device_h2d_s=device_h2d_s,
        device_consumer_s=device_consumer_s,
        synchronization_barrier_s=sample.synchronization_barrier_s,
        critical_path_s=sample.wall_s,
    )


def _median_cold_compact_variant(
    samples: list[ColdCompactProtocolSample],
) -> ColdCompactVariant:
    variants = [_cold_compact_variant(sample) for sample in samples]
    first = variants[0]
    values = {
        name: statistics.median(getattr(item, name) for item in variants)
        for name in ColdCompactVariant.__dataclass_fields__
        if name not in {"flag", "protocol", "consumer_modes"}
    }
    return ColdCompactVariant(
        flag=first.flag,
        protocol=first.protocol,
        consumer_modes=first.consumer_modes,
        **values,
    )


def build_cold_compact_ab_report(
    args: Namespace,
    samples: dict[str, list[ColdCompactProtocolSample]],
) -> ColdCompactABReport:
    """Summarize paired executable flag-off/on protocol samples."""
    if set(samples) != {"flag_unset", "flag_true"} or any(
        not values for values in samples.values()
    ):
        raise ValueError("cold-compact A/B requires non-empty off/on samples")
    sample_count = min(len(values) for values in samples.values())
    paired = []
    for index in range(sample_count):
        off = _cold_compact_variant(samples["flag_unset"][index])
        on = _cold_compact_variant(samples["flag_true"][index])
        paired.append(
            {
                "sample": index + 1,
                "flag_unset_critical_path_s": off.critical_path_s,
                "flag_true_critical_path_s": on.critical_path_s,
                "delta_s": on.critical_path_s - off.critical_path_s,
                "speedup": off.critical_path_s / on.critical_path_s,
            }
        )
    off = _median_cold_compact_variant(samples["flag_unset"])
    on = _median_cold_compact_variant(samples["flag_true"])
    median_delta_s = statistics.median(item["delta_s"] for item in paired)
    return ColdCompactABReport(
        schema=1,
        model="executed_production_ordered_protocol_v2",
        parameters={
            "num_tokens": args.num_tokens,
            "num_layers": args.num_layers,
            "chunk_size": args.chunk_size,
            "prefix_mode": args.ab_prefix_mode,
            "h2d_gbps": args.h2d_gbps,
            "dense_consumer_layer_ms": args.dense_consumer_layer_ms,
            "normal_sync_ms": args.normal_sync_ms,
            "cold_compact_sync_ms": args.cold_compact_sync_ms,
            "cold_compact_barrier_ms": args.cold_compact_barrier_ms,
            "sample_count": sample_count,
            "measured": "CPU, Mooncake emulator, generator order, consumer wall",
            "synthetic": "H2D rate, per-layer consumer, sync/barrier delays",
        },
        variants={
            "flag_unset": off,
            "flag_true": on,
        },
        comparison={
            "delta_s": median_delta_s,
            "speedup": statistics.median(item["speedup"] for item in paired),
            "slowdown": statistics.median(
                1 / item["speedup"] for item in paired
            ),
        },
        samples=paired,
    )


def print_cold_compact_ab(report: ColdCompactABReport) -> None:
    """Print the concise current cold-compact A/B report."""
    print("\nLMCACHE_ENABLE_DSA_COLD_COMPACT_LOAD A/B (median ms)")
    print(
        f"{'setting':12} {'lookup/meta':>12} {'Mooncake':>10} "
        f"{'resolver CPU':>13} {'ptr/handle':>11} {'H2D':>9} {'consumer':>10} "
        f"{'sync/barrier':>13} {'critical':>10}"
    )
    for name in ("flag_unset", "flag_true"):
        item = report.variants[name]
        print(
            f"{item.flag:12} "
            f"{_ms(item.lookup_metadata_s):12.3f} "
            f"{_ms(item.mooncake_s):10.3f} "
            f"{_ms(item.resolver_cpu_s):13.3f} "
            f"{_ms(item.pointer_handle_s):11.3f} "
            f"{_ms(item.device_h2d_s):9.3f} "
            f"{_ms(item.device_consumer_s):10.3f} "
            f"{_ms(item.synchronization_barrier_s):13.3f} "
            f"{_ms(item.critical_path_s):10.3f}"
        )
    comparison = report.comparison
    direction = "slower" if comparison["delta_s"] > 0 else "faster"
    print(
        f"  flag=true is {abs(_ms(comparison['delta_s'])):.3f} ms {direction}; "
        f"off/on={comparison['speedup']:.3f}x"
    )
    print(
        "  device payload: "
        f"unset={report.variants['flag_unset'].device_bytes / GB:.3f} GB, "
        f"true={report.variants['flag_true'].device_bytes / GB:.3f} GB"
    )


def run_layer_merged_cold_compact_matrix(args: Namespace) -> dict[str, Any]:
    """Run the four feature combinations through the existing protocols."""
    cases = [(False, False), (True, False), (False, True), (True, True)]
    samples: dict[str, list[ColdCompactProtocolSample]] = {
        f"merged_{int(merged)}_compact_{int(compact)}": []
        for merged, compact in cases
    }
    for repeat in range(args.warmup + args.repeats):
        order = cases[repeat % len(cases) :] + cases[: repeat % len(cases)]
        for merged, compact in order:
            sample = _run_cold_compact_protocol_once(
                args, enabled=compact, layer_merged=merged
            )
            if repeat >= args.warmup:
                samples[f"merged_{int(merged)}_compact_{int(compact)}"].append(
                    sample
                )
    variants = {}
    for name, case_samples in samples.items():
        medians = {
            field: statistics.median(
                sum(getattr(group, field) for group in sample.groups.values())
                for sample in case_samples
            )
            for field in (
                "physical_pages",
                "logical_entries",
                "legacy_tail_objects",
                "pointer_s",
            )
        }
        pointer_seal_s = medians.pop("pointer_s")
        variants[name] = {
            "status": "measured",
            "critical_path_s": statistics.median(s.wall_s for s in case_samples),
            **medians,
            "pointer_seal_s": pointer_seal_s,
            "passive_wait_s": None,
            "passive_wait_status": "unavailable_single_rank_benchmark",
        }
    return {
        "schema": 1,
        "model": "executed_four_case_single_rank_protocol_v1",
        "measurement": {
            "critical_path": "measured",
            "passive_wait": "unavailable_single_rank_benchmark",
            "production_linux_npu_required": True,
        },
        "variants": variants,
    }


def print_layer_merged_cold_compact_matrix(report: dict[str, Any]) -> None:
    print("\nLayer-merged objects x cold-compact load (median ms)")
    print(
        f"{'case':28} {'critical':>10} {'pages':>8} {'logical':>10} "
        f"{'tails':>8} {'ptr seal':>10} {'passive wait':>16}"
    )
    for name, item in report["variants"].items():
        print(
            f"{name:28} {_ms(item['critical_path_s']):10.3f} "
            f"{int(item['physical_pages']):8d} {int(item['logical_entries']):10d} "
            f"{int(item['legacy_tail_objects']):8d} "
            f"{_ms(item['pointer_seal_s']):10.3f} {'unavailable':>16}"
        )
    print(
        "  Mooncake includes allocation/transfer; resolver CPU excludes its "
        "nested Mooncake call. Consumer and synchronization are executed "
        "synthetic delays."
    )


def _ms(seconds: float) -> float:
    return seconds * 1000


def print_results(results: list[BenchmarkResult]) -> None:
    """Print timing and topology summaries for all measured configurations."""
    print("\nCold bootstrap wall time (median ms; prime is the blocking first next())")
    print(
        f"{'case':38} {'prime':>9} {'cold total':>11} {'warm':>9}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.cold_prime_s):9.3f} "
            f"{_ms(stats.cold_total_s):11.3f} "
            f"{_ms(stats.warm_total_s):9.3f}"
        )

    print(
        "\nCold bootstrap timing breakdown (median ms; "
        "token keys are inside metadata, page keys are inside remote)"
    )
    print(
        f"{'case':38} {'metadata':>10} {'token keys':>11} {'page keys':>10} "
        f"{'local probe':>12} {'capacity':>10} {'remote':>9} "
        f"{'pointers':>9} {'handles':>9}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.metadata_s):10.3f} "
            f"{_ms(stats.token_process_s):11.3f} "
            f"{_ms(stats.page_key_s):10.3f} "
            f"{_ms(stats.local_probe_s):12.3f} "
            f"{_ms(stats.capacity_s):10.3f} "
            f"{_ms(stats.remote_s):9.3f} "
            f"{_ms(stats.pointer_s):9.3f} "
            f"{_ms(stats.handle_s):9.3f}"
        )

    print(
        "\nFirst-yield attribution (median ms; remote is nested inside resolver)"
    )
    print(
        f"{'case':38} {'prime':>9} {'metadata':>10} {'probe':>9} "
        f"{'capacity':>10} {'resolver':>10} {'remote':>9} "
        f"{'resolve CPU':>12} {'other':>9}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.cold_prime_s):9.3f} "
            f"{_ms(stats.metadata_s):10.3f} "
            f"{_ms(stats.local_probe_s):9.3f} "
            f"{_ms(stats.capacity_s):10.3f} "
            f"{_ms(stats.resolver_s):10.3f} "
            f"{_ms(stats.remote_s):9.3f} "
            f"{_ms(stats.resolver_cpu_s):12.3f} "
            f"{_ms(stats.prime_other_s):9.3f}"
        )

    print("\nResolver attribution (median ms; remote get excluded from CPU)")
    print(
        f"{'case':38} {'windows':>9} {'remote get':>11} {'results':>9} "
        f"{'adopt/check':>11} {'pin':>9} {'legacy val':>10} {'scatter':>9} "
        f"{'unattributed':>13} {'rollback':>10}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.resolver_windows_s):9.3f} "
            f"{_ms(stats.resolver_remote_get_s):11.3f} "
            f"{_ms(stats.resolver_results_s):9.3f} "
            f"{_ms(stats.resolver_classification_s):11.3f} "
            f"{_ms(stats.resolver_pinning_s):9.3f} "
            f"{_ms(stats.resolver_validation_s):10.3f} "
            f"{_ms(stats.resolver_scatter_s):9.3f} "
            f"{_ms(stats.resolver_unattributed_s):13.3f} "
            f"{_ms(stats.resolver_rollback_s):10.3f}"
        )

    print(
        "\nPost-prime layer attribution "
        "(median ms; pointer is nested inside cache append)"
    )
    print(
        f"{'case':38} {'layers':>9} {'cache append':>13} {'append CPU':>11} "
        f"{'pointer':>9} {'handles':>9} {'handle cache':>13} "
        f"{'broadcast':>10} {'fence':>8} {'other':>9} {'close':>9}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.cold_layers_s):9.3f} "
            f"{_ms(stats.cache_append_s):13.3f} "
            f"{_ms(stats.cache_append_cpu_s):11.3f} "
            f"{_ms(stats.pointer_s):9.3f} "
            f"{_ms(stats.handle_s):9.3f} "
            f"{_ms(stats.handle_cache_s):13.3f} "
            f"{_ms(stats.broadcast_s):10.3f} "
            f"{_ms(stats.fence_s):8.3f} "
            f"{_ms(stats.layer_other_s):9.3f} "
            f"{_ms(stats.cold_close_s):9.3f}"
        )

    print("\nPage-first remote breakdown (median ms)")
    if any(result.object_mode == "production" for result in results):
        print("  production buffer stage includes metadata, allocation, and wrappers")
    print(
        f"{'case':38} {'layout':>9} {'lookup':>9} {'buffers':>9} "
        f"{'wrappers':>9} {'ptr/size':>10} {'transfer':>10} "
        f"{'scatter':>9} {'hot cache':>11}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:38} "
            f"{_ms(stats.remote_layout_s):9.3f} "
            f"{_ms(stats.remote_lookup_s):9.3f} "
            f"{_ms(stats.remote_buffer_s):9.3f} "
            f"{_ms(stats.remote_wrapper_s):9.3f} "
            f"{_ms(stats.remote_pointer_s):10.3f} "
            f"{_ms(stats.remote_transfer_s):10.3f} "
            f"{_ms(stats.remote_scatter_s):9.3f} "
            f"{_ms(stats.remote_writeback_s):11.3f}"
        )

    print("\nPage-first topology and metadata counts (median)")
    print(
        f"{'case':38} {'probe calls':>12} {'key probes':>11} "
        f"{'storage calls':>13} {'lookups':>8} {'xfers':>7} "
        f"{'LMCache objs':>13} {'Mooncake files':>15} "
        f"{'buffers/file':>13} {'plan reuse':>11}"
    )
    for result in results:
        stats = result.median
        buffers_per_file = (
            stats.lmcache_memory_objects / stats.mooncake_page_objects
            if stats.mooncake_page_objects
            else 0
        )
        print(
            f"{result.name:38} "
            f"{int(stats.local_probe_calls):12d} "
            f"{int(stats.local_key_probes):11d} "
            f"{int(stats.remote_calls):13d} "
            f"{int(stats.remote_lookup_calls):8d} "
            f"{int(stats.remote_transfer_calls):7d} "
            f"{int(stats.lmcache_memory_objects):13d} "
            f"{int(stats.mooncake_page_objects):15d} "
            f"{buffers_per_file:13.1f} "
            f"{int(stats.hash_plan_reuses):11d}"
        )

    result_map = {
        (result.kv_group, result.prefix_mode, result.chunk_plan_mode): result
        for result in results
    }
    print("\nCross-layer LocalCPU probe speedup")
    for chunk_plan_mode in ("independent", "reuse"):
        for kv_group in (0, 1):
            baseline = result_map.get((kv_group, "per-layer", chunk_plan_mode))
            optimized = result_map.get((kv_group, "batched", chunk_plan_mode))
            if baseline is None or optimized is None:
                continue
            print(
                f"  kv_group={kv_group} plan={chunk_plan_mode}: "
                f"prime="
                f"{baseline.median.cold_prime_s / optimized.median.cold_prime_s:.3f}x, "
                f"local_probe="
                f"{baseline.median.local_probe_s / optimized.median.local_probe_s:.3f}x"
            )

    print("\nCross-group chunk-plan speedup")
    for prefix_mode in ("per-layer", "batched"):
        baseline = [
            result_map.get((kv_group, prefix_mode, "independent"))
            for kv_group in (0, 1)
        ]
        optimized = [
            result_map.get((kv_group, prefix_mode, "reuse"))
            for kv_group in (0, 1)
        ]
        if any(result is None for result in baseline + optimized):
            continue
        baseline = [result for result in baseline if result is not None]
        optimized = [result for result in optimized if result is not None]
        baseline_prime = sum(item.median.cold_prime_s for item in baseline)
        optimized_prime = sum(item.median.cold_prime_s for item in optimized)
        baseline_metadata = sum(item.median.metadata_s for item in baseline)
        optimized_metadata = sum(item.median.metadata_s for item in optimized)
        index_baseline = baseline[1].median
        index_optimized = optimized[1].median
        print(
            f"  prefix={prefix_mode}: "
            f"prime={baseline_prime / optimized_prime:.3f}x, "
            f"metadata={baseline_metadata / optimized_metadata:.3f}x, "
            f"indexer metadata="
            f"{index_baseline.metadata_s / index_optimized.metadata_s:.3f}x"
        )


def parse_args() -> Namespace:
    """Parse and validate command-line benchmark settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=61)
    parser.add_argument(
        "--kv-groups",
        default="0,1",
        help="Comma-separated KV groups: 0=MLA latent, 1=DSA index.",
    )
    parser.add_argument(
        "--prefix-modes",
        default="per-layer,batched",
        help="Comma-separated LocalCPU probe modes.",
    )
    parser.add_argument(
        "--chunk-plan-modes",
        default="independent,reuse",
        help="Compare independent group hashing with latent-to-indexer plan reuse.",
    )
    parser.add_argument(
        "--latent-bytes-per-token",
        type=int,
        default=576 * 2,
        help="GLM-5.1 TP8 latent bytes per token per layer.",
    )
    parser.add_argument(
        "--indexer-bytes-per-token",
        type=int,
        default=128 * 2,
        help="GLM-5.1 TP8 DSA-index bytes per token per layer.",
    )
    parser.add_argument(
        "--remote-gbps",
        type=float,
        default=0.0,
        help="Synthetic Mooncake payload rate in decimal GB/s; 0 is metadata only.",
    )
    parser.add_argument(
        "--rpc-overhead-us",
        type=float,
        default=0.0,
        help="Synthetic latency per Mooncake lookup or transfer call.",
    )
    parser.add_argument(
        "--object-mode",
        choices=("production", "synthetic"),
        default="production",
        help="Use real LMCache allocation/MemoryObj/LRU or lightweight stand-ins.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--cold-compact-ab",
        action="store_true",
        help="Compare the production ordering with the cold-compact flag off/on.",
    )
    parser.add_argument(
        "--layer-merged-cold-compact-matrix",
        action="store_true",
        help="Run all four layer-merged object x cold-compact combinations.",
    )
    parser.add_argument(
        "--ab-prefix-mode",
        choices=("per-layer", "batched"),
        default="batched",
        help="Measured LocalCPU probe mode used by the A/B report.",
    )
    parser.add_argument(
        "--h2d-gbps",
        type=float,
        default=22.0,
        help="Per-rank payload rate used by the synthetic H2D consumer.",
    )
    parser.add_argument(
        "--dense-consumer-layer-ms",
        type=float,
        default=0.0,
        help="Dense consumer CPU/launch cost per layer and consumed KV group.",
    )
    parser.add_argument(
        "--normal-sync-ms",
        type=float,
        default=0.0,
        help="Final foreground synchronization cost with the flag unset.",
    )
    parser.add_argument(
        "--cold-compact-sync-ms",
        type=float,
        default=0.0,
        help="Dense-load stream synchronization cost with the flag enabled.",
    )
    parser.add_argument(
        "--cold-compact-barrier-ms",
        type=float,
        default=0.0,
        help="Additional scheduler/completion barrier cost with the flag enabled.",
    )
    parser.add_argument(
        "--ab-output-json",
        type=Path,
        help="Write only the filtered cold-compact A/B report and samples.",
    )
    parser.add_argument(
        "--matrix-output-json",
        type=Path,
        help="Write only the filtered four-case feature matrix.",
    )
    args = parser.parse_args()

    if args.num_tokens < 1 or args.chunk_size < 1 or args.num_layers < 1:
        parser.error("token, chunk, and layer counts must be positive")
    if any(
        value < 2 or value % 2
        for value in (
            args.latent_bytes_per_token,
            args.indexer_bytes_per_token,
        )
    ):
        parser.error("bytes per token must be positive bfloat16-aligned values")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    if args.remote_gbps < 0 or args.rpc_overhead_us < 0:
        parser.error("synthetic remote timings must be non-negative")
    if args.h2d_gbps <= 0:
        parser.error("--h2d-gbps must be positive")
    if any(
        value < 0
        for value in (
            args.dense_consumer_layer_ms,
            args.normal_sync_ms,
            args.cold_compact_sync_ms,
            args.cold_compact_barrier_ms,
        )
    ):
        parser.error(
            "A/B consumer, synchronization, and barrier timings must be non-negative"
        )
    args.kv_groups = [int(value) for value in args.kv_groups.split(",") if value]
    if not args.kv_groups or any(value not in (0, 1) for value in args.kv_groups):
        parser.error("--kv-groups must contain 0 and/or 1")
    args.prefix_modes = [
        value.strip() for value in args.prefix_modes.split(",") if value.strip()
    ]
    valid_modes = {"per-layer", "batched"}
    if not args.prefix_modes or any(
        value not in valid_modes for value in args.prefix_modes
    ):
        parser.error("--prefix-modes must contain per-layer and/or batched")
    args.chunk_plan_modes = [
        value.strip()
        for value in args.chunk_plan_modes.split(",")
        if value.strip()
    ]
    valid_plan_modes = {"independent", "reuse"}
    if not args.chunk_plan_modes or any(
        value not in valid_plan_modes for value in args.chunk_plan_modes
    ):
        parser.error("--chunk-plan-modes must contain independent and/or reuse")
    if "reuse" in args.chunk_plan_modes and args.kv_groups != [0, 1]:
        parser.error("chunk-plan reuse requires --kv-groups 0,1")
    if args.cold_compact_ab:
        if args.kv_groups != [0, 1]:
            parser.error("--cold-compact-ab requires --kv-groups 0,1")
        if args.ab_prefix_mode not in args.prefix_modes:
            parser.error("--ab-prefix-mode must also be selected by --prefix-modes")
    elif args.ab_output_json is not None:
        parser.error("--ab-output-json requires --cold-compact-ab")
    if args.layer_merged_cold_compact_matrix:
        if args.kv_groups != [0, 1] or args.object_mode != "production":
            parser.error(
                "feature matrix requires --kv-groups 0,1 --object-mode production"
            )
    elif args.matrix_output_json is not None:
        parser.error(
            "--matrix-output-json requires --layer-merged-cold-compact-matrix"
        )
    return args


def main() -> None:
    """Run the requested cold-bootstrap benchmark matrix."""
    args = parse_args()
    chunks = math.ceil(args.num_tokens / args.chunk_size)
    full_pages, tail_tokens = divmod(args.num_tokens, args.chunk_size)
    mooncake_files = full_pages + (args.num_layers if tail_tokens else 0)
    print(
        f"tokens={args.num_tokens} chunk_size={args.chunk_size} "
        f"chunks={chunks} layers={args.num_layers} "
        "layout=mooncake_page_first_multi_buffer "
        f"objects={args.object_mode}"
    )
    print(
        "Each KV group retrieves "
        f"{chunks * args.num_layers} LMCache MemoryObjs from {mooncake_files} "
        f"Mooncake files ({full_pages} multi-buffer pages"
        f" + {args.num_layers if tail_tokens else 0} legacy tail files)."
    )
    print(
        "chunk plan modes="
        f"{','.join(args.chunk_plan_modes)} "
        "(reuse shares latent prefix hashes with the indexer group)"
    )
    if args.remote_gbps == 0:
        print("remote model=metadata-only (no synthetic payload delay)")
    else:
        print(
            f"remote model={args.remote_gbps:.3f} GB/s + "
            f"{args.rpc_overhead_us:.1f} us per Mooncake call"
        )

    results = []
    ab_report = None
    matrix_report = None
    if args.layer_merged_cold_compact_matrix:
        print("\n[progress] running interleaved four-case feature matrix")
        matrix_report = run_layer_merged_cold_compact_matrix(args)
        print_layer_merged_cold_compact_matrix(matrix_report)
    elif args.cold_compact_ab:
        print("\n[progress] running interleaved cold-compact off/on protocols")
        ab_report = build_cold_compact_ab_report(
            args,
            run_cold_compact_ab_cases(args),
        )
        print_cold_compact_ab(ab_report)
    else:
        for case_index, prefix_mode in enumerate(args.prefix_modes, start=1):
            print(
                f"[progress] case {case_index}/{len(args.prefix_modes)}: "
                f"prefix_mode={prefix_mode}; chunk plans run interleaved",
                flush=True,
            )
            results.extend(run_pair_cases(args, prefix_mode=prefix_mode))
        print_results(results)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                key: value
                for key, value in vars(args).items()
                if key not in {"output_json", "ab_output_json", "matrix_output_json"}
            },
            "results": [asdict(result) for result in results],
        }
        if ab_report is not None:
            payload["cold_compact_ab"] = asdict(ab_report)
        if matrix_report is not None:
            payload["layer_merged_cold_compact_matrix"] = matrix_report
        args.output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.output_json}")
    if args.ab_output_json is not None:
        assert ab_report is not None
        args.ab_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.ab_output_json.write_text(
            json.dumps(asdict(ab_report), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote filtered A/B JSON: {args.ab_output_json}")
    if args.matrix_output_json is not None:
        assert matrix_report is not None
        args.matrix_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.matrix_output_json.write_text(
            json.dumps(matrix_report, indent=2), encoding="utf-8"
        )
        print(f"Wrote filtered feature matrix: {args.matrix_output_json}")


if __name__ == "__main__":
    main()
