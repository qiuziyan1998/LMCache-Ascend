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

The remote store is synthetic so CPU metadata can be measured independently
from a particular Mooncake deployment. Its page accounting matches
``mooncake_page_first_multi_buffer=true``: one token chunk is one Mooncake
page/file, while every layer in that page remains a separate LMCache
``MemoryObj`` and destination buffer.
"""

# Standard
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional
import json
import math
import statistics
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_page_key
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
    LocalCPUPrefixGetResult,
)
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


GB = 1_000_000_000


@dataclass
class StageStats:
    """Stage timings and operation counts collected from one bootstrap."""

    metadata_s: float = 0.0
    local_probe_s: float = 0.0
    capacity_s: float = 0.0
    remote_s: float = 0.0
    pointer_s: float = 0.0
    handle_s: float = 0.0
    cold_prime_s: float = 0.0
    cold_total_s: float = 0.0
    warm_total_s: float = 0.0
    local_probe_calls: int = 0
    local_key_probes: int = 0
    remote_calls: int = 0
    remote_logical_keys: int = 0
    mooncake_page_objects: int = 0
    lmcache_memory_objects: int = 0
    pointer_objects: int = 0
    handle_objects: int = 0
    broadcast_envelopes: int = 0
    transferred_bytes: int = 0


@dataclass
class BenchmarkResult:
    """Median result for one benchmark case."""

    name: str
    prefix_mode: str
    kv_group: int
    num_layers: int
    num_chunks: int
    bytes_per_token: int
    repeats: int
    median: StageStats
    samples: list[StageStats] = field(default_factory=list)


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


class SyntheticLocalCPUBackend:
    """LocalCPU hot-cache surface with measurable prefix lookup modes."""

    def __init__(
        self,
        *,
        stats: StageStats,
        slab_bytes: int,
        prefix_mode: str,
    ) -> None:
        self.stats = stats
        self.prefix_mode = prefix_mode
        self.hot_cache: dict[CacheEngineKey, SyntheticMemoryObj] = {}
        self.cpu_lock = threading.RLock()
        root_allocator = SimpleNamespace(
            buffer=SyntheticBuffer(slab_bytes),
            address_manager=SyntheticAddressManager(slab_bytes),
        )
        self.memory_allocator = SimpleNamespace(
            _allocator=root_allocator,
            align_bytes=4096,
        )
        if prefix_mode == "per-layer":
            # The production code deliberately supports an older LMCache that
            # does not expose the cross-layer method.
            self.batched_get_prefixes_with_misses = None

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
        local_backend: SyntheticLocalCPUBackend,
        num_layers: int,
        num_tokens: int,
        chunk_size: int,
        bytes_per_token: int,
        remote_gbps: float,
        rpc_overhead_us: float,
    ) -> None:
        self.stats = stats
        self.local_backend = local_backend
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.chunk_size = chunk_size
        self.bytes_per_token = bytes_per_token
        self.remote_gbps = remote_gbps
        self.rpc_overhead_us = rpc_overhead_us
        self.shared_tensor = torch.empty(1, dtype=torch.uint8)

    def _tokens_for_chunk(self, chunk_index: int, num_chunks: int) -> int:
        if chunk_index + 1 < num_chunks:
            return self.chunk_size
        return self.num_tokens - self.chunk_size * (num_chunks - 1)

    def batched_get(
        self,
        keys: list[CacheEngineKey],
        location: Optional[str] = None,
    ) -> list[SyntheticMemoryObj]:
        if location != "RemoteBackend":
            raise ValueError(f"unexpected synthetic remote location: {location}")
        started = time.perf_counter()
        self.stats.remote_calls += 1
        self.stats.remote_logical_keys += len(keys)

        indices_by_page: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            page_id = mooncake_page_key(key, self.num_layers)
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
        transferred_bytes = sum(
            bytes_by_hash[mooncake_page_key(key, self.num_layers)] for key in keys
        )
        self.stats.mooncake_page_objects += num_chunks
        self.stats.lmcache_memory_objects += len(keys)
        self.stats.transferred_bytes += transferred_bytes

        delay_s = self.rpc_overhead_us / 1_000_000
        if self.remote_gbps > 0:
            delay_s += transferred_bytes / (self.remote_gbps * GB)
        if delay_s > 0:
            time.sleep(delay_s)

        results: list[SyntheticMemoryObj] = []
        for key in keys:
            logical_bytes = bytes_by_hash[mooncake_page_key(key, self.num_layers)]
            memory_obj = SyntheticMemoryObj(logical_bytes, self.shared_tensor)
            # StorageManager normally installs a cache-owned reference before
            # returning the caller-owned object.
            memory_obj.ref_count_up()
            self.local_backend.hot_cache[key] = memory_obj
            results.append(memory_obj)

        self.stats.remote_s += time.perf_counter() - started
        return results


class SyntheticGPUConnector:
    """Connector surface needed by cold and prepared sparse generators."""

    def __init__(self, stats: StageStats, num_layers: int) -> None:
        self.stats = stats
        self.num_layers = num_layers

    def append_sparse_chunk_ptr_cache_for_layer(
        self,
        layer_id: int,
        new_tensors: list[torch.Tensor],
        cached_chunk_dev_ptrs: list[list[int]],
        cached_chunk_ptrs_npu: list[Optional[torch.Tensor]],
    ) -> None:
        started = time.perf_counter()
        while len(cached_chunk_dev_ptrs) < self.num_layers:
            cached_chunk_dev_ptrs.append([])
        while len(cached_chunk_ptrs_npu) < self.num_layers:
            cached_chunk_ptrs_npu.append(None)
        new_ptrs = [id(tensor) for tensor in new_tensors]
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
        payload = yield
        while payload is not None:
            payload = yield
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
    config = LMCacheEngineConfig.from_legacy(
        backend="cpu",
        chunk_size=chunk_size,
        save_unfull_chunk=True,
    )
    config.dsa_two_groups = True
    config.extra_config = {"save_only_first_rank": True}
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
    bytes_per_token = (
        args.latent_bytes_per_token if kv_group == 0 else args.indexer_bytes_per_token
    )
    logical_bytes = args.num_tokens * bytes_per_token * args.num_layers
    slab_bytes = max(logical_bytes * 2, 1)
    local_backend = SyntheticLocalCPUBackend(
        stats=stats,
        slab_bytes=slab_bytes,
        prefix_mode=prefix_mode,
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
    )
    connector = SyntheticGPUConnector(stats, args.num_layers)

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = args.num_layers
    engine.use_layerwise = True
    engine.enable_shared_cpu_cache = True
    engine.save_only_first_rank = True
    engine.save_indexer_only_first_rank = True
    engine.shared_cpu_cache_generation = 1
    engine.metadata = metadata
    engine.token_database = token_database
    engine.storage_manager = storage_manager
    engine.gpu_connector = connector
    engine._shared_cpu_request_leases = {}
    engine.config = SimpleNamespace(
        experimental_sampled_layerwise_lookup=True,
        shared_cpu_remote_layers_per_batch=1,
        dsa_two_groups=True,
        extra_config={
            "mooncake_page_first_multi_buffer": True,
            "save_only_first_rank": True,
        },
    )
    engine.is_healthy = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._shared_local_cpu_backend = lambda: local_backend
    engine._is_rank0_shared_mem_obj = lambda _obj: True
    engine._shared_cpu_estimated_physical_chunk_bytes = (
        lambda _kv_group, num_tokens=None: (
            (num_tokens or args.chunk_size) * bytes_per_token
        )
    )
    engine._validate_rank0_shared_mem_obj = lambda _obj, **_kwargs: None
    engine.shared_cpu_rank0_request_object_ids = lambda _req_id, _kv_group: set()

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
    engine._broadcast_shared_envelope = lambda _envelope: setattr(
        stats,
        "broadcast_envelopes",
        stats.broadcast_envelopes + 1,
    )

    _timed_method(engine, "_ensure_retrieve_chunk_metadata", stats, "metadata_s")
    _timed_method(engine, "_shared_cpu_runtime_capacity_details", stats, "capacity_s")

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
) -> tuple[float, float]:
    started = time.perf_counter()
    first_mask = next(retriever)
    prime_s = time.perf_counter() - started
    if first_mask is None or int(first_mask.sum().item()) != num_tokens:
        raise AssertionError("cold bootstrap did not retrieve the complete token mask")
    selected = torch.arange(min(num_tokens, 2048), dtype=torch.long)
    for _ in range(num_layers):
        retriever.send((selected, 0))
    retriever.close()
    return prime_s, time.perf_counter() - started


def run_once(
    args: Namespace,
    *,
    kv_group: int,
    prefix_mode: str,
) -> StageStats:
    """Execute one cold bootstrap followed by one prepared warm retrieve."""
    stats = StageStats()
    engine, storage_manager, caches = _make_engine(
        args,
        stats,
        kv_group=kv_group,
        prefix_mode=prefix_mode,
    )
    tokens = list(range(args.num_tokens))
    ret_mask = torch.zeros(args.num_tokens, dtype=torch.bool)
    cold = engine.retrieve_layer_head_token_wise(
        tokens,
        ret_mask=ret_mask,
        kv_group=kv_group,
        req_id=f"cold-g{kv_group}-{prefix_mode}",
        **caches,
    )
    stats.cold_prime_s, stats.cold_total_s = _drive_retriever(
        cold,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
    )

    num_chunks = math.ceil(args.num_tokens / args.chunk_size)
    chunk_counts = [args.chunk_size] * num_chunks
    chunk_counts[-1] = args.num_tokens - args.chunk_size * (num_chunks - 1)
    prepared = build_prepared_sparse_source(
        caches["cached_tensors"],
        caches["cached_chunk_ptrs_npu"],
        num_layers=args.num_layers,
        total_tokens=args.num_tokens,
        chunk_token_counts=chunk_counts,
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
    _prime_s, stats.warm_total_s = _drive_retriever(
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
    if stats.remote_calls != 1:
        raise AssertionError(
            "page-first cold bootstrap used "
            f"{stats.remote_calls} remote calls, expected 1"
        )
    if stats.mooncake_page_objects != expected_chunks:
        raise AssertionError(
            "page-first Mooncake object count mismatch: "
            f"{stats.mooncake_page_objects} != {expected_chunks}"
        )
    if stats.lmcache_memory_objects != expected_memory_objects:
        raise AssertionError(
            "LMCache MemoryObj count mismatch: "
            f"{stats.lmcache_memory_objects} != {expected_memory_objects}"
        )
    return stats


def _median_stats(samples: list[StageStats]) -> StageStats:
    values: dict[str, Any] = {}
    for field_name in StageStats.__dataclass_fields__:
        field_values = [getattr(sample, field_name) for sample in samples]
        values[field_name] = statistics.median(field_values)
    return StageStats(**values)


def run_case(
    args: Namespace,
    *,
    kv_group: int,
    prefix_mode: str,
) -> BenchmarkResult:
    """Run warmup and measured repetitions for one group/probe configuration."""
    for _ in range(args.warmup):
        run_once(args, kv_group=kv_group, prefix_mode=prefix_mode)
    samples = [
        run_once(args, kv_group=kv_group, prefix_mode=prefix_mode)
        for _ in range(args.repeats)
    ]
    bytes_per_token = (
        args.latent_bytes_per_token if kv_group == 0 else args.indexer_bytes_per_token
    )
    return BenchmarkResult(
        name=f"g{kv_group}_{prefix_mode.replace('-', '_')}",
        prefix_mode=prefix_mode,
        kv_group=kv_group,
        num_layers=args.num_layers,
        num_chunks=math.ceil(args.num_tokens / args.chunk_size),
        bytes_per_token=bytes_per_token,
        repeats=args.repeats,
        median=_median_stats(samples),
        samples=samples,
    )


def _ms(seconds: float) -> float:
    return seconds * 1000


def print_results(results: list[BenchmarkResult]) -> None:
    """Print timing and topology summaries for all measured configurations."""
    print("\nCold bootstrap wall time (median ms; prime is the blocking first next())")
    print(
        f"{'case':24} {'prime':>9} {'cold total':>11} {'warm':>9} "
        f"{'metadata':>10} {'local probe':>12} {'capacity':>10} {'remote':>9}"
    )
    for result in results:
        stats = result.median
        print(
            f"{result.name:24} "
            f"{_ms(stats.cold_prime_s):9.3f} "
            f"{_ms(stats.cold_total_s):11.3f} "
            f"{_ms(stats.warm_total_s):9.3f} "
            f"{_ms(stats.metadata_s):10.3f} "
            f"{_ms(stats.local_probe_s):12.3f} "
            f"{_ms(stats.capacity_s):10.3f} "
            f"{_ms(stats.remote_s):9.3f}"
        )

    print("\nPage-first topology and metadata counts (median)")
    print(
        f"{'case':24} {'probe calls':>12} {'key probes':>11} "
        f"{'remote calls':>13} {'LMCache objs':>13} {'Mooncake files':>15} "
        f"{'buffers/file':>13}"
    )
    for result in results:
        stats = result.median
        buffers_per_file = (
            stats.lmcache_memory_objects / stats.mooncake_page_objects
            if stats.mooncake_page_objects
            else 0
        )
        print(
            f"{result.name:24} "
            f"{int(stats.local_probe_calls):12d} "
            f"{int(stats.local_key_probes):11d} "
            f"{int(stats.remote_calls):13d} "
            f"{int(stats.lmcache_memory_objects):13d} "
            f"{int(stats.mooncake_page_objects):15d} "
            f"{buffers_per_file:13.1f}"
        )

    by_group: dict[int, dict[str, BenchmarkResult]] = {}
    for result in results:
        by_group.setdefault(result.kv_group, {})[result.prefix_mode] = result
    print("\nCross-layer LocalCPU probe speedup")
    for kv_group, modes in sorted(by_group.items()):
        baseline = modes.get("per-layer")
        optimized = modes.get("batched")
        if baseline is None or optimized is None:
            continue
        prime_speedup = baseline.median.cold_prime_s / optimized.median.cold_prime_s
        probe_speedup = (
            baseline.median.local_probe_s / optimized.median.local_probe_s
            if optimized.median.local_probe_s
            else float("inf")
        )
        print(
            f"  kv_group={kv_group}: prime={prime_speedup:.3f}x, "
            f"local_probe={probe_speedup:.3f}x"
        )

    for prefix_mode in ("per-layer", "batched"):
        group_results = [
            result for result in results if result.prefix_mode == prefix_mode
        ]
        if len(group_results) < 2:
            continue
        combined_prime_ms = sum(
            _ms(result.median.cold_prime_s) for result in group_results
        )
        combined_total_ms = sum(
            _ms(result.median.cold_total_s) for result in group_results
        )
        print(
            f"  both groups ({prefix_mode}): prime={combined_prime_ms:.3f} ms, "
            f"cold_total={combined_total_ms:.3f} ms"
        )


def _serialize_result(result: BenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    return payload


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
        help="Synthetic fixed latency for the single page-first batch call.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.num_tokens < 1 or args.chunk_size < 1 or args.num_layers < 1:
        parser.error("token, chunk, and layer counts must be positive")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    if args.remote_gbps < 0 or args.rpc_overhead_us < 0:
        parser.error("synthetic remote timings must be non-negative")
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
    return args


def main() -> None:
    """Run the requested cold-bootstrap benchmark matrix."""
    args = parse_args()
    chunks = math.ceil(args.num_tokens / args.chunk_size)
    print(
        f"tokens={args.num_tokens} chunk_size={args.chunk_size} "
        f"chunks={chunks} layers={args.num_layers} "
        "layout=mooncake_page_first_multi_buffer"
    )
    print(
        "Each KV group retrieves "
        f"{chunks * args.num_layers} LMCache MemoryObjs from {chunks} "
        f"Mooncake page objects/files ({args.num_layers} buffers per page)."
    )
    if args.remote_gbps == 0:
        print("remote model=metadata-only (no synthetic payload delay)")
    else:
        print(
            f"remote model={args.remote_gbps:.3f} GB/s + "
            f"{args.rpc_overhead_us:.1f} us per page-first batch call"
        )

    results = []
    total_cases = len(args.kv_groups) * len(args.prefix_modes)
    cases = (
        (kv_group, prefix_mode)
        for kv_group in args.kv_groups
        for prefix_mode in args.prefix_modes
    )
    for case_index, (kv_group, prefix_mode) in enumerate(cases, start=1):
        print(
            f"[progress] case {case_index}/{total_cases}: "
            f"kv_group={kv_group} prefix_mode={prefix_mode}",
            flush=True,
        )
        results.append(
            run_case(
                args,
                kv_group=kv_group,
                prefix_mode=prefix_mode,
            )
        )
    print_results(results)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                key: value for key, value in vars(args).items() if key != "output_json"
            },
            "results": [_serialize_result(result) for result in results],
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
