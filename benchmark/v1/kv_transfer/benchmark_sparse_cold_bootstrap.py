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
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)
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
    hash_plan_reuses: int = 0


@dataclass
class BenchmarkResult:
    """Median result for one benchmark case."""

    name: str
    prefix_mode: str
    chunk_plan_mode: str
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
        self.shared_tensor = torch.empty(
            chunk_size * bytes_per_token,
            dtype=torch.uint8,
        )

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
        transferred_bytes = sum(
            bytes_by_hash[page_id] for page_id in page_ids
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
        for key, page_id in zip(keys, page_ids, strict=True):
            logical_bytes = bytes_by_hash[page_id]
            memory_obj = SyntheticMemoryObj(
                logical_bytes,
                self.shared_tensor[:logical_bytes],
            )
            # StorageManager normally installs a cache-owned reference before
            # returning the caller-owned object.
            memory_obj.ref_count_up()
            self.local_backend.hot_cache[key] = memory_obj
            results.append(memory_obj)

        self.stats.remote_s += time.perf_counter() - started
        return results


class SyntheticGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    """Connector surface needed by cold and prepared sparse generators."""

    def __init__(self, stats: StageStats, num_layers: int) -> None:
        # Deliberately skip the production connector initializer: it creates
        # device streams and buffers that a metadata-only benchmark must not
        # require. Inheriting the concrete layerwise type keeps this test
        # double compatible with LMCache's nominal connector assertion.
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


def run_pair_case(
    args: Namespace,
    *,
    prefix_mode: str,
    chunk_plan_mode: str,
) -> list[BenchmarkResult]:
    """Run one paired latent/indexer cold-bootstrap configuration."""
    for _ in range(args.warmup):
        _run_pair_once(
            args,
            prefix_mode=prefix_mode,
            chunk_plan_mode=chunk_plan_mode,
        )
    pair_samples = [
        _run_pair_once(
            args,
            prefix_mode=prefix_mode,
            chunk_plan_mode=chunk_plan_mode,
        )
        for _ in range(args.repeats)
    ]
    results = []
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

    print("\nPage-first topology and metadata counts (median)")
    print(
        f"{'case':38} {'probe calls':>12} {'key probes':>11} "
        f"{'remote calls':>13} {'LMCache objs':>13} {'Mooncake files':>15} "
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
            f"{args.rpc_overhead_us:.1f} us per page-first batch call"
        )

    results = []
    total_cases = len(args.chunk_plan_modes) * len(args.prefix_modes)
    cases = (
        (chunk_plan_mode, prefix_mode)
        for chunk_plan_mode in args.chunk_plan_modes
        for prefix_mode in args.prefix_modes
    )
    for case_index, (chunk_plan_mode, prefix_mode) in enumerate(cases, start=1):
        print(
            f"[progress] case {case_index}/{total_cases}: "
            f"chunk_plan={chunk_plan_mode} prefix_mode={prefix_mode}",
            flush=True,
        )
        results.extend(
            run_pair_case(
                args,
                prefix_mode=prefix_mode,
                chunk_plan_mode=chunk_plan_mode,
            )
        )
    print_results(results)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                key: value for key, value in vars(args).items() if key != "output_json"
            },
            "results": [asdict(result) for result in results],
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
