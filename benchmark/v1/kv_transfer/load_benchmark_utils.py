# SPDX-License-Identifier: Apache-2.0
"""Helpers for direct-vs-staging KV load micro-benchmarks."""
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union
import random
import statistics
import sys


def ensure_ascend_host_memory_registered() -> None:
    """Register Ascend aclrtHostRegister for PinMemoryAllocator buffers."""
    try:
        import lmcache_ascend  # noqa: F401
    except ImportError:
        pass

    try:
        import lmcache_ascend.c_ops as ascend_c_ops
    except ImportError:
        return

    sys.modules["lmcache.non_cuda_equivalents"] = ascend_c_ops

    import lmcache.v1.memory_management as mm

    mm.lmc_ops = ascend_c_ops


ensure_ascend_host_memory_registered()

# Third Party
from lmcache.v1.memory_management import MemoryObj, PinMemoryAllocator
import torch

# First Party
from lmcache_ascend.v1.npu_connector.utils import (
    batched_fused_single_layer_kv_transfer,
    prepare_sparse_direct_layer_state,
    sparse_mla_dsa_batched_direct_kv_transfer,
    sparse_mla_dsa_batched_direct_kv_transfer_fast,
)

KV_FORMAT_MLA = 3
KV_FORMAT_DSA = 4
ALIGN_BYTES = 4096


@dataclass(frozen=True)
class ChunkPartition:
    chunk_sizes: List[int]
    chunk_offsets: List[int]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_sizes)


@dataclass(frozen=True)
class MlaDsaDims:
    k_hidden_dims: int
    v_hidden_dims: int
    dsa_hidden_dims: int
    plane_elems: int


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    samples_ms: Tuple[float, ...]


@dataclass(frozen=True)
class LoadBenchmarkCase:
    format_name: str
    num_tokens: int
    chunk_size: int
    num_layers: int
    num_chunks: int
    num_selected: int
    host_bytes_moved: int
    staging_median_ms: float
    direct_median_ms: float
    direct_min_ms: float
    direct_fast_median_ms: float

    @property
    def direct_bandwidth_gbps(self) -> float:
        return bandwidth_gb_per_s(self.host_bytes_moved, self.direct_median_ms)

    @property
    def direct_bandwidth_peak_gbps(self) -> float:
        return bandwidth_gb_per_s(self.host_bytes_moved, self.direct_min_ms)

    @property
    def direct_vs_staging_ratio(self) -> float:
        """direct_time / staging_time (<1 means direct is faster)."""
        if self.staging_median_ms <= 0:
            return 0.0
        return self.direct_median_ms / self.staging_median_ms


@dataclass
class LoadBenchmarkHarness:
    """Prepared tensors for one benchmark case."""

    fmt: str
    kv_format: int
    num_tokens: int
    chunk_size: int
    num_layers: int
    num_selected: int
    partition: ChunkPartition
    dims: MlaDsaDims
    dtype: torch.dtype
    src_kv_cache: List[Tuple[torch.Tensor, ...]]
    staging_cache: torch.Tensor
    dst_staging: List[Tuple[torch.Tensor, ...]]
    dst_direct: List[Tuple[torch.Tensor, ...]]
    dst_direct_fast: List[Tuple[torch.Tensor, ...]]
    stacked_by_layer: List[List[torch.Tensor]]
    chunk_ptrs_by_layer: List[torch.Tensor]
    layer_states: list
    slot_mapping_full: torch.Tensor
    slot_mapping_packed: torch.Tensor
    selected_token_idx: torch.Tensor
    stacked_cpu_mem_objs: List[MemoryObj]
    pin_allocators: List[PinMemoryAllocator]

    @property
    def dense_load(self) -> bool:
        return self.num_selected == self.num_tokens

    def close(self) -> None:
        release_pin_memory_objects(self.stacked_cpu_mem_objs)
        for allocator in self.pin_allocators:
            allocator.close()


def bandwidth_gb_per_s(bytes_moved: int, time_ms: float) -> float:
    """Effective GB/s (decimal 1e9 bytes)."""
    if time_ms <= 0 or bytes_moved <= 0:
        return 0.0
    return (bytes_moved / 1e9) / (time_ms / 1000.0)


def compute_direct_host_bytes(
    *,
    num_selected: int,
    num_layers: int,
    plane_elems: int,
    element_size: int,
) -> int:
    return num_selected * plane_elems * element_size * num_layers


def format_bytes_short(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1e9:.3f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1e6:.3f} MB"
    return f"{num_bytes / 1e3:.3f} KB"


def recommended_num_blocks(num_tokens: int, block_size: int, margin: int = 8) -> int:
    return (num_tokens + block_size - 1) // block_size + margin


def compute_chunk_partition(num_tokens: int, chunk_size: int) -> ChunkPartition:
    chunk_sizes = [chunk_size] * (num_tokens // chunk_size)
    remainder = num_tokens % chunk_size
    if remainder:
        chunk_sizes.append(remainder)
    offsets: List[int] = []
    offset = 0
    for size in chunk_sizes:
        offsets.append(offset)
        offset += size
    return ChunkPartition(chunk_sizes=chunk_sizes, chunk_offsets=offsets)


def mla_dsa_dims_from_layer(
    layer: Tuple[torch.Tensor, ...],
    kv_format: int,
) -> MlaDsaDims:
    k_hidden = layer[0].shape[-2] * layer[0].shape[-1]
    v_hidden = layer[1].shape[-2] * layer[1].shape[-1]
    dsa_hidden = 0
    if kv_format == KV_FORMAT_DSA:
        dsa_hidden = layer[2].shape[-2] * layer[2].shape[-1]
    return MlaDsaDims(
        k_hidden_dims=k_hidden,
        v_hidden_dims=v_hidden,
        dsa_hidden_dims=dsa_hidden,
        plane_elems=k_hidden + v_hidden + dsa_hidden,
    )


def as_vllm_kv_caches(
    layer: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    return layer if isinstance(layer, torch.Tensor) else tuple(layer)


def _align_up(value: int, align: int = ALIGN_BYTES) -> int:
    return (value + align - 1) // align * align


def create_pin_memory_allocator(
    partition: ChunkPartition,
    dims: MlaDsaDims,
    dtype: torch.dtype,
) -> PinMemoryAllocator:
    ensure_ascend_host_memory_registered()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.current_device()

    element_size = torch.empty(0, dtype=dtype).element_size()
    pinned_bytes = sum(
        _align_up(chunk_tokens * dims.plane_elems * element_size)
        for chunk_tokens in partition.chunk_sizes
    )
    pinned_bytes = max(pinned_bytes, 64 * 1024 * 1024)
    allocator = PinMemoryAllocator(int(pinned_bytes))
    if allocator.buffer.numel() == 0:
        raise RuntimeError(
            f"PinMemoryAllocator returned empty buffer (requested {pinned_bytes} B)"
        )
    return allocator


def allocate_stacked_cpu_chunks(
    mem_allocator: PinMemoryAllocator,
    partition: ChunkPartition,
    dims: MlaDsaDims,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[MemoryObj]]:
    chunks: List[torch.Tensor] = []
    mem_objs: List[MemoryObj] = []
    for chunk_tokens in partition.chunk_sizes:
        mem_obj = mem_allocator.allocate(
            torch.Size([chunk_tokens * dims.plane_elems]), dtype
        )
        if mem_obj is None or mem_obj.tensor is None:
            raise RuntimeError("PinMemoryAllocator OOM while allocating CPU chunk")
        chunks.append(mem_obj.tensor)
        mem_objs.append(mem_obj)
    return chunks, mem_objs


def release_pin_memory_objects(mem_objs: Sequence[MemoryObj]) -> None:
    for mem_obj in mem_objs:
        mem_obj.ref_count_down()


def populate_stacked_cpu_from_src(
    *,
    src_layer: Tuple[torch.Tensor, ...],
    stacked_cpu_chunks: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    slot_mapping_full: torch.Tensor,
    partition: ChunkPartition,
    dims: MlaDsaDims,
    kv_format: int,
) -> None:
    batched_fused_single_layer_kv_transfer(
        list(stacked_cpu_chunks),
        staging_cache,
        as_vllm_kv_caches(src_layer),
        slot_mapping_full,
        partition.chunk_offsets,
        partition.chunk_sizes,
        True,
        kv_format,
        False,
        False,
        dims.k_hidden_dims,
        dims.v_hidden_dims,
        dims.dsa_hidden_dims,
    )


def build_chunk_ptrs_npu(
    cpu_tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    import lmcache_ascend.c_ops as lmc_ops

    dev_ptrs = []
    for i, tensor in enumerate(cpu_tensors):
        dev_ptr = int(lmc_ops.get_device_ptr(int(tensor.data_ptr())))
        if dev_ptr == 0:
            raise RuntimeError(
                f"get_device_ptr returned 0 for CPU chunk {i}; "
                "ensure host memory is aclrtHostRegister'd"
            )
        dev_ptrs.append(dev_ptr)
    return torch.tensor(dev_ptrs, dtype=torch.long, device=device)


def lmc_host_interleaved_for_kv_format(kv_format: int) -> bool:
    return kv_format not in (KV_FORMAT_MLA, KV_FORMAT_DSA)


def build_selected_token_idx(
    num_selected: int,
    total_tokens: int,
    *,
    device: torch.device,
    dense: bool,
    seed: int,
) -> torch.Tensor:
    if dense:
        indices = list(range(num_selected))
    else:
        indices = sorted(random.Random(seed).sample(range(total_tokens), num_selected))
    return torch.tensor(indices, dtype=torch.int32, device=device)


def build_slot_mapping_packed(
    slot_mapping_full: torch.Tensor,
    selected_token_idx: torch.Tensor,
) -> torch.Tensor:
    return slot_mapping_full.index_select(0, selected_token_idx.to(torch.long))


def empty_paged_kv_cache_like(
    kv_cache: List[Tuple[torch.Tensor, ...]],
) -> List[Tuple[torch.Tensor, ...]]:
    return [tuple(torch.empty_like(t) for t in layer) for layer in kv_cache]


def run_staging_load_layer(
    *,
    stacked_cpu_chunks: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    dst_layer: Tuple[torch.Tensor, ...],
    slot_mapping_full: torch.Tensor,
    partition: ChunkPartition,
    dims: MlaDsaDims,
    kv_format: int,
) -> None:
    batched_fused_single_layer_kv_transfer(
        list(stacked_cpu_chunks),
        staging_cache,
        as_vllm_kv_caches(dst_layer),
        slot_mapping_full,
        partition.chunk_offsets,
        partition.chunk_sizes,
        False,
        kv_format,
        False,
        False,
        dims.k_hidden_dims,
        dims.v_hidden_dims,
        dims.dsa_hidden_dims,
    )


def run_direct_load_layer(
    *,
    stacked_cpu_chunks: Sequence[torch.Tensor],
    dst_layer: Tuple[torch.Tensor, ...],
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    dims: MlaDsaDims,
    kv_format: int,
    chunk_ptrs_npu: torch.Tensor,
) -> None:
    sparse_mla_dsa_batched_direct_kv_transfer(
        list(stacked_cpu_chunks),
        as_vllm_kv_caches(dst_layer),
        slot_mapping_packed,
        selected_token_idx,
        chunk_size,
        total_tokens,
        kv_format,
        False,
        False,
        dims.k_hidden_dims,
        dims.v_hidden_dims,
        dims.dsa_hidden_dims,
        lmc_host_interleaved_for_kv_format(kv_format),
        chunk_ptrs_npu,
    )


def run_direct_fast_load_layer(
    *,
    layer_state,
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_ptrs_npu: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kv_format: int,
) -> None:
    sparse_mla_dsa_batched_direct_kv_transfer_fast(
        layer_state,
        slot_mapping_packed,
        selected_token_idx,
        chunk_ptrs_npu,
        chunk_size,
        total_tokens,
        lmc_host_interleaved_for_kv_format(kv_format),
        False,
    )


def time_npu_callable(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> TimingStats:
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()

    samples: List[float] = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return TimingStats(
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        samples_ms=tuple(samples),
    )


def build_load_benchmark_harness(
    *,
    fmt: str,
    src_kv_cache: List[Tuple[torch.Tensor, ...]],
    num_tokens: int,
    chunk_size: int,
    num_layers: int,
    num_selected: int,
    seed: int,
) -> LoadBenchmarkHarness:
    device = src_kv_cache[0][0].device
    dtype = src_kv_cache[0][0].dtype
    kv_format = KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA
    partition = compute_chunk_partition(num_tokens, chunk_size)
    dims = mla_dsa_dims_from_layer(src_kv_cache[0], kv_format)

    page_buffer_size = src_kv_cache[0][0].shape[0] * src_kv_cache[0][0].shape[1]
    slot_mapping_full = torch.tensor(
        random.Random(seed).sample(range(page_buffer_size), num_tokens),
        device=device,
        dtype=torch.long,
    )
    dense_load = num_selected == num_tokens
    selected_token_idx = build_selected_token_idx(
        num_selected, num_tokens, device=device, dense=dense_load, seed=seed
    )
    slot_mapping_packed = build_slot_mapping_packed(
        slot_mapping_full, selected_token_idx
    )

    staging_cache = torch.empty(num_tokens * dims.plane_elems, dtype=dtype, device=device)
    dst_staging = empty_paged_kv_cache_like(src_kv_cache)
    dst_direct = empty_paged_kv_cache_like(src_kv_cache)
    dst_direct_fast = empty_paged_kv_cache_like(src_kv_cache)

    stacked_by_layer: List[List[torch.Tensor]] = []
    chunk_ptrs_by_layer: List[torch.Tensor] = []
    layer_states = []
    stacked_cpu_mem_objs: List[MemoryObj] = []
    pin_allocators: List[PinMemoryAllocator] = []

    for layer_id in range(num_layers):
        mem_allocator = create_pin_memory_allocator(partition, dims, dtype)
        pin_allocators.append(mem_allocator)
        stacked, layer_mem_objs = allocate_stacked_cpu_chunks(
            mem_allocator, partition, dims, dtype
        )
        stacked_cpu_mem_objs.extend(layer_mem_objs)
        populate_stacked_cpu_from_src(
            src_layer=src_kv_cache[layer_id],
            stacked_cpu_chunks=stacked,
            staging_cache=staging_cache,
            slot_mapping_full=slot_mapping_full,
            partition=partition,
            dims=dims,
            kv_format=kv_format,
        )
        torch.npu.synchronize()
        stacked_by_layer.append(stacked)
        chunk_ptrs = build_chunk_ptrs_npu(stacked, device)
        chunk_ptrs_by_layer.append(chunk_ptrs)
        layer_states.append(
            prepare_sparse_direct_layer_state(
                stacked[0],
                as_vllm_kv_caches(dst_direct_fast[layer_id]),
                slot_mapping_full,
                False,
                False,
                kv_format,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
                num_tokens,
            )
        )

    return LoadBenchmarkHarness(
        fmt=fmt,
        kv_format=kv_format,
        num_tokens=num_tokens,
        chunk_size=chunk_size,
        num_layers=num_layers,
        num_selected=num_selected,
        partition=partition,
        dims=dims,
        dtype=dtype,
        src_kv_cache=src_kv_cache,
        staging_cache=staging_cache,
        dst_staging=dst_staging,
        dst_direct=dst_direct,
        dst_direct_fast=dst_direct_fast,
        stacked_by_layer=stacked_by_layer,
        chunk_ptrs_by_layer=chunk_ptrs_by_layer,
        layer_states=layer_states,
        slot_mapping_full=slot_mapping_full,
        slot_mapping_packed=slot_mapping_packed,
        selected_token_idx=selected_token_idx,
        stacked_cpu_mem_objs=stacked_cpu_mem_objs,
        pin_allocators=pin_allocators,
    )


def run_all_staging_layers(h: LoadBenchmarkHarness) -> None:
    for layer_id in range(h.num_layers):
        run_staging_load_layer(
            stacked_cpu_chunks=h.stacked_by_layer[layer_id],
            staging_cache=h.staging_cache,
            dst_layer=h.dst_staging[layer_id],
            slot_mapping_full=h.slot_mapping_full,
            partition=h.partition,
            dims=h.dims,
            kv_format=h.kv_format,
        )


def run_all_direct_layers(h: LoadBenchmarkHarness) -> None:
    for layer_id in range(h.num_layers):
        run_direct_load_layer(
            stacked_cpu_chunks=h.stacked_by_layer[layer_id],
            dst_layer=h.dst_direct[layer_id],
            slot_mapping_packed=h.slot_mapping_packed,
            selected_token_idx=h.selected_token_idx,
            chunk_size=h.chunk_size,
            total_tokens=h.num_tokens,
            dims=h.dims,
            kv_format=h.kv_format,
            chunk_ptrs_npu=h.chunk_ptrs_by_layer[layer_id],
        )


def run_all_direct_fast_layers(h: LoadBenchmarkHarness) -> None:
    for layer_id in range(h.num_layers):
        run_direct_fast_load_layer(
            layer_state=h.layer_states[layer_id],
            slot_mapping_packed=h.slot_mapping_packed,
            selected_token_idx=h.selected_token_idx,
            chunk_ptrs_npu=h.chunk_ptrs_by_layer[layer_id],
            chunk_size=h.chunk_size,
            total_tokens=h.num_tokens,
            kv_format=h.kv_format,
        )


def verify_load_benchmark_harness(
    h: LoadBenchmarkHarness,
    check_fn: Callable[..., None],
    **check_kwargs,
) -> None:
    if h.dense_load:
        run_all_staging_layers(h)
    run_all_direct_layers(h)
    run_all_direct_fast_layers(h)
    torch.npu.synchronize()

    if h.dense_load:
        check_fn(
            h.src_kv_cache,
            h.dst_staging,
            h.slot_mapping_full,
            label="staging",
            **check_kwargs,
        )
    check_fn(
        h.src_kv_cache,
        h.dst_direct,
        h.slot_mapping_packed,
        label="direct",
        **check_kwargs,
    )
    check_fn(
        h.src_kv_cache,
        h.dst_direct_fast,
        h.slot_mapping_packed,
        label="direct_fast",
        **check_kwargs,
    )


def benchmark_load_harness(
    h: LoadBenchmarkHarness,
    *,
    warmup: int,
    iters: int,
) -> LoadBenchmarkCase:
    # Reset dst buffers before timing (verify may have filled them).
    h.dst_staging = empty_paged_kv_cache_like(h.src_kv_cache)
    h.dst_direct = empty_paged_kv_cache_like(h.src_kv_cache)
    h.dst_direct_fast = empty_paged_kv_cache_like(h.src_kv_cache)

    staging_stats = time_npu_callable(
        lambda: run_all_staging_layers(h), warmup=warmup, iters=iters
    )
    direct_stats = time_npu_callable(
        lambda: run_all_direct_layers(h), warmup=warmup, iters=iters
    )
    direct_fast_stats = time_npu_callable(
        lambda: run_all_direct_fast_layers(h), warmup=warmup, iters=iters
    )

    host_bytes = compute_direct_host_bytes(
        num_selected=h.num_selected,
        num_layers=h.num_layers,
        plane_elems=h.dims.plane_elems,
        element_size=h.dtype.itemsize,
    )
    return LoadBenchmarkCase(
        format_name=h.fmt,
        num_tokens=h.num_tokens,
        chunk_size=h.chunk_size,
        num_layers=h.num_layers,
        num_chunks=h.partition.num_chunks,
        num_selected=h.num_selected,
        host_bytes_moved=host_bytes,
        staging_median_ms=staging_stats.median_ms,
        direct_median_ms=direct_stats.median_ms,
        direct_min_ms=direct_stats.min_ms,
        direct_fast_median_ms=direct_fast_stats.median_ms,
    )
