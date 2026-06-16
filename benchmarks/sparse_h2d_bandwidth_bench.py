#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare CPU->NPU bandwidth under access patterns similar to LMCache sparse decode.

Tiers (same logical payload bytes unless noted):
  1. contiguous_copy      - single pinned buffer, one copy_  (peak DMA upper bound)
  2. multi_chunk_copy     - N pinned chunks, sequential copy_ to contiguous NPU
  3. indexed_gather_copy  - CPU gather by selected_token_idx, then one copy_
  4. lmcache_sparse_kernel - sparse_mla_dsa_batched_direct_kv_transfer (production op)
  5. lmcache_sparse_8layer - tier 4 repeated num_layers times (matches decode step)
  6. lmcache_pghs_kernel   - pipelined gather + H2D + scatter (optional path)
  7. lmcache_pghs_multilayer - tier 6 repeated num_layers times

Defaults mirror a typical sparse-decode profile (2048 tokens, ~258 chunks, MLA 576-d plane).

Usage:
  python sparse_h2d_bandwidth_bench.py --device npu:0
  python sparse_h2d_bandwidth_bench.py --num-sparse 512 --num-chunks 64 --chunk-size 8
"""

from __future__ import annotations

import argparse
import ctypes
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

try:
    import lmcache_ascend.c_ops as lmc_ops
except ImportError:  # pragma: no cover - optional on non-Ascend hosts
    lmc_ops = None

# Must match lmcache_ascend.v1.kv_format.KVCacheFormat enum order in C++.
KV_FMT_MLA = 3
KV_FMT_DSA = 4


@dataclass
class BenchResult:
    name: str
    bytes_moved: int
    wall_ms: float
    kernel_ms: Optional[float]
    gb_s: float
    kernel_gb_s: Optional[float]
    note: str = ""


def _gb_s(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return num_bytes / seconds / 1e9


def _sync(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _plane_elems(k_hidden: int, v_hidden: int, dsa_hidden: int, kv_fmt: int) -> int:
    plane = k_hidden + v_hidden
    if kv_fmt == KV_FMT_DSA:
        plane += dsa_hidden
    return plane


def _payload_bytes(num_sparse: int, plane_elems: int, elem_size: int = 2) -> int:
    return num_sparse * plane_elems * elem_size


def _global_token_location(
    token_idx: int, num_chunks: int, chunk_size: int, last_chunk_tokens: int
) -> Tuple[int, int]:
    """Map global LMCache token index to (chunk_id, offset_in_chunk)."""
    prefix_tokens = (num_chunks - 1) * chunk_size
    if token_idx < prefix_tokens:
        return token_idx // chunk_size, token_idx % chunk_size
    return num_chunks - 1, token_idx - prefix_tokens


@dataclass
class RegisteredCpuPool:
    """Ascend host memory registered via aclrtMallocHost + register_ptr."""

    _host_ptrs: List[int] = field(default_factory=list)
    _backing_buffers: List[object] = field(default_factory=list)

    def alloc_tensor(self, num_elems: int, dtype: torch.dtype) -> torch.Tensor:
        if lmc_ops is None:
            tensor = torch.randn(num_elems, dtype=dtype, pin_memory=True)
            return tensor

        elem_size = dtype.itemsize
        nbytes = num_elems * elem_size
        ptr = lmc_ops.alloc_pinned_ptr(nbytes, 0)
        dev_ptr = lmc_ops.get_device_ptr(ptr)
        if dev_ptr == 0:
            lmc_ops.free_pinned_ptr(ptr)
            raise RuntimeError(
                "alloc_pinned_ptr succeeded but get_device_ptr returned 0"
            )
        self._host_ptrs.append(ptr)

        buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
        self._backing_buffers.append(buf)
        tensor = torch.frombuffer(buf, dtype=dtype, count=num_elems)
        # Keep backing allocation alive via the pool; fill with random data.
        tensor.copy_(torch.randn(num_elems, dtype=dtype))
        return tensor

    def close(self) -> None:
        if lmc_ops is None:
            return
        for ptr in self._host_ptrs:
            lmc_ops.free_pinned_ptr(ptr)
        self._host_ptrs.clear()
        self._backing_buffers.clear()


def _make_cpu_chunks(
    pool: RegisteredCpuPool,
    num_chunks: int,
    chunk_size: int,
    plane_elems: int,
    dtype: torch.dtype,
    last_chunk_tokens: int,
) -> Tuple[List[torch.Tensor], int]:
    """CPU chunks using the same aclrt-registered host memory as LMCache."""
    chunks: List[torch.Tensor] = []
    for i in range(num_chunks):
        tokens = chunk_size if i < num_chunks - 1 else last_chunk_tokens
        chunks.append(pool.alloc_tensor(tokens * plane_elems, dtype))
    total_tokens = (num_chunks - 1) * chunk_size + last_chunk_tokens
    return chunks, total_tokens


def _make_mla_paged_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = torch.zeros(
        num_blocks, block_size, num_kv_heads, kv_lora_rank, dtype=dtype, device=device
    )
    v = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )
    return k, v


def _make_dsa_paged_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dsa_head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v = _make_mla_paged_cache(
        num_blocks,
        block_size,
        num_kv_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        device,
        dtype,
    )
    dsa = torch.zeros(
        num_blocks, block_size, 1, dsa_head_dim, dtype=dtype, device=device
    )
    return k, v, dsa


def _event_time_ms(fn: Callable[[], None], device: str, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    _sync(device)
    return start.elapsed_time(end)


def bench_contiguous_copy(
    pool: RegisteredCpuPool,
    payload_bytes: int,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> BenchResult:
    num_elems = payload_bytes // dtype.itemsize
    cpu = pool.alloc_tensor(num_elems, dtype)
    npu = torch.empty(num_elems, dtype=dtype, device=device)

    for _ in range(warmup):
        npu.copy_(cpu, non_blocking=True)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        npu.copy_(cpu, non_blocking=True)
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = payload_bytes * iters
    return BenchResult(
        name="contiguous_copy",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=None,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=None,
        note="Peak DMA upper bound (aclrt-registered host buffer).",
    )


def bench_multi_chunk_copy(
    chunks: Sequence[torch.Tensor],
    payload_bytes: int,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> BenchResult:
    total_elems = sum(c.numel() for c in chunks)
    npu = torch.empty(total_elems, dtype=dtype, device=device)

    def run_once() -> None:
        off = 0
        for chunk in chunks:
            n = chunk.numel()
            npu[off : off + n].copy_(chunk, non_blocking=True)
            off += n

    for _ in range(warmup):
        run_once()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = payload_bytes * iters
    return BenchResult(
        name="multi_chunk_copy",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=None,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=None,
        note=f"{len(chunks)} pinned chunks, sequential copy_ (no scatter).",
    )


def bench_indexed_gather_copy(
    chunks: Sequence[torch.Tensor],
    selected_token_idx: torch.Tensor,
    plane_elems: int,
    payload_bytes: int,
    device: str,
    dtype: torch.dtype,
    chunk_size: int,
    num_chunks: int,
    last_chunk_tokens: int,
    iters: int,
    warmup: int,
) -> BenchResult:
    """Gather selected tokens from multi-chunk CPU memory, then H2D."""
    idx_cpu = selected_token_idx.cpu().long()
    gather_rows: List[torch.Tensor] = []
    for t in idx_cpu.tolist():
        cid, within = _global_token_location(
            int(t), num_chunks, chunk_size, last_chunk_tokens
        )
        start = within * plane_elems
        end = start + plane_elems
        gather_rows.append(chunks[cid][start:end])
    gathered = torch.stack(gather_rows, dim=0).contiguous()  # [num_sparse, plane]
    npu = torch.empty_like(gathered, device=device)

    def run_once() -> None:
        npu.copy_(gathered, non_blocking=True)

    for _ in range(warmup):
        run_once()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = payload_bytes * iters
    return BenchResult(
        name="indexed_gather_copy",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=None,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=None,
        note="CPU gather from chunks + one contiguous copy_ (no paged scatter).",
    )


def bench_lmcache_sparse_kernel(
    *,
    chunks: Sequence[torch.Tensor],
    kv_caches,
    slot_mapping: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kv_fmt: int,
    k_hidden: int,
    v_hidden: int,
    dsa_hidden: int,
    payload_bytes: int,
    device: str,
    iters: int,
    warmup: int,
) -> BenchResult:
    if lmc_ops is None:
        return BenchResult(
            name="lmcache_sparse_kernel",
            bytes_moved=0,
            wall_ms=0.0,
            kernel_ms=None,
            gb_s=0.0,
            kernel_gb_s=None,
            note="SKIP: lmcache_ascend.c_ops not available.",
        )

    lmc_list = list(chunks)

    def run_once() -> None:
        lmc_ops.sparse_mla_dsa_batched_direct_kv_transfer(
            lmc_list,
            kv_caches,
            slot_mapping,
            selected_token_idx,
            chunk_size,
            total_tokens,
            kv_fmt,
            False,  # token_major: MLA/DSA CPU layout is interleaved
            False,  # vllm_two_major
            k_hidden,
            v_hidden,
            dsa_hidden,
        )

    kernel_ms = _event_time_ms(run_once, device, warmup=warmup)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = payload_bytes * iters
    return BenchResult(
        name="lmcache_sparse_kernel",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=kernel_ms,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=_gb_s(payload_bytes, kernel_ms / 1000.0),
        note="Production sparse multi-chunk scatter into paged KV.",
    )


def _make_pghs_pool(
    *,
    plane_elems: int,
    dtype: torch.dtype,
    device: str,
    micro_batch_tokens: int,
    max_slot_bytes: int,
) -> object:
    bytes_per_token = plane_elems * dtype.itemsize
    effective = lmc_ops.compute_effective_batch_tokens(
        micro_batch_tokens, max_slot_bytes, bytes_per_token
    )
    return lmc_ops.StagingBufferPool(
        max_slot_bytes,
        effective,
        bytes_per_token,
        dtype,
        torch.device(device),
    )


def bench_lmcache_pghs_kernel(
    *,
    pool: object,
    chunks: Sequence[torch.Tensor],
    kv_caches,
    slot_mapping: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kv_fmt: int,
    k_hidden: int,
    v_hidden: int,
    dsa_hidden: int,
    payload_bytes: int,
    device: str,
    iters: int,
    warmup: int,
    micro_batch_tokens: int,
    gather_thread_num: int,
    scatter_aiv_num: int,
) -> BenchResult:
    if lmc_ops is None:
        return BenchResult(
            name="lmcache_pghs_kernel",
            bytes_moved=0,
            wall_ms=0.0,
            kernel_ms=None,
            gb_s=0.0,
            kernel_gb_s=None,
            note="SKIP: lmcache_ascend.c_ops not available.",
        )

    lmc_list = list(chunks)

    def run_once() -> None:
        lmc_ops.sparse_mla_dsa_pghs_layer_transfer(
            pool,
            lmc_list,
            kv_caches,
            slot_mapping,
            selected_token_idx,
            chunk_size,
            total_tokens,
            kv_fmt,
            k_hidden,
            v_hidden,
            dsa_hidden,
            micro_batch_tokens,
            gather_thread_num,
            scatter_aiv_num,
        )

    kernel_ms = _event_time_ms(run_once, device, warmup=warmup)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = payload_bytes * iters
    return BenchResult(
        name="lmcache_pghs_kernel",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=kernel_ms,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=_gb_s(payload_bytes, kernel_ms / 1000.0),
        note=(
            f"PGHS gather+H2D+scatter micro_batch={micro_batch_tokens} "
            f"gather_threads={gather_thread_num} aiv={scatter_aiv_num}."
        ),
    )


def bench_lmcache_pghs_multilayer(
    *,
    pool: object,
    chunks: Sequence[torch.Tensor],
    kv_caches_layers: Sequence,
    slot_mapping: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kv_fmt: int,
    k_hidden: int,
    v_hidden: int,
    dsa_hidden: int,
    payload_bytes_per_layer: int,
    device: str,
    iters: int,
    warmup: int,
    micro_batch_tokens: int,
    gather_thread_num: int,
    scatter_aiv_num: int,
) -> BenchResult:
    if lmc_ops is None:
        return BenchResult(
            name="lmcache_pghs_multilayer",
            bytes_moved=0,
            wall_ms=0.0,
            kernel_ms=None,
            gb_s=0.0,
            kernel_gb_s=None,
            note="SKIP: lmcache_ascend.c_ops not available.",
        )

    lmc_list = list(chunks)
    num_layers = len(kv_caches_layers)
    total_payload = payload_bytes_per_layer * num_layers

    def run_all_layers_once() -> None:
        for layer_caches in kv_caches_layers:
            lmc_ops.sparse_mla_dsa_pghs_layer_transfer(
                pool,
                lmc_list,
                layer_caches,
                slot_mapping,
                selected_token_idx,
                chunk_size,
                total_tokens,
                kv_fmt,
                k_hidden,
                v_hidden,
                dsa_hidden,
                micro_batch_tokens,
                gather_thread_num,
                scatter_aiv_num,
            )

    kernel_ms = _event_time_ms(run_all_layers_once, device, warmup=warmup)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_all_layers_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = total_payload * iters
    return BenchResult(
        name="lmcache_pghs_multilayer",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=kernel_ms,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=_gb_s(total_payload, kernel_ms / 1000.0),
        note=f"{num_layers} layer PGHS launches (one decode step).",
    )


def bench_lmcache_sparse_multilayer(
    *,
    chunks: Sequence[torch.Tensor],
    kv_caches_layers: Sequence,
    slot_mapping: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kv_fmt: int,
    k_hidden: int,
    v_hidden: int,
    dsa_hidden: int,
    payload_bytes_per_layer: int,
    device: str,
    iters: int,
    warmup: int,
) -> BenchResult:
    if lmc_ops is None:
        return BenchResult(
            name="lmcache_sparse_multilayer",
            bytes_moved=0,
            wall_ms=0.0,
            kernel_ms=None,
            gb_s=0.0,
            kernel_gb_s=None,
            note="SKIP: lmcache_ascend.c_ops not available.",
        )

    lmc_list = list(chunks)
    num_layers = len(kv_caches_layers)
    total_payload = payload_bytes_per_layer * num_layers

    def run_all_layers_once() -> None:
        for layer_caches in kv_caches_layers:
            lmc_ops.sparse_mla_dsa_batched_direct_kv_transfer(
                lmc_list,
                layer_caches,
                slot_mapping,
                selected_token_idx,
                chunk_size,
                total_tokens,
                kv_fmt,
                False,
                False,
                k_hidden,
                v_hidden,
                dsa_hidden,
            )

    kernel_ms = _event_time_ms(run_all_layers_once, device, warmup=warmup)

    t0 = time.perf_counter()
    for _ in range(iters):
        run_all_layers_once()
    _sync(device)
    wall_s = time.perf_counter() - t0
    moved = total_payload * iters
    return BenchResult(
        name="lmcache_sparse_multilayer",
        bytes_moved=moved,
        wall_ms=wall_s * 1000,
        kernel_ms=kernel_ms,
        gb_s=_gb_s(moved, wall_s),
        kernel_gb_s=_gb_s(total_payload, kernel_ms / 1000.0),
        note=f"{num_layers} layer launches (one decode step).",
    )


def _print_result(r: BenchResult) -> None:
    line = (
        f"{r.name:28s}  wall={r.wall_ms:8.2f} ms  "
        f"bw={r.gb_s:6.2f} GB/s  bytes={r.bytes_moved / 1e6:8.2f} MB"
    )
    if r.kernel_ms is not None:
        line += f"  kernel={r.kernel_ms:7.3f} ms  kernel_bw={r.kernel_gb_s:6.2f} GB/s"
    print(line)
    if r.note:
        print(f"{'':28s}  ({r.note})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)

    # Workload shape (defaults from typical sparse-decode perf logs).
    p.add_argument("--num-sparse", type=int, default=2048)
    p.add_argument("--num-chunks", type=int, default=258)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument(
        "--last-chunk-tokens",
        type=int,
        default=0,
        help="Tokens in the last chunk (0 = same as chunk-size).",
    )
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=1)

    # MLA/DSA plane split (576 bf16 elems/token -> 1152 bytes/token).
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope-head-dim", type=int, default=64)
    p.add_argument("--dsa-head-dim", type=int, default=0, help=">0 enables DSA_KV")
    p.add_argument(
        "--kv-format",
        choices=["mla", "dsa"],
        default="mla",
        help="Paged NPU cache format for lmcache kernel tiers.",
    )
    p.add_argument(
        "--selected-mode",
        choices=["full", "random"],
        default="full",
        help="full: arange(num_sparse) like connector fallback; random: sample indices.",
    )
    p.add_argument(
        "--micro-batch-tokens",
        type=int,
        default=512,
        help="PGHS micro-batch size in tokens.",
    )
    p.add_argument(
        "--max-slot-mb",
        type=int,
        default=4,
        help="PGHS per-slot memory cap in MB.",
    )
    p.add_argument(
        "--gather-threads",
        type=int,
        default=4,
        help="PGHS CPU gather thread count.",
    )
    p.add_argument(
        "--scatter-aiv-num",
        type=int,
        default=4,
        help="PGHS scatter kernel aiv_num.",
    )
    p.add_argument(
        "--skip-pghs",
        action="store_true",
        help="Skip PGHS benchmark tiers.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    if device.startswith("npu"):
        torch.npu.set_device(device)

    if lmc_ops is None:
        raise RuntimeError(
            "lmcache_ascend.c_ops is required on Ascend. "
            "Install/build LMCache-Ascend before running this benchmark."
        )

    cpu_pool = RegisteredCpuPool()
    try:
        _run_benchmarks(args, device, dtype, cpu_pool)
    finally:
        cpu_pool.close()


def _run_benchmarks(
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
    cpu_pool: RegisteredCpuPool,
) -> None:
    use_dsa = args.kv_format == "dsa" or args.dsa_head_dim > 0
    kv_fmt = KV_FMT_DSA if use_dsa else KV_FMT_MLA
    dsa_hidden = args.dsa_head_dim if use_dsa else 0
    k_hidden = args.num_kv_heads * args.kv_lora_rank
    v_hidden = args.num_kv_heads * args.qk_rope_head_dim
    plane_elems = _plane_elems(k_hidden, v_hidden, dsa_hidden, kv_fmt)

    num_chunks = args.num_chunks
    chunk_size = args.chunk_size
    last_chunk_tokens = args.last_chunk_tokens or chunk_size
    chunks, total_tokens = _make_cpu_chunks(
        cpu_pool,
        num_chunks,
        chunk_size,
        plane_elems,
        dtype,
        last_chunk_tokens=last_chunk_tokens,
    )

    sparse_count = min(args.num_sparse, total_tokens)
    payload_bytes = _payload_bytes(sparse_count, plane_elems)

    num_blocks = max(256, math.ceil(sparse_count / args.block_size) + 16)
    page_size = num_blocks * args.block_size

    print("=== Sparse H2D bandwidth benchmark ===")
    print(
        f"device={device}  num_sparse={sparse_count}  num_chunks={num_chunks}  "
        f"chunk_size={chunk_size}  last_chunk={last_chunk_tokens}  "
        f"total_tokens={total_tokens}  plane_elems={plane_elems}  "
        f"payload={payload_bytes / 1e6:.4f} MB/layer  layers={args.num_layers}"
    )
    print()

    if args.selected_mode == "full":
        selected = torch.arange(sparse_count, dtype=torch.int32, device=device)
    else:
        selected = torch.tensor(
            torch.randperm(total_tokens)[:sparse_count].tolist(),
            dtype=torch.int32,
            device=device,
        )

    slots = torch.randperm(page_size, device=device)[:sparse_count]
    slot_mapping = slots.to(dtype=torch.long)

    results: List[BenchResult] = []

    results.append(
        bench_contiguous_copy(
            cpu_pool, payload_bytes, device, dtype, args.iters, args.warmup
        )
    )
    results.append(
        bench_multi_chunk_copy(
            chunks, payload_bytes, device, dtype, args.iters, args.warmup
        )
    )
    results.append(
        bench_indexed_gather_copy(
            chunks,
            selected,
            plane_elems,
            payload_bytes,
            device,
            dtype,
            chunk_size,
            num_chunks,
            last_chunk_tokens,
            max(1, args.iters // 5),
            args.warmup,
        )
    )

    if use_dsa:
        layer_cache = _make_dsa_paged_cache(
            num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            args.dsa_head_dim or 128,
            device,
            dtype,
        )
    else:
        layer_cache = _make_mla_paged_cache(
            num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            device,
            dtype,
        )

    results.append(
        bench_lmcache_sparse_kernel(
            chunks=chunks,
            kv_caches=layer_cache,
            slot_mapping=slot_mapping,
            selected_token_idx=selected,
            chunk_size=chunk_size,
            total_tokens=total_tokens,
            kv_fmt=kv_fmt,
            k_hidden=k_hidden,
            v_hidden=v_hidden,
            dsa_hidden=dsa_hidden if use_dsa else 0,
            payload_bytes=payload_bytes,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
        )
    )

    if not args.skip_pghs:
        max_slot_bytes = args.max_slot_mb * 1024 * 1024
        pghs_pool = _make_pghs_pool(
            plane_elems=plane_elems,
            dtype=dtype,
            device=device,
            micro_batch_tokens=args.micro_batch_tokens,
            max_slot_bytes=max_slot_bytes,
        )
        results.append(
            bench_lmcache_pghs_kernel(
                pool=pghs_pool,
                chunks=chunks,
                kv_caches=layer_cache,
                slot_mapping=slot_mapping,
                selected_token_idx=selected,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                kv_fmt=kv_fmt,
                k_hidden=k_hidden,
                v_hidden=v_hidden,
                dsa_hidden=dsa_hidden if use_dsa else 0,
                payload_bytes=payload_bytes,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                micro_batch_tokens=args.micro_batch_tokens,
                gather_thread_num=args.gather_threads,
                scatter_aiv_num=args.scatter_aiv_num,
            )
        )

    kv_layers = []
    for _ in range(args.num_layers):
        if use_dsa:
            kv_layers.append(
                _make_dsa_paged_cache(
                    num_blocks,
                    args.block_size,
                    args.num_kv_heads,
                    args.kv_lora_rank,
                    args.qk_rope_head_dim,
                    args.dsa_head_dim or 128,
                    device,
                    dtype,
                )
            )
        else:
            kv_layers.append(
                _make_mla_paged_cache(
                    num_blocks,
                    args.block_size,
                    args.num_kv_heads,
                    args.kv_lora_rank,
                    args.qk_rope_head_dim,
                    device,
                    dtype,
                )
            )

    results.append(
        bench_lmcache_sparse_multilayer(
            chunks=chunks,
            kv_caches_layers=kv_layers,
            slot_mapping=slot_mapping,
            selected_token_idx=selected,
            chunk_size=chunk_size,
            total_tokens=total_tokens,
            kv_fmt=kv_fmt,
            k_hidden=k_hidden,
            v_hidden=v_hidden,
            dsa_hidden=dsa_hidden if use_dsa else 0,
            payload_bytes_per_layer=payload_bytes,
            device=device,
            iters=max(1, args.iters // 5),
            warmup=args.warmup,
        )
    )

    if not args.skip_pghs:
        results.append(
            bench_lmcache_pghs_multilayer(
                pool=pghs_pool,
                chunks=chunks,
                kv_caches_layers=kv_layers,
                slot_mapping=slot_mapping,
                selected_token_idx=selected,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                kv_fmt=kv_fmt,
                k_hidden=k_hidden,
                v_hidden=v_hidden,
                dsa_hidden=dsa_hidden if use_dsa else 0,
                payload_bytes_per_layer=payload_bytes,
                device=device,
                iters=max(1, args.iters // 5),
                warmup=args.warmup,
                micro_batch_tokens=args.micro_batch_tokens,
                gather_thread_num=args.gather_threads,
                scatter_aiv_num=args.scatter_aiv_num,
            )
        )

    for r in results:
        _print_result(r)

    print()
    print("How to read results:")
    print("  - contiguous_copy:      DMA ceiling with aclrt-registered host memory.")
    print("  - multi_chunk_copy:     same bytes, fragmented pinned sources.")
    print("  - indexed_gather_copy:  adds CPU index/chunk lookup before H2D.")
    print("  - lmcache_sparse_*:     same as production direct multi-chunk path.")
    print("  - lmcache_pghs_*:       pipelined gather + H2D + scatter (optional).")
    print("  - Compare lmcache_sparse_* vs lmcache_pghs_* for end-to-end latency.")
    print("  - kernel_bw on lmcache_* should be comparable to LMCache perf logs.")


if __name__ == "__main__":
    main()
