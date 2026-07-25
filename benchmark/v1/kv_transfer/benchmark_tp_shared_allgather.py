#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare full sparse K load with TP-sharded load plus AllGather.

This is a standalone data-plane benchmark. It does not use or modify the
vLLM/LMCache runtime connector. The benchmark preserves the relevant runtime
layouts and operations:

* rank 0 creates and populates one shared, pinned LMCache CPU slab;
* every rank maps the same single-plane K chunks (chunk size 256 by default);
* the baseline uses the production sparse direct host-to-paged-NPU kernel;
* the proposed path transfers a rank-contiguous 1/world_size token slice into
  an NPU staging cache, AllGathers K, then scatters the full result into a
  vLLM-shaped paged destination;
* isolated H2D, AllGather, and scatter measurements show where time is spent;
* a no-scatter path gives the lower bound when the gathered output can be used
  directly;
* a double-buffered pipeline subdivides each rank's shard so sparse H2D for
  chunk N+1 can overlap AllGather/scatter for chunk N.

The selected-token count must be divisible by the TP world size. Contiguous
rank slices then make the rank-ordered AllGather output match the original
selected-token order without a reorder kernel.

Run under the same NUMA and HCCL settings as the service, for example:

  HCCL_DETERMINISTIC=strict \
  numactl --interleave=all \
  python benchmark/v1/kv_transfer/benchmark_tp_shared_allgather.py \
      --devices 0,1,2,3,4,5,6,7 --verify
"""

from __future__ import annotations

# Standard
import argparse
from dataclasses import dataclass
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import socket
import statistics
import sys
import time
from typing import Any, Callable
import uuid

# Third Party
import torch
import torch.distributed as dist

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    torch_npu = None

# First Party
from load_benchmark_utils import (  # noqa: E402
    ALIGN_BYTES,
    KV_FORMAT_DSA_INDEX,
    LoadBenchmarkHarness,
    MlaDsaDims,
    as_vllm_kv_caches,
    build_load_benchmark_harness,
    compute_chunk_partition,
    format_bytes_short,
    prepare_sparse_direct_destination_state,
    recommended_num_blocks,
    run_all_direct_fast_layers,
    run_direct_fast_load_layer,
    shared_cpu_slab_bytes,
)
from lmcache.v1.memory_management import MixedMemoryAllocator  # noqa: E402
from lmcache.v1.shared_cpu_cache import SharedSlabMapping  # noqa: E402


CASE_NAMES = (
    "src_contiguous__dst_contiguous",
    "src_contiguous__dst_scattered",
    "src_scattered__dst_contiguous",
    "src_scattered__dst_scattered",
)
PATH_DIRECT = "direct_full"
PATH_SHARDED = "shard_h2d_allgather_scatter"
PATH_H2D = "stage_local_h2d"
PATH_ALLGATHER = "stage_allgather"
PATH_SCATTER = "stage_scatter"
PATH_NO_SCATTER = "shard_h2d_allgather_no_scatter"
PATH_PIPELINED = "pipeline_h2d_allgather_scatter"
PATH_NAMES = (
    PATH_DIRECT,
    PATH_SHARDED,
    PATH_NO_SCATTER,
    PATH_PIPELINED,
    PATH_H2D,
    PATH_ALLGATHER,
    PATH_SCATTER,
)
COMPARISON_PATHS = (
    PATH_DIRECT,
    PATH_SHARDED,
    PATH_NO_SCATTER,
    PATH_PIPELINED,
)
STAGE_PATHS = (PATH_H2D, PATH_ALLGATHER, PATH_SCATTER)
HCCL_HOST_PORT_COUNT = 32
HCCL_MIN_BASE_PORT = 1024
HCCL_MAX_BASE_PORT = 65520


@dataclass(frozen=True)
class TransferCase:
    """Source token indices and final paged destination slots for one case."""

    selected_token_idx: torch.Tensor
    target_slot_mapping: torch.Tensor
    local_selected_token_idx: torch.Tensor
    pipeline_target_slot_mappings: tuple[torch.Tensor, ...]


@dataclass
class ShardedAllGatherHarness:
    """NPU buffers and prepared state for the proposed two-stage path."""

    local_staging: tuple[torch.Tensor]
    local_destination_state: Any
    local_slot_mapping: torch.Tensor
    local_k: torch.Tensor
    gathered_k: torch.Tensor
    destinations: list[tuple[torch.Tensor]]
    pipeline_staging: list[tuple[torch.Tensor]]
    pipeline_destination_states: list[Any]
    pipeline_local_k: list[torch.Tensor]
    pipeline_gathered_k: list[torch.Tensor]
    pipeline_local_slot_mapping: torch.Tensor
    pipeline_load_done: list[Any]
    pipeline_buffer_free: list[Any]
    pipeline_buffer_used: list[bool]
    pipeline_entry: Any
    load_stream: Any
    collective_stream: Any
    pipeline_chunk_ranges: tuple[tuple[int, int], ...]
    local_selected: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-tokens", type=int, default=18880)
    parser.add_argument("--num-selected", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=64,
        help=(
            "Accepted for command compatibility but ignored: this benchmark "
            "models only the decode-critical K plane."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--pipeline-chunks",
        type=int,
        default=4,
        help=(
            "Subdivide each rank's 1/world_size shard into this many pieces "
            "for the double-buffered H2D/AllGather pipeline."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--hccl-if-base-port",
        type=int,
        default=None,
        help=(
            "HCCL host-NIC base port. By default, preserve HCCL_IF_BASE_PORT "
            "from the environment or choose a free block automatically."
        ),
    )
    parser.add_argument(
        "--inherit-hccl-socket-ports",
        action="store_true",
        help=(
            "Preserve HCCL_HOST_SOCKET_PORT_RANGE and "
            "HCCL_NPU_SOCKET_PORT_RANGE. By default both are set to 'auto' "
            "to avoid collisions with other or recently stopped HCCL jobs."
        ),
    )
    args = parser.parse_args()

    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if len(devices) < 2:
        parser.error("--devices must contain at least two devices")
    if len(set(devices)) != len(devices):
        parser.error("--devices cannot contain duplicates")
    if not 0 < args.num_selected <= args.num_tokens:
        parser.error("--num-selected must be in [1, --num-tokens]")
    if args.num_selected % len(devices):
        parser.error("--num-selected must be divisible by the number of devices")
    for name in (
        "num_tokens",
        "chunk_size",
        "num_layers",
        "block_size",
        "num_kv_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "iters",
        "repeats",
        "pipeline_chunks",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    local_selected = args.num_selected // len(devices)
    if args.pipeline_chunks > local_selected:
        parser.error(
            "--pipeline-chunks cannot exceed "
            "--num-selected / number of devices"
        )
    if args.hccl_if_base_port is not None and not (
        HCCL_MIN_BASE_PORT
        <= args.hccl_if_base_port
        <= HCCL_MAX_BASE_PORT - HCCL_HOST_PORT_COUNT + 1
    ):
        parser.error(
            "--hccl-if-base-port must leave room for "
            f"{HCCL_HOST_PORT_COUNT} ports in "
            f"[{HCCL_MIN_BASE_PORT}, {HCCL_MAX_BASE_PORT}]"
        )
    args.devices = devices
    return args


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _can_bind_tcp_port_block(base_port: int, count: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(base_port, base_port + count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def _find_free_tcp_port_block(count: int) -> int:
    """Find a free consecutive host-port block outside common ephemeral ports."""
    candidates = list(
        range(
            10_000,
            32_000 - count + 1,
            count,
        )
    )
    random.Random(os.getpid() ^ time.monotonic_ns()).shuffle(candidates)
    for base_port in candidates:
        if _can_bind_tcp_port_block(base_port, count):
            return base_port
    raise RuntimeError(
        f"could not find {count} consecutive free TCP ports for HCCL; "
        "pass --hccl-if-base-port with an available reserved range"
    )


def _configure_hccl_base_port(args: argparse.Namespace) -> tuple[int, str]:
    if args.hccl_if_base_port is not None:
        base_port = args.hccl_if_base_port
        source = "command line"
    elif "HCCL_IF_BASE_PORT" in os.environ:
        try:
            base_port = int(os.environ["HCCL_IF_BASE_PORT"])
        except ValueError as exc:
            raise RuntimeError(
                "HCCL_IF_BASE_PORT must be an integer, got "
                f"{os.environ['HCCL_IF_BASE_PORT']!r}"
            ) from exc
        source = "environment"
    else:
        base_port = _find_free_tcp_port_block(HCCL_HOST_PORT_COUNT)
        source = "auto-selected"

    if not (
        HCCL_MIN_BASE_PORT <= base_port <= HCCL_MAX_BASE_PORT - HCCL_HOST_PORT_COUNT + 1
    ):
        raise RuntimeError(
            f"HCCL_IF_BASE_PORT={base_port} does not leave room for "
            f"{HCCL_HOST_PORT_COUNT} ports in "
            f"[{HCCL_MIN_BASE_PORT}, {HCCL_MAX_BASE_PORT}]"
        )
    if not _can_bind_tcp_port_block(base_port, HCCL_HOST_PORT_COUNT):
        last_port = base_port + HCCL_HOST_PORT_COUNT - 1
        raise RuntimeError(
            f"HCCL host-port range {base_port}-{last_port} is already in use; "
            "stop the conflicting job or pass --hccl-if-base-port with a free "
            "reserved range"
        )

    os.environ["HCCL_IF_BASE_PORT"] = str(base_port)
    args.hccl_if_base_port = base_port
    return base_port, source


def _configure_hccl_socket_ports(args: argparse.Namespace) -> dict[str, str]:
    """Avoid fixed host- and NPU-side HCCL ports used by root-info setup."""
    names = (
        "HCCL_HOST_SOCKET_PORT_RANGE",
        "HCCL_NPU_SOCKET_PORT_RANGE",
    )
    if not args.inherit_hccl_socket_ports:
        if args.hccl_if_base_port is None:
            os.environ.pop("HCCL_IF_BASE_PORT", None)
            os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"
        else:
            os.environ.pop("HCCL_HOST_SOCKET_PORT_RANGE", None)
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
    values = {name: os.environ.get(name, "<unset>") for name in names}
    args.hccl_socket_ports = values
    return values


def _split_evenly(size: int, pieces: int) -> tuple[tuple[int, int], ...]:
    """Return non-empty, near-even half-open ranges covering ``size``."""
    quotient, remainder = divmod(size, pieces)
    ranges = []
    start = 0
    for piece in range(pieces):
        piece_size = quotient + (1 if piece < remainder else 0)
        end = start + piece_size
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _locality_cases(
    *,
    num_tokens: int,
    num_selected: int,
    num_slots: int,
    rank: int,
    world_size: int,
    pipeline_chunks: int,
    seed: int,
    device: torch.device,
) -> dict[str, TransferCase]:
    source_contiguous = torch.arange(
        num_selected,
        dtype=torch.int32,
        device=device,
    )
    source_scattered = torch.tensor(
        random.Random(seed).sample(range(num_tokens), num_selected),
        dtype=torch.int32,
        device=device,
    )
    destination_contiguous = torch.arange(
        num_selected,
        dtype=torch.long,
        device=device,
    )
    destination_scattered = torch.tensor(
        random.Random(seed + 1).sample(range(num_slots), num_selected),
        dtype=torch.long,
        device=device,
    )
    local_selected = num_selected // world_size
    local_start = rank * local_selected
    local_end = local_start + local_selected
    pipeline_chunk_ranges = _split_evenly(local_selected, pipeline_chunks)

    def make_case(
        selected: torch.Tensor,
        destination: torch.Tensor,
    ) -> TransferCase:
        pipeline_target_slot_mappings = []
        for chunk_start, chunk_end in pipeline_chunk_ranges:
            gathered_positions = torch.cat(
                [
                    torch.arange(
                        peer_rank * local_selected + chunk_start,
                        peer_rank * local_selected + chunk_end,
                        dtype=torch.long,
                        device=device,
                    )
                    for peer_rank in range(world_size)
                ]
            )
            pipeline_target_slot_mappings.append(
                destination.index_select(0, gathered_positions)
            )
        return TransferCase(
            selected_token_idx=selected,
            target_slot_mapping=destination,
            local_selected_token_idx=selected[local_start:local_end],
            pipeline_target_slot_mappings=tuple(pipeline_target_slot_mappings),
        )

    return {
        CASE_NAMES[0]: make_case(source_contiguous, destination_contiguous),
        CASE_NAMES[1]: make_case(source_contiguous, destination_scattered),
        CASE_NAMES[2]: make_case(source_scattered, destination_contiguous),
        CASE_NAMES[3]: make_case(source_scattered, destination_scattered),
    }


def _shared_slab_size(args: argparse.Namespace) -> int:
    dims = MlaDsaDims(
        k_hidden_dims=args.num_kv_heads * args.kv_lora_rank,
        v_hidden_dims=0,
        dsa_hidden_dims=args.num_kv_heads * args.kv_lora_rank,
        plane_elems=args.num_kv_heads * args.kv_lora_rank,
    )
    return shared_cpu_slab_bytes(
        compute_chunk_partition(args.num_tokens, args.chunk_size),
        dims,
        torch.bfloat16,
        args.num_layers,
    )


def _build_direct_harness(
    args: argparse.Namespace,
    rank: int,
    shared_cpu_slab: torch.Tensor,
    *,
    populate_cpu_source: bool,
) -> LoadBenchmarkHarness:
    device = torch.device(f"npu:{args.devices[rank]}")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    num_blocks = recommended_num_blocks(args.num_tokens, args.block_size)
    # The decode-critical sparse path under test loads only the latent K plane.
    # DSA_INDEX is the existing production single-plane prepared-transfer
    # format; here it is used mechanically for the K-shaped tensor so the host
    # chunks, sparse H2D kernel, and byte accounting all exclude V.
    source_cache = [
        (
            torch.randn(
                (
                    num_blocks,
                    args.block_size,
                    args.num_kv_heads,
                    args.kv_lora_rank,
                ),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        for _ in range(args.num_layers)
    ]
    return build_load_benchmark_harness(
        fmt="dsa_index",
        src_kv_cache=source_cache,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
        num_selected=args.num_selected,
        seed=args.seed,
        shared_cpu_slab=shared_cpu_slab,
        populate_cpu_source=populate_cpu_source,
    )


def _empty_k_layer(
    *,
    num_slots: int,
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    num_blocks = (num_slots + block_size - 1) // block_size
    return (
        torch.empty(
            (
                num_blocks,
                block_size,
                num_kv_heads,
                kv_lora_rank,
            ),
            dtype=dtype,
            device=device,
        ),
    )


def _build_sharded_harness(
    args: argparse.Namespace,
    direct: LoadBenchmarkHarness,
    world_size: int,
) -> ShardedAllGatherHarness:
    device = direct.src_kv_cache[0][0].device
    local_selected = args.num_selected // world_size
    local_staging = _empty_k_layer(
        num_slots=local_selected,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        kv_lora_rank=args.kv_lora_rank,
        dtype=direct.dtype,
        device=device,
    )
    local_slot_mapping = torch.arange(
        local_selected,
        dtype=torch.long,
        device=device,
    )
    local_destination_state = prepare_sparse_direct_destination_state(
        as_vllm_kv_caches(local_staging),
        local_slot_mapping,
        KV_FORMAT_DSA_INDEX,
        direct.dims.k_hidden_dims,
        direct.dims.v_hidden_dims,
        direct.dims.dsa_hidden_dims,
    )

    local_k = local_staging[0].flatten(0, 1)[:local_selected]
    if not local_k.is_contiguous():
        raise RuntimeError("local K staging view must be contiguous")

    gathered_k = torch.empty(
        (args.num_selected, args.num_kv_heads, args.kv_lora_rank),
        dtype=direct.dtype,
        device=device,
    )
    destinations = [
        tuple(torch.empty_like(plane) for plane in layer)
        for layer in direct.src_kv_cache
    ]

    pipeline_chunk_ranges = _split_evenly(local_selected, args.pipeline_chunks)
    max_pipeline_local = max(end - start for start, end in pipeline_chunk_ranges)
    pipeline_staging = [
        _empty_k_layer(
            num_slots=max_pipeline_local,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads,
            kv_lora_rank=args.kv_lora_rank,
            dtype=direct.dtype,
            device=device,
        )
        for _ in range(2)
    ]
    pipeline_local_slot_mapping = torch.arange(
        max_pipeline_local,
        dtype=torch.long,
        device=device,
    )
    pipeline_destination_states = [
        prepare_sparse_direct_destination_state(
            as_vllm_kv_caches(staging),
            pipeline_local_slot_mapping,
            KV_FORMAT_DSA_INDEX,
            direct.dims.k_hidden_dims,
            0,
            direct.dims.k_hidden_dims,
        )
        for staging in pipeline_staging
    ]
    pipeline_local_k = [
        staging[0].flatten(0, 1)[:max_pipeline_local]
        for staging in pipeline_staging
    ]
    pipeline_gathered_k = [
        torch.empty(
            (
                max_pipeline_local * world_size,
                args.num_kv_heads,
                args.kv_lora_rank,
            ),
            dtype=direct.dtype,
            device=device,
        )
        for _ in range(2)
    ]
    return ShardedAllGatherHarness(
        local_staging=local_staging,
        local_destination_state=local_destination_state,
        local_slot_mapping=local_slot_mapping,
        local_k=local_k,
        gathered_k=gathered_k,
        destinations=destinations,
        pipeline_staging=pipeline_staging,
        pipeline_destination_states=pipeline_destination_states,
        pipeline_local_k=pipeline_local_k,
        pipeline_gathered_k=pipeline_gathered_k,
        pipeline_local_slot_mapping=pipeline_local_slot_mapping,
        pipeline_load_done=[torch.npu.Event() for _ in range(2)],
        pipeline_buffer_free=[torch.npu.Event() for _ in range(2)],
        pipeline_buffer_used=[False, False],
        pipeline_entry=torch.npu.Event(),
        load_stream=torch.npu.Stream(device=device),
        collective_stream=torch.npu.Stream(device=device),
        pipeline_chunk_ranges=pipeline_chunk_ranges,
        local_selected=local_selected,
    )


def _run_direct(
    direct: LoadBenchmarkHarness,
    case: TransferCase,
) -> None:
    direct.selected_token_idx = case.selected_token_idx
    direct.slot_mapping_packed = case.target_slot_mapping
    run_all_direct_fast_layers(direct)


def _scatter_k(
    destination_k: torch.Tensor,
    gathered_k: torch.Tensor,
    target_slot_mapping: torch.Tensor,
    dims: MlaDsaDims,
) -> None:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for this benchmark")
    target_indices = target_slot_mapping.reshape(-1, 1)
    torch_npu.npu_scatter_nd_update_(
        destination_k.view(-1, dims.k_hidden_dims),
        target_indices,
        gathered_k.view(-1, dims.k_hidden_dims),
    )


def _scatter_gathered_layer(
    sharded: ShardedAllGatherHarness,
    layer_id: int,
    target_slot_mapping: torch.Tensor,
    dims: MlaDsaDims,
) -> None:
    (destination_k,) = sharded.destinations[layer_id]
    _scatter_k(
        destination_k,
        sharded.gathered_k,
        target_slot_mapping,
        dims,
    )


def _run_sharded(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    for layer_id in range(direct.num_layers):
        run_direct_fast_load_layer(
            destination_state=sharded.local_destination_state,
            slot_mapping_packed=sharded.local_slot_mapping,
            selected_token_idx=case.local_selected_token_idx,
            chunk_ptrs_npu=direct.chunk_ptrs_by_layer[layer_id],
            chunk_size=direct.chunk_size,
            total_tokens=direct.num_tokens,
            kv_format=direct.kv_format,
        )
        dist.all_gather_into_tensor(
            sharded.gathered_k,
            sharded.local_k,
        )
        _scatter_gathered_layer(
            sharded,
            layer_id,
            case.target_slot_mapping,
            direct.dims,
        )


def _run_local_h2d(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    for layer_id in range(direct.num_layers):
        run_direct_fast_load_layer(
            destination_state=sharded.local_destination_state,
            slot_mapping_packed=sharded.local_slot_mapping,
            selected_token_idx=case.local_selected_token_idx,
            chunk_ptrs_npu=direct.chunk_ptrs_by_layer[layer_id],
            chunk_size=direct.chunk_size,
            total_tokens=direct.num_tokens,
            kv_format=direct.kv_format,
        )


def _run_allgather_only(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
) -> None:
    for _ in range(direct.num_layers):
        dist.all_gather_into_tensor(
            sharded.gathered_k,
            sharded.local_k,
        )


def _run_scatter_only(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    for layer_id in range(direct.num_layers):
        _scatter_gathered_layer(
            sharded,
            layer_id,
            case.target_slot_mapping,
            direct.dims,
        )


def _run_no_scatter(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    """Measure K-only sparse H2D plus one AllGather with direct output use."""
    for layer_id in range(direct.num_layers):
        run_direct_fast_load_layer(
            destination_state=sharded.local_destination_state,
            slot_mapping_packed=sharded.local_slot_mapping,
            selected_token_idx=case.local_selected_token_idx,
            chunk_ptrs_npu=direct.chunk_ptrs_by_layer[layer_id],
            chunk_size=direct.chunk_size,
            total_tokens=direct.num_tokens,
            kv_format=direct.kv_format,
        )
        dist.all_gather_into_tensor(
            sharded.gathered_k,
            sharded.local_k,
        )


def _run_pipelined(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    """Double-buffer local H2D against the preceding chunk's collective."""
    current_stream = torch.npu.current_stream()
    sharded.pipeline_entry.record(current_stream)
    sharded.load_stream.wait_event(sharded.pipeline_entry)
    issued = 0
    used_buffers: set[int] = set()
    pending_work = None
    pending_destination_k = None
    pending_gathered_k = None
    pending_target_slot_mapping = None
    pending_buffer_id = None

    for layer_id in range(direct.num_layers):
        (destination_k,) = sharded.destinations[layer_id]
        for chunk_id, (chunk_start, chunk_end) in enumerate(
            sharded.pipeline_chunk_ranges
        ):
            buffer_id = issued % 2
            chunk_tokens = chunk_end - chunk_start
            used_buffers.add(buffer_id)

            with torch.npu.stream(sharded.load_stream):
                if sharded.pipeline_buffer_used[buffer_id]:
                    sharded.load_stream.wait_event(
                        sharded.pipeline_buffer_free[buffer_id]
                    )
                run_direct_fast_load_layer(
                    destination_state=sharded.pipeline_destination_states[
                        buffer_id
                    ],
                    slot_mapping_packed=sharded.pipeline_local_slot_mapping[
                        :chunk_tokens
                    ],
                    selected_token_idx=case.local_selected_token_idx[
                        chunk_start:chunk_end
                    ],
                    chunk_ptrs_npu=direct.chunk_ptrs_by_layer[layer_id],
                    chunk_size=direct.chunk_size,
                    total_tokens=direct.num_tokens,
                    kv_format=direct.kv_format,
                )
                sharded.pipeline_load_done[buffer_id].record(
                    sharded.load_stream
                )

            # H2D for the current chunk is now in flight. Complete and place
            # the preceding chunk while that independent load stream runs.
            if pending_work is not None:
                assert pending_destination_k is not None
                assert pending_gathered_k is not None
                assert pending_target_slot_mapping is not None
                assert pending_buffer_id is not None
                with torch.npu.stream(sharded.collective_stream):
                    pending_work.wait()
                    _scatter_k(
                        pending_destination_k,
                        pending_gathered_k,
                        pending_target_slot_mapping,
                        direct.dims,
                    )
                    sharded.pipeline_buffer_free[pending_buffer_id].record(
                        sharded.collective_stream
                    )
                    sharded.pipeline_buffer_used[pending_buffer_id] = True

            with torch.npu.stream(sharded.collective_stream):
                sharded.collective_stream.wait_event(
                    sharded.pipeline_load_done[buffer_id]
                )
                local_k = sharded.pipeline_local_k[buffer_id][
                    :chunk_tokens
                ]
                gathered_k = sharded.pipeline_gathered_k[buffer_id][
                    : chunk_tokens * dist.get_world_size()
                ]
                pending_work = dist.all_gather_into_tensor(
                    gathered_k,
                    local_k,
                    async_op=True,
                )
                pending_destination_k = destination_k
                pending_gathered_k = gathered_k
                pending_target_slot_mapping = (
                    case.pipeline_target_slot_mappings[chunk_id]
                )
                pending_buffer_id = buffer_id
            issued += 1

        # A production attention layer cannot consume its cache until every
        # chunk is placed, so finish the final collective at each layer
        # boundary instead of assuming cross-layer prefetch.
        if pending_work is not None:
            assert pending_destination_k is not None
            assert pending_gathered_k is not None
            assert pending_target_slot_mapping is not None
            assert pending_buffer_id is not None
            with torch.npu.stream(sharded.collective_stream):
                pending_work.wait()
                _scatter_k(
                    pending_destination_k,
                    pending_gathered_k,
                    pending_target_slot_mapping,
                    direct.dims,
                )
                sharded.pipeline_buffer_free[pending_buffer_id].record(
                    sharded.collective_stream
                )
                sharded.pipeline_buffer_used[pending_buffer_id] = True
            pending_work = None
            pending_destination_k = None
            pending_gathered_k = None
            pending_target_slot_mapping = None
            pending_buffer_id = None

    # Make the caller's timing stream depend on every pipeline buffer. This
    # keeps the NPU end event honest without a host-side synchronize per run.
    for buffer_id in used_buffers:
        current_stream.wait_event(sharded.pipeline_buffer_free[buffer_id])


def _assert_paged_destinations_match(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    target_slots: torch.Tensor,
) -> None:
    for layer_id, (direct_layer, sharded_layer) in enumerate(
        zip(direct.dst_direct_fast, sharded.destinations, strict=True)
    ):
        for plane_id, (direct_plane, sharded_plane) in enumerate(
            zip(direct_layer, sharded_layer, strict=True)
        ):
            expected = direct_plane.flatten(0, 1).index_select(0, target_slots)
            actual = sharded_plane.flatten(0, 1).index_select(0, target_slots)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if expected.numel() == 0:
                raise AssertionError(
                    f"empty verification payload at layer={layer_id}, plane={plane_id}"
                )


def _verify_case(
    direct: LoadBenchmarkHarness,
    sharded: ShardedAllGatherHarness,
    case: TransferCase,
) -> None:
    target_slots = case.target_slot_mapping.to(torch.long)
    _run_direct(direct, case)
    _run_sharded(direct, sharded, case)
    torch.npu.synchronize()
    _assert_paged_destinations_match(direct, sharded, target_slots)

    # Poison the previous result so a missing pipeline scatter cannot be
    # hidden by the already-correct non-pipelined destination.
    for layer in sharded.destinations:
        for plane in layer:
            plane.fill_(float("nan"))
    _run_pipelined(direct, sharded, case)
    torch.npu.synchronize()
    _assert_paged_destinations_match(direct, sharded, target_slots)

    # The no-scatter lower bound leaves the rank-ordered selected K rows in
    # gathered_k. Verify this layout independently of paged placement.
    _run_no_scatter(direct, sharded, case)
    torch.npu.synchronize()
    expected_gathered = (
        direct.dst_direct_fast[-1][0]
        .flatten(0, 1)
        .index_select(0, target_slots)
    )
    torch.testing.assert_close(
        sharded.gathered_k,
        expected_gathered,
        rtol=0,
        atol=0,
    )


def _time_path(
    run: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    barrier: Any,
) -> list[dict[str, float | int]]:
    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    samples: list[dict[str, float | int]] = []
    for _ in range(repeats):
        barrier.wait(timeout=300)
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        wall_start_ns = time.monotonic_ns()
        start_event.record()
        for _ in range(iters):
            run()
        end_event.record()
        end_event.synchronize()
        wall_end_ns = time.monotonic_ns()
        samples.append(
            {
                "event_ms": float(start_event.elapsed_time(end_event)),
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
            }
        )
        barrier.wait(timeout=300)
    return samples


def _worker(
    rank: int,
    args: argparse.Namespace,
    barrier: Any,
    output: Any,
) -> None:
    direct = None
    shared_owner = None
    shared_mapping = None
    process_group_initialized = False
    try:
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for this benchmark")
        device = torch.device(f"npu:{args.devices[rank]}")
        torch.npu.set_device(device)
        dist.init_process_group(
            backend="hccl",
            init_method=args.init_method,
            rank=rank,
            world_size=len(args.devices),
        )
        process_group_initialized = True

        slab_size = _shared_slab_size(args)
        if rank == 0:
            shared_owner = MixedMemoryAllocator(
                slab_size,
                shm_name=args.shared_slab_name,
                align_bytes=ALIGN_BYTES,
            )
            shared_cpu_slab = shared_owner.buffer
        else:
            shared_cpu_slab = None
        barrier.wait(timeout=300)
        if rank != 0:
            shared_mapping = SharedSlabMapping.attach(
                shm_name=args.shared_slab_name,
                size=slab_size,
                generation=0,
                writable=True,
            )
            shared_cpu_slab = shared_mapping.tensor

        assert shared_cpu_slab is not None
        if rank == 0:
            direct = _build_direct_harness(
                args,
                rank,
                shared_cpu_slab,
                populate_cpu_source=True,
            )
        barrier.wait(timeout=300)
        if rank != 0:
            direct = _build_direct_harness(
                args,
                rank,
                shared_cpu_slab,
                populate_cpu_source=False,
            )
        barrier.wait(timeout=300)

        assert direct is not None
        sharded = _build_sharded_harness(args, direct, len(args.devices))
        num_slots = (
            direct.dst_direct_fast[0][0].shape[0]
            * direct.dst_direct_fast[0][0].shape[1]
        )
        cases = _locality_cases(
            num_tokens=args.num_tokens,
            num_selected=args.num_selected,
            num_slots=num_slots,
            rank=rank,
            world_size=len(args.devices),
            pipeline_chunks=args.pipeline_chunks,
            seed=args.seed,
            device=device,
        )

        # Initialize the HCCL communicator outside measured regions.
        warmup_tensor = torch.full((1,), rank, dtype=torch.int32, device=device)
        warmup_output = torch.empty(
            len(args.devices),
            dtype=torch.int32,
            device=device,
        )
        dist.all_gather_into_tensor(warmup_output, warmup_tensor)
        torch.npu.synchronize()

        results: dict[
            str,
            dict[str, list[dict[str, float | int]]],
        ] = {}
        for case_name in CASE_NAMES:
            case = cases[case_name]
            if args.verify:
                _verify_case(direct, sharded, case)
            direct_run = partial(_run_direct, direct, case)
            sharded_run = partial(_run_sharded, direct, sharded, case)
            path_runs = {
                PATH_DIRECT: direct_run,
                PATH_SHARDED: sharded_run,
                PATH_NO_SCATTER: partial(
                    _run_no_scatter,
                    direct,
                    sharded,
                    case,
                ),
                PATH_PIPELINED: partial(
                    _run_pipelined,
                    direct,
                    sharded,
                    case,
                ),
                PATH_H2D: partial(
                    _run_local_h2d,
                    direct,
                    sharded,
                    case,
                ),
                PATH_ALLGATHER: partial(
                    _run_allgather_only,
                    direct,
                    sharded,
                ),
                PATH_SCATTER: partial(
                    _run_scatter_only,
                    direct,
                    sharded,
                    case,
                ),
            }
            results[case_name] = {
                path_name: _time_path(
                    path_run,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    barrier=barrier,
                )
                for path_name, path_run in path_runs.items()
            }

        output.put(
            {
                "rank": rank,
                "device": args.devices[rank],
                "plane_elems": direct.dims.plane_elems,
                "element_size": torch.empty(
                    0,
                    dtype=direct.dtype,
                ).element_size(),
                "local_selected": sharded.local_selected,
                "results": results,
            }
        )
    except BaseException as exc:
        try:
            barrier.abort()
        except Exception:
            pass
        output.put(
            {
                "rank": rank,
                "device": args.devices[rank],
                "error": repr(exc),
            }
        )
        raise
    finally:
        try:
            if process_group_initialized:
                dist.destroy_process_group()
        except Exception:
            pass
        try:
            barrier.wait(timeout=300)
        except Exception:
            pass
        if direct is not None:
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            direct.close()
        if shared_mapping is not None:
            shared_mapping.close()
        if shared_owner is not None:
            shared_owner.close()


def _critical_wall_samples_ms(
    workers: list[dict[str, Any]],
    case_name: str,
    path_name: str,
    repeats: int,
) -> list[float]:
    samples = []
    for repeat in range(repeats):
        starts = [
            worker["results"][case_name][path_name][repeat]["wall_start_ns"]
            for worker in workers
        ]
        ends = [
            worker["results"][case_name][path_name][repeat]["wall_end_ns"]
            for worker in workers
        ]
        samples.append((max(ends) - min(starts)) / 1e6)
    return samples


def _summarize(
    args: argparse.Namespace,
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    plane_elems = workers[0]["plane_elems"]
    element_size = workers[0]["element_size"]
    full_k_bytes_per_invocation = (
        args.num_selected * plane_elems * element_size * args.num_layers
    )
    world_size = len(workers)
    local_k_bytes_per_invocation = full_k_bytes_per_invocation // world_size
    full_k_bytes_per_layer = full_k_bytes_per_invocation // args.num_layers
    local_k_bytes_per_layer = local_k_bytes_per_invocation // args.num_layers
    pipeline_chunk_tokens = [
        end - start
        for start, end in _split_evenly(
            args.num_selected // world_size,
            args.pipeline_chunks,
        )
    ]
    pipeline_chunk_local_bytes = [
        tokens * plane_elems * element_size for tokens in pipeline_chunk_tokens
    ]

    summary: dict[str, Any] = {
        "devices": args.devices,
        "world_size": world_size,
        "hccl_if_base_port": args.hccl_if_base_port,
        "hccl_socket_ports": args.hccl_socket_ports,
        "num_tokens": args.num_tokens,
        "num_selected": args.num_selected,
        "local_selected": args.num_selected // world_size,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "block_size": args.block_size,
        "pipeline_chunks": args.pipeline_chunks,
        "plane_elems": plane_elems,
        "element_size": element_size,
        "payload_planes": ["k"],
        "cpu_slab": "shared-rank0-owned",
        "full_k_bytes_per_invocation": full_k_bytes_per_invocation,
        "local_k_bytes_per_invocation": local_k_bytes_per_invocation,
        "full_k_bytes_per_layer": full_k_bytes_per_layer,
        "allgather_local_input_bytes_per_layer": local_k_bytes_per_layer,
        "pipeline_chunk_tokens": pipeline_chunk_tokens,
        "pipeline_chunk_local_bytes": pipeline_chunk_local_bytes,
        "cases": {},
    }

    print(
        f"devices={args.devices} tokens={args.num_tokens} "
        f"selected={args.num_selected} local_selected="
        f"{args.num_selected // world_size} layers={args.num_layers} "
        f"chunk_size={args.chunk_size} block_size={args.block_size} "
        f"pipeline_chunks={args.pipeline_chunks}"
    )
    print(
        "K-only payload per invocation: full="
        f"{format_bytes_short(full_k_bytes_per_invocation)} local/rank="
        f"{format_bytes_short(local_k_bytes_per_invocation)}"
    )
    print(
        "AllGather per layer: local input/rank="
        f"{format_bytes_short(local_k_bytes_per_layer)} assembled output="
        f"{format_bytes_short(full_k_bytes_per_layer)}; pipeline local chunks="
        f"{min(pipeline_chunk_tokens)}-{max(pipeline_chunk_tokens)} tokens "
        f"({format_bytes_short(min(pipeline_chunk_local_bytes))}-"
        f"{format_bytes_short(max(pipeline_chunk_local_bytes))})"
    )
    if args.num_layers * args.iters < 100:
        print(
            "WARNING: fewer than 100 timed layer transfers per repeat; "
            "cross-rank wall metrics may be dominated by process start skew. "
            "Use --num-layers 8 --iters 50 for throughput measurements."
        )
    for case_name in CASE_NAMES:
        path_summary: dict[str, Any] = {}
        for path_name in PATH_NAMES:
            event_ms = [
                worker["results"][case_name][path_name][repeat]["event_ms"]
                for worker in workers
                for repeat in range(args.repeats)
            ]
            critical_wall_ms = _critical_wall_samples_ms(
                workers,
                case_name,
                path_name,
                args.repeats,
            )
            median_critical_ms = statistics.median(critical_wall_ms)
            median_wall_ms_per_layer = median_critical_ms / (
                args.iters * args.num_layers
            )
            path_bytes_per_invocation = (
                local_k_bytes_per_invocation
                if path_name == PATH_H2D
                else full_k_bytes_per_invocation
            )
            path_bytes_per_repeat = path_bytes_per_invocation * args.iters
            effective_wall_gbps = (
                path_bytes_per_repeat
                / (median_critical_ms / 1000.0)
                / 1e9
            )
            rank_event_bandwidths = [
                path_bytes_per_repeat / (duration_ms / 1000.0) / 1e9
                for duration_ms in event_ms
            ]
            median_rank_event_ms = statistics.median(event_ms)
            start_skew_ms = []
            for repeat in range(args.repeats):
                starts = [
                    worker["results"][case_name][path_name][repeat]["wall_start_ns"]
                    for worker in workers
                ]
                start_skew_ms.append((max(starts) - min(starts)) / 1e6)
            path_summary[path_name] = {
                "median_critical_wall_ms": median_critical_ms,
                "median_wall_ms_per_layer": median_wall_ms_per_layer,
                "effective_k_payload_wall_gbps": effective_wall_gbps,
                "median_rank_event_ms": median_rank_event_ms,
                "median_rank_event_ms_per_layer": median_rank_event_ms
                / (args.iters * args.num_layers),
                "rank_event_effective_k_payload_gbps": statistics.median(
                    rank_event_bandwidths
                ),
                "min_rank_event_effective_k_payload_gbps": min(
                    rank_event_bandwidths
                ),
                "max_rank_event_effective_k_payload_gbps": max(
                    rank_event_bandwidths
                ),
                "bytes_per_invocation": path_bytes_per_invocation,
                "min_rank_event_ms": min(event_ms),
                "max_rank_event_ms": max(event_ms),
                "median_start_skew_ms": statistics.median(start_skew_ms),
                "max_start_skew_ms": max(start_skew_ms),
                "critical_wall_samples_ms": critical_wall_ms,
            }
            if path_name == PATH_ALLGATHER:
                local_input_bytes_per_repeat = (
                    local_k_bytes_per_invocation * args.iters
                )
                path_summary[path_name][
                    "effective_local_input_wall_gbps"
                ] = (
                    local_input_bytes_per_repeat
                    / (median_critical_ms / 1000.0)
                    / 1e9
                )
                local_input_event_bandwidths = [
                    local_input_bytes_per_repeat
                    / (duration_ms / 1000.0)
                    / 1e9
                    for duration_ms in event_ms
                ]
                path_summary[path_name][
                    "rank_event_effective_local_input_gbps"
                ] = statistics.median(local_input_event_bandwidths)

        speedups: dict[str, Any] = {}
        for alternative in COMPARISON_PATHS[1:]:
            paired_wall_speedups = [
                direct_sample / alternative_sample
                for direct_sample, alternative_sample in zip(
                    path_summary[PATH_DIRECT]["critical_wall_samples_ms"],
                    path_summary[alternative]["critical_wall_samples_ms"],
                    strict=True,
                )
                if alternative_sample > 0
            ]
            paired_event_speedups = [
                worker["results"][case_name][PATH_DIRECT][repeat]["event_ms"]
                / worker["results"][case_name][alternative][repeat]["event_ms"]
                for worker in workers
                for repeat in range(args.repeats)
                if (
                    worker["results"][case_name][alternative][repeat][
                        "event_ms"
                    ]
                    > 0
                )
            ]
            speedups[alternative] = {
                "paired_wall": {
                    "median": statistics.median(paired_wall_speedups),
                    "min": min(paired_wall_speedups),
                    "max": max(paired_wall_speedups),
                },
                "paired_rank_event": {
                    "median": statistics.median(paired_event_speedups),
                    "min": min(paired_event_speedups),
                    "max": max(paired_event_speedups),
                },
            }
        summary["cases"][case_name] = {
            "paths": path_summary,
            "speedup_vs_direct": speedups,
        }

    print(
        "\nEnd-to-end critical wall time (ms/layer; speedup is direct/alternative)"
    )
    print(
        "case                                      "
        "direct   sharded   no-scatter   pipeline   "
        "shard spd   no-scat spd   pipe spd"
    )
    for case_name in CASE_NAMES:
        case_summary = summary["cases"][case_name]
        paths = case_summary["paths"]
        speedups = case_summary["speedup_vs_direct"]
        print(
            f"{case_name:42} "
            f"{paths[PATH_DIRECT]['median_wall_ms_per_layer']:7.3f}  "
            f"{paths[PATH_SHARDED]['median_wall_ms_per_layer']:8.3f}  "
            f"{paths[PATH_NO_SCATTER]['median_wall_ms_per_layer']:10.3f}  "
            f"{paths[PATH_PIPELINED]['median_wall_ms_per_layer']:8.3f}  "
            f"{speedups[PATH_SHARDED]['paired_wall']['median']:8.3f}x  "
            f"{speedups[PATH_NO_SCATTER]['paired_wall']['median']:10.3f}x  "
            f"{speedups[PATH_PIPELINED]['paired_wall']['median']:7.3f}x"
        )

    print("\nEnd-to-end effective full-K GB/s (critical wall)")
    print(
        "case                                      "
        "direct   sharded   no-scatter   pipeline"
    )
    for case_name in CASE_NAMES:
        paths = summary["cases"][case_name]["paths"]
        print(
            f"{case_name:42} "
            f"{paths[PATH_DIRECT]['effective_k_payload_wall_gbps']:7.2f}  "
            f"{paths[PATH_SHARDED]['effective_k_payload_wall_gbps']:8.2f}  "
            f"{paths[PATH_NO_SCATTER]['effective_k_payload_wall_gbps']:10.2f}  "
            f"{paths[PATH_PIPELINED]['effective_k_payload_wall_gbps']:8.2f}"
        )

    print(
        "\nIsolated stage critical wall time "
        "(us/layer; AG assembled GB/s is comparable full-K throughput)"
    )
    print(
        "case                                      "
        "local H2D   AllGather   scatter   stage sum   "
        "AG assembled GB/s   AG local-in GB/s"
    )
    for case_name in CASE_NAMES:
        paths = summary["cases"][case_name]["paths"]
        h2d_us = paths[PATH_H2D]["median_wall_ms_per_layer"] * 1000
        allgather_us = (
            paths[PATH_ALLGATHER]["median_wall_ms_per_layer"] * 1000
        )
        scatter_us = paths[PATH_SCATTER]["median_wall_ms_per_layer"] * 1000
        print(
            f"{case_name:42} "
            f"{h2d_us:9.2f}  "
            f"{allgather_us:10.2f}  "
            f"{scatter_us:8.2f}  "
            f"{h2d_us + allgather_us + scatter_us:9.2f}  "
            f"{paths[PATH_ALLGATHER]['effective_k_payload_wall_gbps']:17.2f}  "
            f"{paths[PATH_ALLGATHER]['effective_local_input_wall_gbps']:16.2f}"
        )
    return summary


def main() -> None:
    args = _parse_args()
    hccl_socket_ports = _configure_hccl_socket_ports(args)
    hccl_base_port, hccl_port_source = _configure_hccl_base_port(args)
    args.shared_slab_name = (
        f"/lmcache_tp_allgather_bench_{os.getpid()}_{uuid.uuid4().hex}"
    )
    args.init_method = f"tcp://127.0.0.1:{_find_free_tcp_port()}"
    print(f"shared CPU slab: {args.shared_slab_name}")
    print(f"HCCL init method: {args.init_method}")
    print(
        "HCCL host ports: "
        f"{hccl_base_port}-{hccl_base_port + HCCL_HOST_PORT_COUNT - 1} "
        f"({hccl_port_source})"
    )
    for name, value in hccl_socket_ports.items():
        print(f"{name}={value}")

    context = mp.get_context("spawn")
    barrier = context.Barrier(len(args.devices))
    output = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, args, barrier, output),
        )
        for rank in range(len(args.devices))
    ]
    for process in processes:
        process.start()

    workers = []
    try:
        for _ in processes:
            workers.append(output.get(timeout=1800))
    except Empty as exc:
        raise RuntimeError("timed out waiting for benchmark workers") from exc
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        SharedSlabMapping.unlink(args.shared_slab_name)

    errors = [worker for worker in workers if "error" in worker]
    if errors:
        raise RuntimeError(f"benchmark worker errors: {errors}")
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise RuntimeError(f"benchmark workers failed: exit codes {failed}")

    workers.sort(key=lambda worker: worker["rank"])
    summary = _summarize(args, workers)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {output_path}")


if __name__ == "__main__":
    main()
