#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark prepared sparse MLA retrieval under four address-locality patterns.

``selected_token_idx`` controls source locality and ``slot_mapping_packed``
controls destination locality. The benchmark invokes the production prepared
sparse kernel without changing its payload, chunk layout, or AIV allocation.
Scattered indices are unique, deterministic, and randomly ordered across the
full source or destination allocation.

Examples:
  python benchmark/v1/kv_transfer/benchmark_sparse_locality.py --verify

  python benchmark/v1/kv_transfer/benchmark_sparse_locality.py \
      --devices 0,1,2,3,4,5,6,7 --verify
"""

from __future__ import annotations

# Standard
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path
from queue import Empty
from typing import Any

# Third Party
import torch

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from kv_cache_fixtures import generate_mla_kv_cache  # noqa: E402
from load_benchmark_utils import (  # noqa: E402
    ALIGN_BYTES,
    LoadBenchmarkHarness,
    MlaDsaDims,
    build_load_benchmark_harness,
    compute_chunk_partition,
    format_bytes_short,
    recommended_num_blocks,
    run_all_direct_fast_layers,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--num-tokens", type=int, default=18880)
    parser.add_argument("--num-selected", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--shared-cpu-slab",
        action="store_true",
        help="make all ranks read the same rank-0-created pinned CPU slab",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if not devices:
        parser.error("--devices must contain at least one device")
    if len(set(devices)) != len(devices):
        parser.error("--devices cannot contain duplicates")
    if not 0 < args.num_selected <= args.num_tokens:
        parser.error("--num-selected must be in [1, --num-tokens]")
    for name in ("num_layers", "chunk_size", "iters", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    args.devices = devices
    return args


def _locality_cases(
    *,
    num_tokens: int,
    num_selected: int,
    num_slots: int,
    seed: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    source_contiguous = torch.arange(num_selected, dtype=torch.int32, device=device)
    source_scattered = torch.tensor(
        random.Random(seed).sample(range(num_tokens), num_selected),
        dtype=torch.int32,
        device=device,
    )
    destination_contiguous = torch.arange(num_selected, dtype=torch.long, device=device)
    destination_scattered = torch.tensor(
        random.Random(seed + 1).sample(range(num_slots), num_selected),
        dtype=torch.long,
        device=device,
    )
    return {
        CASE_NAMES[0]: (source_contiguous, destination_contiguous),
        CASE_NAMES[1]: (source_contiguous, destination_scattered),
        CASE_NAMES[2]: (source_scattered, destination_contiguous),
        CASE_NAMES[3]: (source_scattered, destination_scattered),
    }


def _set_case(
    harness: LoadBenchmarkHarness,
    case: tuple[torch.Tensor, torch.Tensor],
) -> None:
    harness.selected_token_idx, harness.slot_mapping_packed = case


def _verify_case(
    harness: LoadBenchmarkHarness,
    case: tuple[torch.Tensor, torch.Tensor],
) -> None:
    selected, destination_slots = case
    _set_case(harness, case)
    run_all_direct_fast_layers(harness)
    torch.npu.synchronize()

    source_slots = harness.slot_mapping_full.index_select(0, selected.to(torch.long))
    for source_layer, destination_layer in zip(
        harness.src_kv_cache, harness.dst_direct_fast, strict=True
    ):
        for source_plane, destination_plane in zip(
            source_layer, destination_layer, strict=True
        ):
            expected = source_plane.flatten(0, 1).index_select(0, source_slots)
            actual = destination_plane.flatten(0, 1).index_select(0, destination_slots)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _build_harness(
    args: argparse.Namespace,
    rank: int,
    shared_cpu_slab: torch.Tensor | None = None,
    *,
    populate_cpu_source: bool = True,
) -> tuple[LoadBenchmarkHarness, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    device = torch.device(f"npu:{args.devices[rank]}")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed if args.shared_cpu_slab else args.seed + rank)

    num_blocks = recommended_num_blocks(args.num_tokens, args.block_size)
    source_cache = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )
    harness = build_load_benchmark_harness(
        fmt="mla_latent",
        src_kv_cache=source_cache,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
        num_selected=args.num_selected,
        seed=args.seed,
        shared_cpu_slab=shared_cpu_slab,
        populate_cpu_source=populate_cpu_source,
    )
    num_slots = (
        harness.dst_direct_fast[0][0].shape[0] * harness.dst_direct_fast[0][0].shape[1]
    )
    return harness, _locality_cases(
        num_tokens=args.num_tokens,
        num_selected=args.num_selected,
        num_slots=num_slots,
        seed=args.seed,
        device=device,
    )


def _shared_slab_size(args: argparse.Namespace) -> int:
    k_hidden_dims = args.num_kv_heads * args.kv_lora_rank
    v_hidden_dims = args.num_kv_heads * args.qk_rope_head_dim
    return shared_cpu_slab_bytes(
        compute_chunk_partition(args.num_tokens, args.chunk_size),
        MlaDsaDims(
            k_hidden_dims=k_hidden_dims,
            v_hidden_dims=v_hidden_dims,
            dsa_hidden_dims=0,
            plane_elems=k_hidden_dims + v_hidden_dims,
        ),
        torch.bfloat16,
        args.num_layers,
    )


def _worker(
    rank: int,
    args: argparse.Namespace,
    barrier: Any,
    output: Any,
) -> None:
    harness = None
    shared_owner = None
    shared_mapping = None
    try:
        if args.shared_cpu_slab:
            device = torch.device(f"npu:{args.devices[rank]}")
            torch.npu.set_device(device)
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

            if rank == 0:
                harness, cases = _build_harness(
                    args, rank, shared_cpu_slab, populate_cpu_source=True
                )
            barrier.wait(timeout=300)
            if rank != 0:
                harness, cases = _build_harness(
                    args, rank, shared_cpu_slab, populate_cpu_source=False
                )
            barrier.wait(timeout=300)
        else:
            harness, cases = _build_harness(args, rank)

        results: dict[str, list[dict[str, float | int]]] = {}
        for name in CASE_NAMES:
            case = cases[name]
            if args.verify:
                _verify_case(harness, case)
            _set_case(harness, case)
            for _ in range(args.warmup):
                run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            samples = []
            for _ in range(args.repeats):
                barrier.wait(timeout=300)
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                wall_start_ns = time.monotonic_ns()
                start_event.record()
                for _ in range(args.iters):
                    run_all_direct_fast_layers(harness)
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
            results[name] = samples
        output.put(
            {
                "rank": rank,
                "device": args.devices[rank],
                "plane_elems": harness.dims.plane_elems,
                "element_size": torch.empty(0, dtype=harness.dtype).element_size(),
                "results": results,
            }
        )
    except BaseException as exc:
        barrier.abort()
        output.put({"rank": rank, "device": args.devices[rank], "error": repr(exc)})
        raise
    finally:
        if args.shared_cpu_slab:
            try:
                barrier.wait(timeout=300)
            except Exception:
                pass
        if harness is not None:
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            harness.close()
        if shared_mapping is not None:
            shared_mapping.close()
        if shared_owner is not None:
            shared_owner.close()


def _summarize(
    args: argparse.Namespace, workers: list[dict[str, Any]]
) -> dict[str, Any]:
    plane_elems = workers[0]["plane_elems"]
    element_size = workers[0]["element_size"]
    bytes_per_invocation = (
        args.num_selected * plane_elems * element_size * args.num_layers
    )
    bytes_per_repeat = bytes_per_invocation * args.iters

    summary: dict[str, Any] = {
        "devices": args.devices,
        "num_tokens": args.num_tokens,
        "num_selected": args.num_selected,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "plane_elems": plane_elems,
        "element_size": element_size,
        "cpu_slab": "shared" if args.shared_cpu_slab else "private-per-rank",
        "bytes_per_rank_per_invocation": bytes_per_invocation,
        "cases": {},
    }
    print(
        f"devices={args.devices} tokens={args.num_tokens} "
        f"selected={args.num_selected} layers={args.num_layers} "
        f"chunk_size={args.chunk_size} cpu_slab={summary['cpu_slab']}"
    )
    print(
        f"payload per rank per invocation: {format_bytes_short(bytes_per_invocation)}"
    )
    print(
        "\ncase                                      ms/layer  "
        "rank GB/s       aggregate GB/s  wall GB/s"
    )

    for name in CASE_NAMES:
        event_ms = [
            worker["results"][name][repeat]["event_ms"]
            for worker in workers
            for repeat in range(args.repeats)
        ]
        rank_bandwidths = [
            bytes_per_repeat / (duration_ms / 1000) / 1e9 for duration_ms in event_ms
        ]
        aggregate_bandwidths = []
        aggregate_wall_bandwidths = []
        start_skews_ms = []
        for repeat in range(args.repeats):
            starts = [
                worker["results"][name][repeat]["wall_start_ns"] for worker in workers
            ]
            ends = [
                worker["results"][name][repeat]["wall_end_ns"] for worker in workers
            ]
            event_ends = [
                start + int(worker["results"][name][repeat]["event_ms"] * 1_000_000)
                for start, worker in zip(starts, workers, strict=True)
            ]
            event_span_seconds = (max(event_ends) - min(starts)) / 1e9
            aggregate_bandwidths.append(
                bytes_per_repeat * len(workers) / event_span_seconds / 1e9
            )
            wall_span_seconds = (max(ends) - min(starts)) / 1e9
            aggregate_wall_bandwidths.append(
                bytes_per_repeat * len(workers) / wall_span_seconds / 1e9
            )
            start_skews_ms.append((max(starts) - min(starts)) / 1e6)

        milliseconds_per_layer = statistics.median(event_ms) / (
            args.iters * args.num_layers
        )
        rank_median = statistics.median(rank_bandwidths)
        rank_min = min(rank_bandwidths)
        rank_max = max(rank_bandwidths)
        aggregate_median = statistics.median(aggregate_bandwidths)
        aggregate_min = min(aggregate_bandwidths)
        aggregate_max = max(aggregate_bandwidths)
        wall_median = statistics.median(aggregate_wall_bandwidths)
        print(
            f"{name:42} {milliseconds_per_layer:8.3f}  "
            f"{rank_median:6.2f} [{rank_min:5.2f},{rank_max:5.2f}]  "
            f"{aggregate_median:7.2f} "
            f"[{aggregate_min:6.2f},{aggregate_max:6.2f}]  "
            f"{wall_median:7.2f}"
        )
        summary["cases"][name] = {
            "median_ms_per_layer": milliseconds_per_layer,
            "rank_bandwidth_gbps": {
                "median": rank_median,
                "min": rank_min,
                "max": rank_max,
            },
            "aggregate_bandwidth_gbps": {
                "median": aggregate_median,
                "min": aggregate_min,
                "max": aggregate_max,
            },
            "aggregate_wall_bandwidth_gbps": {
                "median": wall_median,
                "min": min(aggregate_wall_bandwidths),
                "max": max(aggregate_wall_bandwidths),
            },
            "max_start_skew_ms": max(start_skews_ms),
        }
    return summary


def main() -> None:
    args = _parse_args()
    args.shared_slab_name = (
        f"/lmcache_sparse_bench_{os.getpid()}_{uuid.uuid4().hex}"
        if args.shared_cpu_slab
        else None
    )
    if args.shared_cpu_slab:
        print(f"shared CPU slab: {args.shared_slab_name}")
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(args.devices))
    output = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, args, barrier, output))
        for rank in range(len(args.devices))
    ]
    for process in processes:
        process.start()

    workers = []
    try:
        for _ in processes:
            workers.append(output.get(timeout=600))
    except Empty as exc:
        raise RuntimeError("timed out waiting for benchmark workers") from exc
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        if args.shared_cpu_slab:
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
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {path}")


if __name__ == "__main__":
    main()
