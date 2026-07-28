#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure whether two streams improve sparse MLA H2D transfer throughput.

The benchmark uses the production ``MLA_LATENT`` host chunk layout and the
runtime-tuned sparse transfer kernel.  It copies both the 512-element latent
plane and the trailing 64-element plane by default.

Four cases separate transfer concurrency from fixed launch overhead:

* one N-token transfer;
* one 2N-token transfer;
* two N-token transfers launched sequentially on one stream;
* two N-token transfers launched concurrently on independent streams.

The last three cases move exactly the same useful payload.  The concurrent
case writes two independent destination KV caches, so it has no data race.
With ``--verify``, both planes from every layer are checked exactly for all
four cases before timing.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import statistics
import time
from typing import Any, Sequence
import uuid

# Third Party
import torch

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

# Local benchmark helpers
from benchmark_sparse_k_transfer_tuning import (
    Shape,
    _build_harness,
    _shared_slab_size,
)
from load_benchmark_utils import (
    ALIGN_BYTES,
    KV_FORMAT_MLA_LATENT,
    LoadBenchmarkHarness,
    format_bytes_short,
)

# First Party
from lmcache.v1.memory_management import MixedMemoryAllocator
from lmcache.v1.shared_cpu_cache import SharedSlabMapping
from lmcache_ascend.v1.npu_connector.utils import (
    prepare_sparse_direct_destination_state,
    sparse_k_batched_direct_kv_transfer_experimental,
    sparse_transfer_hardware_aiv_num,
)


SINGLE_SMALL = "single_small"
SINGLE_LARGE = "single_large"
SEQUENTIAL_PAIR = "sequential_pair"
CONCURRENT_PAIR = "concurrent_pair"
CASE_ORDER = (
    SINGLE_SMALL,
    SINGLE_LARGE,
    SEQUENTIAL_PAIR,
    CONCURRENT_PAIR,
)


@dataclass(frozen=True)
class TransferTarget:
    """Immutable source selection and independent prepared destination."""

    name: str
    destination: list[tuple[torch.Tensor, ...]]
    states: list[Any]
    selected: torch.Tensor
    slots: torch.Tensor


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--num-tokens", type=_positive_int, default=20000)
    parser.add_argument("--selected-count", type=_positive_int, default=2048)
    parser.add_argument("--num-layers", type=_positive_int, default=8)
    parser.add_argument("--chunk-size", type=_positive_int, default=256)
    parser.add_argument("--block-size", type=_positive_int, default=128)
    parser.add_argument("--k-hidden-dims", type=_positive_int, default=512)
    parser.add_argument(
        "--host-v-hidden-dims",
        type=_positive_int,
        default=64,
        help="width of the trailing MLA_LATENT V/rope plane",
    )
    parser.add_argument(
        "--aiv-count",
        type=_non_negative_int,
        default=8,
        help="AIVs per transfer kernel; 0 selects the hardware count",
    )
    parser.add_argument("--tile-tokens", type=_positive_int, default=8)
    parser.add_argument(
        "--locality",
        choices=("contiguous", "scattered"),
        default="scattered",
        help="apply contiguous or scattered locality to source and destination",
    )
    parser.add_argument("--warmup", type=_non_negative_int, default=5)
    parser.add_argument("--iters", type=_positive_int, default=50)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-cpu-slab", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--worker-timeout-sec",
        type=_positive_int,
        default=3600,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the test matrix without starting NPU workers",
    )
    args = parser.parse_args()

    try:
        args.devices = [int(value) for value in args.devices.split(",")]
    except ValueError:
        parser.error("--devices must be comma-separated non-negative integers")
    if not args.devices or any(device < 0 for device in args.devices):
        parser.error("--devices must contain non-negative integers")
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices cannot contain duplicates")
    if args.selected_count * 2 >= args.num_tokens:
        parser.error(
            "2 * --selected-count must be smaller than --num-tokens so the "
            "prepared sparse path and a disjoint pair can both be exercised"
        )
    if args.k_hidden_dims * 2 % 32:
        parser.error("--k-hidden-dims with BF16 must make each token 32-byte aligned")
    if args.host_v_hidden_dims * 2 % 32:
        parser.error(
            "--host-v-hidden-dims with BF16 must make each token 32-byte aligned"
        )

    # Private attributes required by the shared tuning harness.
    args.chunk_sizes = (args.chunk_size,)
    args.transfer_planes = "kv"
    args.shared_slab_name = (
        f"/lmcache_sparse_streams_{os.getpid()}_{uuid.uuid4().hex}"
        if args.shared_cpu_slab
        else None
    )
    return args


def _selection_values(
    *,
    num_tokens: int,
    selected_count: int,
    locality: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return two disjoint source selections with deterministic locality."""
    pair_count = selected_count * 2
    if locality == "contiguous":
        return (
            list(range(selected_count)),
            list(range(selected_count, pair_count)),
        )

    generator = random.Random(seed)
    values = generator.sample(range(num_tokens - 1), pair_count - 1)
    # Deterministically cover the partial last CPU chunk when one exists.
    values.append(num_tokens - 1)
    generator.shuffle(values)
    return values[:selected_count], values[selected_count:]


def _prepared_states(
    args: argparse.Namespace,
    harness: LoadBenchmarkHarness,
    destination: list[tuple[torch.Tensor, ...]],
) -> list[Any]:
    return [
        prepare_sparse_direct_destination_state(
            destination[layer_id],
            harness.slot_mapping_full,
            KV_FORMAT_MLA_LATENT,
            args.k_hidden_dims,
            args.host_v_hidden_dims,
            0,
        )
        for layer_id in range(args.num_layers)
    ]


def _prepare_targets(
    args: argparse.Namespace,
    harness: LoadBenchmarkHarness,
    device: torch.device,
) -> dict[str, TransferTarget]:
    values_a, values_b = _selection_values(
        num_tokens=args.num_tokens,
        selected_count=args.selected_count,
        locality=args.locality,
        seed=args.seed,
    )
    selected_a = torch.tensor(values_a, dtype=torch.int32, device=device)
    selected_b = torch.tensor(values_b, dtype=torch.int32, device=device)
    selected_large = torch.cat((selected_a, selected_b))
    num_slots = (
        harness.dst_direct_fast[0][0].shape[0] * harness.dst_direct_fast[0][0].shape[1]
    )
    if args.locality == "contiguous":
        slot_values = list(range(args.selected_count * 2))
    else:
        slot_values = random.Random(args.seed + 1).sample(
            range(num_slots), args.selected_count * 2
        )
    slots_large = torch.tensor(slot_values, dtype=torch.long, device=device)
    slots_a = slots_large[: args.selected_count].clone()
    slots_b = slots_large[args.selected_count :].clone()

    return {
        "a": TransferTarget(
            name="a",
            destination=harness.dst_direct_fast,
            states=harness.layer_states,
            selected=selected_a,
            slots=slots_a,
        ),
        "b": TransferTarget(
            name="b",
            destination=harness.dst_direct,
            states=_prepared_states(args, harness, harness.dst_direct),
            selected=selected_b,
            slots=slots_b,
        ),
        "large": TransferTarget(
            name="large",
            destination=harness.dst_staging,
            states=_prepared_states(args, harness, harness.dst_staging),
            selected=selected_large,
            slots=slots_large,
        ),
    }


def _launch_target(
    args: argparse.Namespace,
    harness: LoadBenchmarkHarness,
    target: TransferTarget,
    *,
    validate_inputs: bool = False,
) -> None:
    for layer_id in range(args.num_layers):
        sparse_k_batched_direct_kv_transfer_experimental(
            target.states[layer_id],
            target.slots,
            target.selected,
            harness.chunk_ptrs_by_layer[layer_id],
            args.chunk_size,
            args.num_tokens,
            1,
            args.aiv_count,
            args.tile_tokens,
            True,
            0,
            1,
            validate_inputs,
            None,
        )


def _zero_target(target: TransferTarget) -> None:
    for layer in target.destination:
        for plane in layer:
            plane.zero_()


def _verify_target(
    harness: LoadBenchmarkHarness,
    target: TransferTarget,
) -> None:
    source_slots = harness.slot_mapping_full.index_select(
        0, target.selected.to(torch.long)
    )
    for layer_id, (source_layer, destination_layer) in enumerate(
        zip(
            harness.src_kv_cache,
            target.destination,
            strict=True,
        )
    ):
        for plane_index in range(2):
            source = source_layer[plane_index].flatten(0, 1)
            destination = destination_layer[plane_index].flatten(0, 1)
            expected = source.index_select(0, source_slots)
            actual = destination.index_select(0, target.slots)
            try:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            except AssertionError as exc:
                raise AssertionError(
                    f"{target.name} layer {layer_id} plane {plane_index} "
                    "verification failed"
                ) from exc


def _run_correctness(
    args: argparse.Namespace,
    harness: LoadBenchmarkHarness,
    targets: dict[str, TransferTarget],
    stream_a: torch.npu.Stream,
    stream_b: torch.npu.Stream,
) -> None:
    # Each target receives one validated launch before any timed launch.
    for target in targets.values():
        _zero_target(target)
        _launch_target(args, harness, target, validate_inputs=True)
        torch.npu.synchronize()
        _verify_target(harness, target)

    # Sequential pair.
    _zero_target(targets["a"])
    _zero_target(targets["b"])
    _launch_target(args, harness, targets["a"])
    _launch_target(args, harness, targets["b"])
    torch.npu.synchronize()
    _verify_target(harness, targets["a"])
    _verify_target(harness, targets["b"])

    # Concurrent pair. Independent destinations make this race-free.
    _zero_target(targets["a"])
    _zero_target(targets["b"])
    torch.npu.synchronize()
    with torch.npu.stream(stream_a):
        _launch_target(args, harness, targets["a"])
    with torch.npu.stream(stream_b):
        _launch_target(args, harness, targets["b"])
    torch.npu.synchronize()
    _verify_target(harness, targets["a"])
    _verify_target(harness, targets["b"])


def _warmup_case(
    args: argparse.Namespace,
    case: str,
    harness: LoadBenchmarkHarness,
    targets: dict[str, TransferTarget],
    stream_a: torch.npu.Stream,
    stream_b: torch.npu.Stream,
) -> None:
    for _ in range(args.warmup):
        if case == SINGLE_SMALL:
            _launch_target(args, harness, targets["a"])
        elif case == SINGLE_LARGE:
            _launch_target(args, harness, targets["large"])
        elif case == SEQUENTIAL_PAIR:
            _launch_target(args, harness, targets["a"])
            _launch_target(args, harness, targets["b"])
        elif case == CONCURRENT_PAIR:
            with torch.npu.stream(stream_a):
                _launch_target(args, harness, targets["a"])
            with torch.npu.stream(stream_b):
                _launch_target(args, harness, targets["b"])
        else:
            raise ValueError(f"unknown benchmark case: {case}")
    torch.npu.synchronize()


def _time_case(
    args: argparse.Namespace,
    case: str,
    harness: LoadBenchmarkHarness,
    targets: dict[str, TransferTarget],
    stream_a: torch.npu.Stream,
    stream_b: torch.npu.Stream,
    barrier: Any,
) -> list[tuple[float, int, int]]:
    _warmup_case(args, case, harness, targets, stream_a, stream_b)
    samples: list[tuple[float, int, int]] = []
    control_stream = torch.npu.current_stream()

    for _ in range(args.repeats):
        torch.npu.synchronize()
        barrier.wait(timeout=300)
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        wall_start_ns = time.monotonic_ns()
        start_event.record(control_stream)

        if case == CONCURRENT_PAIR:
            done_a = torch.npu.Event()
            done_b = torch.npu.Event()
            stream_a.wait_event(start_event)
            stream_b.wait_event(start_event)
            # Alternate A and B batches instead of enqueuing every A
            # iteration before the first B iteration.
            for _ in range(args.iters):
                with torch.npu.stream(stream_a):
                    _launch_target(args, harness, targets["a"])
                with torch.npu.stream(stream_b):
                    _launch_target(args, harness, targets["b"])
            done_a.record(stream_a)
            done_b.record(stream_b)
            control_stream.wait_event(done_a)
            control_stream.wait_event(done_b)
        else:
            for _ in range(args.iters):
                if case == SINGLE_SMALL:
                    _launch_target(args, harness, targets["a"])
                elif case == SINGLE_LARGE:
                    _launch_target(args, harness, targets["large"])
                elif case == SEQUENTIAL_PAIR:
                    _launch_target(args, harness, targets["a"])
                    _launch_target(args, harness, targets["b"])
                else:
                    raise ValueError(f"unknown benchmark case: {case}")

        end_event.record(control_stream)
        end_event.synchronize()
        wall_end_ns = time.monotonic_ns()
        samples.append(
            (
                float(start_event.elapsed_time(end_event)),
                wall_start_ns,
                wall_end_ns,
            )
        )
        barrier.wait(timeout=300)
    return samples


def _worker(
    rank: int,
    args: argparse.Namespace,
    barrier: Any,
    output: Any,
) -> None:
    shared_owner = None
    shared_mapping = None
    harness = None
    try:
        device = torch.device(f"npu:{args.devices[rank]}")
        torch.npu.set_device(device)
        hardware_aiv = sparse_transfer_hardware_aiv_num()
        if args.aiv_count > hardware_aiv:
            raise ValueError(
                f"--aiv-count={args.aiv_count} exceeds device "
                f"{args.devices[rank]} hardware AIV count {hardware_aiv}"
            )

        if args.shared_cpu_slab:
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
        else:
            shared_cpu_slab = None

        shape = Shape(
            chunk_size=args.chunk_size,
            selected_per_row=args.selected_count * 2,
            request_count=0,
            valid_fraction=1.0,
        )
        if args.shared_cpu_slab:
            if rank == 0:
                harness, _, _, _ = _build_harness(
                    args,
                    shape,
                    rank,
                    shared_cpu_slab,
                    populate_cpu_source=True,
                )
            barrier.wait(timeout=300)
            if rank != 0:
                harness, _, _, _ = _build_harness(
                    args,
                    shape,
                    rank,
                    shared_cpu_slab,
                    populate_cpu_source=False,
                )
            barrier.wait(timeout=300)
        else:
            harness, _, _, _ = _build_harness(args, shape, rank)

        targets = _prepare_targets(args, harness, device)
        stream_a = torch.npu.Stream()
        stream_b = torch.npu.Stream()

        if args.verify:
            _run_correctness(
                args,
                harness,
                targets,
                stream_a,
                stream_b,
            )
        else:
            # Validate every immutable target configuration once, then exclude
            # argument checking from the timed host path.
            for target in targets.values():
                _launch_target(
                    args,
                    harness,
                    target,
                    validate_inputs=True,
                )
            torch.npu.synchronize()

        results = {}
        for case in CASE_ORDER:
            if rank == 0:
                print(f"[progress] starting {case}", flush=True)
            results[case] = {
                "samples": _time_case(
                    args,
                    case,
                    harness,
                    targets,
                    stream_a,
                    stream_b,
                    barrier,
                )
            }
        output.put(
            {
                "rank": rank,
                "device": args.devices[rank],
                "hardware_aiv": hardware_aiv,
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


def _collect_workers(
    processes: Sequence[mp.Process],
    output: Any,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    workers = []
    deadline = time.monotonic() + timeout_sec
    while len(workers) < len(processes):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "timed out waiting for benchmark workers; "
                f"received {len(workers)}/{len(processes)} results"
            )
        try:
            workers.append(output.get(timeout=min(1.0, remaining)))
        except Empty:
            if all(not process.is_alive() for process in processes):
                exit_codes = [process.exitcode for process in processes]
                raise RuntimeError(
                    "benchmark workers exited before reporting all results: "
                    f"received {len(workers)}/{len(processes)}, "
                    f"exit codes={exit_codes}"
                ) from None
    return workers


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _summarize(
    args: argparse.Namespace,
    workers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    element_size = torch.empty(0, dtype=torch.bfloat16).element_size()
    small_bytes = (
        args.selected_count
        * (args.k_hidden_dims + args.host_v_hidden_dims)
        * element_size
        * args.num_layers
    )
    payloads = {
        SINGLE_SMALL: small_bytes,
        SINGLE_LARGE: small_bytes * 2,
        SEQUENTIAL_PAIR: small_bytes * 2,
        CONCURRENT_PAIR: small_bytes * 2,
    }
    labels = {
        SINGLE_SMALL: f"single {args.selected_count}",
        SINGLE_LARGE: f"single {args.selected_count * 2}",
        SEQUENTIAL_PAIR: f"sequential 2x{args.selected_count}",
        CONCURRENT_PAIR: f"concurrent 2x{args.selected_count}",
    }
    case_rows: dict[str, dict[str, Any]] = {}

    for case in CASE_ORDER:
        critical_ms = [
            max(worker["results"][case]["samples"][repeat][0] for worker in workers)
            / args.iters
            for repeat in range(args.repeats)
        ]
        wall_gbps = []
        for repeat in range(args.repeats):
            samples = [worker["results"][case]["samples"][repeat] for worker in workers]
            wall_seconds_per_invocation = (
                (
                    max(sample[2] for sample in samples)
                    - min(sample[1] for sample in samples)
                )
                / 1e9
                / args.iters
            )
            wall_gbps.append(
                payloads[case] * len(workers) / wall_seconds_per_invocation / 1e9
            )
        median_critical_ms = _median(critical_ms)
        case_rows[case] = {
            "case": case,
            "label": labels[case],
            "bytes_per_invocation_per_rank": payloads[case],
            "median_critical_ms_per_invocation": median_critical_ms,
            "median_critical_us_per_layer": (
                median_critical_ms * 1000 / args.num_layers
            ),
            "critical_event_gbps_per_rank": (
                payloads[case] / (median_critical_ms / 1000) / 1e9
            ),
            "median_aggregate_wall_gbps": _median(wall_gbps),
            "critical_ms_samples": critical_ms,
            "aggregate_wall_gbps_samples": wall_gbps,
        }

    sequential_ms = case_rows[SEQUENTIAL_PAIR]["median_critical_ms_per_invocation"]
    for case in (SINGLE_LARGE, SEQUENTIAL_PAIR, CONCURRENT_PAIR):
        case_rows[case]["speedup_vs_sequential_pair"] = (
            sequential_ms / case_rows[case]["median_critical_ms_per_invocation"]
        )
    case_rows[SINGLE_SMALL]["speedup_vs_sequential_pair"] = None

    concurrent_ms = case_rows[CONCURRENT_PAIR]["median_critical_ms_per_invocation"]
    large_ms = case_rows[SINGLE_LARGE]["median_critical_ms_per_invocation"]
    comparison = {
        "concurrent_vs_sequential_speedup": sequential_ms / concurrent_ms,
        "concurrent_vs_single_large_speedup": large_ms / concurrent_ms,
        "single_large_vs_sequential_speedup": sequential_ms / large_ms,
    }

    print(
        "\nBoth-plane useful payload per rank/invocation: "
        f"small={format_bytes_short(small_bytes)}, "
        f"equal-payload cases={format_bytes_short(small_bytes * 2)}"
    )
    print(
        "\ncase                      critical us/layer  "
        "event GB/s/rank  aggregate wall GB/s  vs sequential"
    )
    for case in CASE_ORDER:
        row = case_rows[case]
        speedup = row["speedup_vs_sequential_pair"]
        speedup_text = "       n/a" if speedup is None else f"{speedup:9.3f}x"
        print(
            f"{row['label'][:25]:25} "
            f"{row['median_critical_us_per_layer']:17.2f}  "
            f"{row['critical_event_gbps_per_rank']:15.2f}  "
            f"{row['median_aggregate_wall_gbps']:21.2f}  "
            f"{speedup_text}"
        )
    print(
        "\nconcurrent/sequential equal-payload speedup: "
        f"{comparison['concurrent_vs_sequential_speedup']:.3f}x"
    )
    print(
        "concurrent/single-large equal-payload speedup: "
        f"{comparison['concurrent_vs_single_large_speedup']:.3f}x"
    )
    if comparison["concurrent_vs_sequential_speedup"] <= 1.03:
        print(
            "Interpretation: two streams provide no stable >3% gain; the "
            "transfer path is likely already saturated or the streams contend."
        )
    else:
        print(
            "Interpretation: two streams show >3% overlap. Repeat the run and "
            "inspect per-repeat samples before considering integration."
        )

    return {
        "settings": {
            "devices": args.devices,
            "num_tokens": args.num_tokens,
            "selected_count": args.selected_count,
            "num_layers": args.num_layers,
            "chunk_size": args.chunk_size,
            "block_size": args.block_size,
            "k_hidden_dims": args.k_hidden_dims,
            "host_v_hidden_dims": args.host_v_hidden_dims,
            "aiv_count": args.aiv_count,
            "tile_tokens": args.tile_tokens,
            "locality": args.locality,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "verify": args.verify,
            "shared_cpu_slab": args.shared_cpu_slab,
            "hardware_aiv_by_rank": [worker["hardware_aiv"] for worker in workers],
        },
        "cases": case_rows,
        "comparison": comparison,
        "workers": workers,
    }


def main() -> None:
    args = _parse_args()
    small_bytes = (
        args.selected_count
        * (args.k_hidden_dims + args.host_v_hidden_dims)
        * torch.empty(0, dtype=torch.bfloat16).element_size()
        * args.num_layers
    )
    print(
        f"devices={args.devices} tokens={args.num_tokens} "
        f"selected={args.selected_count} layers={args.num_layers} "
        f"chunk_size={args.chunk_size} block_size={args.block_size} "
        f"aiv/kernel={args.aiv_count} tile={args.tile_tokens} "
        f"locality={args.locality}",
        flush=True,
    )
    print(
        f"cases=4 repeats={args.repeats} iters={args.iters}; "
        f"small payload/rank={format_bytes_short(small_bytes)}; "
        f"equal-payload/rank={format_bytes_short(small_bytes * 2)}; "
        f"CPU slab={'shared' if args.shared_cpu_slab else 'private-per-rank'}",
        flush=True,
    )
    if args.shared_cpu_slab:
        print(f"shared CPU slab: {args.shared_slab_name}", flush=True)
    if args.dry_run:
        print("dry run: no NPU workers started", flush=True)
        return

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

    workers: list[dict[str, Any]] = []
    try:
        workers = _collect_workers(
            processes,
            output,
            args.worker_timeout_sec,
        )
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
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {output_path}")


if __name__ == "__main__":
    main()
