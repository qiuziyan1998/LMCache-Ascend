#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tune and compare the original and experimental K-only sparse H2D kernels.

The extension is compiled once.  AIV count, UB tile size, double buffering,
chunk addressing, and work assignment are runtime arguments.  The benchmark
keeps the production ``MLA_LATENT`` host layout (K followed by an unused
V/rope plane) and prepares a one-plane destination through the existing
``DSA_INDEX`` specialization. Neither kernel reads or counts the V plane.

Recommended first run:

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  numactl --interleave=all \
  python3 benchmark/v1/kv_transfer/benchmark_sparse_k_transfer_tuning.py \
    --devices 0,1,2,3,4,5,6,7 \
    --preset smoke --shared-cpu-slab --verify \
    --output-json /workspace/qzy/sparse-k-smoke.json

The ``standard`` and ``full`` presets expand the shape and configuration
matrix.  Every preset can be overridden with the comma-separated list flags.

The ``bandwidth`` preset is deliberately narrow: it sends 256 through 16384
tokens through the existing production and K-only sparse kernels, batches
enough invocations to amortize event overhead, and fits
``time = fixed_latency + bytes / bandwidth``. It also measures raw
``aclrtMemcpyAsync`` on registered memory as a reference; it does not compile
or launch a synthetic bandwidth kernel.
"""

from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import statistics
import sys
import time
from typing import Any, Iterable, Sequence
import uuid

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
    KV_FORMAT_DSA_INDEX,
    LoadBenchmarkHarness,
    MlaDsaDims,
    build_load_benchmark_harness,
    compute_chunk_partition,
    format_bytes_short,
    recommended_num_blocks,
    shared_cpu_slab_bytes,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MixedMemoryAllocator,
    PinMemoryAllocator,
)
from lmcache.v1.shared_cpu_cache import SharedSlabMapping  # noqa: E402
import lmcache_ascend.c_ops as lmc_ops  # noqa: E402
from lmcache_ascend.v1.npu_connector.utils import (  # noqa: E402
    prepare_sparse_direct_destination_state,
    sparse_k_batched_direct_kv_transfer_experimental,
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
    sparse_transfer_hardware_aiv_num,
)


LOCALITY_CASES = (
    "src_contiguous__dst_contiguous",
    "src_contiguous__dst_scattered",
    "src_scattered__dst_contiguous",
    "src_scattered__dst_scattered",
)

ADDRESSING_MODES = {"auto": 0, "divide": 1, "shift": 2}
WORK_ASSIGNMENTS = {"balanced": 0, "striped": 1}
VARIANTS = {"production", "original_tuned", "k_serial", "k_pipeline"}


@dataclass(frozen=True)
class Shape:
    chunk_size: int
    selected_per_row: int
    request_count: int
    valid_fraction: float

    @property
    def rows(self) -> int:
        return max(1, self.request_count)

    @property
    def capacity(self) -> int:
        return self.rows * self.selected_per_row

    @property
    def count_aware(self) -> bool:
        return self.request_count > 0

    @property
    def name(self) -> str:
        count_label = f"r{self.request_count}" if self.count_aware else "nocount"
        return (
            f"k{self.selected_per_row}_c{self.chunk_size}_{count_label}_"
            f"v{self.valid_fraction:g}"
        )


@dataclass(frozen=True)
class KernelConfig:
    name: str
    variant: str
    kernel_variant: int
    aiv_num: int
    tile_tokens: int
    pipeline: bool
    addressing: str
    work_assignment: str


@dataclass(frozen=True)
class Preset:
    selected_counts: tuple[int, ...]
    chunk_sizes: tuple[int, ...]
    request_counts: tuple[int, ...]
    valid_fractions: tuple[float, ...]
    aiv_counts: tuple[int, ...]
    tile_tokens: tuple[int, ...]
    addressing_modes: tuple[str, ...]
    work_assignments: tuple[str, ...]
    localities: tuple[str, ...]


PRESETS = {
    "smoke": Preset(
        selected_counts=(256, 2048),
        chunk_sizes=(256,),
        request_counts=(0, 1),
        valid_fractions=(1.0,),
        aiv_counts=(4, 8, 16, 0),
        tile_tokens=(1, 8, 16),
        addressing_modes=("auto", "divide"),
        work_assignments=("balanced", "striped"),
        localities=(
            "src_contiguous__dst_contiguous",
            "src_scattered__dst_scattered",
        ),
    ),
    "standard": Preset(
        selected_counts=(64, 128, 256, 512, 1024, 2048, 4096),
        chunk_sizes=(256, 1000),
        request_counts=(0, 1),
        valid_fractions=(1.0,),
        aiv_counts=(2, 4, 8, 12, 16, 24, 0),
        tile_tokens=(1, 4, 8, 16, 32),
        addressing_modes=("auto", "divide"),
        work_assignments=("balanced", "striped"),
        localities=LOCALITY_CASES,
    ),
    "bandwidth": Preset(
        selected_counts=(256, 512, 1024, 2048, 4096, 8192, 16384),
        chunk_sizes=(256,),
        request_counts=(0,),
        valid_fractions=(1.0,),
        aiv_counts=(0,),
        tile_tokens=(16,),
        addressing_modes=("auto",),
        work_assignments=("striped",),
        localities=(
            "src_contiguous__dst_contiguous",
            "src_scattered__dst_scattered",
        ),
    ),
    "full": Preset(
        selected_counts=(64, 128, 256, 512, 1024, 2048, 4096),
        chunk_sizes=(64, 128, 256, 512, 1000),
        request_counts=(0, 1, 2, 4, 8),
        valid_fractions=(1.0, 0.75),
        aiv_counts=(2, 4, 8, 12, 16, 24, 32, 0),
        tile_tokens=(1, 2, 4, 8, 16, 32),
        addressing_modes=("auto", "divide"),
        work_assignments=("balanced", "striped"),
        localities=LOCALITY_CASES,
    ),
}


def _csv_values(raw: str, cast) -> tuple:
    return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def _positive_ints(
    raw: str, option: str, *, allow_zero: bool = False
) -> tuple[int, ...]:
    values = _csv_values(raw, int)
    lower = 0 if allow_zero else 1
    if not values or any(value < lower for value in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{option} must contain {qualifier} integers")
    return tuple(dict.fromkeys(values))


def _byte_size(raw: str, option: str, *, allow_zero: bool = False) -> int:
    text = raw.strip().upper().replace("IB", "B")
    multipliers = {
        "B": 1,
        "KB": 1 << 10,
        "K": 1 << 10,
        "MB": 1 << 20,
        "M": 1 << 20,
        "GB": 1 << 30,
        "G": 1 << 30,
    }
    for suffix in ("GB", "MB", "KB", "G", "M", "K", "B"):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            try:
                value = int(float(number) * multipliers[suffix])
            except ValueError as exc:
                raise ValueError(f"{option} is not a byte size: {raw}") from exc
            break
    else:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{option} is not a byte size: {raw}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{option} must be {qualifier}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--num-tokens", type=int, default=20000)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--k-hidden-dims", type=int, default=512)
    parser.add_argument(
        "--host-v-hidden-dims",
        type=int,
        default=64,
        help="unused trailing V plane retained in each MLA_LATENT host chunk",
    )
    parser.add_argument("--selected-counts", default="")
    parser.add_argument("--chunk-sizes", default="")
    parser.add_argument(
        "--request-counts",
        default="",
        help="0 disables selected_token_counts; 1+ creates that many rows",
    )
    parser.add_argument("--valid-fractions", default="")
    parser.add_argument(
        "--aiv-counts",
        default="",
        help="runtime AIV candidates; 0 means the production auto count",
    )
    parser.add_argument("--tile-tokens", default="")
    parser.add_argument(
        "--addressing-modes",
        default="",
        help="comma-separated subset of auto,divide,shift",
    )
    parser.add_argument(
        "--work-assignments",
        default="",
        help="comma-separated subset of balanced,striped",
    )
    parser.add_argument(
        "--variants",
        default="production,original_tuned,k_serial,k_pipeline",
    )
    parser.add_argument(
        "--localities",
        default="",
        help="comma-separated locality names, or all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--target-bytes-per-sample",
        default="",
        help=(
            "dynamically increase iterations until each timed sample moves "
            "this many useful K bytes; bandwidth preset default: 512M; "
            "0 keeps --iters fixed"
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=4096,
        help="cap for target-bytes-derived iterations",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--near-best-percent",
        type=float,
        default=3.0,
        help="prefer the fewest AIVs within this percentage of best latency",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--shared-cpu-slab", action="store_true")
    parser.add_argument(
        "--raw-memcpy-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "measure raw aclrtMemcpyAsync over the same payload sizes "
            "(enabled by default for the bandwidth preset)"
        ),
    )
    parser.add_argument(
        "--fit-min-payload",
        default="",
        help=(
            "minimum useful K bytes per layer in affine bandwidth fits; "
            "bandwidth preset default: 1M; 0 disables fits"
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="rank-0 progress interval in completed configuration cases; 0 disables",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sweep size and exit before starting NPU workers",
    )
    parser.add_argument(
        "--worker-timeout-sec",
        type=int,
        default=86400,
        help="parent wait timeout for long standard/full sweeps",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    args.devices = list(_positive_ints(args.devices, "--devices", allow_zero=True))
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices cannot contain duplicates")
    try:
        args.selected_counts = (
            _positive_ints(args.selected_counts, "--selected-counts")
            if args.selected_counts
            else preset.selected_counts
        )
        args.chunk_sizes = (
            _positive_ints(args.chunk_sizes, "--chunk-sizes")
            if args.chunk_sizes
            else preset.chunk_sizes
        )
        args.request_counts = (
            _positive_ints(
                args.request_counts, "--request-counts", allow_zero=True
            )
            if args.request_counts
            else preset.request_counts
        )
        args.aiv_counts = (
            _positive_ints(args.aiv_counts, "--aiv-counts", allow_zero=True)
            if args.aiv_counts
            else preset.aiv_counts
        )
        args.tile_tokens = (
            _positive_ints(args.tile_tokens, "--tile-tokens")
            if args.tile_tokens
            else preset.tile_tokens
        )
        args.target_bytes_per_sample = _byte_size(
            args.target_bytes_per_sample
            or ("512M" if args.preset == "bandwidth" else "0"),
            "--target-bytes-per-sample",
            allow_zero=True,
        )
        args.fit_min_payload = _byte_size(
            args.fit_min_payload
            or ("1M" if args.preset == "bandwidth" else "0"),
            "--fit-min-payload",
            allow_zero=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.raw_memcpy_reference is None:
        args.raw_memcpy_reference = args.preset == "bandwidth"

    args.valid_fractions = (
        _csv_values(args.valid_fractions, float)
        if args.valid_fractions
        else preset.valid_fractions
    )
    if any(not 0 < fraction <= 1 for fraction in args.valid_fractions):
        parser.error("--valid-fractions values must be in (0, 1]")

    args.addressing_modes = (
        _csv_values(args.addressing_modes, str)
        if args.addressing_modes
        else preset.addressing_modes
    )
    unknown = set(args.addressing_modes) - set(ADDRESSING_MODES)
    if unknown:
        parser.error(f"unknown addressing modes: {sorted(unknown)}")

    args.work_assignments = (
        _csv_values(args.work_assignments, str)
        if args.work_assignments
        else preset.work_assignments
    )
    unknown = set(args.work_assignments) - set(WORK_ASSIGNMENTS)
    if unknown:
        parser.error(f"unknown work assignments: {sorted(unknown)}")

    args.variants = _csv_values(args.variants, str)
    unknown = set(args.variants) - VARIANTS
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    if "production" not in args.variants:
        parser.error("--variants must include production as the comparison baseline")

    if args.localities:
        args.localities = (
            LOCALITY_CASES
            if args.localities == "all"
            else _csv_values(args.localities, str)
        )
    else:
        args.localities = preset.localities
    unknown = set(args.localities) - set(LOCALITY_CASES)
    if unknown:
        parser.error(f"unknown locality cases: {sorted(unknown)}")

    for name in (
        "num_tokens",
        "num_layers",
        "block_size",
        "k_hidden_dims",
        "host_v_hidden_dims",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "iters",
        "max_iters",
        "repeats",
        "top",
        "worker_timeout_sec",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.near_best_percent < 0:
        parser.error("--near-best-percent cannot be negative")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.k_hidden_dims * 2 % 32:
        parser.error(
            "--k-hidden-dims with BF16 must make each token 32-byte aligned"
        )
    return args


def _shapes(args: argparse.Namespace) -> list[Shape]:
    shapes = []
    for chunk_size in args.chunk_sizes:
        for selected in args.selected_counts:
            for request_count in args.request_counts:
                for valid_fraction in args.valid_fractions:
                    if request_count == 0 and valid_fraction != 1.0:
                        continue
                    shape = Shape(
                        chunk_size=chunk_size,
                        selected_per_row=selected,
                        request_count=request_count,
                        valid_fraction=valid_fraction,
                    )
                    if shape.capacity < args.num_tokens:
                        shapes.append(shape)
    if not shapes:
        raise ValueError(
            "no valid sparse shapes: selected_per_row * request_count must "
            "be smaller than --num-tokens"
        )
    return shapes


def _kernel_configs(
    args: argparse.Namespace, max_aiv_num: int, shape: Shape
) -> list[KernelConfig]:
    requested_aivs = [
        value for value in args.aiv_counts if value == 0 or value <= max_aiv_num
    ]
    if not requested_aivs:
        requested_aivs = [0]
    chunk_is_power_of_two = not (
        shape.chunk_size & (shape.chunk_size - 1)
    )
    addressing_candidates = []
    effective_addressing = set()
    for addressing in args.addressing_modes:
        if addressing == "shift" and not chunk_is_power_of_two:
            continue
        effective = (
            "divide"
            if addressing == "divide" or not chunk_is_power_of_two
            else "shift"
        )
        if effective in effective_addressing:
            continue
        addressing_candidates.append(addressing)
        effective_addressing.add(effective)

    configs: list[KernelConfig] = []
    if "production" in args.variants:
        configs.append(
            KernelConfig(
                name="production",
                variant="production",
                kernel_variant=0,
                aiv_num=0,
                tile_tokens=1,
                pipeline=False,
                addressing="auto",
                work_assignment="balanced",
            )
        )

    if "original_tuned" in args.variants:
        for aiv_num in requested_aivs:
            if aiv_num == 0:
                continue
            configs.append(
                KernelConfig(
                    name=f"original_aiv{aiv_num}",
                    variant="original_tuned",
                    kernel_variant=0,
                    aiv_num=aiv_num,
                    tile_tokens=1,
                    pipeline=False,
                    addressing="auto",
                    work_assignment="balanced",
                )
            )

    for variant, pipeline in (("k_serial", False), ("k_pipeline", True)):
        if variant not in args.variants:
            continue
        for aiv_num in requested_aivs:
            for tile_tokens in args.tile_tokens:
                for addressing in addressing_candidates:
                    for assignment in args.work_assignments:
                        configs.append(
                            KernelConfig(
                                name=(
                                    f"{variant}_aiv{aiv_num}_t{tile_tokens}_"
                                    f"{addressing}_{assignment}"
                                ),
                                variant=variant,
                                kernel_variant=1,
                                aiv_num=aiv_num,
                                tile_tokens=tile_tokens,
                                pipeline=pipeline,
                                addressing=addressing,
                                work_assignment=assignment,
                            )
                        )
    return configs


def _valid_counts(shape: Shape) -> list[int]:
    if not shape.count_aware:
        return [shape.capacity]
    base = max(1, int(round(shape.selected_per_row * shape.valid_fraction)))
    spread = max(1, shape.selected_per_row // 64)
    return [
        max(1, min(shape.selected_per_row, base - (row % 4) * spread))
        for row in range(shape.request_count)
    ]


def _useful_k_bytes_per_layer(
    args: argparse.Namespace, shape: Shape
) -> int:
    return (
        sum(_valid_counts(shape))
        * args.k_hidden_dims
        * torch.empty(0, dtype=torch.bfloat16).element_size()
    )


def _timed_iters(
    args: argparse.Namespace, useful_bytes_per_invocation: int
) -> int:
    if args.target_bytes_per_sample <= 0:
        return args.iters
    target_iters = math.ceil(
        args.target_bytes_per_sample / useful_bytes_per_invocation
    )
    return max(args.iters, min(args.max_iters, target_iters))


def _valid_positions(shape: Shape, counts: Sequence[int]) -> list[int]:
    if not shape.count_aware:
        return list(range(shape.capacity))
    return [
        row * shape.selected_per_row + offset
        for row, count in enumerate(counts)
        for offset in range(count)
    ]


def _locality_cases(
    *,
    num_tokens: int,
    capacity: int,
    num_slots: int,
    seed: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    source_contiguous = torch.arange(capacity, dtype=torch.int32, device=device)
    source_scattered = torch.tensor(
        random.Random(seed).sample(range(num_tokens), capacity),
        dtype=torch.int32,
        device=device,
    )
    destination_contiguous = torch.arange(
        capacity, dtype=torch.long, device=device
    )
    destination_scattered = torch.tensor(
        random.Random(seed + 1).sample(range(num_slots), capacity),
        dtype=torch.long,
        device=device,
    )
    return {
        LOCALITY_CASES[0]: (source_contiguous, destination_contiguous),
        LOCALITY_CASES[1]: (source_contiguous, destination_scattered),
        LOCALITY_CASES[2]: (source_scattered, destination_contiguous),
        LOCALITY_CASES[3]: (source_scattered, destination_scattered),
    }


def _build_harness(
    args: argparse.Namespace,
    shape: Shape,
    rank: int,
    shared_cpu_slab: torch.Tensor | None = None,
    *,
    populate_cpu_source: bool = True,
) -> tuple[
    LoadBenchmarkHarness,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor | None,
    list[int],
]:
    device = torch.device(f"npu:{args.devices[rank]}")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed if args.shared_cpu_slab else args.seed + rank)

    num_blocks = recommended_num_blocks(args.num_tokens, args.block_size)
    source_cache = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=args.num_layers,
        num_kv_heads=1,
        kv_lora_rank=args.k_hidden_dims,
        qk_rope_head_dim=args.host_v_hidden_dims,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )
    harness = build_load_benchmark_harness(
        fmt="mla_latent",
        src_kv_cache=source_cache,
        num_tokens=args.num_tokens,
        chunk_size=shape.chunk_size,
        num_layers=args.num_layers,
        num_selected=shape.capacity,
        seed=args.seed,
        shared_cpu_slab=shared_cpu_slab,
        populate_cpu_source=populate_cpu_source,
        copy_expected_cpu_chunks=False,
    )
    # Keep the production MLA_LATENT source-chunk allocation (K followed by
    # the unused V/rope plane), but prepare a one-plane destination so both
    # the original and experimental kernels copy K only.
    harness.layer_states = [
        prepare_sparse_direct_destination_state(
            (harness.dst_direct_fast[layer_id][0],),
            harness.slot_mapping_full,
            KV_FORMAT_DSA_INDEX,
            args.k_hidden_dims,
            0,
            args.k_hidden_dims,
        )
        for layer_id in range(args.num_layers)
    ]
    num_slots = (
        harness.dst_direct_fast[0][0].shape[0]
        * harness.dst_direct_fast[0][0].shape[1]
    )
    counts = _valid_counts(shape)
    selected_counts = (
        torch.tensor(counts, dtype=torch.int32, device=device)
        if shape.count_aware
        else None
    )
    cases = _locality_cases(
        num_tokens=args.num_tokens,
        capacity=shape.capacity,
        num_slots=num_slots,
        seed=args.seed,
        device=device,
    )
    return harness, cases, selected_counts, counts


def _set_case(
    harness: LoadBenchmarkHarness,
    case: tuple[torch.Tensor, torch.Tensor],
) -> None:
    harness.selected_token_idx, harness.slot_mapping_packed = case


def _run_layers(
    harness: LoadBenchmarkHarness,
    config: KernelConfig,
    selected_counts: torch.Tensor | None,
    *,
    validate_inputs: bool = False,
) -> None:
    for layer_id in range(harness.num_layers):
        if config.variant == "production":
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                harness.layer_states[layer_id],
                harness.slot_mapping_packed,
                harness.selected_token_idx,
                harness.chunk_ptrs_by_layer[layer_id],
                harness.chunk_size,
                harness.num_tokens,
                False,
                selected_counts,
            )
            continue
        sparse_k_batched_direct_kv_transfer_experimental(
            harness.layer_states[layer_id],
            harness.slot_mapping_packed,
            harness.selected_token_idx,
            harness.chunk_ptrs_by_layer[layer_id],
            harness.chunk_size,
            harness.num_tokens,
            config.kernel_variant,
            config.aiv_num,
            config.tile_tokens,
            config.pipeline,
            ADDRESSING_MODES[config.addressing],
            WORK_ASSIGNMENTS[config.work_assignment],
            validate_inputs,
            selected_counts,
        )


def _verify(
    harness: LoadBenchmarkHarness,
    shape: Shape,
    counts: Sequence[int],
    config: KernelConfig,
    selected_counts: torch.Tensor | None,
) -> None:
    for layer in harness.dst_direct_fast:
        for plane in layer:
            plane.zero_()
    _run_layers(
        harness,
        config,
        selected_counts,
        validate_inputs=True,
    )
    torch.npu.synchronize()

    valid_positions = torch.tensor(
        _valid_positions(shape, counts),
        dtype=torch.long,
        device=harness.selected_token_idx.device,
    )
    valid_selected = harness.selected_token_idx.index_select(0, valid_positions)
    valid_destinations = harness.slot_mapping_packed.index_select(
        0, valid_positions
    )
    source_slots = harness.slot_mapping_full.index_select(
        0, valid_selected.to(torch.long)
    )
    for source_layer, destination_layer in zip(
        harness.src_kv_cache, harness.dst_direct_fast, strict=True
    ):
        source = source_layer[0].flatten(0, 1)
        destination = destination_layer[0].flatten(0, 1)
        expected = source.index_select(0, source_slots)
        actual = destination.index_select(0, valid_destinations)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        untouched_v = destination_layer[1].flatten(0, 1).index_select(
            0, valid_destinations
        )
        if torch.count_nonzero(untouched_v).item() != 0:
            raise AssertionError(f"{config.name} unexpectedly copied V")

    if len(valid_positions) == shape.capacity:
        return
    valid_set = set(_valid_positions(shape, counts))
    padded_positions = torch.tensor(
        [index for index in range(shape.capacity) if index not in valid_set],
        dtype=torch.long,
        device=harness.selected_token_idx.device,
    )
    padded_destinations = harness.slot_mapping_packed.index_select(
        0, padded_positions
    )
    for destination_layer in harness.dst_direct_fast:
        padded = destination_layer[0].flatten(0, 1).index_select(
            0, padded_destinations
        )
        if torch.count_nonzero(padded).item() != 0:
            raise AssertionError(
                f"{config.name} copied one or more padded row-tail tokens"
            )


def _shared_slab_size(args: argparse.Namespace) -> int:
    dims = MlaDsaDims(
        k_hidden_dims=args.k_hidden_dims,
        v_hidden_dims=args.host_v_hidden_dims,
        dsa_hidden_dims=0,
        plane_elems=args.k_hidden_dims + args.host_v_hidden_dims,
    )
    return max(
        shared_cpu_slab_bytes(
            compute_chunk_partition(args.num_tokens, chunk_size),
            dims,
            torch.bfloat16,
            args.num_layers,
        )
        for chunk_size in args.chunk_sizes
    )


def _worker(
    rank: int,
    args: argparse.Namespace,
    shapes: Sequence[Shape],
    barrier: Any,
    output: Any,
) -> None:
    shared_owner = None
    shared_mapping = None
    harness = None
    raw_allocator = None
    raw_mem_obj = None
    try:
        worker_start = time.monotonic()
        if rank == 0:
            print(
                f"[progress] initializing NPU worker on device "
                f"{args.devices[rank]}",
                flush=True,
            )
        device = torch.device(f"npu:{args.devices[rank]}")
        torch.npu.set_device(device)
        max_aiv_num = sparse_transfer_hardware_aiv_num()
        total_config_cases = sum(
            len(_kernel_configs(args, max_aiv_num, shape))
            * len(args.localities)
            for shape in shapes
        )
        completed_config_cases = 0
        if rank == 0:
            print(
                f"[progress] worker ready: device={args.devices[rank]} "
                f"hardware_aiv={max_aiv_num} "
                f"configuration_cases={total_config_cases}",
                flush=True,
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

        all_results: dict[str, Any] = {}
        for shape_index, shape in enumerate(shapes, start=1):
            if harness is not None:
                harness.close()
                harness = None
            if rank == 0:
                print(
                    f"[progress] preparing shape {shape_index}/{len(shapes)}: "
                    f"{shape.name}",
                    flush=True,
                )
            if args.shared_cpu_slab:
                if rank == 0:
                    harness, cases, selected_counts, counts = _build_harness(
                        args,
                        shape,
                        rank,
                        shared_cpu_slab,
                        populate_cpu_source=True,
                    )
                barrier.wait(timeout=300)
                if rank != 0:
                    harness, cases, selected_counts, counts = _build_harness(
                        args,
                        shape,
                        rank,
                        shared_cpu_slab,
                        populate_cpu_source=False,
                    )
                barrier.wait(timeout=300)
            else:
                harness, cases, selected_counts, counts = _build_harness(
                    args, shape, rank
                )

            configs = _kernel_configs(args, max_aiv_num, shape)
            useful_bytes_per_layer = _useful_k_bytes_per_layer(args, shape)
            timed_iters = _timed_iters(
                args, useful_bytes_per_layer * args.num_layers
            )
            if rank == 0:
                print(
                    f"[progress] running {shape.name}: "
                    f"{len(configs)} configs x {len(args.localities)} "
                    f"localities; payload/layer="
                    f"{format_bytes_short(useful_bytes_per_layer)} "
                    f"timed_iters={timed_iters}",
                    flush=True,
                )
            shape_results: dict[str, Any] = {}
            for locality in args.localities:
                if rank == 0:
                    print(
                        f"[progress] {shape.name}: starting {locality}",
                        flush=True,
                    )
                _set_case(harness, cases[locality])
                locality_results: dict[str, Any] = {}
                for config in configs:
                    if args.verify:
                        _verify(
                            harness,
                            shape,
                            counts,
                            config,
                            selected_counts,
                        )
                    elif config.variant != "production":
                        _run_layers(
                            harness,
                            config,
                            selected_counts,
                            validate_inputs=True,
                        )
                        torch.npu.synchronize()
                    for _ in range(args.warmup):
                        _run_layers(harness, config, selected_counts)
                    torch.npu.synchronize()

                    samples = []
                    for _ in range(args.repeats):
                        barrier.wait(timeout=300)
                        start_event = torch.npu.Event(enable_timing=True)
                        end_event = torch.npu.Event(enable_timing=True)
                        wall_start_ns = time.monotonic_ns()
                        start_event.record()
                        for _ in range(timed_iters):
                            _run_layers(harness, config, selected_counts)
                        end_event.record()
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
                    locality_results[config.name] = {
                        "config": asdict(config),
                        "samples": samples,
                    }
                    completed_config_cases += 1
                    should_report = (
                        args.progress_every > 0
                        and completed_config_cases % args.progress_every == 0
                    )
                    if rank == 0 and (
                        should_report
                        or completed_config_cases == total_config_cases
                    ):
                        elapsed = time.monotonic() - worker_start
                        rate = completed_config_cases / elapsed
                        remaining = total_config_cases - completed_config_cases
                        eta = remaining / rate if rate > 0 else math.inf
                        percentage = (
                            100
                            * completed_config_cases
                            / total_config_cases
                        )
                        print(
                            f"[progress] {completed_config_cases}/"
                            f"{total_config_cases} cases "
                            f"({percentage:.1f}%) "
                            f"elapsed={elapsed / 60:.1f} min "
                            f"eta={eta / 60:.1f} min last={config.name}",
                            flush=True,
                        )
                shape_results[locality] = locality_results
            all_results[shape.name] = {
                "shape": asdict(shape),
                "valid_counts": counts,
                "timed_iters": timed_iters,
                "results": shape_results,
            }
            barrier.wait(timeout=300)

        raw_memcpy_results: dict[str, Any] = {}
        if args.raw_memcpy_reference:
            raw_sizes = sorted(
                {
                    _useful_k_bytes_per_layer(args, shape)
                    for shape in shapes
                }
            )
            max_raw_size = max(raw_sizes)
            if args.shared_cpu_slab:
                raw_source = shared_cpu_slab[:max_raw_size]
            else:
                raw_allocator = PinMemoryAllocator(
                    max(max_raw_size, 64 * 1024 * 1024)
                )
                raw_mem_obj = raw_allocator.allocate(
                    torch.Size([max_raw_size]), torch.uint8
                )
                if raw_mem_obj is None or raw_mem_obj.tensor is None:
                    raise RuntimeError(
                        "failed to allocate raw memcpy registered source"
                    )
                raw_source = raw_mem_obj.tensor
                pattern_size = min(max_raw_size, 1024 * 1024)
                pattern = (
                    torch.arange(pattern_size, dtype=torch.int64)
                    .add(args.seed + rank)
                    .bitwise_xor(
                        torch.arange(pattern_size, dtype=torch.int64) >> 5
                    )
                    .to(torch.uint8)
                )
                for offset in range(0, max_raw_size, pattern_size):
                    count = min(pattern_size, max_raw_size - offset)
                    raw_source[offset : offset + count].copy_(
                        pattern[:count]
                    )
            raw_destination = torch.empty(
                max_raw_size, dtype=torch.uint8, device=device
            )
            host_source_ptr = int(raw_source.data_ptr())
            barrier.wait(timeout=300)
            if args.verify:
                raw_destination.zero_()
                lmc_ops.benchmark_aclrt_memcpy_h2d(
                    raw_destination,
                    host_source_ptr,
                    max_raw_size,
                    True,
                )
                torch.npu.synchronize()
                actual = raw_destination.cpu()
                if not torch.equal(actual, raw_source):
                    mismatch = torch.nonzero(actual != raw_source)
                    first = (
                        int(mismatch[0].item())
                        if mismatch.numel()
                        else -1
                    )
                    raise AssertionError(
                        "raw aclrtMemcpyAsync verification failed at byte "
                        f"{first}"
                    )
            for raw_size in raw_sizes:
                raw_iters = _timed_iters(args, raw_size)
                lmc_ops.benchmark_aclrt_memcpy_h2d(
                    raw_destination,
                    host_source_ptr,
                    raw_size,
                    True,
                )
                for _ in range(args.warmup):
                    lmc_ops.benchmark_aclrt_memcpy_h2d(
                        raw_destination,
                        host_source_ptr,
                        raw_size,
                        False,
                    )
                torch.npu.synchronize()
                samples = []
                for _ in range(args.repeats):
                    barrier.wait(timeout=300)
                    start_event = torch.npu.Event(enable_timing=True)
                    end_event = torch.npu.Event(enable_timing=True)
                    wall_start_ns = time.monotonic_ns()
                    start_event.record()
                    for _ in range(raw_iters):
                        lmc_ops.benchmark_aclrt_memcpy_h2d(
                            raw_destination,
                            host_source_ptr,
                            raw_size,
                            False,
                        )
                    end_event.record()
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
                raw_memcpy_results[str(raw_size)] = {
                    "timed_iters": raw_iters,
                    "samples": samples,
                }
                if rank == 0:
                    print(
                        f"[raw memcpy] payload={format_bytes_short(raw_size)} "
                        f"timed_iters={raw_iters}",
                        flush=True,
                    )

        output.put(
            {
                "rank": rank,
                "device": args.devices[rank],
                "max_aiv_num": max_aiv_num,
                "results": all_results,
                "raw_memcpy": raw_memcpy_results,
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
        if raw_mem_obj is not None:
            raw_mem_obj.ref_count_down()
        if raw_allocator is not None:
            raw_allocator.close()
        if shared_mapping is not None:
            shared_mapping.close()
        if shared_owner is not None:
            shared_owner.close()


def _median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def _ols_bandwidth_fit(
    points: Sequence[tuple[int, float]],
) -> dict[str, float | int]:
    """Fit time_ms = intercept + bytes / bandwidth."""
    if len(points) < 3:
        raise ValueError("bandwidth fit requires at least three points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) / 1000 for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    sxx = sum((value - x_mean) ** 2 for value in xs)
    sxy = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys, strict=True)
    )
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in xs]
    residuals = [
        actual - estimate
        for actual, estimate in zip(ys, predicted, strict=True)
    ]
    sse = sum(value * value for value in residuals)
    syy = sum((value - y_mean) ** 2 for value in ys)
    degrees = len(points) - 2
    slope_se = math.sqrt((sse / degrees) / sxx)
    # Student-t 97.5% quantiles for the small series used here.
    t_value = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(degrees, 1.96)
    slope_low = slope - t_value * slope_se
    slope_high = slope + t_value * slope_se
    return {
        "point_count": len(points),
        "min_payload_bytes": int(min(xs)),
        "max_payload_bytes": int(max(xs)),
        "bandwidth_gbps": 1 / slope / 1e9 if slope > 0 else math.nan,
        "bandwidth_ci95_low_gbps": (
            1 / slope_high / 1e9 if slope_high > 0 else math.nan
        ),
        "bandwidth_ci95_high_gbps": (
            1 / slope_low / 1e9 if slope_low > 0 else math.nan
        ),
        "fixed_latency_us": intercept * 1e6,
        "r_squared": 1 - sse / syy if syy > 0 else 1.0,
        "max_abs_residual_us": (
            max(abs(value) for value in residuals) * 1e6
        ),
    }


def _summarize(
    args: argparse.Namespace,
    shapes: Sequence[Shape],
    workers: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    element_size = torch.empty(0, dtype=torch.bfloat16).element_size()
    summary: dict[str, Any] = {
        "settings": {
            "devices": args.devices,
            "preset": args.preset,
            "num_tokens": args.num_tokens,
            "num_layers": args.num_layers,
            "block_size": args.block_size,
            "k_hidden_dims": args.k_hidden_dims,
            "host_v_hidden_dims": args.host_v_hidden_dims,
            "selected_counts": args.selected_counts,
            "chunk_sizes": args.chunk_sizes,
            "request_counts": args.request_counts,
            "valid_fractions": args.valid_fractions,
            "aiv_counts": args.aiv_counts,
            "tile_tokens": args.tile_tokens,
            "addressing_modes": args.addressing_modes,
            "work_assignments": args.work_assignments,
            "variants": args.variants,
            "localities": args.localities,
            "warmup": args.warmup,
            "iters": args.iters,
            "target_bytes_per_sample": args.target_bytes_per_sample,
            "max_iters": args.max_iters,
            "repeats": args.repeats,
            "near_best_percent": args.near_best_percent,
            "seed": args.seed,
            "verify": args.verify,
            "shared_cpu_slab": args.shared_cpu_slab,
            "raw_memcpy_reference": args.raw_memcpy_reference,
            "fit_min_payload": args.fit_min_payload,
            "worker_timeout_sec": args.worker_timeout_sec,
            "progress_every": args.progress_every,
            "max_aiv_num_by_rank": [
                worker["max_aiv_num"] for worker in workers
            ],
        },
        "shapes": {},
    }
    flat_rows: list[dict[str, Any]] = []
    speedups_by_config: dict[str, list[float]] = {}

    for shape in shapes:
        worker_shape = [worker["results"][shape.name] for worker in workers]
        valid_tokens = sum(worker_shape[0]["valid_counts"])
        timed_iters = int(worker_shape[0]["timed_iters"])
        bytes_per_invocation = (
            valid_tokens
            * args.k_hidden_dims
            * element_size
            * args.num_layers
        )
        bytes_per_repeat = bytes_per_invocation * timed_iters
        shape_summary = {
            "shape": asdict(shape),
            "valid_counts": worker_shape[0]["valid_counts"],
            "bytes_per_invocation": bytes_per_invocation,
            "timed_iters": timed_iters,
            "localities": {},
        }
        print(
            f"\n{shape.name}: valid_tokens={valid_tokens} "
            f"payload={format_bytes_short(bytes_per_invocation)} "
            f"timed_iters={timed_iters}"
        )

        for locality in args.localities:
            config_names = list(worker_shape[0]["results"][locality])
            rows = []
            for config_name in config_names:
                config = worker_shape[0]["results"][locality][config_name][
                    "config"
                ]
                event_ms = [
                    sample[0]
                    for worker_result in worker_shape
                    for sample in worker_result["results"][locality][
                        config_name
                    ]["samples"]
                ]
                critical_ms_per_layer = [
                    max(
                        worker_result["results"][locality][config_name][
                            "samples"
                        ][repeat][0]
                        for worker_result in worker_shape
                    )
                    / (timed_iters * args.num_layers)
                    for repeat in range(args.repeats)
                ]
                rank_bandwidths = [
                    bytes_per_repeat / (duration_ms / 1000) / 1e9
                    for duration_ms in event_ms
                ]
                wall_bandwidths = []
                for repeat in range(args.repeats):
                    samples = [
                        worker_result["results"][locality][config_name][
                            "samples"
                        ][repeat]
                        for worker_result in worker_shape
                    ]
                    wall_seconds = (
                        max(sample[2] for sample in samples)
                        - min(sample[1] for sample in samples)
                    ) / 1e9
                    wall_bandwidths.append(
                        bytes_per_repeat * len(workers) / wall_seconds / 1e9
                    )
                row = {
                    "shape": shape.name,
                    "locality": locality,
                    **config,
                    "chunk_size": shape.chunk_size,
                    "selected_per_row": shape.selected_per_row,
                    "request_count": shape.request_count,
                    "valid_fraction": shape.valid_fraction,
                    "valid_tokens": valid_tokens,
                    "bytes_per_layer": bytes_per_invocation // args.num_layers,
                    "bytes_per_invocation": bytes_per_invocation,
                    "timed_iters": timed_iters,
                    "median_ms_per_layer": _median(event_ms)
                    / (timed_iters * args.num_layers),
                    "median_critical_ms_per_layer": _median(
                        critical_ms_per_layer
                    ),
                    "median_rank_gbps": _median(rank_bandwidths),
                    "min_rank_gbps": min(rank_bandwidths),
                    "median_wall_gbps": _median(wall_bandwidths),
                }
                rows.append(row)

            baseline = next(
                (row for row in rows if row["variant"] == "production"),
                None,
            )
            for row in rows:
                row["speedup_vs_production"] = (
                    baseline["median_ms_per_layer"]
                    / row["median_ms_per_layer"]
                    if baseline is not None
                    else math.nan
                )
                if baseline is not None:
                    speedups_by_config.setdefault(row["name"], []).append(
                        row["speedup_vs_production"]
                    )
            rows.sort(key=lambda row: row["median_ms_per_layer"])
            near_best_limit = rows[0]["median_ms_per_layer"] * (
                1 + args.near_best_percent / 100
            )
            efficient_candidates = [
                row
                for row in rows
                if row["median_ms_per_layer"] <= near_best_limit
            ]
            max_aiv = min(
                worker["max_aiv_num"] for worker in workers
            )
            efficient = min(
                efficient_candidates,
                key=lambda row: (
                    row["aiv_num"] if row["aiv_num"] > 0 else max_aiv,
                    row["median_ms_per_layer"],
                ),
            )
            flat_rows.extend(rows)
            shape_summary["localities"][locality] = {
                "baseline": baseline,
                "best": rows[0],
                "fewest_aiv_near_best": efficient,
                "configs": rows,
            }

            print(f"  {locality}")
            print(
                "    config                                             "
                "us/layer  rank GB/s  wall GB/s  speedup"
            )
            for row in rows[: args.top]:
                print(
                    f"    {row['name'][:50]:50} "
                    f"{row['median_ms_per_layer'] * 1000:8.2f}  "
                    f"{row['median_rank_gbps']:9.2f}  "
                    f"{row['median_wall_gbps']:9.2f}  "
                    f"{row['speedup_vs_production']:7.3f}x"
                )
            print(
                f"    fewest AIV within {args.near_best_percent:g}%: "
                f"{efficient['name']} "
                f"({efficient['median_ms_per_layer'] * 1000:.2f} us/layer)"
            )
        summary["shapes"][shape.name] = shape_summary

    bandwidth_fits = []
    if args.fit_min_payload > 0:
        series: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in flat_rows:
            key = (
                row["locality"],
                row["name"],
                row["chunk_size"],
                row["request_count"],
                row["valid_fraction"],
            )
            series.setdefault(key, []).append(row)
        for key, rows in series.items():
            eligible = [
                row
                for row in rows
                if row["bytes_per_layer"] >= args.fit_min_payload
            ]
            unique_payloads = {
                row["bytes_per_layer"] for row in eligible
            }
            if len(unique_payloads) < 3:
                continue
            eligible.sort(key=lambda row: row["bytes_per_layer"])
            fit = _ols_bandwidth_fit(
                [
                    (
                        row["bytes_per_layer"],
                        row["median_critical_ms_per_layer"],
                    )
                    for row in eligible
                ]
            )
            bandwidth_fits.append(
                {
                    "locality": key[0],
                    "name": key[1],
                    "variant": eligible[0]["variant"],
                    "chunk_size": key[2],
                    "request_count": key[3],
                    "valid_fraction": key[4],
                    **fit,
                }
            )
        bandwidth_fits.sort(
            key=lambda item: (
                item["locality"],
                -float(item["bandwidth_gbps"]),
            )
        )
        print(
            f"\nExisting sparse-kernel affine fits "
            f"(payload/layer >= {format_bytes_short(args.fit_min_payload)})"
        )
        print(
            "locality                                  config"
            "                              GB/s   95% CI"
            "             fixed us   R^2     max residual us"
        )
        for fit in bandwidth_fits:
            print(
                f"{fit['locality'][:40]:40} "
                f"{fit['name'][:35]:35} "
                f"{fit['bandwidth_gbps']:6.2f} "
                f"[{fit['bandwidth_ci95_low_gbps']:6.2f}, "
                f"{fit['bandwidth_ci95_high_gbps']:6.2f}] "
                f"{fit['fixed_latency_us']:10.2f} "
                f"{fit['r_squared']:7.5f} "
                f"{fit['max_abs_residual_us']:15.2f}"
            )
    summary["bandwidth_fits"] = bandwidth_fits

    raw_summary: dict[str, Any] | None = None
    if args.raw_memcpy_reference:
        raw_rows = []
        raw_results = workers[0]["raw_memcpy"]
        print("\nRaw aclrtMemcpyAsync reference (slowest-rank event)")
        print(
            "payload       iters   event us   event GB/s   "
            "aggregate wall GB/s"
        )
        for size_text in sorted(raw_results, key=int):
            size = int(size_text)
            timed_iters = int(raw_results[size_text]["timed_iters"])
            critical_event_ms = []
            aggregate_wall_gbps = []
            for repeat in range(args.repeats):
                samples = [
                    worker["raw_memcpy"][size_text]["samples"][repeat]
                    for worker in workers
                ]
                critical_event_ms.append(
                    max(sample[0] for sample in samples) / timed_iters
                )
                wall_seconds_per_copy = (
                    (
                        max(sample[2] for sample in samples)
                        - min(sample[1] for sample in samples)
                    )
                    / 1e9
                    / timed_iters
                )
                aggregate_wall_gbps.append(
                    size * len(workers) / wall_seconds_per_copy / 1e9
                )
            median_event_ms = _median(critical_event_ms)
            row = {
                "payload_bytes": size,
                "timed_iters": timed_iters,
                "median_critical_event_ms": median_event_ms,
                "median_critical_event_gbps": (
                    size / (median_event_ms / 1000) / 1e9
                ),
                "median_aggregate_wall_gbps": _median(
                    aggregate_wall_gbps
                ),
            }
            raw_rows.append(row)
            print(
                f"{format_bytes_short(size):>10} "
                f"{timed_iters:7d} "
                f"{median_event_ms * 1000:10.2f} "
                f"{row['median_critical_event_gbps']:12.2f} "
                f"{row['median_aggregate_wall_gbps']:22.2f}"
            )
        eligible = [
            row
            for row in raw_rows
            if row["payload_bytes"] >= args.fit_min_payload
        ]
        raw_fit = (
            _ols_bandwidth_fit(
                [
                    (
                        row["payload_bytes"],
                        row["median_critical_event_ms"],
                    )
                    for row in eligible
                ]
            )
            if args.fit_min_payload > 0 and len(eligible) >= 3
            else None
        )
        raw_summary = {"rows": raw_rows, "fit": raw_fit}
        if raw_fit is not None:
            print(
                "raw memcpy affine fit: "
                f"{raw_fit['bandwidth_gbps']:.2f} GB/s "
                f"(95% CI {raw_fit['bandwidth_ci95_low_gbps']:.2f}-"
                f"{raw_fit['bandwidth_ci95_high_gbps']:.2f}), "
                f"fixed={raw_fit['fixed_latency_us']:.2f} us, "
                f"R^2={raw_fit['r_squared']:.5f}"
            )
    summary["raw_aclrt_memcpy_async"] = raw_summary

    stable = []
    for name, speedups in speedups_by_config.items():
        stable.append(
            {
                "name": name,
                "geomean_speedup": math.exp(
                    sum(math.log(value) for value in speedups) / len(speedups)
                ),
                "worst_speedup": min(speedups),
                "best_speedup": max(speedups),
                "cases": len(speedups),
            }
        )
    stable.sort(
        key=lambda item: (item["geomean_speedup"], item["worst_speedup"]),
        reverse=True,
    )
    summary["stable_recommendations"] = stable
    if stable:
        print("\nStable settings across all cases in which each config appears")
        print(
            "config                                             "
            "geomean  worst   best"
        )
        for item in stable[: args.top]:
            print(
                f"{item['name'][:50]:50} "
                f"{item['geomean_speedup']:7.3f}x "
                f"{item['worst_speedup']:7.3f}x "
                f"{item['best_speedup']:7.3f}x"
            )
    return summary, flat_rows


def _write_outputs(
    args: argparse.Namespace,
    summary: dict[str, Any],
    flat_rows: Sequence[dict[str, Any]],
) -> None:
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {path}")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=list(flat_rows[0]) if flat_rows else [],
            )
            if flat_rows:
                writer.writeheader()
                writer.writerows(flat_rows)
        print(f"Wrote CSV: {path}")


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
                )
    return workers


def main() -> None:
    args = _parse_args()
    shapes = _shapes(args)
    args.shared_slab_name = (
        f"/lmcache_sparse_k_tune_{os.getpid()}_{uuid.uuid4().hex}"
        if args.shared_cpu_slab
        else None
    )
    requested_aiv_ceiling = max(
        (aiv_num for aiv_num in args.aiv_counts if aiv_num > 0),
        default=1,
    )
    estimated_config_counts = [
        len(_kernel_configs(args, requested_aiv_ceiling, shape))
        for shape in shapes
    ]
    estimated_config_cases = sum(
        count * len(args.localities) for count in estimated_config_counts
    )
    estimated_layer_launches = 0
    for shape in shapes:
        timed_iters = _timed_iters(
            args, _useful_k_bytes_per_layer(args, shape) * args.num_layers
        )
        for config in _kernel_configs(args, requested_aiv_ceiling, shape):
            setup_invocations = int(args.verify or config.variant != "production")
            invocations = (
                setup_invocations
                + args.warmup
                + timed_iters * args.repeats
            )
            estimated_layer_launches += (
                len(args.localities) * args.num_layers * invocations
            )
    timed_repeats = estimated_config_cases * args.repeats
    print(
        f"devices={args.devices} preset={args.preset} shapes={len(shapes)} "
        f"localities={len(args.localities)} "
        f"estimated_configs/shape="
        f"{min(estimated_config_counts)}-{max(estimated_config_counts)}",
        flush=True,
    )
    print(
        f"estimated configuration cases/rank={estimated_config_cases}; "
        f"timed repeats/rank={timed_repeats}; "
        f"layer launches/rank={estimated_layer_launches:,}; "
        f"CPU slab={'shared' if args.shared_cpu_slab else 'private-per-rank'}",
        flush=True,
    )
    if args.target_bytes_per_sample > 0:
        shape_iters = [
            _timed_iters(
                args,
                _useful_k_bytes_per_layer(args, shape) * args.num_layers,
            )
            for shape in shapes
        ]
        print(
            f"target useful K payload/sample="
            f"{format_bytes_short(args.target_bytes_per_sample)}; "
            f"timed iters/shape={min(shape_iters)}-{max(shape_iters)}; "
            f"raw aclrtMemcpyAsync reference="
            f"{'on' if args.raw_memcpy_reference else 'off'}",
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
            args=(rank, args, shapes, barrier, output),
        )
        for rank in range(len(args.devices))
    ]
    for process in processes:
        process.start()

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
    summary, flat_rows = _summarize(args, shapes, workers)
    _write_outputs(args, summary, flat_rows)


if __name__ == "__main__":
    main()
