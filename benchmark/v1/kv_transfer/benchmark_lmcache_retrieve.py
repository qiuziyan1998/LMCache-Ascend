#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the current prepared LMCache sparse-retrieve hot path.

The default ``both`` format models the two-group layout used by the current
DSA integration: MLA latent KV (group 0) plus DSA index KV (group 1). Timing
contains the production prepared sparse transfer launches for all configured
layers and groups. Cache lookup, storage I/O, index preparation, and first-use
state construction are intentionally outside the timed region.

Effective bandwidth counts useful CPU-to-NPU KV payload bytes once. It excludes
index/count metadata and any device-internal traffic.

Examples:
  python benchmark/v1/kv_transfer/benchmark_lmcache_retrieve.py \
      --mtp 2 --topk 2048 --valid-count 2048 --verify

  python benchmark/v1/kv_transfer/benchmark_lmcache_retrieve.py \
      --kv-format dsa_index --mtp 1 --topk 2048 --compare-fixed-width
"""

from __future__ import annotations

# Standard
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Third Party
import torch

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from kv_cache_fixtures import (  # noqa: E402
    generate_dsa_index_kv_cache,
    generate_mla_kv_cache,
)
from load_benchmark_utils import (  # noqa: E402
    LoadBenchmarkHarness,
    TimingStats,
    bandwidth_gb_per_s,
    build_load_benchmark_harness,
    compute_direct_host_bytes,
    format_bytes_short,
    percentile,
    rate_per_second,
    recommended_num_blocks,
    time_npu_callable,
)
from lmcache_ascend.v1.npu_connector.utils import (  # noqa: E402
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
)

_FORMAT_MLA_LATENT = "mla_latent"
_FORMAT_DSA_INDEX = "dsa_index"


@dataclass(frozen=True)
class _RetrieveCase:
    format_name: str
    harness: LoadBenchmarkHarness
    selected: torch.Tensor
    target_slots: torch.Tensor
    useful_bytes: int
    fixed_width_bytes: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--kv-format",
        choices=("both", _FORMAT_MLA_LATENT, _FORMAT_DSA_INDEX),
        default="both",
        help="KV group(s) to include in the same timed retrieve",
    )
    parser.add_argument("--num-tokens", type=int, default=18880)
    parser.add_argument(
        "--mtp",
        type=int,
        choices=(1, 2),
        default=2,
        help="top-k rows unioned into one packed request row",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=2048,
        help="index top-k per MTP row; packed row width is MTP * top-k",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        default=2048,
        help="valid prefix length in each selected-token row",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="layers launched inside one timed retrieve sample",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--dsa-head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--compare-fixed-width",
        action="store_true",
        help="also time the old behavior that transfers every padded entry",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    row_width = args.mtp * args.topk
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if not 0 <= args.valid_count <= row_width <= args.num_tokens:
        parser.error(
            "require 0 <= --valid-count <= MTP * --topk <= --num-tokens"
        )
    if args.num_layers <= 0 or args.chunk_size <= 0 or args.block_size <= 0:
        parser.error("--num-layers, --chunk-size, and --block-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters positive")
    return args


def _formats_for(name: str) -> tuple[str, ...]:
    if name == "both":
        return (_FORMAT_MLA_LATENT, _FORMAT_DSA_INDEX)
    return (name,)


def _generate_source_cache(
    format_name: str,
    *,
    num_blocks: int,
    device: torch.device,
    args: argparse.Namespace,
):
    if format_name == _FORMAT_MLA_LATENT:
        return generate_mla_kv_cache(
            num_blocks=num_blocks,
            device=device,
            num_layers=args.num_layers,
            num_kv_heads=1,
            kv_lora_rank=args.kv_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            block_size=args.block_size,
            dtype=torch.bfloat16,
        )
    return generate_dsa_index_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=args.num_layers,
        dsa_head_dim=args.dsa_head_dim,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )


def _build_case(
    format_name: str,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> _RetrieveCase:
    row_width = args.mtp * args.topk
    output_slots = row_width
    num_blocks = recommended_num_blocks(
        max(args.num_tokens, output_slots),
        args.block_size,
    )
    source_cache = _generate_source_cache(
        format_name,
        num_blocks=num_blocks,
        device=device,
        args=args,
    )
    harness = build_load_benchmark_harness(
        fmt=format_name,
        src_kv_cache=source_cache,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
        num_selected=row_width,
        seed=123,
    )
    selected = harness.selected_token_idx.reshape(1, row_width)
    target_slots = torch.arange(
        output_slots,
        dtype=torch.long,
        device=device,
    ).reshape(1, row_width)
    element_size = torch.empty((), dtype=harness.dtype).element_size()
    useful_bytes = compute_direct_host_bytes(
        num_selected=args.valid_count,
        num_layers=args.num_layers,
        plane_elems=harness.dims.plane_elems,
        element_size=element_size,
    )
    fixed_width_bytes = compute_direct_host_bytes(
        num_selected=output_slots,
        num_layers=args.num_layers,
        plane_elems=harness.dims.plane_elems,
        element_size=element_size,
    )
    return _RetrieveCase(
        format_name=format_name,
        harness=harness,
        selected=selected,
        target_slots=target_slots,
        useful_bytes=useful_bytes,
        fixed_width_bytes=fixed_width_bytes,
    )


def _run_cases(
    cases: Sequence[_RetrieveCase],
    counts: torch.Tensor | None,
) -> None:
    for case in cases:
        harness = case.harness
        for layer_id in range(harness.num_layers):
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                harness.layer_states[layer_id],
                case.target_slots,
                case.selected,
                harness.chunk_ptrs_by_layer[layer_id],
                harness.chunk_size,
                harness.num_tokens,
                False,
                counts,
            )


def _verify_case(
    case: _RetrieveCase,
    counts: torch.Tensor,
) -> None:
    harness = case.harness
    for layer in harness.dst_direct_fast:
        for plane in layer:
            plane.zero_()
    _run_cases((case,), counts)
    torch.npu.synchronize()

    row_width = case.selected.shape[1]
    valid_mask = (
        torch.arange(row_width, device=case.selected.device).reshape(1, -1)
        < counts.reshape(-1, 1)
    ).reshape(-1)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
    padding_positions = torch.nonzero(~valid_mask, as_tuple=False).reshape(-1)
    source_slots = harness.slot_mapping_full.index_select(
        0,
        case.selected.reshape(-1).to(torch.long),
    )
    destination_slots = case.target_slots.reshape(-1)

    for source_layer, destination_layer in zip(
        harness.src_kv_cache,
        harness.dst_direct_fast,
        strict=True,
    ):
        for source_plane, destination_plane in zip(
            source_layer,
            destination_layer,
            strict=True,
        ):
            source_flat = source_plane.flatten(0, 1)
            destination_flat = destination_plane.flatten(0, 1)
            expected = source_flat.index_select(
                0,
                source_slots.index_select(0, valid_positions),
            )
            actual = destination_flat.index_select(
                0,
                destination_slots.index_select(0, valid_positions),
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if padding_positions.numel():
                padded = destination_flat.index_select(
                    0,
                    destination_slots.index_select(0, padding_positions),
                )
                assert torch.count_nonzero(padded).item() == 0


def _format_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} M/s"
    if value >= 1_000:
        return f"{value / 1_000:.3f} K/s"
    return f"{value:.3f}/s"


def _print_result(
    label: str,
    stats: TimingStats,
    *,
    payload_bytes: int,
    selected_entries: int,
    num_layer_launches: int,
) -> None:
    p95_ms = percentile(stats.samples_ms, 0.95)
    print(f"{label}:")
    print(
        f"  latency p50={stats.median_ms * 1000.0:.1f} us, "
        f"p95={p95_ms * 1000.0:.1f} us, "
        f"mean={stats.mean_ms * 1000.0:.1f} us, "
        f"min={stats.min_ms * 1000.0:.1f} us, "
        f"max={stats.max_ms * 1000.0:.1f} us"
    )
    print(
        "  speed "
        f"retrieves={_format_rate(rate_per_second(1, stats.median_ms))}, "
        "selected_entries="
        f"{_format_rate(rate_per_second(selected_entries, stats.median_ms))}"
    )
    print(
        "  effective H2D payload bandwidth "
        f"p50={bandwidth_gb_per_s(payload_bytes, stats.median_ms):.3f} GB/s, "
        f"p95={bandwidth_gb_per_s(payload_bytes, p95_ms):.3f} GB/s, "
        f"peak={bandwidth_gb_per_s(payload_bytes, stats.min_ms):.3f} GB/s"
    )
    print(
        "  average latency per group-layer launch="
        f"{stats.median_ms * 1000.0 / num_layer_launches:.1f} us"
    )


def main() -> None:
    args = _parse_args()
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    counts = torch.full(
        (1,),
        args.valid_count,
        dtype=torch.int32,
        device=device,
    )
    cases = [
        _build_case(format_name, device=device, args=args)
        for format_name in _formats_for(args.kv_format)
    ]
    try:
        if args.verify:
            for case in cases:
                _verify_case(case, counts)

        count_aware_by_format = [
            (
                case,
                time_npu_callable(
                    lambda case=case: _run_cases((case,), counts),
                    warmup=args.warmup,
                    iters=args.iters,
                ),
            )
            for case in cases
        ]
        count_aware = (
            time_npu_callable(
                lambda: _run_cases(cases, counts),
                warmup=args.warmup,
                iters=args.iters,
            )
            if len(cases) > 1
            else count_aware_by_format[0][1]
        )
        useful_bytes = sum(case.useful_bytes for case in cases)
        fixed_width_bytes = sum(case.fixed_width_bytes for case in cases)
        useful_entries = args.valid_count
        num_layer_launches = args.num_layers * len(cases)
        row_width = args.mtp * args.topk

        print("LMCache prepared sparse retrieve benchmark")
        print(
            f"  formats={','.join(case.format_name for case in cases)}, "
            f"requests=1, mtp={args.mtp}, topk={args.topk}, "
            f"packed_row_width={row_width}, valid_count={args.valid_count}"
        )
        print(
            f"  source_tokens={args.num_tokens}, layers={args.num_layers}, "
            f"chunk_size={args.chunk_size}, dtype=bfloat16"
        )
        print(
            f"  useful payload={format_bytes_short(useful_bytes)} "
            f"({useful_entries} selected entries across all rows)"
        )
        for case in cases:
            print(
                f"    {case.format_name}: "
                f"{format_bytes_short(case.useful_bytes)} useful payload/call"
            )
        for case, stats in count_aware_by_format:
            _print_result(
                f"count-aware {case.format_name}",
                stats,
                payload_bytes=case.useful_bytes,
                selected_entries=useful_entries,
                num_layer_launches=args.num_layers,
            )
        if len(cases) > 1:
            _print_result(
                "count-aware two-group total",
                count_aware,
                payload_bytes=useful_bytes,
                selected_entries=useful_entries,
                num_layer_launches=num_layer_launches,
            )

        if args.compare_fixed_width:
            fixed_width = time_npu_callable(
                lambda: _run_cases(cases, None),
                warmup=args.warmup,
                iters=args.iters,
            )
            fixed_entries = row_width
            _print_result(
                "fixed-width control",
                fixed_width,
                payload_bytes=fixed_width_bytes,
                selected_entries=fixed_entries,
                num_layer_launches=num_layer_launches,
            )
            ratio = (
                fixed_width.median_ms / count_aware.median_ms
                if count_aware.median_ms > 0
                else 0.0
            )
            avoided = fixed_width_bytes - useful_bytes
            print(
                "comparison: "
                f"fixed/count-aware latency={ratio:.3f}x, "
                f"padding payload avoided={format_bytes_short(avoided)}"
            )
    finally:
        for case in cases:
            case.harness.close()


if __name__ == "__main__":
    main()
