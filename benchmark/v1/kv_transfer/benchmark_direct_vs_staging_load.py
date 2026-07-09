#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Micro-benchmark: direct KV load vs dense staging load (MLA/DSA).

Paths timed per layer:
  dense direct cases use dense_mla_dsa_batched_direct_kv_transfer;
  sparse direct cases use sparse_mla_dsa_batched_direct_kv_transfer.
  staging: CPU stacked -> NPU staging -> paged KV (always all tokens)
  direct: CPU pinned GM -> paged KV (selected tokens only)
  direct_fast: same kernel with cached per-layer state

Examples:
  python benchmark/v1/kv_transfer/benchmark_direct_vs_staging_load.py --verify

  python benchmark/v1/kv_transfer/benchmark_direct_vs_staging_load.py \\
      --format mla --num-selected 2048 --verify

  python benchmark/v1/kv_transfer/benchmark_direct_vs_staging_load.py \\
      --format mla --num-tokens 65536 --num-layers 32 --num-blocks 8192 --verify

  python benchmark/v1/kv_transfer/benchmark_direct_vs_staging_load.py \\
      --sweep --format mla --sparse-token-counts 2048,65536
"""
from __future__ import annotations

# Standard
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

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
    check_paged_kv_cache_equal,
    generate_dsa_index_kv_cache,
    generate_dsa_kv_cache,
    generate_mla_kv_cache,
)
from load_benchmark_utils import (  # noqa: E402
    KV_FORMAT_BY_NAME,
    LoadBenchmarkCase,
    LoadBenchmarkHarness,
    benchmark_load_harness,
    benchmark_store_harness,
    build_load_benchmark_harness,
    format_bytes_short,
    kv_format_from_name,
    recommended_num_blocks,
    verify_load_benchmark_harness,
    verify_store_benchmark_harness,
)


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--format",
        choices=tuple(KV_FORMAT_BY_NAME),
        default="mla",
        help="KV format to benchmark: mla/dsa or split two-group formats.",
    )
    p.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Prompt tokens in the benchmark case (default: 4096).",
    )
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Model layers exercised (default: 4).",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=512,
        help="Paged KV blocks per layer (default: 512; increase for large --num-tokens).",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope-head-dim", type=int, default=128)
    p.add_argument("--dsa-head-dim", type=int, default=128)
    p.add_argument(
        "--num-selected",
        type=int,
        default=None,
        help="Tokens for direct path (default: all tokens).",
    )
    p.add_argument(
        "--sparse-token-counts",
        type=str,
        default="",
        help="Extra num_selected values when sweeping (comma-separated).",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--direction",
        choices=("load", "store"),
        default="load",
        help="Benchmark dense load or dense store/offload path.",
    )
    p.add_argument("--verify", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-num-tokens", type=str, default="4096,16384")
    p.add_argument("--sweep-num-layers", type=str, default="1,4")
    p.add_argument("--output-csv", type=str, default="")
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def _generate_src_kv_cache(args: argparse.Namespace, num_layers: int, num_blocks: int):
    common = dict(
        num_blocks=num_blocks,
        device="npu",
        num_layers=num_layers,
        num_kv_heads=args.num_kv_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )
    if args.format in ("mla", "mla_latent"):
        return generate_mla_kv_cache(**common)
    if args.format == "dsa":
        return generate_dsa_kv_cache(dsa_head_dim=args.dsa_head_dim, **common)
    return generate_dsa_index_kv_cache(
        num_blocks=num_blocks,
        device="npu",
        num_layers=num_layers,
        dsa_head_dim=args.dsa_head_dim,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )


def _check_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        num_heads=args.num_kv_heads,
        head_size=128,
        kv_format=kv_format_from_name(args.format),
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        dsa_head_dim=args.dsa_head_dim,
    )


def _run_case(
    args: argparse.Namespace,
    *,
    num_tokens: int,
    num_layers: int,
    num_selected: int,
) -> LoadBenchmarkCase:
    if args.direction == "store" and num_selected != num_tokens:
        raise ValueError("store benchmark only supports dense all-token transfer")
    required_blocks = recommended_num_blocks(num_tokens, args.block_size)
    if args.num_blocks < required_blocks:
        raise ValueError(
            f"num_blocks={args.num_blocks} too small for {num_tokens} tokens "
            f"(need >= {required_blocks - 8})"
        )
    num_blocks = min(args.num_blocks, required_blocks)
    src_kv_cache = _generate_src_kv_cache(args, num_layers, num_blocks)

    harness = build_load_benchmark_harness(
        fmt=args.format,
        src_kv_cache=src_kv_cache,
        num_tokens=num_tokens,
        chunk_size=args.chunk_size,
        num_layers=num_layers,
        num_selected=num_selected,
        seed=args.seed,
    )
    try:
        if args.verify:
            if args.direction == "store":
                verify_store_benchmark_harness(harness)
            else:
                verify_load_benchmark_harness(
                    harness, check_paged_kv_cache_equal, **_check_kwargs(args)
                )
        if args.direction == "store":
            return benchmark_store_harness(
                harness, warmup=args.warmup, iters=args.iters
            )
        return benchmark_load_harness(harness, warmup=args.warmup, iters=args.iters)
    finally:
        harness.close()


def _print_case(case: LoadBenchmarkCase) -> None:
    layers = case.num_layers
    faster = (
        1.0 / case.direct_vs_staging_ratio
        if case.direct_vs_staging_ratio > 0
        else 0.0
    )
    print(
        f"\ndirection={case.direction} format={case.format_name} "
        f"tokens={case.num_tokens} "
        f"selected={case.num_selected} layers={layers} "
        f"chunks={case.num_chunks} chunk_size={case.chunk_size}"
    )
    if case.direction == "load" and case.num_selected != case.num_tokens:
        print("  note: staging loads all tokens; direct loads selected subset only")
    print(
        f"  staging:      {case.staging_median_ms:8.3f} ms "
        f"({case.staging_median_ms / layers:.3f} ms/layer)"
    )
    print(
        f"  direct:       {case.direct_median_ms:8.3f} ms "
        f"({case.direct_median_ms / layers:.3f} ms/layer, "
        f"{faster:.2f}x faster than staging)"
    )
    print(
        f"  direct_fast:  {case.direct_fast_median_ms:8.3f} ms "
        f"({case.direct_fast_median_ms / layers:.3f} ms/layer)"
    )
    print(
        f"  direct bytes: {format_bytes_short(case.host_bytes_moved)}  "
        f"bandwidth: {case.direct_bandwidth_gbps:.1f} GB/s "
        f"(peak {case.direct_bandwidth_peak_gbps:.1f} GB/s)"
    )


def _iter_selected(args: argparse.Namespace, num_tokens: int) -> Iterable[int]:
    counts = {num_tokens}
    if args.num_selected is not None:
        counts.add(args.num_selected)
    for value in _parse_int_list(args.sparse_token_counts):
        if 0 < value <= num_tokens:
            counts.add(value)
    return sorted(counts)


def _write_csv(path: Path, cases: Sequence[LoadBenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "format",
                "direction",
                "num_tokens",
                "num_selected",
                "num_layers",
                "num_chunks",
                "chunk_size",
                "host_bytes_moved",
                "staging_median_ms",
                "direct_median_ms",
                "direct_min_ms",
                "direct_fast_median_ms",
                "direct_bandwidth_gbps",
                "direct_bandwidth_peak_gbps",
                "direct_vs_staging_ratio",
            ]
        )
        for c in cases:
            w.writerow(
                [
                    c.format_name,
                    c.direction,
                    c.num_tokens,
                    c.num_selected,
                    c.num_layers,
                    c.num_chunks,
                    c.chunk_size,
                    c.host_bytes_moved,
                    f"{c.staging_median_ms:.6f}",
                    f"{c.direct_median_ms:.6f}",
                    f"{c.direct_min_ms:.6f}",
                    f"{c.direct_fast_median_ms:.6f}",
                    f"{c.direct_bandwidth_gbps:.6f}",
                    f"{c.direct_bandwidth_peak_gbps:.6f}",
                    f"{c.direct_vs_staging_ratio:.6f}",
                ]
            )


def _write_json(path: Path, cases: Sequence[LoadBenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "format": c.format_name,
            "direction": c.direction,
            "num_tokens": c.num_tokens,
            "num_selected": c.num_selected,
            "num_layers": c.num_layers,
            "num_chunks": c.num_chunks,
            "chunk_size": c.chunk_size,
            "host_bytes_moved": c.host_bytes_moved,
            "staging_median_ms": c.staging_median_ms,
            "direct_median_ms": c.direct_median_ms,
            "direct_min_ms": c.direct_min_ms,
            "direct_fast_median_ms": c.direct_fast_median_ms,
            "direct_bandwidth_gbps": c.direct_bandwidth_gbps,
            "direct_bandwidth_peak_gbps": c.direct_bandwidth_peak_gbps,
            "direct_vs_staging_ratio": c.direct_vs_staging_ratio,
        }
        for c in cases
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    token_counts = (
        _parse_int_list(args.sweep_num_tokens) if args.sweep else [args.num_tokens]
    )
    layer_counts = (
        _parse_int_list(args.sweep_num_layers) if args.sweep else [args.num_layers]
    )

    cases: List[LoadBenchmarkCase] = []
    for num_tokens in token_counts:
        for num_layers in layer_counts:
            selected_values = (
                [num_tokens]
                if args.direction == "store"
                else _iter_selected(args, num_tokens)
            )
            for num_selected in selected_values:
                case = _run_case(
                    args,
                    num_tokens=num_tokens,
                    num_layers=num_layers,
                    num_selected=num_selected,
                )
                cases.append(case)
                _print_case(case)

    if args.output_csv:
        _write_csv(Path(args.output_csv), cases)
        print(f"\nWrote CSV: {args.output_csv}")
    if args.output_json:
        _write_json(Path(args.output_json), cases)
        print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
