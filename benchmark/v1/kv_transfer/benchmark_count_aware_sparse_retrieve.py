#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the production count-aware prepared sparse retrieve path.

The fixed-width case copies the complete row without selected-token counts.
The count-aware case uses the same row buffers but copies only their valid
prefixes.  This explicitly covers the path used by staged MTP union payloads.

Example:
  python benchmark/v1/kv_transfer/benchmark_count_aware_sparse_retrieve.py \
      --row-width 4096 --valid-count 2048 --num-layers 8 --verify
"""

from __future__ import annotations

# Standard
import argparse
import sys
from pathlib import Path

# Third Party
import torch

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from kv_cache_fixtures import generate_dsa_kv_cache  # noqa: E402
from load_benchmark_utils import (  # noqa: E402
    build_load_benchmark_harness,
    compute_direct_host_bytes,
    format_bytes_short,
    recommended_num_blocks,
    time_npu_callable,
)
from lmcache_ascend.v1.npu_connector.utils import (  # noqa: E402
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=18880)
    parser.add_argument("--row-width", type=int, default=4096)
    parser.add_argument("--valid-count", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--dsa-head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    if not 0 < args.valid_count <= args.row_width <= args.num_tokens:
        parser.error("require 0 < --valid-count <= --row-width <= --num-tokens")
    if args.num_layers <= 0 or args.chunk_size <= 0:
        parser.error("--num-layers and --chunk-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters positive")
    return args


def _run_layers(harness, selected, target_slots, counts) -> None:
    for layer_id in range(harness.num_layers):
        sparse_mla_dsa_batched_direct_kv_transfer_prepared(
            harness.layer_states[layer_id],
            target_slots,
            selected,
            harness.chunk_ptrs_by_layer[layer_id],
            harness.chunk_size,
            harness.num_tokens,
            False,
            counts,
        )


def _verify_count_aware(
    harness,
    selected: torch.Tensor,
    target_slots: torch.Tensor,
    counts: torch.Tensor,
    valid_count: int,
) -> None:
    for layer in harness.dst_direct_fast:
        for plane in layer:
            plane.zero_()
    _run_layers(harness, selected, target_slots, counts)
    torch.npu.synchronize()

    valid = torch.arange(
        valid_count,
        dtype=torch.long,
        device=selected.device,
    )
    padding = torch.arange(
        valid_count,
        selected.numel(),
        dtype=torch.long,
        device=selected.device,
    )
    source_slots = harness.slot_mapping_full.index_select(
        0,
        selected.reshape(-1).to(torch.long),
    )
    destination_slots = target_slots.reshape(-1)
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
                source_slots.index_select(0, valid),
            )
            actual = destination_flat.index_select(
                0,
                destination_slots.index_select(0, valid),
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if padding.numel():
                padded = destination_flat.index_select(
                    0,
                    destination_slots.index_select(0, padding),
                )
                assert torch.count_nonzero(padded).item() == 0


def main() -> None:
    args = _parse_args()
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    source_cache = generate_dsa_kv_cache(
        num_blocks=recommended_num_blocks(
            args.num_tokens,
            args.block_size,
        ),
        device=device,
        num_layers=args.num_layers,
        num_kv_heads=1,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        dsa_head_dim=args.dsa_head_dim,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )
    harness = build_load_benchmark_harness(
        fmt="dsa",
        src_kv_cache=source_cache,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
        num_selected=args.row_width,
        seed=123,
    )
    try:
        selected = harness.selected_token_idx.reshape(1, args.row_width)
        target_slots = torch.arange(
            args.row_width,
            dtype=torch.long,
            device=device,
        ).reshape(1, args.row_width)
        counts = torch.tensor(
            [args.valid_count],
            dtype=torch.int32,
            device=device,
        )
        if args.verify:
            _verify_count_aware(
                harness,
                selected,
                target_slots,
                counts,
                args.valid_count,
            )

        fixed_width = time_npu_callable(
            lambda: _run_layers(
                harness,
                selected,
                target_slots,
                None,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        count_aware = time_npu_callable(
            lambda: _run_layers(
                harness,
                selected,
                target_slots,
                counts,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        valid_bytes = compute_direct_host_bytes(
            num_selected=args.valid_count,
            num_layers=args.num_layers,
            plane_elems=harness.dims.plane_elems,
            element_size=torch.empty(
                (),
                dtype=harness.dtype,
            ).element_size(),
        )
        print(
            "count-aware sparse retrieve: "
            f"tokens={args.num_tokens}, row_width={args.row_width}, "
            f"valid_count={args.valid_count}, layers={args.num_layers}, "
            f"valid_bytes={format_bytes_short(valid_bytes)}"
        )
        print(
            f"fixed-width median={fixed_width.median_ms:.4f} ms, "
            f"count-aware median={count_aware.median_ms:.4f} ms, "
            "fixed/count-aware="
            f"{fixed_width.median_ms / count_aware.median_ms:.3f}x"
        )
    finally:
        harness.close()


if __name__ == "__main__":
    main()
