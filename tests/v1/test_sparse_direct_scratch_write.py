# SPDX-License-Identifier: Apache-2.0
"""Sparse direct retrieve must write selected tokens into compact scratch slots."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not _npu_available(),
    reason="sparse direct scratch write test requires NPU",
)


def _add_benchmark_helpers_to_path() -> None:
    benchmark_dir = (
        Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
    )
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))


def _new_sentinel_layer_like(layer: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    sentinel = -7.0
    return tuple(torch.full_like(t, sentinel) for t in layer)


def _changed_slots(flat: torch.Tensor, sentinel: float) -> list[int]:
    changed = (flat != sentinel).any(dim=1).nonzero().flatten()
    return sorted(changed.detach().cpu().tolist())


def _target_changed_slots(
    flat: torch.Tensor,
    slot_mapping_packed: torch.Tensor,
    sentinel: float,
) -> list[int]:
    target = flat.index_select(0, slot_mapping_packed)
    changed = (target != sentinel).any(dim=1).nonzero().flatten()
    if changed.numel() == 0:
        return []
    target_slots = slot_mapping_packed.index_select(0, changed)
    return sorted(target_slots.detach().cpu().tolist())


def _matching_slots_for_expected_rows(
    flat: torch.Tensor,
    expected: torch.Tensor,
    *,
    limit_rows: int = 4,
    limit_matches: int = 8,
) -> dict[int, list[int]]:
    matches: dict[int, list[int]] = {}
    for row in range(min(limit_rows, int(expected.shape[0]))):
        hit = (flat == expected[row]).all(dim=1).nonzero().flatten()
        if hit.numel() > 0:
            matches[row] = hit[:limit_matches].detach().cpu().tolist()
    return matches


def _assert_direct_write_result(
    *,
    src_layer: tuple[torch.Tensor, ...],
    dst_layer: tuple[torch.Tensor, ...],
    slot_mapping_full: torch.Tensor,
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    sentinel: float,
    label: str,
    planes_to_check: int | None = None,
) -> None:
    page_tokens = int(src_layer[0].shape[0] * src_layer[0].shape[1])
    src_slots = slot_mapping_full.index_select(0, selected_token_idx.to(torch.long))
    expected_changed = sorted(slot_mapping_packed.detach().cpu().tolist())

    planes = list(zip(src_layer, dst_layer, strict=True))
    if planes_to_check is not None:
        planes = planes[:planes_to_check]

    for plane_id, (src, dst) in enumerate(planes):
        src_flat = src.reshape(page_tokens, -1)
        dst_flat = dst.reshape(page_tokens, -1)
        expected = src_flat.index_select(0, src_slots)
        actual = dst_flat.index_select(0, slot_mapping_packed)
        if not torch.equal(actual, expected):
            diff = (actual.float() - expected.float()).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            changed_cpu = _changed_slots(dst_flat, sentinel)
            target_changed = _target_changed_slots(
                dst_flat, slot_mapping_packed, sentinel
            )
            expected_matches = _matching_slots_for_expected_rows(dst_flat, expected)
            if not changed_cpu:
                location_hint = "no destination slot changed"
            elif not target_changed:
                location_hint = "kernel wrote outside compact scratch slots"
            else:
                location_hint = "kernel wrote compact scratch slots with wrong data"
            pytest.fail(
                f"{label}: plane {plane_id} did not write selected KV to "
                f"scratch slots; max_abs_diff={max_diff}; "
                f"{location_hint}; target_changed={target_changed[:32]}; "
                f"all_changed={changed_cpu[:32]}; "
                f"expected_row_matches={expected_matches}"
            )

        changed_cpu = _changed_slots(dst_flat, sentinel)
        assert changed_cpu == expected_changed, (
            f"{label}: plane {plane_id} wrote unexpected destination slots; "
            f"changed={changed_cpu[:32]} expected={expected_changed[:32]}"
        )


def _assert_full_staging_roundtrip(
    *,
    src_layer: tuple[torch.Tensor, ...],
    dst_layer: tuple[torch.Tensor, ...],
    slot_mapping_full: torch.Tensor,
    label: str,
    planes_to_check: int | None = None,
) -> None:
    page_tokens = int(src_layer[0].shape[0] * src_layer[0].shape[1])
    planes = list(zip(src_layer, dst_layer, strict=True))
    if planes_to_check is not None:
        planes = planes[:planes_to_check]

    for plane_id, (src, dst) in enumerate(planes):
        src_flat = src.reshape(page_tokens, -1)
        dst_flat = dst.reshape(page_tokens, -1)
        expected = src_flat.index_select(0, slot_mapping_full)
        actual = dst_flat.index_select(0, slot_mapping_full)
        if not torch.equal(actual, expected):
            diff = (actual.float() - expected.float()).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            pytest.fail(
                f"{label}: plane {plane_id} dense staging roundtrip failed; "
                f"max_abs_diff={max_diff}. This points before sparse direct "
                "retrieve: CPU chunk population or dense staging load is wrong."
            )


@pytest.mark.parametrize("fmt", ["mla", "dsa"])
@pytest.mark.parametrize(
    "direct_path",
    ["direct_host_ptrs", "direct_cached_ptrs", "fast_cached_state"],
)
def test_sparse_direct_retrieve_writes_original_tokens_to_compact_scratch_slots(
    fmt: str,
    direct_path: str,
) -> None:
    _add_benchmark_helpers_to_path()
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from kv_cache_fixtures import generate_dsa_kv_cache, generate_mla_kv_cache
    from load_benchmark_utils import (
        KV_FORMAT_DSA,
        KV_FORMAT_MLA,
        allocate_stacked_cpu_chunks,
        as_vllm_kv_caches,
        batched_fused_single_layer_kv_transfer,
        build_chunk_ptrs_npu,
        compute_chunk_partition,
        create_pin_memory_allocator,
        lmc_host_interleaved_for_kv_format,
        mla_dsa_dims_from_layer,
        populate_stacked_cpu_from_src,
        prepare_sparse_direct_layer_state,
        release_pin_memory_objects,
        sparse_mla_dsa_batched_direct_kv_transfer,
        sparse_mla_dsa_batched_direct_kv_transfer_fast,
    )

    device = "npu"
    dtype = torch.bfloat16
    num_blocks = 128
    block_size = 16
    page_tokens = num_blocks * block_size
    num_tokens = 1024
    chunk_size = 256
    num_layers = 1
    num_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 128
    dsa_head_dim = 128
    sentinel = -7.0
    # Sparse attention consumes latent K/PE, i.e. the first two planes. The DSA
    # indexer plane is not part of the LMCache latent scratch path under test.
    planes_to_check = 2

    kv_format = KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA
    common = dict(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    src = (
        generate_mla_kv_cache(**common)
        if fmt == "mla"
        else generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
    )
    src_layer = src[0]
    dst_layer = _new_sentinel_layer_like(src_layer)
    staging_dst_layer = _new_sentinel_layer_like(src_layer)

    source_base = 512
    assert source_base + num_tokens <= page_tokens
    slot_mapping_full = torch.arange(
        source_base,
        source_base + num_tokens,
        dtype=torch.long,
        device=device,
    )
    selected_token_idx = torch.tensor(
        [7, 255, 256, 400, 511, 512, 777, 1023],
        dtype=torch.int32,
        device=device,
    )
    slot_mapping_packed = torch.arange(
        selected_token_idx.numel(),
        dtype=torch.long,
        device=device,
    )

    dims = mla_dsa_dims_from_layer(src_layer, kv_format)
    partition = compute_chunk_partition(num_tokens, chunk_size)
    staging_cache = torch.empty(
        num_tokens * dims.plane_elems,
        dtype=dtype,
        device=device,
    )
    allocator = create_pin_memory_allocator(partition, dims, dtype)
    mem_objs = []
    try:
        stacked_cpu_chunks, mem_objs = allocate_stacked_cpu_chunks(
            allocator, partition, dims, dtype
        )
        populate_stacked_cpu_from_src(
            src_layer=src_layer,
            stacked_cpu_chunks=stacked_cpu_chunks,
            staging_cache=staging_cache,
            slot_mapping_full=slot_mapping_full,
            partition=partition,
            dims=dims,
            kv_format=kv_format,
        )
        chunk_ptrs_npu = build_chunk_ptrs_npu(stacked_cpu_chunks, torch.device(device))
        batched_fused_single_layer_kv_transfer(
            stacked_cpu_chunks,
            staging_cache,
            as_vllm_kv_caches(staging_dst_layer),
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
        torch.npu.synchronize()
        _assert_full_staging_roundtrip(
            src_layer=src_layer,
            dst_layer=staging_dst_layer,
            slot_mapping_full=slot_mapping_full,
            label=(
                f"{fmt}/batched_fused_single_layer_kv_transfer"
                "(populate_cpu_chunks+dense_load)"
            ),
            planes_to_check=planes_to_check,
        )

        if direct_path == "fast_cached_state":
            layer_state = prepare_sparse_direct_layer_state(
                stacked_cpu_chunks[0],
                as_vllm_kv_caches(dst_layer),
                slot_mapping_full,
                False,
                False,
                kv_format,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
                num_tokens,
            )
            sparse_mla_dsa_batched_direct_kv_transfer_fast(
                layer_state,
                slot_mapping_packed,
                selected_token_idx,
                chunk_ptrs_npu,
                chunk_size,
                num_tokens,
                lmc_host_interleaved_for_kv_format(kv_format),
                True,
            )
        else:
            sparse_mla_dsa_batched_direct_kv_transfer(
                stacked_cpu_chunks,
                as_vllm_kv_caches(dst_layer),
                slot_mapping_packed,
                selected_token_idx,
                chunk_size,
                num_tokens,
                kv_format,
                False,
                False,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
                lmc_host_interleaved_for_kv_format(kv_format),
                None if direct_path == "direct_host_ptrs" else chunk_ptrs_npu,
            )

        torch.npu.synchronize()
        kernel_label = (
            "sparse_mla_dsa_batched_direct_kv_transfer_fast"
            if direct_path == "fast_cached_state"
            else "sparse_mla_dsa_batched_direct_kv_transfer"
        )
        _assert_direct_write_result(
            src_layer=src_layer,
            dst_layer=dst_layer,
            slot_mapping_full=slot_mapping_full,
            slot_mapping_packed=slot_mapping_packed,
            selected_token_idx=selected_token_idx,
            sentinel=sentinel,
            label=f"{fmt}/{direct_path}: {kernel_label}",
            planes_to_check=planes_to_check,
        )
    finally:
        release_pin_memory_objects(mem_objs)
        allocator.close()
