# SPDX-License-Identifier: Apache-2.0
"""Store/retrieve roundtrip via sparse_mla_dsa_batched_direct_kv_transfer."""
from __future__ import annotations

# Standard
import random
import sys
import traceback
from pathlib import Path

# Third Party
import pytest
import torch

_BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
)
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from kv_cache_fixtures import (  # noqa: E402
    generate_dsa_kv_cache,
    generate_mla_kv_cache,
)
from load_benchmark_utils import (  # noqa: E402
    KV_FORMAT_DSA,
    KV_FORMAT_DSA_INDEX,
    KV_FORMAT_MLA,
    KV_FORMAT_MLA_LATENT,
    MlaDsaDims,
    allocate_stacked_cpu_chunks,
    build_load_benchmark_harness,
    build_chunk_ptrs_npu,
    compute_chunk_partition,
    create_pin_memory_allocator,
    release_pin_memory_objects,
    run_all_direct_fast_layers,
    run_all_direct_layers,
    run_direct_load_layer,
)
from lmcache_ascend.v1.npu_connector.utils import (  # noqa: E402
    batched_fused_sparse_single_layer_kv_transfer,
    prepare_sparse_direct_destination_state,
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
)


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _sample_packed_indices(num_packed: int, max_samples: int = 32) -> list[int]:
    if num_packed <= max_samples:
        return list(range(num_packed))
    step = max(1, num_packed // max_samples)
    return list(range(0, num_packed, step))[:max_samples]


def _sample_layer_ids(num_layers: int, max_samples: int = 4) -> list[int]:
    if num_layers <= max_samples:
        return list(range(num_layers))
    step = max(1, num_layers // max_samples)
    return list(range(0, num_layers, step))[:max_samples]


def _tail_heavy_selected_tokens(total_tokens: int, num_selected: int) -> list[int]:
    """Unsorted selected ids shaped like indexer top-k output near prompt tail."""
    seed_values = [
        total_tokens - 48,
        total_tokens - 65,
        total_tokens - 69,
        total_tokens - 228,
        total_tokens - 240,
        total_tokens - 424,
        total_tokens - 237,
        total_tokens - 434,
        3,
        257,
        511,
        4096,
        total_tokens - 1,
    ]
    selected: list[int] = []
    seen: set[int] = set()
    for value in seed_values:
        if 0 <= value < total_tokens and value not in seen:
            selected.append(value)
            seen.add(value)

    rng = random.Random(1234)
    while len(selected) < num_selected:
        value = rng.randrange(total_tokens)
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return selected


def _fill_stacked_chunks_with_token_pattern(
    chunks: list[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> None:
    for chunk_id, chunk in enumerate(chunks):
        values = (
            torch.arange(chunk.numel(), dtype=torch.float32)
            .remainder(97)
            .div(16.0)
            .add(float(chunk_id * 8))
        )
        chunk.copy_(values.to(dtype=dtype))


def _check_compact_scratch_from_cpu_chunks(
    *,
    chunks: list[torch.Tensor],
    selected: list[int],
    dst_slots: list[int],
    dst_layer: tuple[torch.Tensor, torch.Tensor],
    chunk_size: int,
    dims: MlaDsaDims,
    label: str,
) -> None:
    dst_k = dst_layer[0].reshape(-1, dims.k_hidden_dims).detach().cpu()
    dst_v = dst_layer[1].reshape(-1, dims.v_hidden_dims).detach().cpu()
    mismatches: list[str] = []
    for packed_idx, token_idx in enumerate(selected):
        chunk_id = token_idx // chunk_size
        local_idx = token_idx % chunk_size
        chunk = chunks[chunk_id].detach().cpu()
        chunk_tokens = chunk.numel() // dims.plane_elems
        k_start = local_idx * dims.k_hidden_dims
        k_end = k_start + dims.k_hidden_dims
        v_start = chunk_tokens * dims.k_hidden_dims + local_idx * dims.v_hidden_dims
        v_end = v_start + dims.v_hidden_dims
        dst_slot = int(dst_slots[packed_idx])
        expected_k = chunk[k_start:k_end]
        expected_v = chunk[v_start:v_end]
        if not torch.equal(dst_k[dst_slot], expected_k):
            mismatches.append(
                f"{label} packed{packed_idx} token{token_idx} "
                f"chunk{chunk_id} local{local_idx} dst{dst_slot} K mismatch "
                f"got_head={dst_k[dst_slot, :4].tolist()} "
                f"expected_head={expected_k[:4].tolist()}"
            )
        if not torch.equal(dst_v[dst_slot], expected_v):
            mismatches.append(
                f"{label} packed{packed_idx} token{token_idx} "
                f"chunk{chunk_id} local{local_idx} dst{dst_slot} V mismatch "
                f"got_head={dst_v[dst_slot, :4].tolist()} "
                f"expected_head={expected_v[:4].tolist()}"
            )
        if len(mismatches) >= 8:
            break
    assert not mismatches, (
        f"compact scratch direct sparse load mismatch: {mismatches}"
    )


def _check_compact_index_from_cpu_chunks(
    *,
    chunks: list[torch.Tensor],
    selected: list[int],
    dst_slots: list[int],
    dst_layer: tuple[torch.Tensor],
    chunk_size: int,
    dims: MlaDsaDims,
    label: str,
) -> None:
    dst_index = dst_layer[0].reshape(-1, dims.dsa_hidden_dims).detach().cpu()
    mismatches: list[str] = []
    for packed_idx, token_idx in enumerate(selected):
        chunk_id = token_idx // chunk_size
        local_idx = token_idx % chunk_size
        chunk = chunks[chunk_id].detach().cpu()
        start = local_idx * dims.dsa_hidden_dims
        end = start + dims.dsa_hidden_dims
        dst_slot = int(dst_slots[packed_idx])
        expected = chunk[start:end]
        if not torch.equal(dst_index[dst_slot], expected):
            mismatches.append(
                f"{label} packed{packed_idx} token{token_idx} "
                f"chunk{chunk_id} local{local_idx} dst{dst_slot} "
                f"DSA_INDEX mismatch got_head="
                f"{dst_index[dst_slot, :4].tolist()} "
                f"expected_head={expected[:4].tolist()}"
            )
        if len(mismatches) >= 8:
            break
    assert not mismatches, (
        f"compact scratch DSA_INDEX sparse load mismatch: {mismatches}"
    )


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
class TestSparseMlaDsaStoreRetrieveRoundtrip:
    """sparse_mla_dsa_batched_direct_kv_transfer store/retrieve roundtrip.

    Comparison mirrors kv_cache_fixtures.check_paged_kv_cache_equal but is
    inlined and done on CPU so that:
      1. MLA/DSA layout is reshaped correctly to [-1, num_heads, kv_lora_rank]
         (NOT [-1, kv_lora_rank] which would read a single head's row).
      2. pytest never reprs an NPU tensor (segfaults on Ascend) -- the only
         frame with NPU tensors is the test method body, whose args are
         (self, fmt, num_selected): plain str/int.
    """

    @pytest.mark.parametrize("fmt", ["mla", "dsa"])
    @pytest.mark.parametrize("num_selected", [256, 2048])
    def test_sparse_direct_matches_src_checksum(
        self, fmt: str, num_selected: int
    ) -> None:
        num_tokens = 1024 if num_selected <= 1024 else 4096
        num_layers = 2
        num_blocks = 512
        block_size = 16
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        kv_format = KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA
        common = dict(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            src = (
                generate_mla_kv_cache(**common)
                if fmt == "mla"
                else generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
            )
            harness = build_load_benchmark_harness(
                fmt=fmt,
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=num_layers,
                num_selected=num_selected,
                seed=42,
            )
            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            slots_npu = harness.slot_mapping_packed
            num_packed = int(slots_npu.numel())
            slots = slots_npu.detach().cpu().tolist()
            packed_idx_to_check = _sample_packed_indices(num_packed)
            layer_ids = _sample_layer_ids(harness.num_layers)

            mismatches: list[str] = []
            for layer_id in layer_ids:
                src_layer = harness.src_kv_cache[layer_id]
                dst_dir = harness.dst_direct[layer_id]
                dst_fast = harness.dst_direct_fast[layer_id]

                src_k = src_layer[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_dir_k = dst_dir[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_fast_k = dst_fast[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                src_v = src_layer[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_dir_v = dst_dir[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_fast_v = dst_fast[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()

                src_dsa = dst_dir_dsa = dst_fast_dsa = None
                if kv_format == KV_FORMAT_DSA:
                    src_dsa = src_layer[2].reshape(-1, dsa_head_dim).detach().cpu()
                    dst_dir_dsa = dst_dir[2].reshape(-1, dsa_head_dim).detach().cpu()
                    dst_fast_dsa = dst_fast[2].reshape(-1, dsa_head_dim).detach().cpu()

                for pi in packed_idx_to_check:
                    slot = int(slots[pi])
                    if not torch.equal(src_k[slot], dst_dir_k[slot]):
                        mismatches.append(
                            f"direct L{layer_id}P{pi}slot{slot} K mismatch"
                        )
                    if not torch.equal(src_v[slot], dst_dir_v[slot]):
                        mismatches.append(
                            f"direct L{layer_id}P{pi}slot{slot} V mismatch"
                        )
                    if not torch.equal(src_k[slot], dst_fast_k[slot]):
                        mismatches.append(
                            f"fast L{layer_id}P{pi}slot{slot} K mismatch"
                        )
                    if not torch.equal(src_v[slot], dst_fast_v[slot]):
                        mismatches.append(
                            f"fast L{layer_id}P{pi}slot{slot} V mismatch"
                        )
                    if src_dsa is not None:
                        if not torch.equal(src_dsa[slot], dst_dir_dsa[slot]):
                            mismatches.append(
                                f"direct L{layer_id}P{pi}slot{slot} DSA mismatch"
                            )
                        if not torch.equal(src_dsa[slot], dst_fast_dsa[slot]):
                            mismatches.append(
                                f"fast L{layer_id}P{pi}slot{slot} DSA mismatch"
                            )
            assert not mismatches, (
                f"sparse scatter roundtrip mismatch fmt={fmt} "
                f"num_selected={num_selected} (showing first 8 of "
                f"{len(mismatches)}): {mismatches[:8]}"
            )
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed fmt={fmt} num_selected={num_selected}: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass

    def test_sparse_direct_mla_compact_scratch_reads_selected_chunks(self) -> None:
        """Direct multi-chunk MLA sparse load must write selected tokens tightly.

        This mirrors the vLLM sparse decode path:
        selected_token_idx contains absolute prompt positions in top-k order,
        while slot_mapping_packed is the compact scratch window's physical
        slot list in the same order.
        """
        total_tokens = 18879
        chunk_size = 256
        num_selected = 2048
        block_size = 128
        dtype = torch.bfloat16
        device = "npu"
        dims = MlaDsaDims(
            k_hidden_dims=512,
            v_hidden_dims=64,
            dsa_hidden_dims=0,
            plane_elems=576,
        )
        partition = compute_chunk_partition(total_tokens, chunk_size)
        mem_allocator = None
        mem_objs = []
        try:
            mem_allocator = create_pin_memory_allocator(partition, dims, dtype)
            chunks, mem_objs = allocate_stacked_cpu_chunks(
                mem_allocator, partition, dims, dtype
            )
            _fill_stacked_chunks_with_token_pattern(chunks, dtype=dtype)

            selected = _tail_heavy_selected_tokens(total_tokens, num_selected)
            selected_npu = torch.tensor(
                selected, dtype=torch.int32, device=device
            )
            scratch_base_block = 5
            scratch_base_slot = scratch_base_block * block_size
            scratch_slots_cpu = list(
                range(scratch_base_slot, scratch_base_slot + num_selected)
            )
            scratch_slots = torch.tensor(
                scratch_slots_cpu, dtype=torch.long, device=device
            )
            num_scratch_blocks = scratch_base_block + (
                (num_selected + block_size - 1) // block_size
            )
            dst_direct = (
                torch.empty(
                    (num_scratch_blocks, block_size, 1, dims.k_hidden_dims),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_scratch_blocks, block_size, 1, dims.v_hidden_dims),
                    dtype=dtype,
                    device=device,
                ),
            )
            dst_fast = (
                torch.empty_like(dst_direct[0]),
                torch.empty_like(dst_direct[1]),
            )
            dst_staged = (
                torch.empty_like(dst_direct[0]),
                torch.empty_like(dst_direct[1]),
            )
            for tensor in (*dst_direct, *dst_fast, *dst_staged):
                tensor.zero_()

            chunk_ptrs_npu = build_chunk_ptrs_npu(chunks, torch.device(device))
            staging = torch.empty(
                total_tokens * dims.plane_elems,
                dtype=dtype,
                device=device,
            )

            batched_fused_sparse_single_layer_kv_transfer(
                chunks,
                staging,
                list(dst_staged),
                scratch_slots,
                selected_npu,
                partition.chunk_offsets,
                partition.chunk_sizes,
                KV_FORMAT_MLA_LATENT,
                False,
                False,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
            )

            run_direct_load_layer(
                stacked_cpu_chunks=chunks,
                dst_layer=dst_direct,
                slot_mapping_packed=scratch_slots,
                selected_token_idx=selected_npu,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                dims=dims,
                kv_format=KV_FORMAT_MLA_LATENT,
                chunk_ptrs_npu=chunk_ptrs_npu,
            )
            destination_state = prepare_sparse_direct_destination_state(
                list(dst_fast),
                scratch_slots,
                KV_FORMAT_MLA_LATENT,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
            )
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                destination_state,
                scratch_slots,
                selected_npu,
                chunk_ptrs_npu,
                chunk_size,
                total_tokens,
                False,
            )
            torch.npu.synchronize()

            _check_compact_scratch_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_staged,
                chunk_size=chunk_size,
                dims=dims,
                label="staged",
            )
            _check_compact_scratch_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_direct,
                chunk_size=chunk_size,
                dims=dims,
                label="direct",
            )
            _check_compact_scratch_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_fast,
                chunk_size=chunk_size,
                dims=dims,
                label="fast",
            )
        finally:
            if mem_objs:
                release_pin_memory_objects(mem_objs)

    def test_sparse_direct_dsa_index_compact_scratch_reads_selected_chunks(
        self,
    ) -> None:
        """DSA_INDEX sparse load must support a one-tensor indexer group."""
        total_tokens = 18879
        chunk_size = 256
        num_selected = 2048
        block_size = 128
        dtype = torch.bfloat16
        device = "npu"
        dims = MlaDsaDims(
            k_hidden_dims=128,
            v_hidden_dims=0,
            dsa_hidden_dims=128,
            plane_elems=128,
        )
        partition = compute_chunk_partition(total_tokens, chunk_size)
        mem_allocator = None
        mem_objs = []
        try:
            mem_allocator = create_pin_memory_allocator(partition, dims, dtype)
            chunks, mem_objs = allocate_stacked_cpu_chunks(
                mem_allocator, partition, dims, dtype
            )
            _fill_stacked_chunks_with_token_pattern(chunks, dtype=dtype)

            selected = _tail_heavy_selected_tokens(total_tokens, num_selected)
            selected_npu = torch.tensor(
                selected, dtype=torch.int32, device=device
            )
            scratch_base_block = 5
            scratch_base_slot = scratch_base_block * block_size
            scratch_slots_cpu = list(
                range(scratch_base_slot, scratch_base_slot + num_selected)
            )
            scratch_slots = torch.tensor(
                scratch_slots_cpu, dtype=torch.long, device=device
            )
            num_scratch_blocks = scratch_base_block + (
                (num_selected + block_size - 1) // block_size
            )
            dst_direct = (
                torch.empty(
                    (num_scratch_blocks, block_size, 1, dims.dsa_hidden_dims),
                    dtype=dtype,
                    device=device,
                ),
            )
            dst_fast = (torch.empty_like(dst_direct[0]),)
            dst_staged = (torch.empty_like(dst_direct[0]),)
            for tensor in (*dst_direct, *dst_fast, *dst_staged):
                tensor.zero_()

            chunk_ptrs_npu = build_chunk_ptrs_npu(chunks, torch.device(device))
            staging = torch.empty(
                total_tokens * dims.plane_elems,
                dtype=dtype,
                device=device,
            )

            batched_fused_sparse_single_layer_kv_transfer(
                chunks,
                staging,
                list(dst_staged),
                scratch_slots,
                selected_npu,
                partition.chunk_offsets,
                partition.chunk_sizes,
                KV_FORMAT_DSA_INDEX,
                False,
                False,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
            )
            run_direct_load_layer(
                stacked_cpu_chunks=chunks,
                dst_layer=dst_direct,
                slot_mapping_packed=scratch_slots,
                selected_token_idx=selected_npu,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                dims=dims,
                kv_format=KV_FORMAT_DSA_INDEX,
                chunk_ptrs_npu=chunk_ptrs_npu,
            )
            destination_state = prepare_sparse_direct_destination_state(
                list(dst_fast),
                scratch_slots,
                KV_FORMAT_DSA_INDEX,
                dims.k_hidden_dims,
                dims.v_hidden_dims,
                dims.dsa_hidden_dims,
            )
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                destination_state,
                scratch_slots,
                selected_npu,
                chunk_ptrs_npu,
                chunk_size,
                total_tokens,
                False,
            )
            torch.npu.synchronize()

            _check_compact_index_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_staged,
                chunk_size=chunk_size,
                dims=dims,
                label="staged",
            )
            _check_compact_index_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_direct,
                chunk_size=chunk_size,
                dims=dims,
                label="direct",
            )
            _check_compact_index_from_cpu_chunks(
                chunks=chunks,
                selected=selected,
                dst_slots=scratch_slots_cpu,
                dst_layer=dst_fast,
                chunk_size=chunk_size,
                dims=dims,
                label="fast",
            )
        finally:
            if mem_objs:
                release_pin_memory_objects(mem_objs)

    @pytest.mark.parametrize(
        ("row_width", "row_counts"),
        [
            (4096, (2048,)),
            (2048, (1537, 509)),
        ],
    )
    def test_prepared_sparse_count_rows_copy_only_valid_prefixes(
        self,
        row_width: int,
        row_counts: tuple[int, ...],
    ) -> None:
        """Count-aware rows use all valid tokens without copying padded tails."""
        num_tokens = 8192
        row_count = len(row_counts)
        payload_capacity = row_count * row_width
        block_size = 128
        common = dict(
            num_blocks=72,
            device="npu",
            num_layers=1,
            num_kv_heads=1,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            block_size=block_size,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            source_cache = generate_dsa_kv_cache(
                dsa_head_dim=32,
                **common,
            )
            harness = build_load_benchmark_harness(
                fmt="dsa",
                src_kv_cache=source_cache,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=1,
                num_selected=payload_capacity,
                seed=321,
            )
            selected = harness.selected_token_idx.reshape(
                row_count,
                row_width,
            )
            target_slots = torch.arange(
                payload_capacity,
                dtype=torch.long,
                device=selected.device,
            ).reshape(row_count, row_width)
            selected_counts = torch.tensor(
                row_counts,
                dtype=torch.int32,
                device=selected.device,
            )
            destination = harness.dst_direct_fast[0]
            for plane in destination:
                plane.zero_()

            destination_state = prepare_sparse_direct_destination_state(
                list(destination),
                target_slots,
                KV_FORMAT_DSA,
                harness.dims.k_hidden_dims,
                harness.dims.v_hidden_dims,
                harness.dims.dsa_hidden_dims,
            )
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                destination_state,
                target_slots,
                selected,
                harness.chunk_ptrs_by_layer[0],
                harness.chunk_size,
                harness.num_tokens,
                False,
                selected_counts,
            )
            torch.npu.synchronize()

            valid_positions = []
            padded_positions = []
            for row, count in enumerate(row_counts):
                row_start = row * row_width
                valid_positions.extend(range(row_start, row_start + count))
                padded_positions.extend(
                    range(row_start + count, row_start + row_width)
                )
            valid_positions_npu = torch.tensor(
                valid_positions,
                dtype=torch.long,
                device=selected.device,
            )
            padded_positions_npu = torch.tensor(
                padded_positions,
                dtype=torch.long,
                device=selected.device,
            )
            selected_flat = selected.reshape(-1).to(torch.long)
            source_slots = harness.slot_mapping_full.index_select(
                0,
                selected_flat,
            )
            target_slots_flat = target_slots.reshape(-1)

            for source_plane, destination_plane in zip(
                harness.src_kv_cache[0],
                destination,
                strict=True,
            ):
                source_flat = source_plane.flatten(0, 1)
                destination_flat = destination_plane.flatten(0, 1)
                expected = source_flat.index_select(
                    0,
                    source_slots.index_select(0, valid_positions_npu),
                )
                actual = destination_flat.index_select(
                    0,
                    target_slots_flat.index_select(
                        0,
                        valid_positions_npu,
                    ),
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                padding = destination_flat.index_select(
                    0,
                    target_slots_flat.index_select(
                        0,
                        padded_positions_npu,
                    ),
                )
                assert torch.count_nonzero(padding).item() == 0
        finally:
            if harness is not None:
                harness.close()

    @pytest.mark.parametrize("frontier", [6400, 7168])
    def test_prepared_sparse_reuses_payload_addresses_for_three_steps(
        self,
        frontier: int,
    ) -> None:
        """Replay the long-prompt MTP latent load without step barriers.

        Production submits one prepared sparse-direct transfer for each of the
        79 latent-cache layers.  Each layer owns fixed selected/count/target
        workspaces and a 4096-slot scratch, then reuses those addresses across
        decode steps.  Keep that ordering here: all three steps are submitted
        on one stream and synchronize only once, after layer 78 of step three.

        The fixed row is MTP(2) * top-k(2048) = 4096 entries.  Invalid tails
        contain the resident planner's -1 token sentinel, while every target
        mapping remains in bounds.  Consuming a stale count is therefore seen
        as an invalid source read, a wrong final value, or a guard overwrite.
        """
        row_width = 4096
        num_layers = 79
        num_steps = 3
        block_size = 128
        chunk_size = 256
        dtype = torch.bfloat16
        device = torch.device("npu")
        guard_value = -512.0
        dims = MlaDsaDims(
            k_hidden_dims=512,
            v_hidden_dims=64,
            dsa_hidden_dims=0,
            plane_elems=576,
        )
        partition = compute_chunk_partition(frontier, chunk_size)
        mem_allocator = None
        mem_objs = []
        destination_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            # Keep independent registered source backing for the first layer,
            # the final target layer, and the MTP layer.  The remaining target
            # layers share layer 0's source only to keep this regression test
            # comfortably below the memory footprint of a full server.
            mem_allocator = create_pin_memory_allocator(partition, dims, dtype)
            source_chunks: dict[int, list[torch.Tensor]] = {}
            source_ptrs: dict[int, torch.Tensor] = {}
            for layer_id, bias in ((0, 0.0), (77, 64.0), (78, 128.0)):
                chunks, layer_mem_objs = allocate_stacked_cpu_chunks(
                    mem_allocator,
                    partition,
                    dims,
                    dtype,
                )
                mem_objs.extend(layer_mem_objs)
                _fill_stacked_chunks_with_token_pattern(chunks, dtype=dtype)
                if bias:
                    for chunk in chunks:
                        chunk.add_(bias)
                source_chunks[layer_id] = chunks
                source_ptrs[layer_id] = build_chunk_ptrs_npu(chunks, device)

            source_chunks_by_layer = [source_chunks[0]] * num_layers
            chunk_ptrs_by_layer = [source_ptrs[0]] * num_layers
            source_chunks_by_layer[77] = source_chunks[77]
            source_chunks_by_layer[78] = source_chunks[78]
            chunk_ptrs_by_layer[77] = source_ptrs[77]
            chunk_ptrs_by_layer[78] = source_ptrs[78]

            selected_by_layer = [
                torch.empty(
                    (1, row_width),
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(num_layers)
            ]
            selected_counts_by_layer = [
                torch.empty(
                    (1,),
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(num_layers)
            ]
            target_slots_by_layer = [
                torch.empty(
                    (1, row_width),
                    dtype=torch.long,
                    device=device,
                )
                for _ in range(num_layers)
            ]
            payload_ptrs = [
                (
                    selected_by_layer[layer_id].data_ptr(),
                    selected_counts_by_layer[layer_id].data_ptr(),
                    target_slots_by_layer[layer_id].data_ptr(),
                )
                for layer_id in range(num_layers)
            ]

            # Every layer gets different payload contents while its bridge
            # tensors retain the same data_ptr across steps.  This catches a
            # queued transfer that observes the next step's overwrite instead
            # of the values preceding its launch.
            selected_cpu = torch.full(
                (num_steps, num_layers, row_width),
                -1,
                dtype=torch.int32,
            )
            target_cpu = torch.empty(
                (num_steps, num_layers, row_width),
                dtype=torch.long,
            )
            counts_cpu = torch.empty(
                (num_steps, num_layers),
                dtype=torch.int32,
            )
            positions = torch.arange(row_width, dtype=torch.long)
            base_counts = (4096, 3072, 2048)
            for step, base_count in enumerate(base_counts):
                for layer_id in range(num_layers):
                    if layer_id == 78:
                        # The draft layer has one top-k row, not the target
                        # model's two-row union.
                        count = 2048 - step * 512
                    elif layer_id & 1:
                        count = max(1, base_count - 1024)
                    else:
                        count = base_count
                    counts_cpu[step, layer_id] = count
                    selected_cpu[step, layer_id, :count] = (
                        positions[:count] * 137
                        + step * 211
                        + layer_id * 43
                    ).remainder(frontier).to(torch.int32)
                    target_cpu[step, layer_id] = (
                        positions * 127
                        + step * 193
                        + layer_id * 47
                    ).remainder(row_width)

            selected_by_launch = selected_cpu.to(device=device)
            counts_by_launch = counts_cpu.to(device=device)
            target_by_launch = target_cpu.to(device=device)

            # Production keeps exactly 32 * 128 = 4096 latent scratch slots;
            # it does not allocate a disjoint window for each decode step.
            num_destination_blocks = row_width // block_size
            destination_states = []
            for layer_id in range(num_layers):
                destination = (
                    torch.full(
                        (
                            num_destination_blocks,
                            block_size,
                            1,
                            dims.k_hidden_dims,
                        ),
                        guard_value,
                        dtype=dtype,
                        device=device,
                    ),
                    torch.full(
                        (
                            num_destination_blocks,
                            block_size,
                            1,
                            dims.v_hidden_dims,
                        ),
                        guard_value,
                        dtype=dtype,
                        device=device,
                    ),
                )
                destination_layers.append(destination)
                destination_states.append(
                    prepare_sparse_direct_destination_state(
                        list(destination),
                        target_slots_by_layer[layer_id],
                        KV_FORMAT_MLA_LATENT,
                        dims.k_hidden_dims,
                        dims.v_hidden_dims,
                        dims.dsa_hidden_dims,
                    )
                )

            # Complete setup before the barrier-free replay begins.  There is
            # deliberately no synchronize in either loop below.
            torch.npu.synchronize()
            for step in range(num_steps):
                for layer_id in range(num_layers):
                    selected_by_layer[layer_id].copy_(
                        selected_by_launch[step, layer_id]
                    )
                    selected_counts_by_layer[layer_id].copy_(
                        counts_by_launch[step, layer_id].reshape(1),
                    )
                    target_slots_by_layer[layer_id].copy_(
                        target_by_launch[step, layer_id]
                    )
                    sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                        destination_states[layer_id],
                        target_slots_by_layer[layer_id],
                        selected_by_layer[layer_id],
                        chunk_ptrs_by_layer[layer_id],
                        chunk_size,
                        frontier,
                        False,
                        selected_counts_by_layer[layer_id],
                    )
            torch.npu.synchronize()

            assert payload_ptrs == [
                (
                    selected_by_layer[layer_id].data_ptr(),
                    selected_counts_by_layer[layer_id].data_ptr(),
                    target_slots_by_layer[layer_id].data_ptr(),
                )
                for layer_id in range(num_layers)
            ]

            # Layer 78 is the MTP drafter latent cache; layer 77 is the final
            # target layer.  Layer 0 catches a failure that starts earlier.
            for layer_id in (0, 77, 78):
                # Scratch is persistent and is not cleared between decode
                # steps.  Derive the final contents from the last legal write
                # to each slot rather than assuming only step three survives.
                last_token_by_slot = torch.full(
                    (row_width,),
                    -1,
                    dtype=torch.int32,
                )
                for step in range(num_steps):
                    count = int(counts_cpu[step, layer_id])
                    slots = target_cpu[step, layer_id, :count]
                    last_token_by_slot[slots] = selected_cpu[
                        step, layer_id, :count
                    ]
                written_slots = torch.nonzero(
                    last_token_by_slot >= 0,
                    as_tuple=False,
                ).reshape(-1)
                chunks = source_chunks_by_layer[layer_id]
                _check_compact_scratch_from_cpu_chunks(
                    chunks=chunks,
                    selected=[
                        int(last_token_by_slot[slot]) for slot in written_slots
                    ],
                    dst_slots=[int(slot) for slot in written_slots],
                    dst_layer=destination_layers[layer_id],
                    chunk_size=chunk_size,
                    dims=dims,
                    label=(
                        "prepared-three-step "
                        f"frontier={frontier} layer={layer_id}"
                    ),
                )

                unwritten_slots = torch.nonzero(
                    last_token_by_slot < 0,
                    as_tuple=False,
                ).reshape(-1)
                if unwritten_slots.numel():
                    for plane in destination_layers[layer_id]:
                        plane_cpu = plane.reshape(
                            -1, plane.shape[-1]
                        ).detach().cpu()
                        unwritten = plane_cpu.index_select(0, unwritten_slots)
                        assert torch.all(unwritten == guard_value), (
                            "prepared sparse transfer consumed an invalid "
                            "padding sentinel: "
                            f"frontier={frontier} layer={layer_id} "
                            f"unwritten_slots={unwritten_slots.numel()}"
                        )
        finally:
            if mem_objs:
                release_pin_memory_objects(mem_objs)
            if mem_allocator is not None:
                mem_allocator.close()

    @pytest.mark.parametrize("fmt", ["mla", "dsa"])
    def test_sparse_direct_compact_scratch_slots_match_selected_source(
        self, fmt: str
    ) -> None:
        num_tokens = 4096
        num_selected = 2048
        num_layers = 1
        num_blocks = 64
        block_size = 128
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        kv_format = KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA
        common = dict(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            src = (
                generate_mla_kv_cache(**common)
                if fmt == "mla"
                else generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
            )
            harness = build_load_benchmark_harness(
                fmt=fmt,
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=num_layers,
                num_selected=num_selected,
                seed=123,
            )
            selected_values = _tail_heavy_selected_tokens(
                num_tokens, num_selected
            )
            harness.selected_token_idx = torch.tensor(
                selected_values,
                device=harness.selected_token_idx.device,
                dtype=torch.int32,
            )
            compact_base_slot = 5 * block_size
            harness.slot_mapping_packed = torch.arange(
                compact_base_slot,
                compact_base_slot + num_selected,
                device=harness.selected_token_idx.device,
                dtype=torch.long,
            )

            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            selected = harness.selected_token_idx.detach().cpu().tolist()
            full_slots = harness.slot_mapping_full.detach().cpu().tolist()
            compact_slots = harness.slot_mapping_packed.detach().cpu().tolist()
            packed_idx_to_check = sorted(
                set([0, 1, 2, 3, 4, 127, 255, 256, 511, 1023, 2047])
            )

            src_layer = harness.src_kv_cache[0]
            dst_dir = harness.dst_direct[0]
            dst_fast = harness.dst_direct_fast[0]
            src_k = src_layer[0].reshape(
                -1, num_kv_heads, kv_lora_rank
            ).detach().cpu()
            src_v = src_layer[1].reshape(
                -1, num_kv_heads, qk_rope_head_dim
            ).detach().cpu()
            dst_dir_k = dst_dir[0].reshape(
                -1, num_kv_heads, kv_lora_rank
            ).detach().cpu()
            dst_dir_v = dst_dir[1].reshape(
                -1, num_kv_heads, qk_rope_head_dim
            ).detach().cpu()
            dst_fast_k = dst_fast[0].reshape(
                -1, num_kv_heads, kv_lora_rank
            ).detach().cpu()
            dst_fast_v = dst_fast[1].reshape(
                -1, num_kv_heads, qk_rope_head_dim
            ).detach().cpu()

            src_dsa = dst_dir_dsa = dst_fast_dsa = None
            if kv_format == KV_FORMAT_DSA:
                src_dsa = src_layer[2].reshape(-1, dsa_head_dim).detach().cpu()
                dst_dir_dsa = dst_dir[2].reshape(-1, dsa_head_dim).detach().cpu()
                dst_fast_dsa = dst_fast[2].reshape(-1, dsa_head_dim).detach().cpu()

            mismatches: list[str] = []
            for pi in packed_idx_to_check:
                src_slot = int(full_slots[int(selected[pi])])
                dst_slot = int(compact_slots[pi])
                if not torch.equal(dst_dir_k[dst_slot], src_k[src_slot]):
                    mismatches.append(
                        f"direct P{pi}src{src_slot}->dst{dst_slot} K mismatch"
                    )
                if not torch.equal(dst_dir_v[dst_slot], src_v[src_slot]):
                    mismatches.append(
                        f"direct P{pi}src{src_slot}->dst{dst_slot} V mismatch"
                    )
                if not torch.equal(dst_fast_k[dst_slot], src_k[src_slot]):
                    mismatches.append(
                        f"fast P{pi}src{src_slot}->dst{dst_slot} K mismatch"
                    )
                if not torch.equal(dst_fast_v[dst_slot], src_v[src_slot]):
                    mismatches.append(
                        f"fast P{pi}src{src_slot}->dst{dst_slot} V mismatch"
                    )
                if src_dsa is not None:
                    assert dst_dir_dsa is not None
                    assert dst_fast_dsa is not None
                    if not torch.equal(dst_dir_dsa[dst_slot], src_dsa[src_slot]):
                        mismatches.append(
                            f"direct P{pi}src{src_slot}->dst{dst_slot} DSA mismatch"
                        )
                    if not torch.equal(dst_fast_dsa[dst_slot], src_dsa[src_slot]):
                        mismatches.append(
                            f"fast P{pi}src{src_slot}->dst{dst_slot} DSA mismatch"
                        )
            assert not mismatches, (
                f"compact sparse scratch mismatch fmt={fmt} "
                f"(showing first 8 of {len(mismatches)}): {mismatches[:8]}"
            )
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed fmt={fmt}: {type(exc).__name__}: "
                f"{exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass

    def test_direct_and_fast_paths_agree(self) -> None:
        num_tokens = 512
        num_layers = 2
        num_selected = 512
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        common = dict(
            num_blocks=256,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=16,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            src = generate_mla_kv_cache(**common)
            harness = build_load_benchmark_harness(
                fmt="mla",
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=num_layers,
                num_selected=num_selected,
                seed=7,
            )
            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            slots = harness.slot_mapping_packed.detach().cpu().tolist()
            packed_idx_to_check = _sample_packed_indices(len(slots))
            layer_ids = _sample_layer_ids(num_layers)
            mismatches: list[str] = []
            for layer_id in layer_ids:
                dst_dir_k = harness.dst_direct[layer_id][0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_fast_k = harness.dst_direct_fast[layer_id][0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_dir_v = harness.dst_direct[layer_id][1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_fast_v = harness.dst_direct_fast[layer_id][1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                for pi in packed_idx_to_check:
                    slot = int(slots[pi])
                    if not torch.equal(dst_dir_k[slot], dst_fast_k[slot]):
                        mismatches.append(f"L{layer_id}P{pi}slot{slot} K direct/fast differ")
                    if not torch.equal(dst_dir_v[slot], dst_fast_v[slot]):
                        mismatches.append(f"L{layer_id}P{pi}slot{slot} V direct/fast differ")
            assert not mismatches, (
                f"direct vs fast disagree (first 8 of {len(mismatches)}): "
                f"{mismatches[:8]}"
            )
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
class TestDsaPopulateVsScatterIsolation:
    """Isolate whether DSA is lost in src->cpu (populate) or cpu->dst (scatter).

    Uses a single-chunk DSA layout so localTokenIdx == globalTokenIdx, making
    the CPU chunk DSA plane offset trivially auditable.

    CPU chunk (non-interleaved) layout for DSA, per chunk of N tokens:
        [ K plane: N*kH | V plane: N*vH | DSA plane: N*dsaH ]
    DSA for local token t lives at:
        offset = N*(kH+vH) + t*dsaH, length dsaH
    """

    def test_dsa_populate_and_scatter_isolation(self) -> None:
        num_tokens = 256
        chunk_size = 256  # single chunk
        num_layers = 1
        num_blocks = 512
        block_size = 16
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        kH = num_kv_heads * kv_lora_rank  # 4096
        vH = num_kv_heads * qk_rope_head_dim  # 1024
        dsaH = dsa_head_dim  # 128
        common = dict(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            src = generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
            harness = build_load_benchmark_harness(
                fmt="dsa",
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=chunk_size,
                num_layers=num_layers,
                num_selected=num_tokens,  # dense select: all tokens
                seed=42,
            )
            # populate already ran inside build_load_benchmark_harness.
            slot_mapping_full = harness.slot_mapping_full.detach().cpu().tolist()
            chunk0 = harness.stacked_by_layer[0][0].detach().cpu()
            num_lmc = chunk_size  # single full chunk

            # ---- Stage 1: src DSA vs CPU chunk DSA plane (populate check) ----
            src_dsa = harness.src_kv_cache[0][2].reshape(-1, dsaH).detach().cpu()
            dsa_plane_start = num_lmc * (kH + vH)
            populate_mismatches: list[str] = []
            for t in range(num_tokens):
                cpu_dsa_t = chunk0[dsa_plane_start + t * dsaH : dsa_plane_start + (t + 1) * dsaH]
                slot = int(slot_mapping_full[t])
                if not torch.equal(src_dsa[slot], cpu_dsa_t):
                    populate_mismatches.append(f"token{t}slot{slot}")

            # ---- Stage 2: scatter cpu->dst_direct DSA ----
            run_all_direct_layers(harness)
            torch.npu.synchronize()
            slots_packed = harness.slot_mapping_packed.detach().cpu().tolist()
            selected = harness.selected_token_idx.detach().cpu().tolist()
            dst_dsa = harness.dst_direct[0][2].reshape(-1, dsaH).detach().cpu()
            scatter_mismatches: list[str] = []
            for pi, gtok in enumerate(selected):
                local_t = gtok % chunk_size
                cpu_dsa_t = chunk0[dsa_plane_start + local_t * dsaH : dsa_plane_start + (local_t + 1) * dsaH]
                slot = int(slots_packed[pi])
                if not torch.equal(dst_dsa[slot], cpu_dsa_t):
                    scatter_mismatches.append(f"packed{pi}gtok{gtok}slot{slot}")

            populate_ok = not populate_mismatches
            scatter_ok = not scatter_mismatches
            summary = (
                f"populate(src->cpu)={'OK' if populate_ok else 'BROKEN'} "
                f"({len(populate_mismatches)}/{num_tokens} mismatched, "
                f"first 5: {populate_mismatches[:5]}); "
                f"scatter(cpu->dst)={'OK' if scatter_ok else 'BROKEN'} "
                f"({len(scatter_mismatches)}/{len(selected)} mismatched, "
                f"first 5: {scatter_mismatches[:5]})"
            )
            # Don't hard-fail: report which stage is broken. But assert so the
            # test surfaces in CI with the summary.
            if not (populate_ok and scatter_ok):
                raise AssertionError(f"DSA isolation: {summary}")
            # If both OK but the full roundtrip test still fails, the issue is
            # elsewhere (layout interpretation in the roundtrip checker).
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
def test_chunk_partition_matches_long_prompt() -> None:
    """Sanity: 18879-token partition matches production chunk layout."""
    part = compute_chunk_partition(18879, 256)
    assert sum(part.chunk_sizes) == 18879
    assert part.num_chunks == 74
