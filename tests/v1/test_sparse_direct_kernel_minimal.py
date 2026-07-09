# SPDX-License-Identifier: Apache-2.0
"""Minimal reproducer for the MLA/DSA sparse multi-chunk direct kernel."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not _npu_available(),
    reason="minimal sparse direct kernel test requires NPU",
)


def _add_benchmark_helpers_to_path() -> None:
    benchmark_dir = (
        Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
    )
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))


def test_mla_sparse_multi_chunk_direct_kernel_writes_selected_tokens() -> None:
    """Exercise only single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk.

    This bypasses LMCache store, dense staging load, connector row mapping, and
    fast-state caching. The Python wrapper below dispatches to:

      csrc/mem_kernels.cpp:launch_sparse_multi_chunk_direct_kernel
        -> single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk
    """
    _add_benchmark_helpers_to_path()
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        KV_FORMAT_MLA,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (
        sparse_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()

    device = torch.device("npu")
    dtype = torch.bfloat16
    chunk_size = 4
    num_chunks = 2
    total_tokens = chunk_size * num_chunks
    selected_tokens = [0, 3, 4, 7]
    scratch_slots = [0, 1, 2, 3]
    block_size = 16
    num_blocks = 4
    k_hidden_dims = 512
    v_hidden_dims = 128
    sentinel = -7.0

    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    mem_objs = []
    cpu_chunks = []
    try:
        for chunk_id in range(num_chunks):
            mem_obj = allocator.allocate(
                torch.Size([chunk_size * (k_hidden_dims + v_hidden_dims)]),
                dtype,
            )
            assert mem_obj is not None and mem_obj.tensor is not None
            mem_objs.append(mem_obj)
            chunk = mem_obj.tensor
            cpu_chunks.append(chunk)

            k_plane = chunk[: chunk_size * k_hidden_dims].reshape(
                chunk_size, k_hidden_dims
            )
            v_plane = chunk[chunk_size * k_hidden_dims :].reshape(
                chunk_size, v_hidden_dims
            )
            for local_token in range(chunk_size):
                token = chunk_id * chunk_size + local_token
                k_plane[local_token].fill_(float(token + 1))
                v_plane[local_token].fill_(float(token + 101))

        k_dst = torch.full(
            (num_blocks, block_size, 1, k_hidden_dims),
            sentinel,
            dtype=dtype,
            device=device,
        )
        v_dst = torch.full(
            (num_blocks, block_size, 1, v_hidden_dims),
            sentinel,
            dtype=dtype,
            device=device,
        )
        slot_mapping_packed = torch.tensor(
            scratch_slots,
            dtype=torch.long,
            device=device,
        )
        selected_token_idx = torch.tensor(
            selected_tokens,
            dtype=torch.int32,
            device=device,
        )
        chunk_ptrs_npu = build_chunk_ptrs_npu(cpu_chunks, device)

        sparse_mla_dsa_batched_direct_kv_transfer(
            cpu_chunks,
            (k_dst, v_dst),
            slot_mapping_packed,
            selected_token_idx,
            chunk_size,
            total_tokens,
            KV_FORMAT_MLA,
            False,  # CPU chunk layout is stacked K plane followed by V plane.
            False,
            k_hidden_dims,
            v_hidden_dims,
            0,
            False,
            chunk_ptrs_npu,
        )
        torch.npu.synchronize()

        k_flat = k_dst.reshape(-1, k_hidden_dims)
        v_flat = v_dst.reshape(-1, v_hidden_dims)
        for row, token in enumerate(selected_tokens):
            slot = scratch_slots[row]
            expected_k = torch.full(
                (k_hidden_dims,),
                float(token + 1),
                dtype=dtype,
                device=device,
            )
            expected_v = torch.full(
                (v_hidden_dims,),
                float(token + 101),
                dtype=dtype,
                device=device,
            )
            actual_k = k_flat[slot]
            actual_v = v_flat[slot]
            assert torch.equal(actual_k, expected_k), (
                "single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk "
                f"wrote wrong K for selected_token_idx={token} "
                f"at scratch_slot={slot}; "
                f"actual_sample={actual_k[:8].detach().cpu().tolist()} "
                f"expected_sample={expected_k[:8].detach().cpu().tolist()}"
            )
            assert torch.equal(actual_v, expected_v), (
                "single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk "
                f"wrote wrong V for selected_token_idx={token} "
                f"at scratch_slot={slot}; "
                f"actual_sample={actual_v[:8].detach().cpu().tolist()} "
                f"expected_sample={expected_v[:8].detach().cpu().tolist()}"
            )

        changed_k = (k_flat != sentinel).any(dim=1).nonzero().flatten()
        changed_v = (v_flat != sentinel).any(dim=1).nonzero().flatten()
        expected_changed = torch.tensor(scratch_slots, device=device)
        assert torch.equal(changed_k, expected_changed)
        assert torch.equal(changed_v, expected_changed)
    finally:
        for mem_obj in mem_objs:
            mem_obj.ref_count_down()
        allocator.close()


@pytest.mark.parametrize(
    "kv_format_name",
    [
        "mla",
        "mla_latent",
    ],
)
def test_mla_dense_multi_chunk_direct_kernel_loads_and_offloads_all_tokens(
    kv_format_name: str,
) -> None:
    """Exercise dense direct MLA/MLA_LATENT load and offload without staging."""
    _add_benchmark_helpers_to_path()
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        KV_FORMAT_MLA,
        KV_FORMAT_MLA_LATENT,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (
        dense_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()

    device = torch.device("npu")
    dtype = torch.bfloat16
    kv_format = (
        KV_FORMAT_MLA if kv_format_name == "mla" else KV_FORMAT_MLA_LATENT
    )
    chunk_size = 4
    num_chunks = 2
    total_tokens = chunk_size * num_chunks
    block_size = 16
    num_blocks = 2
    k_hidden_dims = 512
    v_hidden_dims = 128
    sentinel = -11.0

    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    source_mem_objs = []
    offload_mem_objs = []
    source_chunks = []
    offload_chunks = []
    try:
        for chunk_id in range(num_chunks):
            source_obj = allocator.allocate(
                torch.Size([chunk_size * (k_hidden_dims + v_hidden_dims)]),
                dtype,
            )
            offload_obj = allocator.allocate(
                torch.Size([chunk_size * (k_hidden_dims + v_hidden_dims)]),
                dtype,
            )
            assert source_obj is not None and source_obj.tensor is not None
            assert offload_obj is not None and offload_obj.tensor is not None
            source_mem_objs.append(source_obj)
            offload_mem_objs.append(offload_obj)
            source_chunk = source_obj.tensor
            offload_chunk = offload_obj.tensor
            source_chunks.append(source_chunk)
            offload_chunks.append(offload_chunk)
            offload_chunk.fill_(sentinel)

            k_plane = source_chunk[: chunk_size * k_hidden_dims].reshape(
                chunk_size, k_hidden_dims
            )
            v_plane = source_chunk[chunk_size * k_hidden_dims :].reshape(
                chunk_size, v_hidden_dims
            )
            for local_token in range(chunk_size):
                token = chunk_id * chunk_size + local_token
                k_plane[local_token].fill_(float(token + 1))
                v_plane[local_token].fill_(float(token + 101))

        k_dst = torch.full(
            (num_blocks, block_size, 1, k_hidden_dims),
            sentinel,
            dtype=dtype,
            device=device,
        )
        v_dst = torch.full(
            (num_blocks, block_size, 1, v_hidden_dims),
            sentinel,
            dtype=dtype,
            device=device,
        )
        slots = [3, 5, 9, 11, 13, 15, 17, 19]
        slot_mapping_full = torch.tensor(slots, dtype=torch.long, device=device)
        dummy_metadata = torch.empty(1, dtype=torch.int32, device=device)

        dense_mla_dsa_batched_direct_kv_transfer(
            source_chunks,
            (k_dst, v_dst),
            slot_mapping_full,
            dummy_metadata,
            dummy_metadata,
            total_tokens,
            kv_format,
            False,
            False,
            k_hidden_dims,
            v_hidden_dims,
            0,
            False,
            False,
            build_chunk_ptrs_npu(source_chunks, device),
            chunk_size,
        )
        torch.npu.synchronize()

        k_flat = k_dst.reshape(-1, k_hidden_dims)
        v_flat = v_dst.reshape(-1, v_hidden_dims)
        for token, slot in enumerate(slots):
            expected_k = torch.full(
                (k_hidden_dims,),
                float(token + 1),
                dtype=dtype,
                device=device,
            )
            expected_v = torch.full(
                (v_hidden_dims,),
                float(token + 101),
                dtype=dtype,
                device=device,
            )
            assert torch.equal(k_flat[slot], expected_k)
            assert torch.equal(v_flat[slot], expected_v)

        dense_mla_dsa_batched_direct_kv_transfer(
            offload_chunks,
            (k_dst, v_dst),
            slot_mapping_full,
            dummy_metadata,
            dummy_metadata,
            total_tokens,
            kv_format,
            False,
            False,
            k_hidden_dims,
            v_hidden_dims,
            0,
            False,
            True,
            build_chunk_ptrs_npu(offload_chunks, device),
            chunk_size,
        )
        torch.npu.synchronize()

        for source_chunk, offload_chunk in zip(
            source_chunks, offload_chunks, strict=False
        ):
            assert torch.equal(offload_chunk, source_chunk)
    finally:
        for mem_obj in source_mem_objs + offload_mem_objs:
            mem_obj.ref_count_down()
        allocator.close()


def test_dsa_index_dense_multi_chunk_direct_kernel_roundtrip() -> None:
    """Exercise dense direct DSA_INDEX load/offload for two-group DSA mode."""
    _add_benchmark_helpers_to_path()
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        KV_FORMAT_DSA_INDEX,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (
        dense_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()

    device = torch.device("npu")
    dtype = torch.bfloat16
    chunk_size = 4
    num_chunks = 2
    total_tokens = chunk_size * num_chunks
    block_size = 16
    num_blocks = 2
    dsa_hidden_dims = 256
    sentinel = -13.0

    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    source_mem_objs = []
    offload_mem_objs = []
    source_chunks = []
    offload_chunks = []
    try:
        for chunk_id in range(num_chunks):
            source_obj = allocator.allocate(
                torch.Size([chunk_size * dsa_hidden_dims]),
                dtype,
            )
            offload_obj = allocator.allocate(
                torch.Size([chunk_size * dsa_hidden_dims]),
                dtype,
            )
            assert source_obj is not None and source_obj.tensor is not None
            assert offload_obj is not None and offload_obj.tensor is not None
            source_mem_objs.append(source_obj)
            offload_mem_objs.append(offload_obj)
            source_chunk = source_obj.tensor
            offload_chunk = offload_obj.tensor
            source_chunks.append(source_chunk)
            offload_chunks.append(offload_chunk)
            offload_chunk.fill_(sentinel)

            index_plane = source_chunk.reshape(chunk_size, dsa_hidden_dims)
            for local_token in range(chunk_size):
                token = chunk_id * chunk_size + local_token
                index_plane[local_token].fill_(float(token + 301))

        index_dst = torch.full(
            (num_blocks, block_size, 1, dsa_hidden_dims),
            sentinel,
            dtype=dtype,
            device=device,
        )
        slots = [2, 4, 6, 8, 10, 12, 14, 16]
        slot_mapping_full = torch.tensor(slots, dtype=torch.long, device=device)
        dummy_metadata = torch.empty(1, dtype=torch.int32, device=device)

        dense_mla_dsa_batched_direct_kv_transfer(
            source_chunks,
            (index_dst,),
            slot_mapping_full,
            dummy_metadata,
            dummy_metadata,
            total_tokens,
            KV_FORMAT_DSA_INDEX,
            False,
            False,
            dsa_hidden_dims,
            0,
            dsa_hidden_dims,
            False,
            False,
            build_chunk_ptrs_npu(source_chunks, device),
            chunk_size,
        )
        torch.npu.synchronize()

        index_flat = index_dst.reshape(-1, dsa_hidden_dims)
        for token, slot in enumerate(slots):
            expected = torch.full(
                (dsa_hidden_dims,),
                float(token + 301),
                dtype=dtype,
                device=device,
            )
            assert torch.equal(index_flat[slot], expected)

        dense_mla_dsa_batched_direct_kv_transfer(
            offload_chunks,
            (index_dst,),
            slot_mapping_full,
            dummy_metadata,
            dummy_metadata,
            total_tokens,
            KV_FORMAT_DSA_INDEX,
            False,
            False,
            dsa_hidden_dims,
            0,
            dsa_hidden_dims,
            False,
            True,
            build_chunk_ptrs_npu(offload_chunks, device),
            chunk_size,
        )
        torch.npu.synchronize()

        for source_chunk, offload_chunk in zip(
            source_chunks, offload_chunks, strict=False
        ):
            assert torch.equal(offload_chunk, source_chunk)
    finally:
        for mem_obj in source_mem_objs + offload_mem_objs:
            mem_obj.ref_count_down()
        allocator.close()
