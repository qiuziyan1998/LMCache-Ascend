# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from types import SimpleNamespace
from unittest.mock import patch
import random

# Third Party
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator

# TODO (gingfung): once we have sglang kernel, re-enable test_sglang_connector_with_gpu_and_mla
from lmcache_tests.v1.test_gpu_connector import (
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench,
)
import pytest
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    SGLangLayerwiseNPUConnector,
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
    _accumulate_kvcache_fingerprint,
    _kvcache_fingerprint_message,
    _layerwise_transfer_diag_message,
    _new_kvcache_fingerprint,
)
from tests.v1.utils import check_sglang_npu_kv_cache_equal, generate_sglang_npu_kv_cache
import lmcache_ascend.c_ops as lmc_ops


def test_dense_prefix_layerwise_transfer_diag_message() -> None:
    layout = SimpleNamespace(
        kv_format=SimpleNamespace(name="DSA_INDEX"),
        vllm_two_major=False,
        k_hidden_dims=16,
        v_hidden_dims=0,
        dsa_hidden_dims=16,
    )
    message = _layerwise_transfer_diag_message(
        operation="npu_dense_prefix_load",
        req_id="req-diag",
        kv_group=1,
        starts=[0, 4],
        ends=[4, 8],
        slot_mapping_full=torch.arange(20, 28, dtype=torch.long),
        layout=layout,
        token_major=False,
        expected_fmt=MemoryFormat.KV_T2D,
        kvcaches_ref=[(torch.zeros((2, 4, 1, 16)),)],
    )

    assert "operation=npu_dense_prefix_load" in message
    assert "req_id=req-diag" in message
    assert "kv_group=1" in message
    assert "group=dsa_index" in message
    assert "kv_format=DSA_INDEX" in message
    assert "chunks=2" in message
    assert "total_span_tokens=8" in message
    assert "slot_mapping_full.len=8" in message
    assert "slot_mapping_full.min=20" in message
    assert "slot_mapping_full.max=27" in message
    assert "kvcache_layer0_shapes=[(2, 4, 1, 16)]" in message


def test_dense_prefix_kvcache_fingerprint_message() -> None:
    fingerprint = _new_kvcache_fingerprint()
    layer_cache = (torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2),)

    _accumulate_kvcache_fingerprint(
        fingerprint,
        layer_cache,
        torch.tensor([1, 3], dtype=torch.long),
    )
    message = _kvcache_fingerprint_message(
        operation="npu_dense_prefix_load_kvcache_fingerprint",
        req_id="req-diag",
        kv_group=1,
        fingerprint=fingerprint,
    )

    assert "operation=npu_dense_prefix_load_kvcache_fingerprint" in message
    assert "req_id=req-diag" in message
    assert "kv_group=1" in message
    assert "group=dsa_index" in message
    assert "layers=1" in message
    assert "numel=4" in message
    assert "sum=1.800000e+01" in message
    assert "abs_sum=1.800000e+01" in message
    assert "sq_sum=9.800000e+01" in message
    assert "min=2.0" in message
    assert "max=7.0" in message


@pytest.mark.parametrize("use_npu", [True])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
    ],
)
def test_layerwise_vllm_paged_connector_with_npu(use_npu, gpu_kv_format):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu, gpu_kv_format)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)


def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("use_mla", [True, False])
def test_sglang_layerwise_connector_with_npu(use_gpu, use_mla):
    """
    Test SGLang NPU integration with LMCache-Ascend.

    This test verifies the complete workflow of SGLang NPU with LMCache-Ascend:
    1. Generate SGLang NPU Layer-Concatenated format KV cache
    2. Test KV cache transfer from NPU to CPU (store)
    3. Test KV cache transfer from CPU to NPU (load)
    4. Verify the data integrity after round-trip transfer

    KV cache format: [2, layer_nums, num_blocks, block_size, num_heads, head_dim]
    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "npu"
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size

    num_tokens = num_blocks * block_size // 2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )
    gpu_kv_dst = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_sglang_npu_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    connector = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector2.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )

    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_sglang_npu_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    allocator.close()
