# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
from lmcache.v1.memory_management import MemoryFormat

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
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
    _accumulate_kvcache_fingerprint,
    _diag_selected_chunk_summary,
    _kvcache_fingerprint_message,
    _layerwise_transfer_diag_message,
    _new_kvcache_fingerprint,
)
import lmcache_ascend.c_ops as lmc_ops


def _make_sparse_pack_connector() -> VLLMPagedMemLayerwiseNPUConnector:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    connector._layerwise_sparse_idx_cache = None
    return connector


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


def test_diag_selected_chunk_summary() -> None:
    selected = torch.tensor([18831, 18814, 3, 257], dtype=torch.int32)

    message = _diag_selected_chunk_summary(selected, chunk_size=256, sample=4)

    assert message == (
        "selected_chunk_local.head="
        "[(18831, 73, 143), (18814, 73, 126), (3, 0, 3), (257, 1, 1)]"
    )


def test_sparse_pack_requires_compact_scratch_slot_mapping() -> None:
    """Sparse selected-token load uses slot_mapping as destination rows.

    The connector does not derive compact scratch slots from selected token ids.
    If the caller passes a full-prefix mapping, LMCache writes to the first N
    full-prefix slots instead of the compact scratch rows consumed by SFA.
    """
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651, 18639, 18455, 18642, 18445],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    packed, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
        )
    )

    assert torch.equal(selected_out, selected)
    assert packed.tolist() == list(range(256, 256 + selected.numel()))
    assert packed.tolist() != list(range(selected.numel()))

    compact_scratch_slots = torch.arange(selected.numel(), dtype=torch.long)
    packed_compact, selected_compact = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            compact_scratch_slots,
            selected,
            0,
        )
    )
    assert torch.equal(selected_compact, selected)
    assert packed_compact.tolist() == list(range(selected.numel()))


def test_sparse_pack_uses_target_slot_mapping_when_provided() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    target_slots = torch.tensor([901, 902, 903, 904], dtype=torch.long)

    packed, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=target_slots,
        )
    )

    assert torch.equal(selected_out, selected)
    assert torch.equal(packed, target_slots)


def test_sparse_pack_legacy_slots_miss_compact_scratch_window() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    compact_scratch_slots = torch.tensor([900, 901, 902, 903], dtype=torch.long)

    legacy_slots, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
        )
    )
    assert torch.equal(selected_out, selected)
    assert not torch.equal(legacy_slots, compact_scratch_slots)

    source_by_token = {
        int(token): float(idx + 1) for idx, token in enumerate(selected.tolist())
    }
    scratch = torch.zeros(1024, dtype=torch.float32)
    for token, dst_slot in zip(selected_out.tolist(), legacy_slots.tolist()):
        scratch[int(dst_slot)] = source_by_token[int(token)]

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert not torch.equal(scratch[compact_scratch_slots], expected)

    fixed_slots, selected_fixed = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=compact_scratch_slots,
        )
    )
    scratch.zero_()
    for token, dst_slot in zip(selected_fixed.tolist(), fixed_slots.tolist()):
        scratch[int(dst_slot)] = source_by_token[int(token)]

    assert torch.equal(scratch[compact_scratch_slots], expected)


def test_sparse_pack_rejects_target_slot_mapping_length_mismatch() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([18831, 18814, 18810, 18651], dtype=torch.int32)
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    target_slots = torch.tensor([901, 902, 903], dtype=torch.long)

    with pytest.raises(ValueError, match="target_slot_mapping"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=target_slots,
        )


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
