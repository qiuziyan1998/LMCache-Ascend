# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from contextlib import nullcontext
from unittest.mock import patch

# Third Party
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
)
import lmcache_ascend.v1.npu_connector.npu_connectors as npu_connectors
import lmcache_ascend.c_ops as lmc_ops


def _make_sparse_pack_connector() -> VLLMPagedMemLayerwiseNPUConnector:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    connector._layerwise_sparse_idx_cache = None
    return connector


def test_sparse_pack_rejects_compact_without_target_slot_mapping(
    monkeypatch,
) -> None:
    """Compact selected-token loads must use explicit destination slots."""
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651, 18639, 18455, 18642, 18445],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)

    with pytest.raises(ValueError, match="Compact sparse load"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
        )

    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    with pytest.raises(ValueError, match="target_slot_mapping"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
        )


def test_sparse_pack_uses_target_slot_mapping_when_provided(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
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


def test_sparse_pack_compact_target_slot_mapping_writes_scratch_window(
    monkeypatch,
) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    compact_scratch_slots = torch.tensor([900, 901, 902, 903], dtype=torch.long)

    source_by_token = {
        int(token): float(idx + 1) for idx, token in enumerate(selected.tolist())
    }
    scratch = torch.zeros(1024, dtype=torch.float32)
    fixed_slots, selected_fixed = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=compact_scratch_slots,
        )
    )
    for token, dst_slot in zip(selected_fixed.tolist(), fixed_slots.tolist()):
        scratch[int(dst_slot)] = source_by_token[int(token)]

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(scratch[compact_scratch_slots], expected)


def test_sparse_pack_rejects_target_slot_mapping_length_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
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


def test_sparse_pack_full_width_uses_legacy_path_when_compact_disabled() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.arange(2048, dtype=torch.int32)
    full_width_slots = torch.arange(256, 256 + 2048, dtype=torch.long)

    packed, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_width_slots,
            selected,
            0,
        )
    )

    assert torch.equal(selected_out, selected)
    assert torch.equal(packed, full_width_slots)


def test_sparse_pack_rejects_compact_explicit_when_disabled() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([18831, 18814, 18810, 18651], dtype=torch.int32)
    target_slots = torch.tensor([900, 901, 902, 903], dtype=torch.long)

    with pytest.raises(ValueError, match="compact_load_enabled=False"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            kv_group=0,
        )


def test_sparse_pack_rejects_compact_for_dsa_index_group(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([18831, 18814, 18810, 18651], dtype=torch.int32)
    target_slots = torch.tensor([900, 901, 902, 903], dtype=torch.long)

    with pytest.raises(ValueError, match="kv_group=0"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            kv_group=1,
        )


def test_sparse_pack_rejects_compact_python_lists(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    connector = _make_sparse_pack_connector()

    with pytest.raises(ValueError, match="torch.Tensor payloads"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            [18831, 18814, 18810, 18651],
            [900, 901, 902, 903],
            kv_group=0,
        )


def test_sparse_pack_compact_2d_flattens_row_major(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [[18831, 18814, 18810], [18651, 18639, 18455]],
        dtype=torch.int32,
    )
    target_slots = torch.tensor(
        [[900, 901, 902], [2948, 2949, 2950]],
        dtype=torch.long,
    )

    packed, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected.t().t(),
            target_slots.t().t(),
            kv_group=0,
        )
    )

    assert torch.equal(selected_out, selected.reshape(-1))
    assert torch.equal(packed, target_slots.reshape(-1))


def test_sparse_pack_rejects_compact_shape_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_COMPACT_LOAD", True)
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [[18831, 18814, 18810], [18651, 18639, 18455]],
        dtype=torch.int32,
    )
    target_slots = torch.tensor([[900, 901, 902]], dtype=torch.long)

    with pytest.raises(ValueError, match="shape must match"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            kv_group=0,
        )


def test_sparse_direct_explicit_payload_uses_fast_path(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    connector._sparse_direct_layer_states = None
    connector._sparse_direct_validated_layers = set()

    class _Stream:
        def wait_stream(self, stream):
            pass

        def wait_event(self, event):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    class _TensorLike:
        def __init__(self, numel: int):
            self._numel = numel

        def numel(self):
            return self._numel

        def record_stream(self, stream):
            pass

    fast_calls = []
    slow_calls = []
    layer_state = object()

    def _prepare_state(*args, **kwargs):
        return layer_state

    def _fast(*args, **kwargs):
        fast_calls.append((args, kwargs))

    def _slow(*args, **kwargs):
        slow_calls.append((args, kwargs))

    monkeypatch.setattr(
        npu_connectors, "prepare_sparse_direct_layer_state", _prepare_state
    )
    monkeypatch.setattr(
        npu_connectors, "sparse_mla_dsa_batched_direct_kv_transfer_fast", _fast
    )
    monkeypatch.setattr(
        npu_connectors, "sparse_mla_dsa_batched_direct_kv_transfer", _slow
    )

    lmc_chunk = object()
    slot_mapping = _TensorLike(2)
    selected = _TensorLike(2)
    chunk_ptrs = _TensorLike(1)

    connector._run_sparse_direct_kv_transfer_layer(
        kvcaches_ref=[(object(), object())],
        kv_group=0,
        layer_id=0,
        load_stream=_Stream(),
        current_stream=_Stream(),
        slot_mapping_packed=slot_mapping,
        selected_token_idx=selected,
        chunk_size=4,
        total_tokens=4,
        chunk_ptrs_npu=chunk_ptrs,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
        sparse_host_interleaved=False,
        layer_tensors=[lmc_chunk],
        slot_mapping_ref=slot_mapping,
        cpu_tensors=[lmc_chunk],
        explicit_sparse_payload=True,
    )

    assert len(fast_calls) == 1
    assert slow_calls == []
    assert fast_calls[0][0][0] is layer_state


def test_sparse_head_token_wise_uses_cached_token_count(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.load_stream_idx = 0
    connector.load_stream_num = 1
    connector.load_stream_list = [object()]
    connector.lmcache_chunk_size = 256

    class _Stream:
        pass

    class _Layout:
        k_hidden_dims = 1
        v_hidden_dims = 1
        dsa_hidden_dims = 0
        kv_format = type("_Fmt", (), {"value": 0})()
        vllm_two_major = False

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer",
        lambda kvcaches, kv_group=0: _Layout(),
    )
    monkeypatch.setattr(
        connector,
        "_layerwise_token_major",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_sparse_lmc_host_interleaved",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_pack_sparse_layer_inputs",
        lambda slot_mapping, selected_token_idx, token_start_index: (
            slot_mapping,
            selected_token_idx,
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors, cached_chunk_ptrs_npu: torch.tensor(
            [123], dtype=torch.long
        ),
    )

    def _unexpected_total_tokens(*args, **kwargs):
        raise AssertionError("sparse total token shape walk should be skipped")

    monkeypatch.setattr(
        connector,
        "_sparse_total_tokens_from_layer_chunks",
        _unexpected_total_tokens,
    )

    calls = []

    def _run_sparse_direct(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        connector,
        "_run_sparse_direct_kv_transfer_layer",
        _run_sparse_direct,
    )

    gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=[[torch.zeros(4)]],
        lmcache_cached_tokens=18879,
        kv_group=0,
    )
    next(gen)
    gen.send(([], torch.arange(4, dtype=torch.int32), 0))

    assert len(calls) == 1
    assert calls[0]["total_tokens"] == 18879


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
