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
from lmcache.v1.memory_management import MemoryFormat
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


class _NoopStream:
    def wait_stream(self, stream):
        pass

    def synchronize(self):
        pass


class _RecordableTensor:
    def __init__(self, numel: int, dtype=torch.long):
        self._numel = numel
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.recorded_streams = []

    def numel(self):
        return self._numel

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _TrackingStream:
    def __init__(self, name: str):
        self.name = name
        self.events = []

    def wait_stream(self, stream):
        self.events.append(("wait_stream", stream.name))

    def synchronize(self):
        self.events.append("synchronize")


class _DenseLayout:
    k_hidden_dims = 1
    v_hidden_dims = 1
    dsa_hidden_dims = 0
    kv_format = type("_Fmt", (), {"value": 0})()
    vllm_two_major = False
    gpu_buffer_allocator = None


class _MemoryObj:
    def __init__(self, tensor, fmt=MemoryFormat.KV_MLA_LATENT_FMT):
        self.tensor = tensor
        self.metadata = type("_Metadata", (), {"fmt": fmt})()


def test_sparse_memory_update_resets_fast_direct_state() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = {(123, 0, 0): object()}
    connector._sparse_direct_kvcaches_id = 123
    connector._sparse_direct_validated_layers = {(0, 0)}

    connector.notify_sparse_memory_objs_updated()

    assert connector._sparse_direct_layer_states is None
    assert connector._sparse_direct_kvcaches_id is None
    assert connector._sparse_direct_validated_layers == set()


def test_shared_cpu_store_publication_fences_store_stream() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.store_stream = _TrackingStream("store")

    connector.synchronize_shared_cpu_store_publication()

    assert connector.store_stream.events == ["synchronize"]


def test_shared_cpu_sparse_load_fences_current_stream(monkeypatch) -> None:
    current_stream = _TrackingStream("current")

    class _Npu:
        def current_stream(self):
            return current_stream

    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)

    VLLMPagedMemLayerwiseNPUConnector.synchronize_shared_cpu_sparse_load()

    assert current_stream.events == ["synchronize"]


def test_sparse_pointer_cache_reuse_debug_rejects_stale_ptrs(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    cpu_tensors = [torch.zeros(1), torch.ones(1)]
    cached_npu_ptrs = torch.tensor([111, 222], dtype=torch.long)

    monkeypatch.setattr(
        npu_connectors,
        "_SPARSE_POINTER_CACHE_REUSE_VALIDATE_PTRS",
        True,
    )
    monkeypatch.setattr(
        connector,
        "_resolve_registered_cpu_tensor_device_ptr",
        lambda tensor, **_kwargs: int(tensor.data_ptr()),
    )

    with pytest.raises(RuntimeError, match="do not match current CPU tensors"):
        connector._resolve_sparse_chunk_ptrs_npu(
            0,
            cpu_tensors,
            [cached_npu_ptrs],
            [[333, 444]],
        )


def test_sparse_direct_state_key_includes_source_layout(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = None
    kvcaches_ref = [
        (
            torch.zeros((1, 4), dtype=torch.bfloat16),
            torch.zeros((1, 4), dtype=torch.bfloat16),
        )
    ]
    slot_mapping = torch.arange(4, dtype=torch.long)
    source = torch.zeros(8, dtype=torch.bfloat16)
    prepared = []

    def _prepare_state(*args, **kwargs):
        state = object()
        prepared.append(state)
        return state

    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_layer_state",
        _prepare_state,
    )

    first = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same_shape_new_source = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[torch.zeros(8, dtype=torch.bfloat16)],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same_shape_new_slot_mapping = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=torch.arange(4, dtype=torch.long),
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    changed = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[torch.zeros(10, dtype=torch.bfloat16)],
        slot_mapping_ref=slot_mapping,
        total_tokens=5,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )

    assert first is same
    assert same_shape_new_source is first
    assert same_shape_new_slot_mapping is first
    assert changed is not first
    assert len(prepared) == 2


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
    for token, dst_slot in zip(
        selected_out.tolist(), legacy_slots.tolist(), strict=False
    ):
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
    for token, dst_slot in zip(
        selected_fixed.tolist(), fixed_slots.tolist(), strict=False
    ):
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
            self.dtype = torch.long
            self.device = torch.device("cpu")

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

    lmc_chunk = torch.zeros(8, dtype=torch.bfloat16)
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


def test_dense_direct_records_local_metadata_inputs_when_enabled(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = None
    connector._sparse_direct_validated_layers = set()

    transfer_stream = _NoopStream()
    current_stream = _NoopStream()
    slot_mapping = _RecordableTensor(8)
    chunk_ptrs = _RecordableTensor(2)
    chunk_offsets = _RecordableTensor(2, dtype=torch.int32)
    chunk_sizes = _RecordableTensor(2, dtype=torch.int32)
    layer_state = object()
    fast_calls = []
    slow_calls = []

    monkeypatch.setattr(npu_connectors, "_SPARSE_DIRECT_RECORD_STREAM", True)
    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_layer_state",
        lambda *args, **kwargs: layer_state,
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer_fast",
        lambda *args, **kwargs: fast_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer",
        lambda *args, **kwargs: slow_calls.append((args, kwargs)),
    )

    connector._run_dense_direct_kv_transfer_layer(
        kvcaches_ref=[(object(), object())],
        kv_group=0,
        layer_id=0,
        transfer_stream=transfer_stream,
        current_stream=current_stream,
        slot_mapping_full=slot_mapping,
        chunk_ptrs_npu=chunk_ptrs,
        chunk_offsets_npu=chunk_offsets,
        chunk_sizes_npu=chunk_sizes,
        total_tokens=8,
        fixed_chunk_size=0,
        dense_kv_format=5,
        dense_token_major=False,
        dense_vllm_two_major=False,
        dense_k_hidden_dims=512,
        dense_v_hidden_dims=64,
        dense_dsa_hidden_dims=0,
        dense_host_interleaved=False,
        layer_tensors=[torch.zeros(8, dtype=torch.bfloat16)],
        direction=True,
    )

    assert slow_calls == []
    assert len(fast_calls) == 1
    assert fast_calls[0][0][0] is layer_state
    assert fast_calls[0][0][1] is slot_mapping
    assert fast_calls[0][0][2] is chunk_ptrs
    assert fast_calls[0][0][3] is chunk_offsets
    assert fast_calls[0][0][4] is chunk_sizes
    assert fast_calls[0][0][7] is True
    for tensor in (slot_mapping, chunk_ptrs, chunk_offsets, chunk_sizes):
        assert tensor.recorded_streams == [transfer_stream]


def test_dense_direct_fast_state_cache_separates_load_and_store(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = None
    connector._sparse_direct_validated_layers = set()

    transfer_stream = _NoopStream()
    current_stream = _NoopStream()
    slot_mapping = _RecordableTensor(8)
    chunk_ptrs = _RecordableTensor(2)
    chunk_offsets = _RecordableTensor(2, dtype=torch.int32)
    chunk_sizes = _RecordableTensor(2, dtype=torch.int32)
    layer_tensors = [torch.zeros(8, dtype=torch.bfloat16)]
    kvcaches_ref = [(object(), object())]
    prepared = []
    fast_calls = []

    def _prepare_state(*args, **kwargs):
        state = object()
        prepared.append(state)
        return state

    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_layer_state",
        _prepare_state,
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer_fast",
        lambda *args, **kwargs: fast_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer",
        lambda *args, **kwargs: pytest.fail("slow dense direct path used"),
    )

    common_kwargs = dict(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        transfer_stream=transfer_stream,
        current_stream=current_stream,
        slot_mapping_full=slot_mapping,
        chunk_ptrs_npu=chunk_ptrs,
        chunk_offsets_npu=chunk_offsets,
        chunk_sizes_npu=chunk_sizes,
        total_tokens=8,
        fixed_chunk_size=256,
        dense_kv_format=5,
        dense_token_major=False,
        dense_vllm_two_major=False,
        dense_k_hidden_dims=512,
        dense_v_hidden_dims=64,
        dense_dsa_hidden_dims=0,
        dense_host_interleaved=False,
        layer_tensors=layer_tensors,
    )

    connector._run_dense_direct_kv_transfer_layer(
        **common_kwargs,
        direction=False,
    )
    connector._run_dense_direct_kv_transfer_layer(
        **common_kwargs,
        direction=True,
    )

    assert len(prepared) == 2
    assert len(fast_calls) == 2
    assert fast_calls[0][0][0] is prepared[0]
    assert fast_calls[1][0][0] is prepared[1]
    assert fast_calls[0][0][7] is False
    assert fast_calls[1][0][7] is True


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
        lambda layer_id, cpu_tensors, cached_chunk_ptrs_npu,
        cached_chunk_dev_ptrs=None: torch.tensor([123], dtype=torch.long),
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


def test_sparse_head_token_wise_sees_late_cached_tensors(monkeypatch) -> None:
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

    class _MemoryObj:
        def __init__(self, tensor):
            self.tensor = tensor

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
        lambda layer_id, cpu_tensors, cached_chunk_ptrs_npu,
        cached_chunk_dev_ptrs=None: torch.tensor([123], dtype=torch.long),
    )

    calls = []

    def _run_sparse_direct(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        connector,
        "_run_sparse_direct_kv_transfer_layer",
        _run_sparse_direct,
    )

    cached_tensors = []
    gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=cached_tensors,
        lmcache_cached_tokens=4,
        kv_group=0,
    )
    next(gen)

    cached_tensor = torch.zeros(4)
    fallback_tensor = torch.ones(4)
    cached_tensors.extend([[cached_tensor]])
    gen.send(([_MemoryObj(fallback_tensor)], torch.arange(4, dtype=torch.int32), 0))

    assert len(calls) == 1
    assert calls[0]["cpu_tensors"][0] is cached_tensor
    assert calls[0]["layer_tensors"][0] is cached_tensor


def test_sparse_head_token_wise_can_disable_direct_path(monkeypatch) -> None:
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

    monkeypatch.setattr(npu_connectors, "_SPARSE_DIRECT_DISABLE", True)
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
        lambda layer_id, cpu_tensors, cached_chunk_ptrs_npu,
        cached_chunk_dev_ptrs=None: torch.tensor([123], dtype=torch.long),
    )

    staging_calls = []

    def _run_sparse_staging(**kwargs):
        staging_calls.append(kwargs)

    def _run_sparse_direct(**kwargs):
        raise AssertionError("direct sparse path should be disabled")

    monkeypatch.setattr(
        connector,
        "_run_sparse_staging_kv_transfer_layer",
        _run_sparse_staging,
    )
    monkeypatch.setattr(
        connector,
        "_run_sparse_direct_kv_transfer_layer",
        _run_sparse_direct,
    )

    gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=[[torch.zeros(4)]],
        lmcache_cached_tokens=4,
        kv_group=0,
    )
    next(gen)
    gen.send(([], torch.arange(4, dtype=torch.int32), 0))

    assert len(staging_calls) == 1
    assert staging_calls[0]["total_tokens"] == 4


def test_dense_direct_fixed_chunk_size_detects_sparse_like_layout() -> None:
    assert (
        VLLMPagedMemLayerwiseNPUConnector._dense_direct_fixed_chunk_size(
            [0, 256, 512],
            [256, 256, 17],
        )
        == 256
    )
    assert (
        VLLMPagedMemLayerwiseNPUConnector._dense_direct_fixed_chunk_size(
            [0, 128, 384],
            [128, 256, 17],
        )
        == 0
    )


def test_dense_batched_to_gpu_direct_path_skips_staging(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.load_stream = _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _NoopStream())
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    init_staging_values = []

    def _lazy_initialize_buffer_with_staging(kvcaches, *, kv_group, init_staging):
        init_staging_values.append(init_staging)
        return _DenseLayout()

    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        _lazy_initialize_buffer_with_staging,
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector,
        "_check_layerwise_transfer_invariants",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not allocate staging")
        ),
    )

    metadata_calls = []

    def _metadata(chunk_offsets, chunk_sizes, *, total_tokens, kv_group):
        metadata_calls.append(
            (list(chunk_offsets), list(chunk_sizes), total_tokens, kv_group)
        )
        return (
            torch.tensor(chunk_offsets, dtype=torch.int32),
            torch.tensor(chunk_sizes, dtype=torch.int32),
        )

    monkeypatch.setattr(connector, "_dense_direct_chunk_metadata_tensors", _metadata)

    pointer_calls = []

    def _resolve_chunk_ptrs(
        layer_id,
        cpu_tensors,
        cached_chunk_ptrs_arg=None,
        cached_chunk_dev_ptrs_arg=None,
    ):
        pointer_calls.append(
            (
                layer_id,
                list(cpu_tensors),
                cached_chunk_ptrs_arg,
                cached_chunk_dev_ptrs_arg,
            )
        )
        return torch.tensor([123, 456], dtype=torch.long)

    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        _resolve_chunk_ptrs,
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    gen = connector.batched_to_gpu(
        [0, 256],
        [256, 273],
        slot_mapping=torch.arange(273, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.send(
        [
            _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
        ]
    )
    gen.close()

    assert init_staging_values == [False]
    assert metadata_calls == []
    assert len(pointer_calls) == 1
    assert pointer_calls[0][2] is None
    assert pointer_calls[0][3] is None
    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is False
    assert direct_calls[0]["total_tokens"] == 273
    assert direct_calls[0]["fixed_chunk_size"] == 256


def test_dense_batched_to_gpu_direct_path_passes_variable_chunk_metadata(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.load_stream = _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _NoopStream())
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **kwargs: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda kvcaches, *, kv_group, init_staging: _DenseLayout(),
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **kwargs: None
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not allocate staging")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors: torch.tensor([111, 222, 333], dtype=torch.long),
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    starts = [0, 128, 384]
    ends = [128, 384, 401]
    gen = connector.batched_to_gpu(
        starts,
        ends,
        slot_mapping=torch.arange(401, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.send(
        [
            _MemoryObj(torch.zeros(128, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
        ]
    )
    gen.close()

    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is False
    assert direct_calls[0]["fixed_chunk_size"] == 0
    assert direct_calls[0]["total_tokens"] == 401
    assert direct_calls[0]["chunk_offsets_npu"].tolist() == starts
    assert direct_calls[0]["chunk_sizes_npu"].tolist() == [128, 256, 17]


def test_dense_batched_from_gpu_direct_path_skips_staging(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.store_stream = _NoopStream()

    class _Npu:
        def current_stream(self):
            return _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_STORE_DISABLE", False)
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    init_staging_values = []

    def _lazy_initialize_buffer_with_staging(kvcaches, *, kv_group, init_staging):
        init_staging_values.append(init_staging)
        return _DenseLayout()

    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        _lazy_initialize_buffer_with_staging,
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector,
        "_check_layerwise_transfer_invariants",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not allocate staging")
        ),
    )

    metadata_calls = []

    def _metadata(chunk_offsets, chunk_sizes, *, total_tokens, kv_group):
        metadata_calls.append(
            (list(chunk_offsets), list(chunk_sizes), total_tokens, kv_group)
        )
        return (
            torch.tensor(chunk_offsets, dtype=torch.int32),
            torch.tensor(chunk_sizes, dtype=torch.int32),
        )

    monkeypatch.setattr(connector, "_dense_direct_chunk_metadata_tensors", _metadata)

    pointer_calls = []

    def _resolve_chunk_ptrs(
        layer_id,
        cpu_tensors,
        cached_chunk_ptrs_arg=None,
        cached_chunk_dev_ptrs_arg=None,
    ):
        pointer_calls.append(
            (
                layer_id,
                list(cpu_tensors),
                cached_chunk_ptrs_arg,
                cached_chunk_dev_ptrs_arg,
            )
        )
        return torch.tensor([123, 456], dtype=torch.long)

    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        _resolve_chunk_ptrs,
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    gen = connector.batched_from_gpu(
        [
            [
                _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
            ]
        ],
        [0, 256],
        [256, 273],
        slot_mapping=torch.arange(273, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.close()

    assert init_staging_values == [False]
    assert metadata_calls == []
    assert len(pointer_calls) == 1
    assert pointer_calls[0][2] is None
    assert pointer_calls[0][3] is None
    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is True
    assert direct_calls[0]["total_tokens"] == 273
    assert direct_calls[0]["fixed_chunk_size"] == 256


def test_dense_batched_from_gpu_direct_path_passes_variable_chunk_metadata(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.store_stream = _NoopStream()

    class _Npu:
        def current_stream(self):
            return _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_STORE_DISABLE", False)
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **kwargs: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda kvcaches, *, kv_group, init_staging: _DenseLayout(),
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **kwargs: None
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not allocate staging")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors: torch.tensor([111, 222, 333], dtype=torch.long),
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    starts = [0, 128, 384]
    ends = [128, 384, 401]
    gen = connector.batched_from_gpu(
        [
            [
                _MemoryObj(torch.zeros(128, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
            ]
        ],
        starts,
        ends,
        slot_mapping=torch.arange(401, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.close()

    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is True
    assert direct_calls[0]["fixed_chunk_size"] == 0
    assert direct_calls[0]["total_tokens"] == 401
    assert direct_calls[0]["chunk_offsets_npu"].tolist() == starts
    assert direct_calls[0]["chunk_sizes_npu"].tolist() == [128, 256, 17]


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
