# SPDX-License-Identifier: Apache-2.0
"""Run actual pointer-cache methods with CPU torch and no NPU imports."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


class TorchProbe:
    def __init__(self):
        self.allocations = []
        self.reject_allocations = False

    def __getattr__(self, name):
        original = getattr(torch, name)
        if name not in {"tensor", "as_tensor", "empty", "stack"}:
            return original

        def call(*args, **kwargs):
            self.allocations.append(name)
            if self.reject_allocations:
                raise AssertionError("Raw prefill must not construct device metadata")
            return original(*args, **kwargs)

        return call


def connector(*, prefill=True):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_is_deferred_sparse_pointer_cache",
        "prepare_layerwise_prefill_source_pointers",
        "materialize_sparse_chunk_ptr_cache",
        "append_sparse_chunk_ptr_cache_for_layer",
        "_append_sparse_chunk_ptr_cache_for_layer_legacy",
        "append_sparse_chunk_ptr_cache_for_layers",
        "_append_sparse_chunk_ptr_rows",
        "_append_sparse_chunk_ptr_rows_legacy",
        "release_sparse_chunk_ptr_cache",
        "_resolve_sparse_chunk_ptrs_npu",
    }
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    probe = TorchProbe()
    ns = {
        "torch": probe,
        "LayerPageSource": type("LayerPageSource", (), {}),
        "serving_perf_enabled": lambda: False,
        "_layer_source_memory_objs": lambda sources, layer: sources,
        "nullcontext": nullcontext,
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *methods], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    resolved = []

    def resolve(source_obj, **kwargs):
        resolved.append((kwargs["layer_id"], source_obj))
        return source_obj

    obj = SimpleNamespace(
        _layerwise_prefill_dma=prefill,
        kv_device=torch.device("cpu"),
        _expected_group_layers=lambda group: 2,
        _layer_page_pointer_rows=lambda sources: None,
        _resolve_registered_cpu_source_device_ptr=resolve,
        _stream_context_or_null=lambda stream: nullcontext(),
    )
    for name in names:
        setattr(obj, name, MethodType(ns[name], obj))
    return obj, probe, resolved


def test_raw_prefill_and_store_append_keep_host_rows_without_tensor_allocation():
    obj, probe, resolved = connector()
    host, device = [], []
    probe.reject_allocations = True
    obj.prepare_layerwise_prefill_source_pointers(
        [[11, 12], [21, 22]], host, device, prefill_dma=True
    )
    obj.prepare_layerwise_prefill_source_pointers(
        [[13], [23]], host, device, prefill_dma=True
    )
    obj.append_sparse_chunk_ptr_cache_for_layer(0, [14], host, device)
    obj.append_sparse_chunk_ptr_cache_for_layer(1, [24], host, device)
    obj.append_sparse_chunk_ptr_cache_for_layers([[15], [25]], host, device)

    assert host == [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]
    assert device == [None, None]
    assert len(resolved) == 10  # Each newly appended pointer was resolved once.
    assert probe.allocations == []


def test_first_sparse_consumer_materializes_all_layers_once_and_reuses_rows():
    obj, probe, resolved = connector()
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11, 12], [21, 22]], host, device, prefill_dma=True
    )
    row = obj._resolve_sparse_chunk_ptrs_npu(
        0, [], device, expected_num_chunks=2, cached_chunk_dev_ptrs=host
    )
    obj.materialize_sparse_chunk_ptr_cache(host, device, kv_group=0)
    other = obj._resolve_sparse_chunk_ptrs_npu(
        1, [], device, expected_num_chunks=2, cached_chunk_dev_ptrs=host
    )

    assert row is device[0] and other is device[1]
    assert [value.tolist() for value in device] == host
    assert probe.allocations == ["tensor"]
    assert len(resolved) == 4
    assert not obj._is_deferred_sparse_pointer_cache(device)

    obj.append_sparse_chunk_ptr_cache_for_layers([[13], [23]], host, device)
    assert [value.tolist() for value in device] == host


def test_raw_prepare_invalidates_device_table_and_release_drops_identity():
    obj, probe, _ = connector()
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers([[11], [21]], host, device)
    assert all(isinstance(row, torch.Tensor) for row in device)
    probe.allocations.clear()
    probe.reject_allocations = True
    obj.prepare_layerwise_prefill_source_pointers(
        [[12], [22]], host, device, prefill_dma=True
    )
    assert device == [None, None]
    assert id(device) not in obj._layerwise_pointer_tables
    assert obj._is_deferred_sparse_pointer_cache(device)

    obj.release_sparse_chunk_ptr_cache(device)
    assert not obj._is_deferred_sparse_pointer_cache(device)
    assert obj._deferred_sparse_pointer_caches == {}
    assert probe.allocations == []


@pytest.mark.parametrize("corruption", ["ragged", "device_row", "missing_layer"])
def test_invalid_deferred_cache_is_rejected_before_tensor_allocation(corruption):
    obj, probe, _ = connector()
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11], [21]], host, device, prefill_dma=True
    )
    if corruption == "ragged":
        host[0].append(12)
    elif corruption == "device_row":
        device[0] = torch.tensor([11])
    else:
        host.pop()
    with pytest.raises(ValueError, match="Deferred sparse pointer cache"):
        obj.materialize_sparse_chunk_ptr_cache(host, device, kv_group=0)
    assert probe.allocations == []
    assert obj._is_deferred_sparse_pointer_cache(device)


def test_failed_upload_keeps_deferred_state_for_retry():
    obj, probe, _ = connector()
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11], [21]], host, device, prefill_dma=True
    )
    probe.reject_allocations = True
    with pytest.raises(AssertionError, match="Raw prefill"):
        obj.materialize_sparse_chunk_ptr_cache(host, device, kv_group=0)
    assert device == [None, None]
    assert obj._is_deferred_sparse_pointer_cache(device)
    probe.reject_allocations = False
    obj.materialize_sparse_chunk_ptr_cache(host, device, kv_group=0)
    assert [row.tolist() for row in device] == host


def test_ordinary_cache_is_not_silently_treated_as_deferred():
    obj, probe, _ = connector()
    host, device = [[11], [21]], [None, None]
    obj.materialize_sparse_chunk_ptr_cache(host, device, kv_group=0)
    assert device == [None, None]
    assert probe.allocations == []
    with pytest.raises(ValueError, match="Sparse pointer prefix is incomplete"):
        obj.append_sparse_chunk_ptr_cache_for_layers([[12], [22]], host, device)


def test_deferred_source_returning_to_ordinary_dense_materializes_prefix():
    obj, _, _ = connector()
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11], [21]], host, device, prefill_dma=True
    )
    obj.prepare_layerwise_prefill_source_pointers([[12], [22]], host, device)
    assert [row.tolist() for row in device] == host
    assert not obj._is_deferred_sparse_pointer_cache(device)


def test_deferred_groups_materialize_and_release_independently():
    obj, probe, _ = connector()
    host0, device0, host1, device1 = [], [], [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11], [21]], host0, device0, kv_group=0, prefill_dma=True
    )
    obj.prepare_layerwise_prefill_source_pointers(
        [[31], [41]], host1, device1, kv_group=1, prefill_dma=True
    )
    obj.materialize_sparse_chunk_ptr_cache(host1, device1, kv_group=1)
    assert [row.tolist() for row in device1] == host1
    assert device0 == [None, None]
    assert obj._is_deferred_sparse_pointer_cache(device0)
    obj.release_sparse_chunk_ptr_cache(device1)
    assert obj._is_deferred_sparse_pointer_cache(device0)
    obj.materialize_sparse_chunk_ptr_cache(host0, device0, kv_group=0)
    assert [row.tolist() for row in device0] == host0
    assert probe.allocations == ["tensor", "tensor"]


def test_decoder_group_append_rebuilds_complete_rows_and_preserves_old_storage():
    obj, probe, _ = connector(prefill=False)
    host, device = [], []
    obj.append_sparse_chunk_ptr_cache_for_layers([[11, 12], [21, 22]], host, device)
    old = tuple(device)
    obj.append_sparse_chunk_ptr_cache_for_layers([[13], [23]], host, device)
    assert [row.tolist() for row in device] == [[11, 12, 13], [21, 22, 23]]
    assert [row.tolist() for row in old] == [[11, 12], [21, 22]]
    assert old[0].untyped_storage().data_ptr() != device[0].untyped_storage().data_ptr()
    assert probe.allocations == ["tensor", "tensor"]
    assert not hasattr(obj, "_layerwise_pointer_tables")


def test_decoder_layer_append_and_metadata_kwargs_cannot_enable_prefill_tables():
    obj, probe, _ = connector(prefill=False)
    host, device = [], []
    obj.prepare_layerwise_prefill_source_pointers(
        [[11], [21]], host, device, prefill_dma=True,
    )
    old = tuple(device)
    obj.append_sparse_chunk_ptr_cache_for_layer(0, [12], host, device)
    obj.append_sparse_chunk_ptr_cache_for_layer(1, [22], host, device)
    assert [row.tolist() for row in device] == [[11, 12], [21, 22]]
    assert [row.tolist() for row in old] == [[11], [21]]
    assert not obj._is_deferred_sparse_pointer_cache(device)
    assert not hasattr(obj, "_layerwise_pointer_tables")
    assert probe.allocations == ["tensor", "tensor", "tensor"]


def test_decoder_deferred_copy_uploads_full_table_once_on_the_load_stream():
    obj, probe, _ = connector(prefill=False)
    staged = []

    def stage(value, **kwargs):
        staged.append(value.tolist())
        return value.clone()

    obj.stage_dense_load_tensor = stage
    host, device = [], []
    obj._append_sparse_chunk_ptr_rows([[11], [21]], host, device, defer_copy=True)
    obj._append_sparse_chunk_ptr_rows([[12], [22]], host, device, defer_copy=True)
    assert staged == [[[11], [21]], [[11, 12], [21, 22]]]
    assert [row.tolist() for row in device] == host
    assert probe.allocations == ["tensor", "tensor"]


def test_prefill_group_append_still_reuses_its_capacity_table():
    obj, _, _ = connector(prefill=True)
    host, device = [], []
    obj.append_sparse_chunk_ptr_cache_for_layers([[11], [21]], host, device)
    old_address = device[0].untyped_storage().data_ptr()
    obj.append_sparse_chunk_ptr_cache_for_layers([[12], [22]], host, device)
    assert device[0].untyped_storage().data_ptr() == old_address
    assert [row.tolist() for row in device] == [[11, 12], [21, 22]]
