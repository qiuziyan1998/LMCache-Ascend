# SPDX-License-Identifier: Apache-2.0
"""Exercise the ordinary store generator with real merged-page allocations.

Only NPU initialization and the native transfer are substituted. The production
generator, layer accessor and slot slicing run on CPU, including partial chunks.
Run with the matching LMCache checkout installed or on PYTHONPATH.
"""

import ast
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


@pytest.fixture(scope="module")
def store_implementation():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helpers = {
        "_layer_memory_tensor",
        "_slice_layerwise_slot_mapping",
        "_cached_layerwise_slot_mapping",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in helpers]
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    nodes.append(
        next(
            node
            for node in cls.body
            if getattr(node, "name", None) == "batched_from_gpu"
        )
    )

    class TorchWithCPUStream:
        npu = NS(current_stream=lambda: "compute")

        def __getattr__(self, name):
            return getattr(torch, name)

    namespace = {
        "torch": TorchWithCPUStream(),
        "LayerPageMemoryObj": LayerPageMemoryObj,
        "_DENSE_DIRECT_STORE_DISABLE": False,
        "_mtp_dw_deep_diag_enabled": lambda: False,
        "logger": logging.getLogger(__name__),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["batched_from_gpu"]


def memory_obj(fmt, tokens, width, layers=None):
    shape = torch.Size([tokens * width])
    count = layers or 1
    raw = torch.empty(shape.numel() * count * 2, dtype=torch.uint8)
    raw.view(torch.bfloat16).fill_(-1)
    meta = MemoryObjMetadata(
        shape=shape,
        dtype=torch.bfloat16,
        address=0,
        phy_size=raw.numel(),
        ref_count=1,
        fmt=fmt,
        shapes=[shape] * count,
        dtypes=[torch.bfloat16] * count,
    )
    if layers is not None:
        return LayerPageMemoryObj(
            raw, meta, None, num_layers=layers, valid_tokens=tokens
        )
    return TensorMemoryObj(raw, meta, None)


def connector_for_store(implementation, group, layers):
    fmt = (
        MemoryFormat.KV_MLA_LATENT_FMT if group == 0 else MemoryFormat.KV_DSA_INDEX_FMT
    )
    widths = (3, 2) if group == 0 else (2,)
    caches = [
        tuple(
            (torch.arange(8 * width).reshape(8, width) + 40 * layer + 100 * plane).to(
                torch.bfloat16
            )
            for plane, width in enumerate(widths)
        )
        for layer in range(layers)
    ]
    calls, syncs = [], []
    obj = NS(
        initialize_kvcaches_ptr=lambda **kwargs: None,
        kvcaches=caches,
        use_gpu=True,
        store_stream=NS(synchronize=lambda: syncs.append("store")),
        _lazy_initialize_buffer_with_staging=lambda *args, **kwargs: NS(
            kv_format=NS(value=group),
            vllm_two_major=False,
            k_hidden_dims=3 if group == 0 else 0,
            v_hidden_dims=2 if group == 0 else 0,
            dsa_hidden_dims=2 if group == 1 else 0,
        ),
        _is_mla_dsa_format=lambda group: True,
        _check_layerwise_transfer_invariants=lambda **kwargs: None,
        _slot_mapping_on_kv_device=lambda mapping, stream: mapping,
        _layerwise_token_major=lambda group: False,
        _expected_memory_format=lambda group: fmt,
        _sparse_lmc_host_interleaved=lambda group: False,
        _prepare_dense_direct_chunk_metadata=lambda offsets, sizes, **kwargs: (
            4,
            torch.tensor(offsets),
            torch.tensor(sizes),
        ),
        _expected_group_layers=lambda group: layers,
        _resolve_sparse_chunk_ptrs_npu=lambda layer, tensors: torch.tensor(
            [tensor.data_ptr() for tensor in tensors], dtype=torch.long
        ),
    )

    def native_transfer(**kwargs):
        assert kwargs["direction"] is True
        assert kwargs["defer_consumer_wait"] is False
        assert kwargs["require_prepared_state"] is False
        assert kwargs["kv_group"] == group
        assert kwargs["kvcaches_ref"] is caches
        assert kwargs["transfer_stream"] is obj.store_stream
        layer = kwargs["layer_id"]
        slots = kwargs["slot_mapping_full"]
        for tensor, ptr, offset, size in zip(
            kwargs["layer_tensors"],
            kwargs["chunk_ptrs_npu"].tolist(),
            kwargs["chunk_offsets_npu"].tolist(),
            kwargs["chunk_sizes_npu"].tolist(),
            strict=True,
        ):
            assert ptr == tensor.data_ptr()
            expected = torch.cat(
                [
                    plane[slots[offset : offset + size]].flatten()
                    for plane in caches[layer]
                ]
            )
            tensor.copy_(expected)
        calls.append(kwargs)

    obj._run_dense_direct_kv_transfer_layer = native_transfer
    obj.batched_from_gpu = MethodType(implementation, obj)
    return obj, fmt, sum(widths), calls, syncs


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("merged", [False, True])
def test_ordinary_store_writes_only_selected_layer(store_implementation, group, merged):
    layers = 3
    connector, fmt, width, calls, syncs = connector_for_store(
        store_implementation, group, layers
    )
    if merged:
        pages = [memory_obj(fmt, count, width, layers) for count in (4, 2)]
        # A real merged page deliberately disallows unqualified tensor access.
        with pytest.raises(RuntimeError, match="requires layer_tensor"):
            _ = pages[0].tensor
        destinations = [pages] * layers
        views = [
            [page.layer_tensor(layer) for page in pages] for layer in range(layers)
        ]
    else:
        destinations = [
            [memory_obj(fmt, count, width) for count in (4, 2)] for _ in range(layers)
        ]
        views = [[obj.tensor for obj in row] for row in destinations]
    slots = torch.tensor([7, 2, 4, 1, 0, 5])
    # Nonzero base models a later chunked-prefill store and includes a partial page.
    gen = connector.batched_from_gpu(
        destinations,
        [8, 12],
        [12, 14],
        slot_mapping=slots,
        slot_mapping_base=8,
        sync=False,
        kv_group=group,
    )
    try:
        for layer in range(layers):
            next(gen)
            assert len(calls) == layer + 1
            for chunk, (offset, count) in enumerate(((0, 4), (4, 2))):
                expected = torch.cat(
                    [
                        plane[slots[offset : offset + count]].flatten()
                        for plane in connector.kvcaches[layer]
                    ]
                )
                torch.testing.assert_close(
                    views[layer][chunk], expected, rtol=0, atol=0
                )
            for future in views[layer + 1 :]:
                assert all(torch.all(tensor == -1) for tensor in future)
        assert list(gen) == [None]
        assert len(syncs) == layers  # Existing ordinary-store drain is preserved.
    finally:
        gen.close()


@pytest.mark.parametrize("merged", [False, True])
def test_ordinary_store_rejects_invalid_storage(store_implementation, merged):
    connector, fmt, width, calls, _ = connector_for_store(store_implementation, 1, 1)
    obj = memory_obj(fmt, 4, width, 1 if merged else None)
    obj.invalidate()
    gen = connector.batched_from_gpu(
        [[obj]], [0], [4], slot_mapping=torch.arange(4), sync=False, kv_group=1
    )
    with pytest.raises((RuntimeError, ValueError), match="no longer valid|no tensor"):
        next(gen)
    assert not calls


@pytest.mark.parametrize("prefill", [False, True])
def test_store_stream_snapshot_and_early_close_follow_selected_mode(
    store_implementation, monkeypatch, prefill,
):
    connector, fmt, width, calls, syncs = connector_for_store(
        store_implementation, 1, 3,
    )
    connector._layerwise_prefill_dma = prefill
    snapshots = []

    def current_stream():
        stream = object()
        snapshots.append(stream)
        return stream

    monkeypatch.setattr(
        store_implementation.__globals__["torch"].npu,
        "current_stream", current_stream,
    )
    pages = [memory_obj(fmt, 4, width, 3)]
    gen = connector.batched_from_gpu(
        [pages] * 3, [0], [4],
        slot_mapping=torch.arange(4), sync=False, kv_group=1,
    )
    next(gen)
    next(gen)
    assert len(snapshots) == (2 if prefill else 1)
    assert [call["current_stream"] for call in calls] == (
        snapshots if prefill else snapshots * 2
    )
    gen.close()
    # Dense D stores retain baseline close semantics; P may have a pending DMA.
    assert len(syncs) == (2 if prefill else 1)
