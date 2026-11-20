# SPDX-License-Identifier: Apache-2.0
"""Actual prepared generator and eager physical-plane fallback; CPU native doubles."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import time

import pytest
import torch


def method(name, namespace, *, graph=False):
    root = Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector"
    path = root / ("sparse_graph.py" if graph else "npu_connectors.py")
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if not graph:
        body = next(
            n
            for n in body
            if getattr(n, "name", None) == "VLLMPagedMemLayerwiseNPUConnector"
        ).body
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(next(n for n in body if getattr(n, "name", None) == name))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("group", [0, 1])
def test_actual_prepared_generator_forwards_certified_physical_capacity(
    monkeypatch, group
):
    stream = object()
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: stream), raising=False)
    namespace = dict(
        torch=torch,
        time=time,
        serving_perf_enabled=lambda: False,
        serving_perf_detailed_enabled=lambda: False,
        npu_content_diagnostics_enabled=lambda: False,
        _mtp_dw_deep_diag_enabled=lambda: False,
        _should_capture_deep_payload=lambda **kw: False,
    )
    prepared = method("_batched_to_gpu_head_token_wise_prepared", namespace)
    source = NS(
        layers=(NS(chunk_ptrs_npu=torch.tensor([1000])),),
        total_tokens=1,
        chunk_token_counts=(1,),
        validated_chunk_size=4,
        physical_chunk_token_counts=((4,),),
        compact_transfer_metadata={},
        owner_lease=object(),
    )
    slots, calls = torch.tensor([7]), []
    layout = NS(
        k_hidden_dims=0,
        v_hidden_dims=0,
        dsa_hidden_dims=0,
        kv_format=NS(value=5),
        kv_device=torch.device("cpu"),
        indexer_c8=group == 1,
    )
    connector = NS(
        _group_layouts={group: layout},
        lmcache_chunk_size=4,
        compact_source_importer=NS(
            layout=NS(groups=(NS(plane_token_bytes=((64, 32),)),))
        ),
        _sparse_lmc_host_interleaved=lambda g: False,
        _get_or_create_sparse_destination_plan=lambda **kw: object(),
        _unpack_sparse_dynamic_request=lambda payload: (None, None, None, None, None),
        _normalize_sparse_selection=lambda *a: (None, False),
        _pack_sparse_layer_inputs=lambda *a: (
            slots,
            torch.tensor([0], dtype=torch.int32),
        ),
        _maybe_limit_sparse_transfer_inputs=lambda mapping, selected, **kw: (
            mapping,
            selected,
        ),
        native_source_lifetime=NS(note=lambda *a: None),
        _run_c8_indexer_bootstrap=lambda *a, **kw: calls.append((a, kw)),
        _run_prepared_sparse_direct_kv_transfer_layer=lambda **kw: calls.append(
            ((), kw)
        ),
    )
    generator = prepared(
        connector,
        {
            "prepared_sparse_source": source,
            "kvcaches": [object()],
            "slot_mapping": slots,
            "kv_group": group,
        },
    )
    next(generator)
    generator.send(None)
    generator.close()
    args, kwargs = calls[0]
    if group == 1:
        assert args[4] == 1
        assert kwargs["physical_chunk_token_counts"] == (4,)
        assert kwargs["metadata_cache"] is source.compact_transfer_metadata
    else:
        assert kwargs["total_tokens"] == 1
        assert kwargs["compact_source"] is source and kwargs["source_k_bytes"] == 64


@pytest.mark.parametrize("requests", [1, 2])
@pytest.mark.parametrize("explicit_counts", [False, True])
def test_eager_fallback_preserves_physical_planes_logical_clipping_and_operand_reuse(
    monkeypatch, requests, explicit_counts
):
    stream = object()
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: stream), raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    prepare = method("prepare_source_pointer_pairs", {"torch": torch}, graph=True)
    monkeypatch.setitem(
        sys.modules,
        "lmcache_ascend.v1.npu_connector.sparse_graph",
        NS(prepare_source_pointer_pairs=prepare),
    )
    source = NS(
        layers=(NS(chunk_ptrs_npu=torch.tensor([1000])),),
        pointer_device=torch.device("cpu"),
        validated_chunk_size=4,
        total_tokens=1,
        chunk_token_counts=(1,),
        physical_chunk_token_counts=((4,),),
        graph_pointer_pairs=None,
        compact_transfer_metadata={},
    )
    calls = []

    def launch(state, slots, selected, counts, pointers, limits, chunk_size):
        calls.append((slots, selected, counts, pointers, limits))
        assert limits.tolist() == [1] * requests
        assert pointers.tolist() == [[1000] * requests, [1000 + 4 * 64] * requests]
        # Same bounds as csrc/graph/sparse_graph_kernel.cpp: invalid logical
        # selections never dereference the longer physical packet or padding.
        copied = []
        for row in range(requests):
            count = counts[row].reshape(-1)[0].item()
            for column in range(min(count, selected.shape[1])):
                token, slot = selected[row, column].item(), slots[row, column].item()
                if 0 <= token < limits[row] and slot >= 0:
                    copied.append((row, column, token))
        assert copied == [(row, 0, 0) for row in range(requests)]

    scope = dict(
        torch=torch,
        lmc_ops=NS(sparse_graph_kv_transfer=launch),
        sparse_mla_dsa_batched_direct_kv_transfer_prepared=lambda *a: pytest.fail(
            "conflated physical/logical native path"
        ),
    )
    run = method("_run_prepared_sparse_direct_kv_transfer_layer", scope)
    validate = method("_validate_sparse_fixed_chunk_coverage", {})
    connector = NS(_validate_sparse_fixed_chunk_coverage=validate)
    selected = torch.tensor([[0, 3, -1, 0]] * requests, dtype=torch.int32)
    slots = torch.tensor([[0, 1, 2, -1]] * requests)
    counts = (
        torch.tensor([[4, 99]] * requests, dtype=torch.int32)
        if explicit_counts
        else None
    )
    for width in (4, 2):
        active_selected = selected[:, :width].contiguous()
        active_slots = slots[:, :width].contiguous()
        run(
            connector,
            plan=NS(states=[object()]),
            chunk_ptrs_npu=source.layers[0].chunk_ptrs_npu,
            layer_id=0,
            slot_mapping_packed=(
                active_slots.reshape(-1) if requests == 1 else active_slots
            ),
            selected_token_idx=(
                active_selected.reshape(-1) if requests == 1 else active_selected
            ),
            chunk_size=4,
            total_tokens=1,
            sparse_host_interleaved=False,
            selected_token_counts=counts,
            compact_source=source,
            source_k_bytes=64,
        )
    assert all(calls[0][i] is calls[1][i] for i in (2, 3, 4))
    assert source.total_tokens == 1 and source.chunk_token_counts == (1,)
