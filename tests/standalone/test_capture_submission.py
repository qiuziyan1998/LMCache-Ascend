# SPDX-License-Identifier: Apache-2.0
"""Exercise actual group preparation/submission around a mocked native API."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from typing import Any, List, Union

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    events = []
    stream = NS(
        wait_stream=lambda s: events.append("dependency"),
        synchronize=lambda: events.append("sync"),
    )

    def tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch.tensor(*args, **kwargs)

    fake_torch = NS(
        Tensor=torch.Tensor,
        tensor=tensor,
        cat=torch.cat,
        int32=torch.int32,
        int64=torch.int64,
        long=torch.long,
        npu=NS(
            current_stream=lambda: stream,
            Event=lambda: NS(record=lambda s: events.append("event")),
        ),
    )
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return [[10], [20]], torch.tensor([[10], [20]])

    ns = dict(
        torch=fake_torch,
        Any=Any,
        List=List,
        Union=Union,
        MemoryObj=Any,
        PreparedGroupCapture=None,
        dataclass=dataclass,
        _layer_memory_tensor=lambda obj, layer: obj.tensor,
        prepare_sparse_direct_layer_state=lambda *a: object(),
        dense_mla_dsa_group_direct_kv_transfer_fast=native,
        lmc_ops=NS(
            get_device_ptr=lambda p, size: p,
            dense_mla_dsa_group_direct_kv_transfer_prepared=lambda *a: events.append(
                "native"
            ),
        ),
    )
    source = ast.parse(
        (ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py").read_text(
            encoding="utf-8"
        )
    )
    nodes = [
        next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "PreparedGroupCapture"
        )
    ]
    names = {
        "prepare_group_capture",
        "enqueue_group_capture",
        "batched_from_gpu_group",
    }
    nodes += [
        n
        for n in ast.walk(source)
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, *nodes], type_ignores=[])
            ),
            "group_capture",
            "exec",
        ),
        ns,
    )
    layout = NS(
        kv_format=NS(value=1),
        vllm_two_major=False,
        k_hidden_dims=2,
        v_hidden_dims=0,
        dsa_hidden_dims=0,
    )
    obj = NS(
        num_layers=2,
        kvcaches=[object(), object()],
        kv_device=torch.device("cpu"),
        store_stream=stream,
        _sparse_direct_validated_layers=set(),
        initialize_kvcaches_ptr=lambda **kw: None,
        _lazy_initialize_buffer_with_staging=lambda *a, **kw: layout,
        supports_batched_from_gpu_group=lambda g: True,
        _check_layerwise_transfer_invariants=lambda **kw: None,
        _slot_mapping_on_kv_device=lambda slots, s: slots,
        _expected_memory_format=lambda g: "fmt",
        _layerwise_token_major=lambda g: True,
        _sparse_lmc_host_interleaved=lambda g: True,
        _prepare_dense_direct_chunk_metadata=lambda *a, **kw: (
            4,
            torch.empty(1),
            torch.empty(1),
        ),
        _get_or_create_sparse_direct_layer_state=lambda **kw: (
            object(),
            kw["layer_id"],
        ),
        _stream_context_or_null=lambda s: nullcontext(),
    )
    for name in names:
        setattr(obj, name, MethodType(ns[name], obj))
    rows = [[NS(tensor=torch.zeros(8), metadata=NS(fmt="fmt"))] for _ in range(2)]
    return obj, rows, events, calls


def test_ordinary_group_store_keeps_its_existing_blocking_dispatch():
    obj, rows, events, calls = fixture()
    host, ptrs = obj.batched_from_gpu_group(
        rows, [0], [4], slot_mapping=torch.arange(4), kv_group=0
    )
    assert host == [[10], [20]]
    assert ptrs.shape == (2, 1)
    assert len(calls) == 1 and calls[0][1]["fixed_chunk_size"] == 4
    assert events == ["dependency", "dependency", "sync"]


def test_checkpoint_enqueue_does_not_wait_or_populate_warm_state_cache():
    obj, rows, events, calls = fixture()
    obj._get_or_create_sparse_direct_layer_state = lambda **kw: pytest.fail(
        "warm state cache polluted"
    )
    plan = obj.prepare_group_capture(
        rows, [0], [4], slot_mapping=torch.arange(4), kv_group=0
    )
    assert events == [] and calls == []
    event = obj.enqueue_group_capture(plan)
    assert event is not None
    assert events == ["dependency", "native", "event"]
    assert not obj._sparse_direct_validated_layers
    assert len(plan.host_metadata) == 4


def test_fragmented_group_capture_builds_one_pointer_matrix_without_extra_fences():
    obj, rows, events, calls = fixture()
    rows = [[row[0], row[0], row[0]] for row in rows]
    plan = obj.prepare_group_capture(
        rows,
        [8, 12, 16],
        [12, 16, 18],
        slot_mapping=torch.arange(10),
        slot_mapping_base=8,
        kv_group=0,
    )
    assert plan.pointers.shape == (2, 3)
    assert plan.offsets.tolist() == [0, 4, 8]
    assert plan.sizes.tolist() == [4, 4, 2]
    assert len(plan.states) == 2 and events == []
    obj.enqueue_group_capture(plan)
    assert events == ["dependency", "native", "event"]
