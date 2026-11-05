# SPDX-License-Identifier: Apache-2.0
"""Check C8 dispatch ordering/lifetimes without simulating native copies."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("fixed", [0, 1024])
@pytest.mark.parametrize("from_npu", [False, True])
@pytest.mark.parametrize("shared_stream", [False, True])
def test_c8_dispatch_reuses_metadata_and_orders_streams(fixed, from_npu, shared_stream):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_run_dense_direct_kv_transfer_layer"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append((self.name, "wait", other.name))

    class Tensor:
        def __init__(self, name, size):
            self.name, self.size, self.device = name, size, "npu:0"

        def numel(self):
            return self.size

        def record_stream(self, stream):
            events.append((self.name, "retain", stream.name))

    native = Mock(side_effect=lambda *args, **kw: events.append("copy"))
    ns = {"lmc_ops": SimpleNamespace(indexer_c8_transfer_prepared=native)}
    exec(compile(tree, str(source), "exec"), ns)
    current = Stream("compute")
    transfer = current if shared_stream else Stream("transfer")
    plan = SimpleNamespace(states=[object()])
    impl = SimpleNamespace(
        indexer_c8_layout=object(),
        _stream_context_or_null=lambda _: nullcontext(),
        _get_or_create_sparse_destination_plan=Mock(return_value=plan),
    )
    slots, pointers = Tensor("slots", 1025), Tensor("pointers", 2)
    offsets, counts = Tensor("offsets", 2), Tensor("counts", 2)
    ns[method.name](
        impl,
        kvcaches_ref=[(Tensor("keys", 1), Tensor("scales", 1))],
        kv_group=1,
        layer_id=0,
        transfer_stream=transfer,
        current_stream=current,
        slot_mapping_full=slots,
        chunk_ptrs_npu=pointers,
        chunk_offsets_npu=offsets,
        chunk_sizes_npu=counts,
        total_tokens=1025,
        fixed_chunk_size=fixed,
        dense_kv_format=0,
        dense_token_major=False,
        dense_vllm_two_major=False,
        dense_k_hidden_dims=130,
        dense_v_hidden_dims=0,
        dense_dsa_hidden_dims=130,
        dense_host_interleaved=False,
        layer_tensors=[],
        direction=from_npu,
        destination_plan=None if from_npu else plan,
        c8_chunk_capacity=1024,
    )
    assert native.call_args.args == (
        plan.states[0],
        pointers,
        offsets,
        counts,
        slots,
        1024,
        from_npu,
    )
    assert native.call_args.kwargs == {"fixed_chunks": bool(fixed)}
    assert impl._get_or_create_sparse_destination_plan.call_count == int(from_npu)
    retained = [
        (name, "retain", transfer.name)
        for name in ("pointers", "slots", "offsets", "counts")
    ]
    expected = retained + (
        ["copy"]
        if shared_stream
        else [("transfer", "wait", "compute"), "copy", ("compute", "wait", "transfer")]
    )
    assert events == expected


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_completion_is_published_only_after_group_submission(c8, fail):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "enqueue_group_capture"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    events = []

    def submit(*args, **kw):
        events.append("submit")
        if fail:
            raise RuntimeError("submission failed")

    c8_copy, legacy_copy = Mock(side_effect=submit), Mock(side_effect=submit)
    event = SimpleNamespace(record=lambda stream: events.append("record"))
    npu = SimpleNamespace(current_stream=lambda: "compute", Event=lambda: event)
    ns = dict(
        torch=SimpleNamespace(npu=npu),
        lmc_ops=SimpleNamespace(
            indexer_c8_group_transfer_prepared=c8_copy,
            dense_mla_dsa_group_direct_kv_transfer_prepared=legacy_copy,
        ),
    )
    exec(compile(tree, str(source), "exec"), ns)
    plan = SimpleNamespace(
        c8_group=object() if c8 else None,
        c8_chunk_capacity=7,
        states=[],
        slots=object(),
        pointers=object(),
        offsets=object(),
        sizes=object(),
        total_tokens=7,
        interleaved=False,
        validate=False,
        fixed_chunk_size=0,
    )
    impl = SimpleNamespace(
        store_stream=SimpleNamespace(wait_stream=lambda _: events.append("wait")),
        _stream_context_or_null=lambda _: nullcontext(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="submission failed"):
            ns[method.name](impl, plan)
    else:
        assert ns[method.name](impl, plan) is event
    assert c8_copy.call_count == int(c8)
    assert legacy_copy.call_count == int(not c8)
    assert events == (["wait", "submit"] if fail else ["wait", "submit", "record"])
