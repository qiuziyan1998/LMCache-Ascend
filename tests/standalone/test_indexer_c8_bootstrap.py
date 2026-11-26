# SPDX-License-Identifier: Apache-2.0
"""Full-prefix Group-1 loads through the sparse-generator interface."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_bootstrap_retains_prefix_and_registers_failure_fence(
    monkeypatch, joined, fail
):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {"_run_c8_indexer_bootstrap", "_validate_sparse_fixed_chunk_coverage"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(
        ast.ClassDef(
            name="Connector", bases=[], keywords=[], body=methods, decorator_list=[]
        )
    )
    compute = SimpleNamespace()
    load = SimpleNamespace(wait_stream=Mock())
    native = Mock(side_effect=RuntimeError("launch") if fail else None)
    ns = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: compute)),
        lmc_ops=SimpleNamespace(indexer_c8_transfer_prepared=native),
    )
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    connector = ns["Connector"]()
    join = (
        SimpleNamespace(used_stream_indices=set(), compute_stream=compute)
        if joined
        else None
    )
    connector._active_sparse_load_join = join
    connector.load_stream_list = [load]
    connector._stream_context_or_null = lambda _: nullcontext()
    dummy = torch.empty(1, dtype=torch.int32)
    connector._dense_direct_dummy_metadata_tensor = lambda: dummy
    retained = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: retained.append((tensor, stream)),
    )
    plan = SimpleNamespace(states=[object()])
    pointers = torch.tensor([100, 200])
    slots = torch.arange(17)
    if fail:
        with pytest.raises(RuntimeError, match="launch"):
            connector._run_c8_indexer_bootstrap(plan, pointers, 0, slots, 9, 8, False)
    else:
        connector._run_c8_indexer_bootstrap(plan, pointers, 0, slots, 9, 8, False)
    args = native.call_args.args
    assert args[0] is plan.states[0] and args[1] is pointers
    assert args[2] is args[3] is dummy
    assert torch.equal(args[4], slots[:9])
    assert args[5:] == (8, False)
    assert native.call_args.kwargs == {"fixed_chunks": True}
    assert len(retained) == 3
    assert all(stream is (load if joined else compute) for _, stream in retained)
    if joined:
        assert join.used_stream_indices == {0}
        load.wait_stream.assert_called_once_with(compute)
    else:
        load.wait_stream.assert_not_called()
    native.reset_mock()
    with pytest.raises(ValueError, match="implicit full-prefix"):
        connector._run_c8_indexer_bootstrap(plan, pointers, 0, slots, 9, 8, True)
    native.assert_not_called()


@pytest.mark.parametrize("capacities,valid", [((4,), 1), ((4, 3), 5), ((4, 4), 5)])
def test_compact_c8_uses_physical_scale_offset_without_copying_unconsumed_rows(
    monkeypatch, capacities, valid
):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {"_run_c8_indexer_bootstrap", "_validate_sparse_fixed_chunk_coverage"}
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(
        ast.ClassDef(
            name="Connector",
            bases=[],
            keywords=[],
            body=[
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            decorator_list=[],
        )
    )
    stream = object()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: stream), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    # Byte-faithful simulation of the actual kernel's packet + count*key_bytes
    # scale address. Only the device launch is injected here.
    packets = [
        torch.cat(
            (
                torch.full((count * 2,), 11, dtype=torch.uint8),
                torch.full((count * 2,), 77, dtype=torch.uint8),
            )
        )
        for count in capacities
    ]
    captured = []
    outputs = torch.full((valid, 2), 255, dtype=torch.uint8)

    def native(
        state, pointers, offsets, counts, slots, capacity, direction, fixed_chunks=False
    ):
        assert not fixed_chunks and direction is False
        captured.append((offsets, counts, slots))
        for chunk, count in enumerate(counts.tolist()):
            for row in range(count):
                slot = slots[offsets[chunk].item() + row].item()
                if slot >= 0:
                    outputs[slot].copy_(
                        packets[chunk][count * 2 + row * 2 : count * 2 + (row + 1) * 2]
                    )

    ns = dict(torch=torch, lmc_ops=SimpleNamespace(indexer_c8_transfer_prepared=native))
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    connector = ns["Connector"]()
    connector._stream_context_or_null = lambda _: nullcontext()
    connector._dense_direct_dummy_metadata_tensor = lambda: torch.empty(
        1, dtype=torch.int32
    )
    plan = SimpleNamespace(states=[object()])
    cache = {}
    for _ in range(2):
        connector._run_c8_indexer_bootstrap(
            plan,
            torch.arange(len(capacities)),
            0,
            torch.arange(valid),
            valid,
            4,
            False,
            physical_chunk_token_counts=capacities,
            metadata_cache=cache,
        )
    assert torch.all(outputs == 77)
    assert captured[0][1].tolist() == list(capacities)
    assert captured[0][2].tolist() == list(range(valid)) + [-1] * (
        sum(capacities) - valid
    )
    assert all(
        first is second for first, second in zip(captured[0], captured[1], strict=True)
    )
