# SPDX-License-Identifier: Apache-2.0
"""CPU tests for graph source tables; native capture is tested separately."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call as mock_call

import pytest
import torch


@pytest.fixture
def transfer_module(monkeypatch):
    calls = []
    modules = {
        "lmcache.v1.gpu_connector.sparse": {"PreparedSparseSource": object},
        "lmcache_ascend.v1.kv_format": {
            "KVCacheFormat": SimpleNamespace(DSA_INDEX=SimpleNamespace(value=6))
        },
        "lmcache_ascend.v1.npu_connector.utils": {
            "prepare_sparse_direct_destination_state": lambda caches, *args: caches[0],
            "sparse_mla_dsa_batched_direct_kv_transfer_prepared": lambda *args: (
                calls.append(args)
            ),
        },
    }
    for name, attrs in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/sparse_graph.py"
    )
    spec = importlib.util.spec_from_file_location("tested_sparse_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def make_source(ptrs, counts, total=None):
    return SimpleNamespace(
        chunk_token_counts=tuple(counts),
        total_tokens=sum(counts) if total is None else total,
        layers=(SimpleNamespace(chunk_ptrs_npu=torch.tensor(ptrs, dtype=torch.int64)),),
    )


def make_transfer(module):
    return module.SparseGraphTransfer(
        (torch.zeros((2, 16, 1, 512)), torch.zeros((2, 16, 1, 64))),
        torch.zeros((1, 4), dtype=torch.int64),
        256,
        1024,
    )


def test_tail_growth_and_request_replacement_keep_addresses(transfer_module):
    module, _ = transfer_module
    transfer = make_transfer(module)
    addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
    transfer.bind(make_source([1000], [13]), 0)
    assert transfer.ptrs[1, 0] == 1000 + 13 * 512 * 4
    transfer.bind(make_source([2000, 3000], [256, 17]), 0)
    assert transfer.ptrs[1, 1] == 3000 + 17 * 512 * 4
    assert transfer.valid_tokens == 273
    transfer.bind(make_source([9000], [256]), 0)
    assert torch.equal(transfer.ptrs[0], torch.tensor([9000, 0, 0, 0]))
    assert addresses == (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())


def test_device_selection_masks_invalid_tokens_and_empty_source(transfer_module):
    module, calls = transfer_module
    transfer = make_transfer(module)
    transfer.bind(make_source([1000], [13]), 0)
    selected = torch.tensor([[0, 12, 13, -1]])
    counts = torch.tensor([4])
    slots = torch.tensor([[1, 2, 3, 4]])
    transfer.load(selected, counts, slots)
    assert len(calls) == 2  # K and PE, not two graph replays.
    for call in calls:
        assert call[2].dtype == torch.int32 and call[-1].dtype == torch.int32
        assert torch.equal(call[1], torch.tensor([[1, 2, -1, -1]]))
        assert torch.equal(call[2], torch.tensor([[0, 12, 0, 0]], dtype=torch.int32))
    transfer.clear_source()
    transfer.load(selected, counts.zero_(), slots)
    assert calls[-1][1].eq(-1).all()


@pytest.mark.parametrize(
    "counts,total",
    [([13, 256], 269), ([256] * 5, 1280), ([257], 257), ([256], 257), ([], 0)],
)
def test_invalid_source_is_rejected_before_replay(transfer_module, counts, total):
    module, _ = transfer_module
    transfer = make_transfer(module)
    with pytest.raises(ValueError, match="bounded chunk prefix"):
        transfer.bind(make_source([1000] * len(counts), counts, total), 0)


def test_native_graph_setup_initializes_npu_before_pinned_allocation():
    """Check the native test's bootstrap order without importing torch_npu."""
    path = Path(__file__).with_name("test_sparse_graph_transfer_npu.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    native_test = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_one_capture_replays_live_topk_and_growing_cpu_history"
    )
    # Execute the actual setup through the first pinned allocation. Imports
    # and NPU calls are stubbed; the real device behavior stays in the NPU test.
    setup_statements = []
    for statement in native_test.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        setup_statements.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "allocator"
            for target in statement.targets
        ):
            break
    else:
        pytest.fail("Native graph test has no pinned allocator setup")

    runtime = Mock()
    runtime.device.return_value = "npu:0"
    namespace = {
        "__file__": str(path),
        "Path": Path,
        "sys": SimpleNamespace(path=[]),
        "torch": runtime,
        "ensure_ascend_host_memory_registered": Mock(),
        "PinMemoryAllocator": runtime.host_allocator,
    }
    setup = ast.Module(body=setup_statements, type_ignores=[])
    exec(compile(setup, str(path), "exec"), namespace)
    runtime.assert_has_calls(
        [
            mock_call.npu.set_device("npu:0"),
            mock_call.zeros(1, device="npu:0"),
            mock_call.npu.synchronize(),
            mock_call.host_allocator(64 * 1024 * 1024),
        ]
    )
