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
            "KVCacheFormat": SimpleNamespace(MLA_LATENT=SimpleNamespace(value=5))
        },
        "lmcache_ascend.v1.npu_connector.utils": {
            "prepare_sparse_direct_destination_state": lambda caches, *args: tuple(
                caches
            ),
            "sparse_graph_kv_transfer": lambda *args: calls.append(args),
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "counts,total",
    [([1], 1), ([256], 17), ([256, 1], 257), ([256, 255], 500), ([256, 256], 263)],
)
def test_bind_uses_device_pointer_arithmetic_without_count_upload(
    transfer_module,
    monkeypatch,
    dtype,
    counts,
    total,
):
    module, _ = transfer_module
    transfer = module.SparseGraphTransfer(
        (
            torch.zeros((2, 16, 1, 512), dtype=dtype),
            torch.zeros((2, 16, 1, 64), dtype=dtype),
        ),
        torch.zeros((1, 4), dtype=torch.int64),
        256,
        1024,
    )
    source = make_source(
        [2**55 + i * 1234567 for i in range(len(counts))], counts, total
    )
    expected = source.layers[0].chunk_ptrs_npu + torch.tensor(counts) * transfer.k_bytes
    addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
    with monkeypatch.context() as patcher:
        patcher.setattr(
            module.torch,
            "tensor",
            Mock(side_effect=AssertionError("CPU count tensor upload")),
        )
        transfer.bind(source, 0)
    torch.testing.assert_close(
        transfer.ptrs[1, : len(counts)], expected, rtol=0, atol=0
    )
    assert transfer.valid_tokens.item() == total
    assert transfer.ptrs[:, len(counts) :].eq(0).all()
    assert addresses == (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())


def test_load_passes_live_inputs_once_without_tensor_preprocessing(transfer_module):
    module, calls = transfer_module
    transfer = make_transfer(module)
    transfer.bind(make_source([1000], [13]), 0)
    selected = torch.tensor([[0, 12, 13, -1]])
    counts = torch.tensor([4], dtype=torch.int32)
    slots = torch.tensor([[1, 2, 3, 4]])
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoTensorOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            pytest.fail(f"Graph load added a preprocessing operator: {func}")

    with NoTensorOps():
        transfer.load(selected, counts, slots)
    assert len(calls) == 1
    state, sent_slots, sent_selected, sent_counts, ptrs, limits, chunk = calls[0]
    assert state is transfer.state and len(state) == 2
    assert sent_slots is slots and sent_selected is selected and sent_counts is counts
    assert ptrs is transfer.ptrs and limits is transfer.valid_tokens and chunk == 256
    # Invalid/empty selections are interpreted by the fused device kernel, not
    # by separately launched masks/conversions. No host data read is required.
    transfer.clear_source()
    transfer.load(selected, counts.zero_(), slots)
    assert calls[-1][4] is transfer.ptrs and calls[-1][5].eq(0).all()
    assert len(calls) == 2


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


@pytest.mark.parametrize("capacity", [4, 8, 12, 16])
def test_batch_lanes_never_read_another_request_or_padding(transfer_module, capacity):
    module, calls = transfer_module
    transfer = module.SparseGraphTransfer(
        (torch.zeros((32, 16, 1, 512)), torch.zeros((32, 16, 1, 64))),
        torch.zeros((capacity, 4), dtype=torch.int64),
        256,
        1024,
        request_capacity=capacity,
    )
    source_a = make_source([1000], [13])
    source_b = make_source([2000, 3000], [256, 17])
    addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
    transfer.bind_batch((source_b, None, source_a), 0)
    assert transfer.ptrs[0, 0] == 2000
    assert transfer.ptrs[0, 8] == 1000
    selected = torch.tensor([[0, 12, 256, 273]] * capacity)
    slots = torch.arange(capacity * 4).reshape(capacity, 4)
    counts = torch.full((capacity, 16), 4, dtype=torch.int32)
    transfer.load(selected, counts, slots)
    assert len(calls) == 1
    sent = calls[-1]
    assert sent[1] is slots and sent[2] is selected and sent[3] is counts
    assert sent[4] is transfer.ptrs and sent[5] is transfer.valid_tokens
    assert sent[5].view(-1).tolist() == [273, 0, 13] + [0] * (capacity - 3)
    assert transfer.ptrs[1, 8] == 1000 + 13 * 512 * 4
    transfer.bind_batch((source_a, source_b), 0)
    assert transfer.valid_tokens[2:].eq(0).all()
    assert addresses == (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
    transfer.clear_source()
    transfer.load(selected, counts, slots)
    assert calls[-1][5].eq(0).all()
