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


def make_transfer(module, capacity=1):
    return module.SparseGraphTransfer(
        (torch.zeros((2, 16, 1, 512)), torch.zeros((2, 16, 1, 64))),
        torch.zeros((capacity, 4), dtype=torch.int64),
        256,
        1024,
        request_capacity=capacity,
    )


def test_aiv_limit_is_per_launch_and_serial_default_is_uncapped(transfer_module, monkeypatch):
    module, _ = transfer_module
    native = Mock()
    monkeypatch.setattr(module, "sparse_graph_kv_transfer", native)
    transfer = make_transfer(module)
    selected, counts, slots = torch.zeros((1, 4), dtype=torch.int32), torch.ones(1, dtype=torch.int32), torch.zeros((1, 4), dtype=torch.int64)
    transfer.load(selected, counts, slots)
    transfer.load(selected, counts, slots, max_aiv_cores=12)
    transfer.load(selected, counts, slots)
    assert [call.kwargs for call in native.call_args_list] == [{}, {"max_aiv_cores": 12}, {}]


def test_graph_transfer_wrapper_preserves_default_native_signature():
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/npu_connector/utils.py"
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "sparse_graph_kv_transfer")
    native = Mock()
    ns = {"torch": torch, "lmc_ops": SimpleNamespace(sparse_graph_kv_transfer=native)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    arguments = (object(),) * 6 + (1024,)
    ns[fn.name](*arguments)
    ns[fn.name](*arguments, max_aiv_cores=12)
    assert native.call_args_list == [mock_call(*arguments), mock_call(*arguments, 12)]


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


@pytest.mark.parametrize("capacity", [1, 4, 16])
def test_incremental_tables_match_full_binding_through_transitions(
    transfer_module, capacity
):
    import random

    module, _ = transfer_module
    full, delta = make_transfer(module, capacity), make_transfer(module, capacity)
    addresses = delta.ptrs.data_ptr(), delta.valid_tokens.data_ptr()
    rng = random.Random(918)
    previous, sources = None, []
    used_delta = False
    for step in range(100):
        if not sources or rng.random() < 0.25:
            sources = [None] * rng.randrange(capacity + 1)
        elif rng.random() < 0.3:
            rng.shuffle(sources)
        if sources:
            lane = rng.randrange(len(sources))
            count = rng.randrange(1, 5)
            sources[lane] = (
                None
                if rng.random() < 0.2
                else make_source(
                    [10000 * (step + 1) + i * 1000 for i in range(count)],
                    [256] * (count - 1) + [rng.randrange(1, 257)],
                )
            )
        full.bind_batch(sources, 0)
        lanes = (
            None
            if previous is None
            else tuple(
                i
                for i in range(max(len(previous), len(sources)))
                if (previous[i] if i < len(previous) else None)
                is not (sources[i] if i < len(sources) else None)
            )
        )
        selected = None if lanes is None else delta.plan_bind_update(sources, lanes)
        used_delta |= selected is not None
        delta.bind_batch(sources, 0, lanes=selected)
        torch.testing.assert_close(delta.ptrs, full.ptrs, rtol=0, atol=0)
        torch.testing.assert_close(
            delta.valid_tokens, full.valid_tokens, rtol=0, atol=0
        )
        assert addresses == (delta.ptrs.data_ptr(), delta.valid_tokens.data_ptr())
        previous = tuple(sources)
    assert used_delta


@pytest.mark.parametrize("lanes", [(-1,), (4,), (1, 1), ("0",)])
def test_bad_delta_lanes_fail_before_writing(transfer_module, lanes):
    module, _ = transfer_module
    transfer = make_transfer(module, 4)
    transfer.ptrs.fill_(77)
    transfer.valid_tokens.fill_(33)
    with pytest.raises(ValueError, match="lane"):
        transfer.bind_batch([], 0, lanes=lanes)
    assert torch.all(transfer.ptrs == 77) and torch.all(transfer.valid_tokens == 33)


def test_delta_plan_avoids_more_writes_and_preserves_unchanged_lanes(transfer_module):
    module, _ = transfer_module
    transfer = make_transfer(module, 4)
    sources = [make_source([1000 + i], [256]) for i in range(4)]
    transfer.bind_batch(sources, 0)
    old = transfer.ptrs.clone()
    sources[1] = make_source([9000, 10000], [256, 17])
    assert transfer.plan_bind_update(sources, (1,)) == (1,)
    assert transfer.plan_bind_update(sources, (0, 1, 2, 3)) is None
    transfer.bind_batch(sources, 0, lanes=(1,))
    assert torch.equal(old[:, :4], transfer.ptrs[:, :4])
    assert torch.equal(old[:, 8:], transfer.ptrs[:, 8:])
    sources[1] = None
    transfer.bind_batch(sources, 0, lanes=(1,))
    assert transfer.ptrs[:, 4:8].eq(0).all() and transfer.valid_tokens[1] == 0
    assert transfer.plan_bind_update(sources, ()) == ()


def test_delta_validates_all_affected_sources_before_clearing(transfer_module):
    module, _ = transfer_module
    transfer = make_transfer(module, 4)
    transfer.ptrs.fill_(77)
    with pytest.raises(ValueError, match="bounded chunk prefix"):
        transfer.bind_batch(
            [None, make_source([1000, 2000], [17, 256])], 0, lanes=(0, 1)
        )
    assert transfer.ptrs.eq(77).all()


def test_single_lane_update_reduces_table_write_dispatches(transfer_module):
    from collections import Counter
    from torch.utils._python_dispatch import TorchDispatchMode

    module, _ = transfer_module
    transfer = make_transfer(module, 16)
    sources = [
        make_source([1000 + i * 100 + j for j in range(4)], [256] * 4)
        for i in range(16)
    ]
    operations = Counter()

    class Writes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) in {
                "aten.zero_.default",
                "aten.copy_.default",
                "aten.add.out",
                "aten.fill_.Scalar",
            }:
                operations[str(func)] += 1
            return func(*args, **(kwargs or {}))

    with Writes():
        transfer.bind_batch(sources, 0)
    assert sum(operations.values()) == 50
    sources[3] = make_source([9000, 10000, 11000, 12000], [256] * 4)
    lanes = transfer.plan_bind_update(sources, (3,))
    assert lanes == (3,)
    operations.clear()
    with Writes():
        transfer.bind_batch(sources, 0, lanes=lanes)
    assert sum(operations.values()) == 3
