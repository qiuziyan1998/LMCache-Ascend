# SPDX-License-Identifier: Apache-2.0
"""CPU tests for graph source tables; native capture is tested separately."""

import importlib.util
from itertools import count
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


SOURCE_IDS = count(1)


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
        binding_id=next(SOURCE_IDS),
        chunk_token_counts=tuple(counts),
        total_tokens=sum(counts) if total is None else total,
        layers=(SimpleNamespace(chunk_ptrs_npu=torch.tensor(ptrs, dtype=torch.int64)),),
    )


def make_transfer(module, request_capacity=1):
    return module.SparseGraphTransfer(
        (torch.zeros((2, 16, 1, 512)), torch.zeros((2, 16, 1, 64))),
        torch.zeros((request_capacity, 4), dtype=torch.int64),
        256,
        1024,
        request_capacity=request_capacity,
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


def test_batch_sources_use_disjoint_virtual_chunk_ranges(transfer_module):
    module, calls = transfer_module
    transfer = make_transfer(module, request_capacity=3)
    transfer.bind_batch(
        (
            make_source([1000], [13]),
            make_source([2000, 3000], [256, 17]),
        ),
        0,
    )
    selected = torch.tensor(
        [[0, 12, 13, -1], [0, 256, 272, 273], [0, 1, 2, 3]]
    )
    counts = torch.tensor([4, 4, 4])
    slots = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    transfer.load(selected, counts, slots)

    assert torch.equal(
        transfer.ptrs[0],
        torch.tensor(
            [1000, 0, 0, 0, 2000, 3000, 0, 0, 0, 0, 0, 0]
        ),
    )
    for call in calls:
        assert torch.equal(
            call[2],
            torch.tensor(
                [[0, 12, 0, 0], [1024, 1280, 1296, 0], [0, 0, 0, 0]],
                dtype=torch.int32,
            ),
        )
        assert torch.equal(
            call[1],
            torch.tensor([[1, 2, -1, -1], [5, 6, 7, -1], [-1, -1, -1, -1]]),
        )
        assert torch.equal(call[-1], torch.tensor([4, 4, 0], dtype=torch.int32))


@pytest.mark.parametrize(
    "counts,total",
    [([13, 256], 269), ([256] * 5, 1280), ([257], 257), ([256], 257), ([], 0)],
)
def test_invalid_source_is_rejected_before_replay(transfer_module, counts, total):
    module, _ = transfer_module
    transfer = make_transfer(module)
    with pytest.raises(ValueError, match="bounded chunk prefix"):
        transfer.bind(make_source([1000] * len(counts), counts, total), 0)
