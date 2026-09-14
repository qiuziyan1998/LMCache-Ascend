# SPDX-License-Identifier: Apache-2.0
"""The boundary loader uses existing dense transfers and borrows owned pages."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from test_local_checkpoint_restore import implementation


def engine(consumer):
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"load_checkpoint_resident_tail"},
        object,
    )
    obj = cls()
    obj.num_layers = 2
    obj.config = NS(chunk_size=4)
    obj.gpu_connector = NS(batched_to_gpu=consumer)
    return obj


def test_one_boundary_uses_dense_stream_without_reallocating_cpu_pages():
    pages, received, calls = [object(), object()], [], []
    event = object()

    def consumer(starts, ends, **kw):
        calls.append((starts, ends, kw))
        received.append((yield))
        received.append((yield))
        kw["_dense_load_readiness_out"].append(event)
        yield
        yield

    obj = engine(consumer)
    state = NS(
        cached_starts=[0, 4],
        cached_ends=[4, 7],
        cached_memory_objs=[[object(), pages[0]], [object(), pages[1]]],
    )
    slots, caches = torch.tensor([12, 13, 14]), [object(), object()]
    assert obj.load_checkpoint_resident_tail(state, slots, caches) is event
    assert received == [[pages[0]], [pages[1]]]
    starts, ends, kw = calls[0]
    assert starts == [0] and ends == [3]
    assert kw["slot_mapping"] is slots and kw["kvcaches"] is caches
    assert kw["sync"] and kw["kv_group"] == 0


def test_invalid_boundary_fails_before_device_submission():
    obj = engine(lambda *a, **kw: pytest.fail("device submission"))
    state = NS(cached_starts=[5], cached_ends=[7], cached_memory_objs=[[object()]] * 2)
    with pytest.raises(ValueError, match="incomplete CPU coverage"):
        obj.load_checkpoint_resident_tail(state, torch.tensor([1, 2]), [])


@pytest.mark.parametrize("rows", [[2, 1], [2, 3]])
def test_ragged_rows_cannot_substitute_another_chunk_for_the_boundary(rows):
    obj = engine(
        lambda *a, **kw: pytest.fail("invalid source reached device submission")
    )
    state = NS(
        cached_starts=[0, 4],
        cached_ends=[4, 7],
        cached_memory_objs=[[object()] * n for n in rows],
    )
    with pytest.raises(ValueError, match="incomplete CPU coverage"):
        obj.load_checkpoint_resident_tail(
            state, torch.tensor([12, 13, 14]), [object()] * 2
        )


def test_boundary_reuses_resolved_host_pointers():
    calls = []
    event = object()
    resolver_type = implementation(
        "lmcache_ascend/v1/npu_connector/npu_connectors.py",
        "VLLMPagedMemLayerwiseNPUConnector",
        {"_resolve_sparse_chunk_ptrs_npu"},
        object,
    )
    resolver = resolver_type()
    resolver.kv_device = torch.device("cpu")
    resolver._stream_context_or_null = lambda stream: nullcontext()
    resolver._resolve_registered_cpu_source_device_ptr = lambda *a, **kw: pytest.fail(
        "resolved a retained source pointer again"
    )

    def consumer(starts, ends, **kw):
        for layer in range(2):
            sources = yield
            row = resolver._resolve_sparse_chunk_ptrs_npu(
                layer,
                [],
                cached_chunk_dev_ptrs=kw.get("cached_chunk_dev_ptrs"),
                expected_num_chunks=1,
                source_objs=sources,
            )
            calls.append(row.tolist())
        kw["_dense_load_readiness_out"].append(event)
        yield
        yield

    obj = engine(consumer)
    pointers = [[11, 22], [33, 44]]
    state = NS(
        cached_starts=[0, 4],
        cached_ends=[4, 7],
        cached_memory_objs=[[object(), object()] for _ in range(2)],
        cached_chunk_dev_ptrs=pointers,
    )
    assert (
        obj.load_checkpoint_resident_tail(
            state, torch.tensor([12, 13, 14]), [object()] * 2
        )
        is event
    )
    assert calls == [[22], [44]]
    assert pointers == [[11, 22], [33, 44]]


@pytest.mark.parametrize("pointers", [[[11, 22]], [[11, 22], [33]]])
def test_partial_pointer_cache_keeps_the_existing_resolution_fallback(pointers):
    event = object()

    def consumer(*a, **kw):
        assert kw["cached_chunk_dev_ptrs"] is None
        yield
        yield
        kw["_dense_load_readiness_out"].append(event)
        yield
        yield

    obj = engine(consumer)
    state = NS(
        cached_starts=[0, 4],
        cached_ends=[4, 7],
        cached_memory_objs=[[object(), object()] for _ in range(2)],
        cached_chunk_dev_ptrs=pointers,
    )
    assert (
        obj.load_checkpoint_resident_tail(
            state, torch.tensor([12, 13, 14]), [object()] * 2
        )
        is event
    )
