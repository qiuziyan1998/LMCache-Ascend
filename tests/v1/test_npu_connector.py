# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from weakref import WeakSet
import threading

# Third Party
from lmcache_tests.v1.test_gpu_connector import (
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench,
)
from lmcache.v1.gpu_connector.sparse import (
    PreparedSparseSource,
    PreparedSparseSourceLayer,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    LayerPageSource,
    MemoryFormat,
    TensorMemoryAllocator,
)
import pytest
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
)
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
import lmcache_ascend.v1.cache_engine as ascend_cache_engine
import lmcache_ascend.v1.npu_connector.npu_connectors as npu_connectors
import lmcache_ascend.c_ops as lmc_ops


def test_layer_page_source_selects_requested_layer_and_suffix() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    suffix = allocator.allocate(
        torch.Size([4]),
        torch.float16,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert pages is not None and suffix is not None
    pages[0].layer_tensor(0).fill_(1)
    pages[0].layer_tensor(1).fill_(2)
    suffix.tensor.fill_(3)

    tensors = npu_connectors._layer_source_tensors(
        LayerPageSource(tuple(pages), 1, (suffix,)),
        1,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )

    assert [int(tensor[0]) for tensor in tensors] == [2, 3]
    with pytest.raises(ValueError, match="selects 0"):
        npu_connectors._layer_source_tensors(
            LayerPageSource(tuple(pages), 0),
            1,
            MemoryFormat.KV_MLA_LATENT_FMT,
        )
    pages[0].ref_count_down()
    suffix.ref_count_down()


def test_layer_page_pointer_resolution_uses_selected_layer(monkeypatch) -> None:
    allocator = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert pages is not None
    monkeypatch.setattr(lmc_ops, "get_device_ptr", lambda address: address + 7)
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)

    pointer = connector._resolve_registered_cpu_source_device_ptr(
        pages[0], layer_id=1, chunk_index=0, source="test"
    )

    assert pointer == pages[0].layer_data_ptr(1) + 7
    pages[0].ref_count_down()


def test_direct_page_planner_preserves_layer_plane_run_order() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    layout = SimpleNamespace(
        kv_format=npu_connectors.KVCacheFormat.MLA_LATENT,
        k_hidden_dims=2,
        v_hidden_dims=1,
        dsa_hidden_dims=0,
    )
    connector._group_layouts = {0: layout}
    connector._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout
    kvcaches = [
        (
            torch.empty((4, 4, 1, 2), dtype=torch.float16),
            torch.empty((4, 4, 1, 1), dtype=torch.float16),
        )
        for _ in range(2)
    ]
    slots = torch.tensor([0, 1, 4, 5], dtype=torch.long)

    planned = connector.plan_direct_page_sources(
        kvcaches, slots, [0], [4], kv_group=0
    )

    assert planned is not None
    ptrs, sizes, owners = planned
    assert owners == tuple(tensor for layer in kvcaches for tensor in layer)
    assert sizes == [[8, 8, 4, 4, 8, 8, 4, 4]]
    expected = []
    for tensor in owners:
        token_bytes = tensor[0, 0].numel() * tensor.element_size()
        expected.extend(
            [tensor.data_ptr(), tensor.data_ptr() + 4 * token_bytes]
        )
    assert ptrs == [expected]

    layer_ptrs, layer_sizes, _ = connector.plan_direct_page_sources(
        kvcaches, slots, [0], [4], kv_group=0, layerwise=True
    )
    assert layer_ptrs == [expected[:4], expected[4:]]
    assert layer_sizes == [[8, 8, 4, 4], [8, 8, 4, 4]]

    relative = connector.plan_direct_page_sources(
        kvcaches,
        slots,
        [256],
        [260],
        kv_group=0,
        slot_mapping_base=256,
    )
    assert relative is not None
    assert relative[:2] == planned[:2]


def test_direct_page_planner_rejects_invalid_slots() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    layout = SimpleNamespace(
        kv_format=npu_connectors.KVCacheFormat.DSA_INDEX,
        k_hidden_dims=2,
        v_hidden_dims=0,
        dsa_hidden_dims=2,
    )
    connector._group_layouts = {1: layout}
    connector._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout
    kvcaches = [(torch.empty((2, 4, 1, 2), dtype=torch.float16),)]

    assert (
        connector.plan_direct_page_sources(
            kvcaches,
            torch.tensor([0, -1], dtype=torch.long),
            [0],
            [2],
            kv_group=1,
        )
        is None
    )


def test_direct_page_planner_rejects_cache_dtype_mismatch() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.dtype = torch.bfloat16
    layout = SimpleNamespace(
        kv_format=npu_connectors.KVCacheFormat.DSA_INDEX,
        k_hidden_dims=2,
        v_hidden_dims=0,
        dsa_hidden_dims=2,
    )
    connector._group_layouts = {1: layout}
    connector._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout

    assert (
        connector.plan_direct_page_sources(
            [(torch.empty((1, 2, 1, 2), dtype=torch.float16),)],
            torch.arange(2),
            [0],
            [2],
            kv_group=1,
        )
        is None
    )


def test_failed_direct_future_keeps_request_pending_for_cpu_retry() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    state = ascend_cache_engine._DirectStoreRequestState()
    state.pending_keys.add("key")
    engine._direct_store_states = {"request": state}
    engine._pending_store_reqs = {"request": 1}
    engine._direct_completed_futures = WeakSet()
    lock = threading.Lock()
    engine._store_cv = threading.Condition(lock)
    future = Future()
    future.set_exception(RuntimeError("failed"))

    engine._direct_store_done("request", {"key"}, future)

    assert engine._pending_store_reqs == {"request": 1}
    assert state.pending_keys == {"key"}


def test_direct_completion_does_not_publish_out_of_order_frontier() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    state = ascend_cache_engine._DirectStoreRequestState(
        pending_keys={"key"}, submitted_end={0: 512}
    )
    engine._direct_store_states = {"request": state}
    engine._pending_store_reqs = {"request": 1}
    engine._direct_completed_futures = WeakSet()
    engine._store_cv = threading.Condition(threading.Lock())
    future = Future()
    future.set_result(None)

    engine._direct_store_done("request", {"key"}, future)

    assert state.committed_end == {}
    assert state.pending_keys == set()


def test_direct_cpu_retry_restores_submitted_frontier() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    state = ascend_cache_engine._DirectStoreRequestState(
        committed_end={0: 128}, submitted_end={0: 512}
    )
    engine._store_direct_cpu_group = lambda *args: 256

    engine._retry_direct_cpu(
        "request", [0] * 512, {0: [object()]}, {0: object()}, None, state
    )

    assert state.committed_end == {0: 256}
    assert state.submitted_end == {0: 256}


def test_final_cpu_fallback_requires_complete_group_coverage() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(save_unfull_chunk=True)
    state = ascend_cache_engine._DirectStoreRequestState(committed_end={0: 4})

    with pytest.raises(RuntimeError, match="did not cover"):
        engine._finalize_direct_store(
            "request", [0] * 5, (0,), state, final=True
        )

    state.committed_end[0] = 5
    engine._finalize_direct_store(
        "request", [0] * 5, (0,), state, final=True
    )
    assert state.finalized


def test_direct_failure_retries_each_group_in_submission_order() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    first, second = Future(), Future()
    first.set_exception(RuntimeError("latent failed"))
    second.set_exception(RuntimeError("indexer failed"))
    state = ascend_cache_engine._DirectStoreRequestState(
        futures=deque((first, second)),
        pending_keys={"latent", "indexer"},
        submitted_end={0: 4, 1: 6},
    )
    engine._direct_store_states = {"request": state}
    engine._pending_store_reqs = {"request": 2}
    engine._direct_completed_futures = WeakSet()
    engine._store_cv = threading.Condition(threading.Lock())
    engine.config = SimpleNamespace(blocking_timeout_secs=1)
    engine._direct_retry_args = {
        first: (
            "request",
            [0] * 4,
            {0: ["latent-cache"]},
            {0: "latent-slots"},
            None,
            0,
            {"latent"},
            {0: 4},
        ),
        second: (
            "request",
            [0] * 6,
            {1: ["indexer-cache"]},
            {1: "indexer-slots"},
            None,
            0,
            {"indexer"},
            {1: 6},
        ),
    }
    calls = []
    engine._store_direct_cpu_group = (
        lambda _req, tokens, _caches, _slots, group, _start, _configs, _base: (
            calls.append((group, len(tokens))) or len(tokens)
        )
    )

    engine.wait_for_direct_stores(("request",))

    assert calls == [(0, 4), (1, 6)]
    assert state.committed_end == {0: 4, 1: 6}
    assert state.pending_keys == set()


def test_direct_retry_replays_success_between_failed_windows() -> None:
    first, middle, tail = Future(), Future(), Future()
    first.set_exception(RuntimeError("first failed"))
    middle.set_result(None)
    tail.set_exception(RuntimeError("tail failed"))
    state = ascend_cache_engine._DirectStoreRequestState(
        futures=deque((first, middle, tail)),
        pending_keys={"first", "middle", "tail"},
        submitted_end={0: 5},
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine._direct_store_states = {"request": state}
    engine._pending_store_reqs = {"request": 3}
    engine._direct_completed_futures = WeakSet()
    engine._store_cv = threading.Condition(threading.Lock())
    engine.config = SimpleNamespace(blocking_timeout_secs=1)
    engine._direct_retry_args = {
        first: (
            "request", [0] * 2, {0: [0]}, {0: [0]}, None, 0, {"first"}, {0: 2}
        ),
        middle: (
            "request", [0] * 4, {0: [0]}, {0: [0]}, None, 2, {"middle"}, {0: 4}
        ),
        tail: (
            "request", [0] * 5, {0: [0]}, {0: [0]}, None, 4, {"tail"}, {0: 5}
        ),
    }
    starts = []
    engine._store_direct_cpu_group = (
        lambda _req, tokens, _cache, _slots, _group, start, _configs, base: (
            starts.append((start, base)) or len(tokens)
        )
    )

    engine.wait_for_direct_stores(("request",))

    assert starts == [(0, 0), (4, 4)]
    assert state.committed_end == {0: 5}


def test_direct_prefill_reuses_hashes_and_submits_both_groups_once(
    monkeypatch,
) -> None:
    class _Event:
        def record(self) -> None:
            pass

    class _TokenDatabase:
        calls = []

        def process_tokens(
            self, tokens=None, hashes=None, offsets=None, kv_group=0, **kwargs
        ):
            self.calls.append((kv_group, hashes, offsets))
            values = hashes or [11, 22]
            for index, value in enumerate(values):
                yield (
                    index * 2,
                    (index + 1) * 2,
                    CacheEngineKey(
                        "model", 1, 0, value, torch.float16, kv_group=kv_group
                    ),
                )

        def process_tokens_from_prefix(
            self, tokens, prefix_token_count, kv_group=0, **kwargs
        ):
            self.calls.append(("suffix", prefix_token_count, len(tokens)))
            yield (
                prefix_token_count,
                len(tokens),
                CacheEngineKey(
                    "model", 1, 0, 33, torch.float16, kv_group=kv_group
                ),
            )

    class _StorageManager:
        submissions = []

        @staticmethod
        def batched_external_pages_exist(keys):
            return [index % 2 == 1 for index in range(len(keys))]

        def batched_put_external_pages(self, *args):
            self.submissions.append(args)
            future = Future()
            future.set_result(None)
            return future

    class _GPUConnector:
        owner = torch.empty(16, dtype=torch.uint8)
        calls = []

        def plan_direct_page_sources(
            self, kvcaches, slot_mapping, starts, ends, kv_group, **kwargs
        ):
            self.calls.append((starts, ends, len(slot_mapping), kwargs))
            return (
                [[self.owner.data_ptr()] for _ in starts],
                [[2] for _ in starts],
                (self.owner,),
            )

    monkeypatch.setattr(torch.npu, "Event", _Event)
    engine = object.__new__(AscendLMCacheEngine)
    engine._direct_store_enabled = True
    engine._direct_store_states = {}
    engine._direct_store_jobs = deque()
    engine._direct_retry_args = {}
    engine._direct_completed_futures = WeakSet()
    engine._pending_store_reqs = {}
    engine._store_queue_maxsize = 2
    engine._store_cv = threading.Condition(threading.Lock())
    engine.config = SimpleNamespace(
        chunk_size=2,
        blocking_timeout_secs=5,
        get_extra_config_value=lambda name, default: default,
    )
    engine.num_layers = 1
    engine.token_database = _TokenDatabase()
    engine.storage_manager = _StorageManager()
    engine.gpu_connector = _GPUConnector()
    slots = torch.arange(4)

    assert engine.store_direct_prefill(
        "request",
        [1, 2, 3, 4],
        {0: [object()], 1: [object()]},
        {0: slots, 1: slots},
    )

    assert len(engine.storage_manager.submissions) == 1
    assert len(engine.storage_manager.submissions[0][0]) == 2
    assert engine.token_database.calls[1] == (1, [11, 22], [2, 2])
    engine.wait_for_direct_stores(("request",))
    assert engine._direct_store_states["request"].committed_end == {0: 4, 1: 4}

    assert engine.store_direct_prefill(
        "request",
        [1, 2, 3, 4, 5, 6],
        {0: [object()], 1: [object()]},
        {0: slots[:2], 1: slots[:2]},
        slot_mapping_base=4,
    )
    assert engine.gpu_connector.calls[-1] == (
        [4],
        [6],
        2,
        {"slot_mapping_base": 4},
    )
    engine.wait_for_direct_stores(("request",))
    assert engine._direct_store_states["request"].committed_end == {0: 6, 1: 6}


def test_direct_suffix_planning_hashes_only_new_complete_chunks() -> None:
    class _TokenDatabase:
        calls = []

        @staticmethod
        def _key(value, group):
            return CacheEngineKey(
                "model", 1, 0, value, torch.float16, kv_group=group
            )

        def process_tokens(
            self, tokens=None, hashes=None, offsets=None, kv_group=0, **kwargs
        ):
            self.calls.append(("full", kv_group, len(tokens or ()), hashes))
            values = [11, 22] if hashes is None else hashes
            for index, value in enumerate(values):
                yield index * 2, (index + 1) * 2, self._key(value, kv_group)

        def process_tokens_from_prefix(
            self, tokens, prefix_token_count, prefix_hash, kv_group=0, **kwargs
        ):
            self.calls.append(("suffix", prefix_token_count, prefix_hash))
            if prefix_token_count < len(tokens):
                yield prefix_token_count, len(tokens), self._key(33, kv_group)

    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=2)
    engine.token_database = _TokenDatabase()
    state = ascend_cache_engine._DirectStoreRequestState()

    engine._direct_suffix_plans(state, [1, 2, 3, 4], (0, 1), None)
    state.submitted_end = {0: 4, 1: 4}
    plans = engine._direct_suffix_plans(state, [1, 2, 3, 4, 5, 6], (0, 1), None)
    state.submitted_end = {0: 6, 1: 6}
    duplicate = engine._direct_suffix_plans(
        state, [1, 2, 3, 4, 5, 6], (0, 1), None
    )

    assert [(start, end) for start, end, _ in plans[0]] == [(4, 6)]
    assert [(start, end) for start, end, _ in plans[1]] == [(4, 6)]
    assert state.planned_end == 6 and state.planned_hash == 33
    assert ("suffix", 4, 22) in engine.token_database.calls
    assert duplicate == {0: [], 1: []}

    late_group = ascend_cache_engine._DirectStoreRequestState(
        submitted_end={0: 4}, planned_end=4, planned_hash=22
    )
    rebuilt = engine._direct_suffix_plans(
        late_group, [1, 2, 3, 4], (0, 1), None
    )
    assert [(start, end) for start, end, _ in rebuilt[1]] == [(0, 2), (2, 4)]


def test_direct_tail_uses_legacy_layer_keys(monkeypatch) -> None:
    class _Event:
        def record(self) -> None:
            pass

    class _TokenDatabase:
        @staticmethod
        def _key(value, group):
            return CacheEngineKey(
                "model", 1, 0, value, torch.float16, kv_group=group
            )

        def process_tokens_from_prefix(self, tokens, prefix_token_count, **kwargs):
            yield prefix_token_count, len(tokens), self._key(33, 0)

        def process_tokens(self, hashes, offsets, kv_group=0, **kwargs):
            yield 0, offsets[0], self._key(hashes[0], kv_group)

    class _StorageManager:
        submissions = []
        hit_groups = set()

        @staticmethod
        def batched_contains(keys, search_range=None):
            assert search_range == ["RemoteBackend"]
            return (
                len(keys) if keys[0].kv_group in _StorageManager.hit_groups else 0,
                {},
            )

        def batched_put_external_pages(self, *args):
            self.submissions.append(args)
            future = Future()
            future.set_result(None)
            return future

    monkeypatch.setattr(torch.npu, "Event", _Event)
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.token_database = _TokenDatabase()
    engine.storage_manager = _StorageManager()
    engine._direct_store_states = {
        "request": ascend_cache_engine._DirectStoreRequestState(
            planned_end=4, planned_hash=22
        )
    }
    engine._direct_store_jobs = deque()
    engine._direct_retry_args = {}
    engine._direct_completed_futures = WeakSet()
    engine._pending_store_reqs = {}
    engine._store_queue_maxsize = 2
    engine._store_cv = threading.Condition(threading.Lock())
    engine.config = SimpleNamespace(
        blocking_timeout_secs=5,
        get_extra_config_value=lambda name, default: default,
    )
    owner = torch.empty(1)

    def planner(*args, **kwargs):
        assert kwargs == {"layerwise": True, "slot_mapping_base": 0}
        return [[1], [2]], [[4], [4]], (owner,)

    assert engine._submit_direct_tail(
        "request",
        [1, 2, 3, 4, 5],
        {0: [object()], 1: [object()]},
        {0: torch.arange(5), 1: torch.arange(5)},
        None,
        engine._direct_store_states["request"],
        planner,
    )
    assert engine._submit_direct_tail(
        "request",
        [1, 2, 3, 4, 5],
        {0: [object()], 1: [object()]},
        {0: torch.arange(5), 1: torch.arange(5)},
        None,
        engine._direct_store_states["request"],
        planner,
    )
    submission = engine.storage_manager.submissions[0]
    assert len(engine.storage_manager.submissions) == 1
    assert len(submission[0]) == 4
    assert [key.layer_id for key in submission[0]] == [0, 1, 0, 1]
    assert [key.kv_group for key in submission[0]] == [0, 0, 1, 1]
    assert submission[-1] == "request"
    engine.wait_for_direct_stores(("request",))
    assert engine._direct_store_states["request"].committed_end == {0: 5, 1: 5}

    _StorageManager.hit_groups = {0}
    engine._direct_store_states["existing"] = (
        ascend_cache_engine._DirectStoreRequestState(planned_end=4, planned_hash=22)
    )
    assert engine._submit_direct_tail(
        "existing",
        [1, 2, 3, 4, 5],
        {0: [object()]},
        {0: torch.arange(5)},
        None,
        engine._direct_store_states["existing"],
        planner,
    )
    assert len(engine.storage_manager.submissions) == 1
    assert engine.direct_store_committed_ends("existing") == {0: 5}

    engine._direct_store_states["mixed"] = (
        ascend_cache_engine._DirectStoreRequestState(planned_end=4, planned_hash=22)
    )
    assert engine._submit_direct_tail(
        "mixed",
        [1, 2, 3, 4, 5],
        {0: [object()], 1: [object()]},
        {0: torch.arange(5), 1: torch.arange(5)},
        None,
        engine._direct_store_states["mixed"],
        planner,
    )
    retry = next(
        value for value in engine._direct_retry_args.values() if value[0] == "mixed"
    )
    assert retry[-1] == {1: 5}


def test_direct_prefill_skips_disabled_unfull_tail() -> None:
    class _TokenDatabase:
        calls = 0

        @classmethod
        def process_tokens(cls, **kwargs):
            cls.calls += 1
            return iter(())

    class _StorageManager:
        @staticmethod
        def batched_external_pages_exist(keys):
            assert not keys
            return []

    class _GPUConnector:
        @staticmethod
        def plan_direct_page_sources(*args, **kwargs):
            raise AssertionError("unaligned tail must not be planned")

    engine = object.__new__(AscendLMCacheEngine)
    engine._direct_store_enabled = True
    engine._direct_store_states = {}
    engine._direct_store_jobs = deque()
    engine._direct_retry_args = {}
    engine._direct_completed_futures = WeakSet()
    engine._pending_store_reqs = {}
    engine._store_queue_maxsize = 2
    engine._store_cv = threading.Condition(threading.Lock())
    engine.config = SimpleNamespace(
        chunk_size=2,
        save_unfull_chunk=False,
        blocking_timeout_secs=5,
        get_extra_config_value=lambda name, default: default,
    )
    engine.token_database = _TokenDatabase()
    engine.storage_manager = _StorageManager()
    engine.gpu_connector = _GPUConnector()
    engine._store_direct_cpu_group = lambda *args: (_ for _ in ()).throw(
        AssertionError("disabled tail must not use CPU fallback")
    )

    assert engine.store_direct_prefill(
        "request",
        [1],
        {0: [object()]},
        {0: torch.arange(1)},
        final=True,
    )
    assert engine.direct_store_committed_ends("request") == {0: 0}
    assert engine.store_direct_prefill(
        "request",
        [1],
        {0: [object()]},
        {0: torch.arange(1)},
        final=True,
    )
    assert engine._direct_store_states["request"].finalized
    assert _TokenDatabase.calls == 1

    assert engine.store_direct_prefill(
        "window",
        list(range(6)),
        {0: [object()]},
        {0: torch.arange(2)},
        slot_mapping_base=4,
    )
    assert engine.direct_store_committed_ends("window") == {0: 4}


def _make_layer_page_sources():
    allocator = TensorMemoryAllocator(torch.zeros(32768, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=2,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    suffix = [
        allocator.allocate(
            torch.Size([4]), torch.float16, MemoryFormat.KV_MLA_LATENT_FMT
        )
        for _ in range(2)
    ]
    assert pages is not None and all(obj is not None for obj in suffix)
    return allocator, pages, suffix, [
        LayerPageSource(tuple(pages), layer_id, (suffix[layer_id],))
        for layer_id in range(2)
    ]


def test_group_pointer_append_resolves_layer_pages_once(monkeypatch) -> None:
    _allocator, pages, suffix, sources = _make_layer_page_sources()
    connector = _make_sparse_pack_connector()
    connector.num_layers = 2
    calls = []
    monkeypatch.setattr(
        lmc_ops,
        "get_device_ptr",
        lambda address: calls.append(address) or address + 1000,
    )
    tensor_calls = 0
    original_tensor = torch.tensor

    def counted_tensor(*args, **kwargs):
        nonlocal tensor_calls
        tensor_calls += 1
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(npu_connectors.torch, "tensor", counted_tensor)
    host_rows, npu_rows = [], []

    connector.append_sparse_chunk_ptr_cache_for_layers(sources, host_rows, npu_rows)

    assert calls[:2] == [page.layer_data_ptr(0) for page in pages]
    assert len(calls) == len(pages) + len(suffix)
    assert host_rows == [
        [page.layer_data_ptr(layer) + 1000 for page in pages]
        + [suffix[layer].data_ptr + 1000]
        for layer in range(2)
    ]
    assert [row.tolist() for row in npu_rows] == host_rows
    assert (
        npu_rows[0].untyped_storage().data_ptr()
        == npu_rows[1].untyped_storage().data_ptr()
    )
    assert tensor_calls == 1
    for obj in [*pages, *suffix]:
        obj.ref_count_down()


def test_group_pointer_append_falls_back_for_legacy_rows(monkeypatch) -> None:
    connector = _make_sparse_pack_connector()
    connector.num_layers = 2
    calls = []
    monkeypatch.setattr(
        connector,
        "_resolve_registered_cpu_source_device_ptr",
        lambda source, *, layer_id, chunk_index, **_kwargs: calls.append(
            (source, layer_id, chunk_index)
        )
        or 100 * layer_id
        + chunk_index,
    )
    rows = [[object(), object()], [object(), object()]]
    host_rows = []

    connector.append_sparse_chunk_ptr_cache_for_layers(rows, host_rows, None)

    assert host_rows == [[0, 1], [100, 101]]
    assert len(calls) == 4


def test_group_pointer_append_falls_back_for_malformed_page_layout(
    monkeypatch,
) -> None:
    _allocator, pages, suffix, sources = _make_layer_page_sources()
    connector = _make_sparse_pack_connector()
    connector.num_layers = 2
    pages[0].group_prefix_sum = (0, pages[0].layer_size, pages[0].layer_size + 1)
    calls = []
    monkeypatch.setattr(
        connector,
        "_resolve_registered_cpu_source_device_ptr",
        lambda _source, *, layer_id, chunk_index, **_kwargs: calls.append(
            (layer_id, chunk_index)
        )
        or 100 * layer_id
        + chunk_index,
    )
    host_rows = []

    connector.append_sparse_chunk_ptr_cache_for_layers(sources, host_rows, None)

    assert host_rows == [[0, 1, 2], [100, 101, 102]]
    assert len(calls) == 6
    for obj in [*pages, *suffix]:
        obj.ref_count_down()


def _make_sparse_pack_connector() -> VLLMPagedMemLayerwiseNPUConnector:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    connector._layerwise_sparse_idx_cache = None
    return connector


def test_sparse_pointer_resolution_prefers_complete_npu_cache(monkeypatch) -> None:
    connector = _make_sparse_pack_connector()
    cached = torch.tensor([101, 202], dtype=torch.long)
    monkeypatch.setattr(
        connector,
        "_resolve_registered_cpu_source_device_ptr",
        lambda *_args, **_kwargs: pytest.fail("cached pointers were rebuilt"),
    )

    resolved = connector._resolve_sparse_chunk_ptrs_npu(
        0,
        [torch.zeros(1)],
        [cached],
        expected_num_chunks=2,
        cached_chunk_dev_ptrs=[[101, 202]],
    )

    assert resolved is cached


def test_sparse_pointer_resolution_rebuilds_npu_cache_from_host_row(
    monkeypatch,
) -> None:
    connector = _make_sparse_pack_connector()
    npu_rows = []
    host_rows = [[101, 202]]
    monkeypatch.setattr(
        connector,
        "_resolve_registered_cpu_source_device_ptr",
        lambda *_args, **_kwargs: pytest.fail("host pointers were rediscovered"),
    )

    resolved = connector._resolve_sparse_chunk_ptrs_npu(
        0,
        [torch.zeros(1)],
        npu_rows,
        expected_num_chunks=2,
        cached_chunk_dev_ptrs=host_rows,
    )

    assert resolved.tolist() == [101, 202]
    assert npu_rows[0] is resolved
    assert host_rows == [[101, 202]]


def test_sparse_pointer_cache_rejects_npu_row_without_host_coverage() -> None:
    connector = _make_sparse_pack_connector()

    with pytest.raises(RuntimeError, match="incomplete host coverage"):
        connector._resolve_sparse_chunk_ptrs_npu(
            0,
            [torch.zeros(1)],
            [torch.tensor([101, 202], dtype=torch.long)],
            expected_num_chunks=2,
            cached_chunk_dev_ptrs=[[101]],
        )


def test_bounded_stable_int_checksum_matches_ascend_producer() -> None:
    values = [12, -1, 999_999_999_999]

    checksum = npu_connectors._bounded_stable_int_checksum(values)

    assert checksum == 12039416201095166938
    assert npu_connectors._bounded_stable_int_checksum([]) == 14695981039346656037


def test_bounded_stable_int_checksum_uses_first_32_aggregate_values() -> None:
    values = list(range(40))

    assert npu_connectors._bounded_stable_int_checksum(
        values
    ) == npu_connectors._bounded_stable_int_checksum(values[:32] + [999] * 8)


def test_mtp_deep_diag_requires_both_gates(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
    monkeypatch.delenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", raising=False)
    assert not npu_connectors._mtp_dw_deep_diag_enabled()

    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
    assert npu_connectors._mtp_dw_deep_diag_enabled()

    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "0")
    assert not npu_connectors._mtp_dw_deep_diag_enabled()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"explicit_payload": False}, False),
        ({"committed_end": 0}, False),
        ({"req_id": None}, False),
        ({"seen": {("req", 0, 256): None}}, False),
    ],
)
def test_deep_payload_capture_is_first_successful_committed_payload(
    overrides, expected
) -> None:
    inputs = {
        "enabled": True,
        "explicit_payload": True,
        "committed_end": 256,
        "req_id": "req",
        "kv_group": 0,
        "seen": {},
    }
    inputs.update(overrides)

    assert npu_connectors._should_capture_deep_payload(**inputs) is expected


def test_deep_payload_capture_repeats_for_new_committed_frontier() -> None:
    inputs = {
        "enabled": True,
        "explicit_payload": True,
        "committed_end": 512,
        "req_id": "req",
        "kv_group": 0,
        "seen": {("req", 0, 256): None},
    }

    assert npu_connectors._should_capture_deep_payload(**inputs)


def test_conflicting_duplicate_target_slots_is_bounded_and_precise() -> None:
    conflicts = npu_connectors._conflicting_duplicate_target_slots(
        [10, 10, 11, 12, 13],
        [4, 4, 5, 5, 6],
    )

    assert conflicts == [{"slot": 5, "first_selected": 11, "selected": 12}]
    assert npu_connectors._conflicting_duplicate_target_slots(
        [10, 10], [4, 4]
    ) == []
    assert npu_connectors._conflicting_duplicate_target_slots(
        [10, 10], [4, 5]
    ) == []
    assert npu_connectors._conflicting_duplicate_target_slots(
        list(range(40)), list(range(39)) + [0]
    ) == [{"slot": 0, "first_selected": 0, "selected": 39}]


def test_remember_bounded_key_evicts_oldest_state(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_MTP_DW_DEEP_SEEN_LIMIT", 2)
    seen = {}

    npu_connectors._remember_bounded_key(seen, "oldest")
    npu_connectors._remember_bounded_key(seen, "middle")
    npu_connectors._remember_bounded_key(seen, "newest")

    assert list(seen) == ["middle", "newest"]


class _NoopStream:
    def wait_stream(self, stream):
        pass

    def synchronize(self):
        pass


class _RecordableTensor:
    def __init__(self, numel: int, dtype=torch.long):
        self._numel = numel
        self.dtype = dtype
        self.device = torch.device("cpu")

    def numel(self):
        return self._numel


class _TrackingStream:
    def __init__(self, name: str):
        self.name = name
        self.events = []

    def wait_stream(self, stream):
        self.events.append(("wait_stream", stream.name))

    def wait_event(self, event):
        self.events.append(("wait_event", event.name))

    def synchronize(self):
        self.events.append("synchronize")


class _TrackingEvent:
    def __init__(self, name: str):
        self.name = name
        self.records = []

    def record(self, stream):
        self.records.append(stream.name)


class _DenseLayout:
    k_hidden_dims = 1
    v_hidden_dims = 1
    dsa_hidden_dims = 0
    kv_format = type("_Fmt", (), {"value": 0})()
    vllm_two_major = False
    gpu_buffer_allocator = None


class _MemoryObj:
    def __init__(self, tensor, fmt=MemoryFormat.KV_MLA_LATENT_FMT):
        self.tensor = tensor
        self.metadata = type("_Metadata", (), {"fmt": fmt})()


def test_sparse_memory_update_resets_fast_direct_state() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = {(123, 0, 0): object()}
    connector._sparse_direct_validated_layers = {(0, 0)}
    destination_plan = object()
    connector._sparse_destination_plans = {0: destination_plan}

    connector.notify_sparse_memory_objs_updated()

    assert connector._sparse_direct_layer_states is None
    assert connector._sparse_direct_validated_layers == set()
    assert connector._sparse_destination_plans[0] is destination_plan


def test_group_store_cat_rows_append_pointer_table() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    memory_objs = [
        [_MemoryObj(torch.zeros(1))],
        [_MemoryObj(torch.zeros(1))],
    ]
    cached_tensors = [[], []]
    cached_dev_ptrs = [[11], [22]]
    cached_npu_ptrs = [
        torch.tensor([11], dtype=torch.long),
        torch.tensor([22], dtype=torch.long),
    ]

    AscendLMCacheEngine._append_group_store_tensors(
        SimpleNamespace(gpu_connector=connector),
        memory_objs,
        cached_tensors,
        cached_dev_ptrs,
        cached_npu_ptrs,
        [[101], [202]],
        torch.tensor([[101], [202]], dtype=torch.long),
    )

    assert [row.tolist() for row in cached_npu_ptrs] == [[11, 101], [22, 202]]


def test_shared_cpu_store_publication_fences_store_stream() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.store_stream = _TrackingStream("store")

    connector.synchronize_shared_cpu_store_publication()

    assert connector.store_stream.events == ["synchronize"]


def test_sparse_direct_state_key_includes_source_layout(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = None
    kvcaches_ref = [
        (
            torch.zeros((1, 4), dtype=torch.bfloat16),
            torch.zeros((1, 4), dtype=torch.bfloat16),
        )
    ]
    slot_mapping = torch.arange(4, dtype=torch.long)
    source = torch.zeros(8, dtype=torch.bfloat16)
    prepared = []

    def _prepare_state(*args, **kwargs):
        state = object()
        prepared.append(state)
        return state

    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_layer_state",
        _prepare_state,
    )

    first = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same_shape_new_source = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[torch.zeros(8, dtype=torch.bfloat16)],
        slot_mapping_ref=slot_mapping,
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    same_shape_new_slot_mapping = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[source],
        slot_mapping_ref=torch.arange(4, dtype=torch.long),
        total_tokens=4,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )
    changed = connector._get_or_create_sparse_direct_layer_state(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        layer_tensors=[torch.zeros(10, dtype=torch.bfloat16)],
        slot_mapping_ref=slot_mapping,
        total_tokens=5,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
    )

    assert first is same
    assert same_shape_new_source is first
    assert same_shape_new_slot_mapping is first
    assert changed is not first
    assert len(prepared) == 2


def test_sparse_pack_requires_compact_scratch_slot_mapping() -> None:
    """Sparse selected-token load uses slot_mapping as destination rows.

    The connector does not derive compact scratch slots from selected token ids.
    If the caller passes a full-prefix mapping, LMCache writes to the first N
    full-prefix slots instead of the compact scratch rows consumed by SFA.
    """
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651, 18639, 18455, 18642, 18445],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    packed, selected_out = VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
        connector,
        full_prefix_slots,
        selected,
        0,
    )

    assert torch.equal(selected_out, selected)
    assert packed.tolist() == list(range(256, 256 + selected.numel()))
    assert packed.tolist() != list(range(selected.numel()))

    compact_scratch_slots = torch.arange(selected.numel(), dtype=torch.long)
    packed_compact, selected_compact = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            compact_scratch_slots,
            selected,
            0,
        )
    )
    assert torch.equal(selected_compact, selected)
    assert packed_compact.tolist() == list(range(selected.numel()))


def test_sparse_pack_uses_target_slot_mapping_when_provided() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    target_slots = torch.tensor([901, 902, 903, 904], dtype=torch.long)

    packed, selected_out = VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
        connector,
        full_prefix_slots,
        selected,
        0,
        target_slot_mapping=target_slots,
    )

    assert torch.equal(selected_out, selected)
    assert torch.equal(packed, target_slots)


def test_sparse_pack_explicit_slots_preserves_fixed_rows_and_counts() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [[3, 91, 249, 0, 0], [0, 17, 0, 0, 0]], dtype=torch.int32
    )
    target_slots = torch.tensor(
        [[900, 901, 902, 1000, 1001], [1100, 1101, 1200, 1201, 1202]],
        dtype=torch.long,
    )

    packed, selected_out, counts = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            torch.tensor([3, 2], dtype=torch.int32),
        )
    )

    assert torch.equal(selected_out, selected)
    assert torch.equal(packed, target_slots)
    assert counts is not None
    assert counts.tolist() == [3, 2]


def test_sparse_pack_explicit_slots_allows_empty_row_payload() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([0, 91, 249], dtype=torch.int32)
    target_slots = torch.tensor([1000, 1001, 1002], dtype=torch.long)

    packed, selected_out, counts = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            torch.tensor([0], dtype=torch.int32),
        )
    )

    assert torch.equal(packed, target_slots)
    assert torch.equal(selected_out, selected)
    assert counts is not None
    assert counts.tolist() == [0]


def test_sparse_pack_explicit_slots_preserves_strided_counts() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([[3, 0], [7, 8]], dtype=torch.int32)
    target_slots = torch.tensor([[900, 0], [901, 902]], dtype=torch.long)
    count_storage = torch.zeros((2, 16), dtype=torch.int32)
    count_storage[:, 0] = torch.tensor([1, 2], dtype=torch.int32)
    strided_counts = count_storage[:, 0]

    packed, selected_out, counts = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_explicit_slot_inputs(
            connector,
            selected,
            target_slots,
            strided_counts,
        )
    )

    assert torch.equal(packed, target_slots)
    assert torch.equal(selected_out, selected)
    assert counts is not None
    assert counts.stride() == (16,)
    assert counts.data_ptr() == strided_counts.data_ptr()


def test_sparse_pack_legacy_slots_miss_compact_scratch_window() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor(
        [18831, 18814, 18810, 18651],
        dtype=torch.int32,
    )
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    compact_scratch_slots = torch.tensor([900, 901, 902, 903], dtype=torch.long)

    legacy_slots, selected_out = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
        )
    )
    assert torch.equal(selected_out, selected)
    assert not torch.equal(legacy_slots, compact_scratch_slots)

    source_by_token = {
        int(token): float(idx + 1) for idx, token in enumerate(selected.tolist())
    }
    scratch = torch.zeros(1024, dtype=torch.float32)
    for token, dst_slot in zip(
        selected_out.tolist(), legacy_slots.tolist(), strict=False
    ):
        scratch[int(dst_slot)] = source_by_token[int(token)]

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert not torch.equal(scratch[compact_scratch_slots], expected)

    fixed_slots, selected_fixed = (
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=compact_scratch_slots,
        )
    )
    scratch.zero_()
    for token, dst_slot in zip(
        selected_fixed.tolist(), fixed_slots.tolist(), strict=False
    ):
        scratch[int(dst_slot)] = source_by_token[int(token)]

    assert torch.equal(scratch[compact_scratch_slots], expected)


def test_sparse_pack_rejects_target_slot_mapping_length_mismatch() -> None:
    connector = _make_sparse_pack_connector()
    selected = torch.tensor([18831, 18814, 18810, 18651], dtype=torch.int32)
    full_prefix_slots = torch.arange(256, 256 + 18879, dtype=torch.long)
    target_slots = torch.tensor([901, 902, 903], dtype=torch.long)

    with pytest.raises(ValueError, match="target_slot_mapping"):
        VLLMPagedMemLayerwiseNPUConnector._pack_sparse_layer_inputs(
            connector,
            full_prefix_slots,
            selected,
            0,
            target_slot_mapping=target_slots,
        )


def test_sparse_transfer_topk_limits_aligned_views(monkeypatch) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_TRANSFER_TOPK", 2)
    slots = torch.tensor([901, 902, 903, 904], dtype=torch.long)
    selected = torch.tensor([31, 17, 9, 4], dtype=torch.int32)

    limited_slots, limited_selected = (
        VLLMPagedMemLayerwiseNPUConnector._limit_sparse_transfer_inputs(
            slots,
            selected,
        )
    )

    assert limited_slots.tolist() == [901, 902]
    assert limited_selected.tolist() == [31, 17]
    assert slots.numel() == selected.numel() == 4


@pytest.mark.parametrize("limit", [0, 4, 8])
def test_sparse_transfer_topk_preserves_shorter_inputs(
    monkeypatch,
    limit: int,
) -> None:
    monkeypatch.setattr(npu_connectors, "_SPARSE_TRANSFER_TOPK", limit)
    slots = torch.arange(4, dtype=torch.long)
    selected = torch.arange(4, dtype=torch.int32)

    limited_slots, limited_selected = (
        VLLMPagedMemLayerwiseNPUConnector._limit_sparse_transfer_inputs(
            slots,
            selected,
        )
    )

    assert limited_slots is slots
    assert limited_selected is selected


def test_sparse_direct_explicit_payload_uses_fast_path(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    connector._sparse_direct_layer_states = None
    connector._sparse_direct_validated_layers = set()

    class _Stream:
        def wait_stream(self, stream):
            pass

        def wait_event(self, event):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    class _TensorLike:
        def __init__(self, numel: int):
            self._numel = numel
            self.dtype = torch.long
            self.device = torch.device("cpu")

        def numel(self):
            return self._numel

    fast_calls = []
    slow_calls = []
    layer_state = object()

    def _prepare_state(*args, **kwargs):
        return layer_state

    def _fast(*args, **kwargs):
        fast_calls.append((args, kwargs))

    def _slow(*args, **kwargs):
        slow_calls.append((args, kwargs))

    monkeypatch.setattr(
        npu_connectors, "prepare_sparse_direct_layer_state", _prepare_state
    )
    monkeypatch.setattr(
        npu_connectors, "sparse_mla_dsa_batched_direct_kv_transfer_fast", _fast
    )
    monkeypatch.setattr(
        npu_connectors, "sparse_mla_dsa_batched_direct_kv_transfer", _slow
    )

    lmc_chunk = torch.zeros(8, dtype=torch.bfloat16)
    slot_mapping = _TensorLike(2)
    selected = _TensorLike(2)
    chunk_ptrs = _TensorLike(1)

    transfer_kwargs = dict(
        kvcaches_ref=[(object(), object())],
        kv_group=0,
        layer_id=0,
        load_stream=_Stream(),
        load_stream_idx=0,
        current_stream=_Stream(),
        slot_mapping_packed=slot_mapping,
        selected_token_idx=selected,
        chunk_size=4,
        total_tokens=4,
        chunk_ptrs_npu=chunk_ptrs,
        sparse_kv_format=0,
        sparse_token_major=False,
        sparse_vllm_two_major=False,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
        sparse_host_interleaved=False,
        layer_tensors=[lmc_chunk],
        slot_mapping_ref=slot_mapping,
        cpu_tensors=[lmc_chunk],
    )
    first_kernel = connector._run_sparse_direct_kv_transfer_layer(**transfer_kwargs)
    second_kernel = connector._run_sparse_direct_kv_transfer_layer(**transfer_kwargs)

    assert first_kernel == "sparse_mla_dsa_batched_direct_kv_transfer_fast"
    assert second_kernel == "sparse_mla_dsa_batched_direct_kv_transfer_fast"
    assert len(fast_calls) == 2
    assert slow_calls == []
    assert fast_calls[0][0][0] is layer_state
    assert fast_calls[1][0][0] is layer_state
    assert fast_calls[0][0][7] is True
    assert fast_calls[1][0][7] is False


def test_dense_direct_fast_state_cache_separates_load_and_store(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._sparse_direct_layer_states = None
    connector._sparse_direct_validated_layers = set()

    transfer_stream = _NoopStream()
    current_stream = _NoopStream()
    slot_mapping = _RecordableTensor(8)
    chunk_ptrs = _RecordableTensor(2)
    chunk_offsets = _RecordableTensor(2, dtype=torch.int32)
    chunk_sizes = _RecordableTensor(2, dtype=torch.int32)
    layer_tensors = [torch.zeros(8, dtype=torch.bfloat16)]
    kvcaches_ref = [(object(), object())]
    prepared = []
    fast_calls = []

    def _prepare_state(*args, **kwargs):
        state = object()
        prepared.append(state)
        return state

    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_layer_state",
        _prepare_state,
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer_fast",
        lambda *args, **kwargs: fast_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_batched_direct_kv_transfer",
        lambda *args, **kwargs: pytest.fail("slow dense direct path used"),
    )

    common_kwargs = dict(
        kvcaches_ref=kvcaches_ref,
        kv_group=0,
        layer_id=0,
        transfer_stream=transfer_stream,
        current_stream=current_stream,
        slot_mapping_full=slot_mapping,
        chunk_ptrs_npu=chunk_ptrs,
        chunk_offsets_npu=chunk_offsets,
        chunk_sizes_npu=chunk_sizes,
        total_tokens=8,
        fixed_chunk_size=256,
        dense_kv_format=5,
        dense_token_major=False,
        dense_vllm_two_major=False,
        dense_k_hidden_dims=512,
        dense_v_hidden_dims=64,
        dense_dsa_hidden_dims=0,
        dense_host_interleaved=False,
        layer_tensors=layer_tensors,
    )

    connector._run_dense_direct_kv_transfer_layer(
        **common_kwargs,
        direction=False,
    )
    connector._run_dense_direct_kv_transfer_layer(
        **common_kwargs,
        direction=True,
    )

    assert len(prepared) == 2
    assert len(fast_calls) == 2
    assert fast_calls[0][0][0] is prepared[0]
    assert fast_calls[1][0][0] is prepared[1]
    assert fast_calls[0][0][7] is False
    assert fast_calls[1][0][7] is True


def test_sparse_head_token_wise_uses_cached_token_count(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    request_kvcaches = [(object(), object())]
    connector.kvcaches = [(object(),)]
    connector.load_stream_idx = 0
    connector.load_stream_num = 1
    connector.load_stream_list = [object()]
    connector.lmcache_chunk_size = 256

    class _Stream:
        pass

    class _Layout:
        k_hidden_dims = 1
        v_hidden_dims = 1
        dsa_hidden_dims = 0
        kv_format = type("_Fmt", (), {"value": 0})()
        vllm_two_major = False

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    def _lazy_initialize_buffer(kvcaches, kv_group=0, init_staging=False):
        assert kvcaches is request_kvcaches
        connector.kvcaches = [(object(),)]
        return _Layout()

    monkeypatch.setattr(connector, "_lazy_initialize_buffer", _lazy_initialize_buffer)
    monkeypatch.setattr(
        connector,
        "_layerwise_token_major",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_sparse_lmc_host_interleaved",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_pack_sparse_layer_inputs",
        lambda slot_mapping, selected_token_idx, token_start_index: (
            slot_mapping,
            selected_token_idx,
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors, cached_chunk_ptrs_npu: torch.tensor(
            [123], dtype=torch.long
        ),
    )

    def _unexpected_total_tokens(*args, **kwargs):
        raise AssertionError("sparse total token shape walk should be skipped")

    monkeypatch.setattr(
        connector,
        "_sparse_total_tokens_from_layer_chunks",
        _unexpected_total_tokens,
    )

    calls = []

    def _run_sparse_direct(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        connector,
        "_run_sparse_direct_kv_transfer_layer",
        _run_sparse_direct,
    )

    gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=[[torch.zeros(4)]],
        lmcache_cached_tokens=18879,
        kv_group=0,
        kvcaches=request_kvcaches,
    )
    next(gen)
    gen.send(([], torch.arange(4, dtype=torch.int32), 0))

    assert len(calls) == 1
    assert calls[0]["total_tokens"] == 18879
    assert calls[0]["kvcaches_ref"] is request_kvcaches


def test_sparse_head_token_wise_sees_late_cached_tensors(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "0")
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.load_stream_idx = 0
    connector.load_stream_num = 1
    connector.load_stream_list = [object()]
    connector.lmcache_chunk_size = 256

    class _Stream:
        pass

    class _Layout:
        k_hidden_dims = 1
        v_hidden_dims = 1
        dsa_hidden_dims = 0
        kv_format = type("_Fmt", (), {"value": 0})()
        vllm_two_major = False

    class _MemoryObj:
        def __init__(self, tensor):
            self._tensor = tensor
            self.tensor_reads = 0

        @property
        def tensor(self):
            self.tensor_reads += 1
            return self._tensor

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer",
        lambda kvcaches, kv_group=0, init_staging=False: _Layout(),
    )
    monkeypatch.setattr(
        connector,
        "_layerwise_token_major",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_sparse_lmc_host_interleaved",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_pack_sparse_layer_inputs",
        lambda slot_mapping, selected_token_idx, token_start_index: (
            slot_mapping,
            selected_token_idx,
        ),
    )
    resolve_calls = []

    def _resolve_ptrs(
        layer_id,
        cpu_tensors,
        cached_chunk_ptrs_npu,
        expected_num_chunks=None,
    ):
        resolve_calls.append(
            (layer_id, cpu_tensors, cached_chunk_ptrs_npu, expected_num_chunks)
        )
        return (
            cached_chunk_ptrs_npu[layer_id]
            if cached_chunk_ptrs_npu
            else torch.tensor([123], dtype=torch.long)
        )

    monkeypatch.setattr(connector, "_resolve_sparse_chunk_ptrs_npu", _resolve_ptrs)

    calls = []

    def _run_sparse_direct(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        connector,
        "_run_sparse_direct_kv_transfer_layer",
        _run_sparse_direct,
    )

    cached_tensors = []
    gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=cached_tensors,
        lmcache_cached_tokens=4,
        kv_group=0,
    )
    next(gen)

    cached_tensor = torch.zeros(4)
    fallback_tensor = torch.ones(4)
    cached_tensors.extend([[cached_tensor]])
    gen.send(([_MemoryObj(fallback_tensor)], torch.arange(4, dtype=torch.int32), 0))

    assert len(calls) == 1
    assert calls[0]["cpu_tensors"][0] is cached_tensor
    assert calls[0]["layer_tensors"][0] is cached_tensor

    pointer_objs = [_MemoryObj(torch.zeros(4)), _MemoryObj(torch.ones(4))]
    pointer_table = torch.tensor([101, 102], dtype=torch.long)
    pointer_gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=[],
        cached_memory_objs=[pointer_objs],
        cached_chunk_ptrs_npu=[pointer_table],
        lmcache_cached_tokens=4,
        kv_group=0,
    )
    next(pointer_gen)
    pointer_gen.send(
        ([pointer_objs[-1]], torch.arange(4, dtype=torch.int32), 0)
    )

    assert [obj.tensor_reads for obj in pointer_objs] == [1, 0]
    assert resolve_calls[-1][3] == 2
    assert calls[-1]["chunk_ptrs_npu"] is pointer_table
    assert len(calls[-1]["cpu_tensors"]) == 1
    assert calls[-1]["cpu_tensors"][0] is pointer_objs[0]._tensor
    allocator = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([4]),
        torch.float32,
        batch_size=1,
        num_layers=1,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert pages is not None
    page_gen = connector.batched_to_gpu_head_token_wise(
        slot_mapping=torch.arange(4, dtype=torch.long),
        sync=True,
        cached_tensors=[],
        cached_memory_objs=[],
        cached_chunk_ptrs_npu=[],
        lmcache_cached_tokens=4,
        kv_group=0,
    )
    next(page_gen)
    page_gen.send(
        (
            LayerPageSource(tuple(pages), 0),
            torch.arange(4, dtype=torch.int32),
            0,
        )
    )
    assert calls[-1]["cpu_tensors"][0] is pages[0].layer_tensor(0)
    page_gen.close()
    pages[0].ref_count_down()
    gen.close()
    pointer_gen.close()


def test_prepared_sparse_head_token_wise_skips_layer_lookups(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "0")
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.lmcache_chunk_size = 256
    connector.kv_device = torch.device("cpu")
    connector._group_layouts = {}

    class _Layout:
        k_hidden_dims = 1
        v_hidden_dims = 1
        dsa_hidden_dims = 0
        kv_format = type("_Fmt", (), {"value": 0})()
        vllm_two_major = False
        kv_device = torch.device("cpu")
        gpu_buffer_allocator = None

    class _Owner:
        @property
        def tensor(self):
            raise AssertionError("prepared transfer materialized an owner tensor")

    owner: Any = _Owner()
    source_layer = PreparedSparseSourceLayer(
        tensors=(),
        chunk_ptrs_npu=torch.tensor([123], dtype=torch.int64),
        memory_objs=(owner,),
    )
    source = PreparedSparseSource(
        layers=(source_layer,),
        total_tokens=4,
    )
    destination_plan = object()
    plan_calls = []
    transfer_calls = []

    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda kvcaches, kv_group, init_staging: _Layout(),
    )
    monkeypatch.setattr(
        connector,
        "_layerwise_token_major",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_sparse_lmc_host_interleaved",
        lambda kv_group: False,
    )
    monkeypatch.setattr(
        connector,
        "_pack_sparse_layer_inputs",
        lambda slot_mapping, selected_token_idx, token_start_index: (
            slot_mapping,
            selected_token_idx,
        ),
    )

    def get_plan(**kwargs):
        plan_calls.append(kwargs)
        return destination_plan

    def fail_lookup(*args, **kwargs):
        raise AssertionError("prepared path performed a per-layer lookup")

    monkeypatch.setattr(
        connector,
        "_get_or_create_sparse_destination_plan",
        get_plan,
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        fail_lookup,
    )
    monkeypatch.setattr(
        connector,
        "_get_or_create_sparse_direct_layer_state",
        fail_lookup,
    )
    monkeypatch.setattr(
        connector,
        "_run_prepared_sparse_direct_kv_transfer_layer",
        lambda **kwargs: transfer_calls.append(kwargs),
    )

    generators = [
        connector.batched_to_gpu_head_token_wise(
            prepared_sparse_source=source,
            kvcaches=[(object(), object())],
            slot_mapping=torch.arange(4, dtype=torch.long),
            sync=False,
            kv_group=0,
        )
        for _ in range(2)
    ]
    for generator in generators:
        next(generator)
    selected = torch.arange(4, dtype=torch.int32)
    for generator in generators:
        generator.send(
            {
                "selected_token_ids": selected,
                "token_start_index": 0,
                "payload_event": object(),
            }
        )

    assert len(plan_calls) == 2
    assert all("source" not in call for call in plan_calls)
    assert all("chunk_size" not in call for call in plan_calls)
    assert len(transfer_calls) == 2
    assert all(call["plan"] is destination_plan for call in transfer_calls)
    assert all(call["source_layer"] is source_layer for call in transfer_calls)
    assert all(
        "load_stream" not in call and "current_stream" not in call
        for call in transfer_calls
    )

    for generator in generators:
        generator.close()


def test_sparse_destination_plan_is_reused_across_step_sizes(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    connector._sparse_destination_plans = {}
    kvcaches = [object(), object()]
    prepare_calls = []
    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_destination_state",
        lambda *args: prepare_calls.append(args) or object(),
    )

    plan_kwargs = {
        "kvcaches_ref": kvcaches,
        "kv_group": 0,
        "sparse_kv_format": 0,
        "sparse_k_hidden_dims": 1,
        "sparse_v_hidden_dims": 1,
        "sparse_dsa_hidden_dims": 0,
        "expected_device": torch.device("cpu"),
    }
    first = connector._get_or_create_sparse_destination_plan(
        slot_mapping_ref=torch.arange(4, dtype=torch.long),
        **plan_kwargs,
    )
    second = connector._get_or_create_sparse_destination_plan(
        slot_mapping_ref=torch.arange(32, dtype=torch.long),
        **plan_kwargs,
    )

    assert second is first
    assert len(prepare_calls) == 2
    assert not hasattr(first, "validated")
    assert not hasattr(first, "source")


def test_prepared_sparse_launch_avoids_load_stream_handoff(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    native_state = object()
    plan = npu_connectors._SparseDestinationPlan(
        [object()],
        (torch.long, "cpu", 0, 1, 1, 0),
        (native_state,),
    )
    chunk_ptrs = torch.tensor([123], dtype=torch.int64)
    source_layer = PreparedSparseSourceLayer(
        tensors=(torch.zeros(4),),
        chunk_ptrs_npu=chunk_ptrs,
    )
    slots = torch.arange(2, dtype=torch.long)
    selected = torch.arange(2, dtype=torch.int32)
    calls = []

    def fail_stream_context(_stream):
        raise AssertionError("prepared sparse launch entered a load stream")

    monkeypatch.setattr(torch.cuda, "stream", fail_stream_context)
    monkeypatch.setattr(
        npu_connectors,
        "sparse_mla_dsa_batched_direct_kv_transfer_prepared",
        lambda *args: calls.append(args),
    )

    connector._run_prepared_sparse_direct_kv_transfer_layer(
        plan=plan,
        source_layer=source_layer,
        layer_id=0,
        slot_mapping_packed=slots,
        selected_token_idx=selected,
        chunk_size=256,
        total_tokens=4,
        sparse_host_interleaved=True,
    )

    assert len(calls) == 1
    args = calls[0]
    assert args[0] is native_state
    assert args[1] is slots
    assert args[2] is selected
    assert args[3] is chunk_ptrs
    assert args[4:] == (256, 4, True)


def test_deferred_sparse_consumer_wait_joins_after_all_submissions(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    compute_stream = _TrackingStream("compute")
    load_streams = [_TrackingStream("load-0"), _TrackingStream("load-1")]
    done_events = [_TrackingEvent("done-0"), _TrackingEvent("done-1")]
    connector.load_stream_list = load_streams
    connector._sparse_load_done_events = done_events
    connector._active_sparse_load_join = None

    connector._sparse_direct_validated_layers = set()
    monkeypatch.setattr(
        npu_connectors.torch,
        "npu",
        SimpleNamespace(current_stream=lambda: compute_stream),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        npu_connectors,
        "sparse_mla_dsa_batched_direct_kv_transfer_fast",
        lambda *args: None,
    )
    monkeypatch.setattr(
        connector,
        "_sparse_direct_pointer_cache_signature",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        connector,
        "_get_or_create_sparse_direct_layer_state",
        lambda **kwargs: (object(), (0, 0)),
    )

    def launch(stream_index: int) -> None:
        connector._run_sparse_direct_kv_transfer_layer(
            kvcaches_ref=[(object(), object())],
            kv_group=0,
            layer_id=0,
            load_stream=load_streams[stream_index],
            load_stream_idx=stream_index,
            current_stream=compute_stream,
            slot_mapping_packed=torch.arange(2, dtype=torch.long),
            selected_token_idx=torch.arange(2, dtype=torch.int32),
            chunk_size=256,
            total_tokens=4,
            chunk_ptrs_npu=torch.tensor([123], dtype=torch.int64),
            sparse_kv_format=0,
            sparse_token_major=False,
            sparse_vllm_two_major=False,
            sparse_k_hidden_dims=1,
            sparse_v_hidden_dims=1,
            sparse_dsa_hidden_dims=0,
            sparse_host_interleaved=True,
            layer_tensors=[torch.zeros(4)],
        )

    launch(0)
    assert compute_stream.events == [("wait_stream", "load-0")]
    compute_stream.events.clear()
    load_streams[0].events.clear()

    with connector.defer_sparse_load_consumer_wait():
        launch(0)
        launch(1)
        assert compute_stream.events == []

    assert load_streams[0].events == [("wait_stream", "compute")]
    assert load_streams[1].events == [("wait_stream", "compute")]
    assert done_events[0].records == ["load-0"]
    assert done_events[1].records == ["load-1"]
    assert compute_stream.events == [
        ("wait_event", "done-0"),
        ("wait_event", "done-1"),
    ]
    assert connector._active_sparse_load_join is None

    load_streams[0].events.clear()
    with pytest.raises(RuntimeError, match="submission failed"):
        with connector.defer_sparse_load_consumer_wait():
            launch(0)
            raise RuntimeError("submission failed")

    assert load_streams[0].events == [
        ("wait_stream", "compute"),
        "synchronize",
    ]
    assert connector._active_sparse_load_join is None

    load_streams[0].events.clear()
    load_streams[1].events.clear()
    original_record = done_events[0].record

    def fail_record(_stream) -> None:
        raise RuntimeError("event record failed")

    monkeypatch.setattr(done_events[0], "record", fail_record)
    with pytest.raises(RuntimeError, match="event record failed"):
        with connector.defer_sparse_load_consumer_wait():
            launch(0)
            launch(1)

    assert load_streams[0].events[-1] == "synchronize"
    assert load_streams[1].events[-1] == "synchronize"
    assert connector._active_sparse_load_join is None

    monkeypatch.setattr(done_events[0], "record", original_record)
    load_streams[0].events.clear()
    load_streams[1].events.clear()

    def fail_wait(_event) -> None:
        raise RuntimeError("event wait failed")

    monkeypatch.setattr(compute_stream, "wait_event", fail_wait)
    with pytest.raises(RuntimeError, match="event wait failed"):
        with connector.defer_sparse_load_consumer_wait():
            launch(0)
            launch(1)

    assert load_streams[0].events[-1] == "synchronize"
    assert load_streams[1].events[-1] == "synchronize"
    assert connector._active_sparse_load_join is None


def test_sparse_destination_plan_replaces_destination_per_group(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector._sparse_destination_plans = {}
    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_destination_state",
        lambda *args: object(),
    )

    def get_plan(kvcaches, kv_group):
        return connector._get_or_create_sparse_destination_plan(
            kvcaches_ref=kvcaches,
            kv_group=kv_group,
            slot_mapping_ref=torch.arange(4, dtype=torch.long),
            sparse_kv_format=0,
            sparse_k_hidden_dims=1,
            sparse_v_hidden_dims=1,
            sparse_dsa_hidden_dims=0,
            expected_device=torch.device("cpu"),
        )

    latent_a = [object()]
    indexer = [object()]
    latent_b = [object()]
    latent_a_plan = get_plan(latent_a, 0)
    indexer_plan = get_plan(indexer, 1)
    latent_b_plan = get_plan(latent_b, 0)

    assert connector._sparse_destination_plans[0] is latent_b_plan
    assert connector._sparse_destination_plans[1] is indexer_plan
    assert latent_a_plan not in connector._sparse_destination_plans.values()

    other_plan = get_plan([object()], 2)
    assert len(connector._sparse_destination_plans) == (
        npu_connectors._SPARSE_DESTINATION_PLAN_CACHE_SIZE
    )
    assert 1 not in connector._sparse_destination_plans
    assert indexer_plan not in connector._sparse_destination_plans.values()
    assert other_plan in connector._sparse_destination_plans.values()


def test_sparse_destination_plan_rebuilds_for_process_abi_change(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector._sparse_destination_plans = {}
    prepare_calls = []
    monkeypatch.setattr(
        npu_connectors,
        "prepare_sparse_direct_destination_state",
        lambda *args: prepare_calls.append(args) or object(),
    )
    kwargs = {
        "kvcaches_ref": [object()],
        "kv_group": 0,
        "sparse_kv_format": 0,
        "sparse_k_hidden_dims": 1,
        "sparse_v_hidden_dims": 1,
        "sparse_dsa_hidden_dims": 0,
        "expected_device": torch.device("cpu"),
    }

    first = connector._get_or_create_sparse_destination_plan(
        slot_mapping_ref=torch.arange(4, dtype=torch.long),
        **kwargs,
    )
    second = connector._get_or_create_sparse_destination_plan(
        slot_mapping_ref=torch.arange(4, dtype=torch.int32),
        **kwargs,
    )

    assert second is not first
    assert len(prepare_calls) == 2


def test_prepare_dense_direct_chunk_metadata() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")

    fixed_size, fixed_offsets, fixed_sizes = (
        connector._prepare_dense_direct_chunk_metadata(
            [0, 256, 512],
            [256, 256, 17],
            total_tokens=529,
            kv_group=0,
        )
    )
    assert fixed_size == 256
    assert fixed_offsets is fixed_sizes
    assert fixed_offsets.dtype == torch.int32

    variable_size, variable_offsets, variable_sizes = (
        connector._prepare_dense_direct_chunk_metadata(
            [0, 128, 384],
            [128, 256, 17],
            total_tokens=401,
            kv_group=0,
        )
    )
    assert variable_size == 0
    assert variable_offsets.tolist() == [0, 128, 384]
    assert variable_sizes.tolist() == [128, 256, 17]

    with pytest.raises(ValueError, match="must be contiguous"):
        connector._prepare_dense_direct_chunk_metadata(
            [0, 255, 512],
            [256, 256, 17],
            total_tokens=529,
            kv_group=0,
        )


def _staging_batched_to_gpu(
    monkeypatch, *, fail_transfer: bool = False
) -> tuple[Any, list[bool]]:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.load_stream = _NoopStream()
    layout = _DenseLayout()
    layout.gpu_buffer_allocator = object()
    releases = []
    staging_obj = SimpleNamespace(
        tensor=torch.zeros(1), ref_count_down=lambda: releases.append(True)
    )

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _NoopStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **_kwargs: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda *_args, **_kwargs: layout,
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda _group=0: False)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda _group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda _group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda _group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda **_kwargs: (staging_obj, staging_obj.tensor),
    )

    def transfer(*_args, **_kwargs):
        if fail_transfer:
            raise RuntimeError("transfer failed")

    monkeypatch.setattr(
        npu_connectors, "batched_fused_single_layer_kv_transfer", transfer
    )
    generator = connector.batched_to_gpu(
        [0],
        [1],
        slot_mapping=torch.arange(1),
        sync=False,
        kv_group=0,
    )
    return generator, releases


def test_staging_batched_to_gpu_releases_buffer_on_completion(monkeypatch) -> None:
    generator, releases = _staging_batched_to_gpu(monkeypatch)
    next(generator)
    generator.send([_MemoryObj(torch.zeros(1))])
    assert releases == []
    next(generator)
    assert releases == [True]
    with pytest.raises(StopIteration):
        next(generator)
    assert releases == [True]


def test_staging_batched_to_gpu_releases_buffer_on_error(monkeypatch) -> None:
    generator, releases = _staging_batched_to_gpu(monkeypatch, fail_transfer=True)
    next(generator)
    with pytest.raises(RuntimeError, match="transfer failed"):
        generator.send([_MemoryObj(torch.zeros(1))])
    assert releases == [True]


def test_staging_batched_to_gpu_releases_buffer_on_close(monkeypatch) -> None:
    generator, releases = _staging_batched_to_gpu(monkeypatch)
    next(generator)
    generator.close()
    generator.close()
    assert releases == [True]


def test_dense_batched_to_gpu_direct_path_skips_staging(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    request_kvcaches = [(object(), object())]
    connector.kvcaches = [(object(),)]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.load_stream = _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _NoopStream())
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    init_staging_values = []

    def _lazy_initialize_buffer_with_staging(kvcaches, *, kv_group, init_staging):
        assert kvcaches is request_kvcaches
        connector.kvcaches = [(object(),)]
        init_staging_values.append(init_staging)
        return _DenseLayout()

    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        _lazy_initialize_buffer_with_staging,
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector,
        "_check_layerwise_transfer_invariants",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not allocate staging")
        ),
    )

    pointer_calls = []

    def _resolve_chunk_ptrs(
        layer_id,
        cpu_tensors,
        cached_chunk_ptrs_arg=None,
        expected_num_chunks=None,
        cached_chunk_dev_ptrs=None,
        source_objs=None,
    ):
        pointer_calls.append(
            (
                layer_id,
                list(cpu_tensors),
                cached_chunk_ptrs_arg,
                expected_num_chunks,
                source_objs,
            )
        )
        resolved = torch.tensor([123, 456], dtype=torch.long)
        if cached_chunk_ptrs_arg is not None:
            while len(cached_chunk_ptrs_arg) <= layer_id:
                cached_chunk_ptrs_arg.append(None)
            cached_chunk_ptrs_arg[layer_id] = resolved
        return resolved

    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        _resolve_chunk_ptrs,
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    cached_chunk_ptrs_npu = []
    gen = connector.batched_to_gpu(
        [0, 256],
        [256, 273],
        slot_mapping=torch.arange(273, dtype=torch.long),
        sync=False,
        kv_group=0,
        cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        kvcaches=request_kvcaches,
    )
    next(gen)
    gen.send(
        [
            _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
        ]
    )
    gen.close()

    assert init_staging_values == [False]
    assert len(pointer_calls) == 1
    assert pointer_calls[0][2] is cached_chunk_ptrs_npu
    assert len(pointer_calls[0][1]) == 1
    assert len(pointer_calls[0][4]) == 2
    assert cached_chunk_ptrs_npu[0].tolist() == [123, 456]
    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is False
    assert direct_calls[0]["total_tokens"] == 273
    assert direct_calls[0]["fixed_chunk_size"] == 256
    assert direct_calls[0]["kvcaches_ref"] is request_kvcaches

    cached_gen = connector.batched_to_gpu(
        [0, 256],
        [256, 273],
        slot_mapping=torch.arange(273, dtype=torch.long),
        sync=False,
        kv_group=0,
        cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        kvcaches=request_kvcaches,
    )
    next(cached_gen)
    cached_gen.send(
        [
            _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
        ]
    )
    cached_gen.close()
    assert len(pointer_calls[-1][1]) == 1
    assert pointer_calls[-1][3] == 2


def test_dense_batched_to_gpu_direct_path_passes_variable_chunk_metadata(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.load_stream = _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _NoopStream())
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **kwargs: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda kvcaches, *, kv_group, init_staging: _DenseLayout(),
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **kwargs: None
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct retrieve should not allocate staging")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors, cached=None, **_kwargs: torch.tensor(
            [111, 222, 333], dtype=torch.long
        ),
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    starts = [0, 128, 384]
    ends = [128, 384, 401]
    gen = connector.batched_to_gpu(
        starts,
        ends,
        slot_mapping=torch.arange(401, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.send(
        [
            _MemoryObj(torch.zeros(128, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
            _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
        ]
    )
    gen.close()

    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is False
    assert direct_calls[0]["fixed_chunk_size"] == 0
    assert direct_calls[0]["total_tokens"] == 401
    assert direct_calls[0]["chunk_offsets_npu"].tolist() == starts
    assert direct_calls[0]["chunk_sizes_npu"].tolist() == [128, 256, 17]


def test_dense_batched_from_gpu_direct_path_skips_staging(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    request_kvcaches = [(object(), object())]
    connector.kvcaches = [(object(),)]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.store_stream = _NoopStream()

    class _Npu:
        def current_stream(self):
            return _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_STORE_DISABLE", False)
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(
        connector,
        "initialize_kvcaches_ptr",
        lambda **kwargs: None,
    )
    init_staging_values = []

    def _lazy_initialize_buffer_with_staging(kvcaches, *, kv_group, init_staging):
        assert kvcaches is request_kvcaches
        connector.kvcaches = [(object(),)]
        init_staging_values.append(init_staging)
        return _DenseLayout()

    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        _lazy_initialize_buffer_with_staging,
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector,
        "_check_layerwise_transfer_invariants",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not allocate staging")
        ),
    )

    pointer_calls = []

    def _resolve_chunk_ptrs(
        layer_id,
        cpu_tensors,
        cached_chunk_ptrs_arg=None,
    ):
        pointer_calls.append(
            (
                layer_id,
                list(cpu_tensors),
                cached_chunk_ptrs_arg,
            )
        )
        return torch.tensor([123, 456], dtype=torch.long)

    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        _resolve_chunk_ptrs,
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    local_slot_mapping = torch.arange(1000, 1273, dtype=torch.long)
    gen = connector.batched_from_gpu(
        [
            [
                _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
            ]
        ],
        [256, 512],
        [512, 529],
        slot_mapping=local_slot_mapping,
        slot_mapping_base=256,
        sync=False,
        kv_group=0,
        kvcaches=request_kvcaches,
    )
    next(gen)
    gen.close()

    assert init_staging_values == [False]
    assert len(pointer_calls) == 1
    assert pointer_calls[0][2] is None
    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is True
    assert direct_calls[0]["total_tokens"] == 273
    assert direct_calls[0]["fixed_chunk_size"] == 256
    assert direct_calls[0]["kvcaches_ref"] is request_kvcaches
    assert torch.equal(direct_calls[0]["slot_mapping_full"], local_slot_mapping)


def test_dense_batched_from_gpu_direct_path_passes_variable_chunk_metadata(
    monkeypatch,
) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kvcaches = [(object(), object())]
    connector.use_gpu = True
    connector.kv_device = torch.device("cpu")
    connector.store_stream = _NoopStream()

    class _Npu:
        def current_stream(self):
            return _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_STORE_DISABLE", False)
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **kwargs: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda kvcaches, *, kv_group, init_staging: _DenseLayout(),
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda kv_group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda kv_group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda kv_group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda kv_group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **kwargs: None
    )
    monkeypatch.setattr(
        connector,
        "_check_staging_transfer_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not check staging tokens")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_allocate_layerwise_staging_buffer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dense direct store should not allocate staging")
        ),
    )
    monkeypatch.setattr(
        connector,
        "_resolve_sparse_chunk_ptrs_npu",
        lambda layer_id, cpu_tensors: torch.tensor([111, 222, 333], dtype=torch.long),
    )

    direct_calls = []
    monkeypatch.setattr(
        connector,
        "_run_dense_direct_kv_transfer_layer",
        lambda **kwargs: direct_calls.append(kwargs),
    )

    starts = [0, 128, 384]
    ends = [128, 384, 401]
    gen = connector.batched_from_gpu(
        [
            [
                _MemoryObj(torch.zeros(128, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(256, dtype=torch.bfloat16)),
                _MemoryObj(torch.zeros(17, dtype=torch.bfloat16)),
            ]
        ],
        starts,
        ends,
        slot_mapping=torch.arange(401, dtype=torch.long),
        sync=False,
        kv_group=0,
    )
    next(gen)
    gen.close()

    assert len(direct_calls) == 1
    assert direct_calls[0]["direction"] is True
    assert direct_calls[0]["fixed_chunk_size"] == 0
    assert direct_calls[0]["total_tokens"] == 401
    assert direct_calls[0]["chunk_offsets_npu"].tolist() == starts
    assert direct_calls[0]["chunk_sizes_npu"].tolist() == [128, 256, 17]


def test_dense_group_store_uses_one_host_dispatch(monkeypatch) -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    connector.kvcaches = [(object(), object()), (object(), object())]
    connector.kv_device = torch.device("cpu")
    connector.store_stream = _NoopStream()
    connector._sparse_direct_validated_layers = set()

    class _Npu:
        def current_stream(self):
            return _NoopStream()

    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_STORE_DISABLE", False)
    monkeypatch.setattr(
        npu_connectors, "_DENSE_DIRECT_GROUP_STORE_DISABLE", False
    )
    monkeypatch.setattr(torch, "npu", _Npu(), raising=False)
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", lambda **_kw: None)
    monkeypatch.setattr(
        connector,
        "_lazy_initialize_buffer_with_staging",
        lambda _caches, *, kv_group, init_staging: _DenseLayout(),
    )
    monkeypatch.setattr(connector, "_is_mla_dsa_format", lambda _group=0: True)
    monkeypatch.setattr(
        connector,
        "_expected_memory_format",
        lambda _group=0: MemoryFormat.KV_MLA_LATENT_FMT,
    )
    monkeypatch.setattr(connector, "_layerwise_token_major", lambda _group=0: False)
    monkeypatch.setattr(
        connector, "_sparse_lmc_host_interleaved", lambda _group=0: False
    )
    monkeypatch.setattr(
        connector, "_check_layerwise_transfer_invariants", lambda **_kw: None
    )
    monkeypatch.setattr(
        connector,
        "_get_or_create_sparse_direct_layer_state",
        lambda **kw: (f"state-{kw['layer_id']}", ("state", kw["layer_id"])),
    )
    group_calls = []
    monkeypatch.setattr(
        npu_connectors,
        "dense_mla_dsa_group_direct_kv_transfer_fast",
        lambda *args, **kwargs: (
            group_calls.append((args, kwargs))
            or ([[100], [110]], torch.tensor([[100], [110]], dtype=torch.long))
        ),
    )
    memory_objs = [
        [_MemoryObj(torch.zeros(4, dtype=torch.bfloat16))],
        [_MemoryObj(torch.zeros(4, dtype=torch.bfloat16))],
    ]

    host_rows, pointer_table = connector.batched_from_gpu_group(
        memory_objs,
        [0],
        [4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        kv_group=0,
    )

    assert host_rows == [[100], [110]]
    assert pointer_table.tolist() == [[100], [110]]
    assert len(group_calls) == 1
    args, kwargs = group_calls[0]
    assert args[0] == ["state-0", "state-1"]
    assert args[5] == 4
    assert args[7] is True
    assert kwargs["validate_inputs"] is True
    assert kwargs["fixed_chunk_size"] == 4


@pytest.mark.parametrize("use_npu", [True])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
    ],
)
def test_layerwise_vllm_paged_connector_with_npu(use_npu, gpu_kv_format):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu, gpu_kv_format)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)


def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)
