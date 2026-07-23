# SPDX-License-Identifier: Apache-2.0
"""Sparse decode cache-state and shared collective-order tests."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.shared_cpu_cache import SharedHandleEnvelope
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult
import lmcache_ascend.v1.cache_engine as ascend_cache_engine
import lmcache_ascend.v1.npu_connector.npu_connectors as npu_connectors
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


def test_layout_probe_does_not_initialize_staging() -> None:
    calls = []

    class _LayoutOnlyConnector:
        kvcaches = None

        def initialize_kvcaches_ptr(self, **kwargs):
            self.kvcaches = kwargs["kvcaches"]

        def _lazy_initialize_buffer(
            self,
            kvcaches,
            *,
            kv_group=0,
            init_staging=None,
        ):
            calls.append((kvcaches, kv_group, init_staging))

    connector = _LayoutOnlyConnector()
    engine = object.__new__(AscendLMCacheEngine)
    engine.gpu_connector = connector
    kvcaches = [object()]

    engine._ensure_layerwise_connector_layout(
        kvcaches=kvcaches,
        kv_group=1,
    )

    assert calls == [(kvcaches, 1, False)]


class _FakePinnedMemObj:
    def __init__(self, ref_count=1):
        self.tensor = torch.empty(1)
        self.is_pinned = True
        self.ref_count = ref_count
        self.unpin_count = 0
        self.release_count = 0

    def unpin(self):
        self.is_pinned = False
        self.unpin_count += 1

    def is_valid(self):
        return self.ref_count > 0

    def ref_count_down(self):
        self.ref_count -= 1
        self.release_count += 1


class _FakeClaimableMemObj:
    def __init__(self):
        self.is_pinned = False
        self.valid = True
        self.ref_up_count = 0
        self.ref_down_count = 0
        self.pin_count = 0
        self.unpin_count = 0

    def ref_count_up(self):
        self.ref_up_count += 1

    def ref_count_down(self):
        self.ref_down_count += 1

    def pin(self):
        self.is_pinned = True
        self.pin_count += 1
        return True

    def unpin(self):
        self.is_pinned = False
        self.unpin_count += 1
        return True

    def is_valid(self):
        return self.valid


class _FakeSparseConsumer:
    def batched_to_gpu_head_token_wise(self, **_kwargs):
        yield
        while True:
            yield


class _OrderedSparseConsumer:
    def __init__(self, events):
        self.events = events

    def batched_to_gpu_head_token_wise(self, **_kwargs):
        payload = yield
        while payload is not None:
            self.events.append("load")
            payload = yield
        yield

    def synchronize_shared_cpu_store_publication(self):
        self.events.append("store_fence")


class _FakeTensorMemObj:
    def __init__(self, tensor):
        self._tensor = tensor
        self.release_count = 0
        self.valid = True

    @property
    def tensor(self):
        return self._tensor

    def is_valid(self):
        return self.valid

    def ref_count_down(self):
        self.release_count += 1
        self.valid = False


class _FakePassiveAllocator:
    def __init__(self, mem_obj):
        self.mem_obj = mem_obj

    def create_view(self, *_args, **_kwargs):
        return self.mem_obj


class _SequencePassiveAllocator:
    def __init__(self, views):
        self.views = iter(views)

    def create_view(self, *_args, **_kwargs):
        view = next(self.views)
        if isinstance(view, Exception):
            raise view
        return view


class _FakeTokenDatabase:
    def __init__(self, keys):
        self.keys = keys if isinstance(keys, list) else [keys]

    def process_tokens(self, **_kwargs):
        for index, key in enumerate(self.keys):
            yield index, index + 1, key


def _make_key(kv_group=0, chunk_hash=1234):
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
        kv_group=kv_group,
    )


def test_sparse_rank0_preflight_error_broadcasts_on_first_layer_send(monkeypatch):
    """Preflight errors must align with passive ranks' first receive point."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "MooncakeStore",
        [0],
        [4],
        [["layer0-key"], ["layer1-key"]],
    )
    engine._find_shared_rank0_chunk_location = lambda _key: None
    broadcasts = []
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2, 3, 4],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    assert broadcasts == []

    with pytest.raises(ValueError, match="missing required chunks"):
        retriever.send(([0], 0))

    assert len(broadcasts) == 1
    assert broadcasts[0].status == "error"
    assert broadcasts[0].layer_id == 0
    assert broadcasts[0].kv_group == 0


def test_sparse_retrieve_allocated_ret_mask_clears_metadata_warm_flag(monkeypatch):
    """A fresh ret_mask must be populated even when request metadata is warm."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = SimpleNamespace(
        layerwise_batched_get=lambda *_args, **_kwargs: iter(())
    )
    engine.gpu_connector = object()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location

    def ensure_metadata(**kwargs):
        retrieve_kwargs = kwargs["retrieve_kwargs"]
        assert "_retrieve_metadata_warm" not in retrieve_kwargs
        kwargs["ret_mask"][0:2] = True
        return "LocalCPUBackend", [0], [2], [["layer0-key"]]

    engine._ensure_retrieve_chunk_metadata = ensure_metadata

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[["layer0-key"]],
        cached_starts=[0],
        cached_ends=[2],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
        _retrieve_metadata_warm=True,
    )

    ret_mask = next(retriever)

    assert ret_mask.tolist() == [True, True]
    retriever.close()


def test_metadata_warm_data_cold_repopulates_stale_ret_mask():
    """Metadata-warm retrieve must rebuild ret_mask when data is not cached."""

    engine = object.__new__(AscendLMCacheEngine)
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False

    cached_keys = [["layer0-key"]]
    cached_starts = [1]
    cached_ends = [3]
    ret_mask = torch.ones(3, dtype=torch.bool)
    retrieve_kwargs = {
        "_retrieve_metadata_warm": True,
        "cached_retrieve_location": "MooncakeStore",
        "kv_group": 0,
    }

    location, starts, ends, keys = AscendLMCacheEngine._ensure_retrieve_chunk_metadata(
        engine,
        tokens=[1, 2, 3],
        mask=None,
        request_configs=None,
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        ret_mask=ret_mask,
        retrieve_kwargs=retrieve_kwargs,
    )

    assert location == "MooncakeStore"
    assert starts is cached_starts
    assert ends is cached_ends
    assert keys is cached_keys
    assert ret_mask.tolist() == [False, True, True]
    assert retrieve_kwargs["_retrieve_metadata_warm"] is True


def test_sampled_metadata_build_skips_per_chunk_contains():
    key = _make_key()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.use_layerwise = True
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=True)
    engine.token_database = _FakeTokenDatabase(key)
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_shared_retrieve_passive = lambda _kv_group: False
    engine._find_shared_rank0_chunk_location = lambda _key: (_ for _ in ()).throw(
        AssertionError("sampled metadata path must not call contains")
    )
    ret_mask = torch.zeros(1, dtype=torch.bool)

    location, starts, ends, keys = engine._ensure_retrieve_chunk_metadata(
        tokens=[1],
        mask=None,
        request_configs=None,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        ret_mask=ret_mask,
        retrieve_kwargs={"kv_group": 0},
    )

    assert location == "mixed"
    assert starts == [0]
    assert ends == [1]
    assert keys == [[key.split_layers(2)[0]], [key.split_layers(2)[1]]]
    assert ret_mask.tolist() == [True]


def test_sparse_request_preflight_error_prevents_partial_latent_publication(
    monkeypatch,
):
    """A kv_group=1 preflight failure must abort before latent handles publish."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "MooncakeStore",
        [0],
        [4],
        [["layer0-key"], ["layer1-key"]],
    )
    engine._find_shared_rank0_chunk_location = lambda _key: "MooncakeStore"
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}
    allocated = [_FakePinnedMemObj(ref_count=2), _FakePinnedMemObj(ref_count=2)]
    resolved = list(allocated)
    engine._resolve_shared_rank0_layer_mem_objs = lambda **_kwargs: [
        resolved.pop(0)
    ]
    broadcasts = []
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)
    shared_preflight_state = {}

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2, 3, 4],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        kv_group=0,
        req_id="req-1",
        shared_cpu_request_preflight_state=shared_preflight_state,
    )

    next(retriever)
    assert broadcasts == []
    shared_preflight_state.update(
        {
            "source_kv_group": 1,
            "source_layer_id": 0,
            "message": "index missing",
            "details": {"missing_locations": [(0, 0)]},
            "error": ValueError("index missing"),
        }
    )

    with pytest.raises(ValueError, match="request preflight failed"):
        retriever.send(([0], 0))

    assert len(broadcasts) == 1
    assert broadcasts[0].status == "error"
    assert broadcasts[0].layer_id == 0
    assert broadcasts[0].kv_group == 0
    assert broadcasts[0].handles == []
    assert broadcasts[0].error_details["source_kv_group"] == 1
    assert not resolved
    assert all(obj.unpin_count == 1 for obj in allocated)
    assert all(obj.release_count == 1 for obj in allocated)
    assert all(obj.ref_count == 1 for obj in allocated)


def test_sparse_rank0_close_after_prime_releases_pre_resolved_memobjs(
    monkeypatch,
):
    """Closing after priming must not leak request-scoped pre-resolved pins."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "MooncakeStore",
        [0],
        [4],
        [["layer0-key"], ["layer1-key"]],
    )
    engine._find_shared_rank0_chunk_location = lambda _key: "MooncakeStore"
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}
    allocated = [_FakePinnedMemObj(), _FakePinnedMemObj()]
    resolved = list(allocated)
    engine._resolve_shared_rank0_layer_mem_objs = lambda **_kwargs: [
        resolved.pop(0)
    ]
    engine._broadcast_shared_envelope = lambda _envelope: None

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2, 3, 4],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    assert not resolved

    retriever.close()

    assert all(obj.unpin_count == 1 for obj in allocated)
    assert all(obj.release_count == 1 for obj in allocated)
    assert all(obj.ref_count == 0 for obj in allocated)


@pytest.mark.parametrize(
    ("page_first", "expected_layers_per_batch"),
    [(False, 2), (True, 5)],
)
def test_sparse_rank0_uses_windowed_remote_preflight_when_enabled(
    monkeypatch, page_first, expected_layers_per_batch
):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    num_layers = 5
    keys_layer_major = [[f"layer-{layer}"] for layer in range(num_layers)]
    allocated = [[_FakePinnedMemObj()] for _ in range(num_layers)]
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = num_layers
    engine.config = SimpleNamespace(
        experimental_sampled_layerwise_lookup=False,
        shared_cpu_remote_layers_per_batch=2,
        extra_config={"mooncake_page_first_multi_buffer": page_first},
    )
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._min_layer_cache_chunks = lambda *_args: 0
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "RemoteBackend",
        [0],
        [1],
        keys_layer_major,
    )
    engine._find_shared_rank0_chunk_location = lambda _key: "RemoteBackend"
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}
    windowed_calls = []

    def resolve_windowed(**kwargs):
        windowed_calls.append(kwargs)
        return allocated

    engine._resolve_shared_rank0_remote_layers_windowed = resolve_windowed
    engine._resolve_shared_rank0_layer_mem_objs = lambda **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("layerwise resolver must be bypassed"))
    engine._broadcast_shared_envelope = lambda _envelope: None

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)

    assert len(windowed_calls) == 1
    assert windowed_calls[0]["keys_layer_major"] == keys_layer_major
    assert windowed_calls[0]["layers_per_batch"] == expected_layers_per_batch

    retriever.close()
    assert all(obj.unpin_count == 1 for layer in allocated for obj in layer)
    assert all(obj.release_count == 1 for layer in allocated for obj in layer)


def test_page_first_windowed_preflight_reads_only_missing_suffix(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    num_layers = 2
    old_keys = [f"layer-{layer}-old" for layer in range(num_layers)]
    new_keys = [f"layer-{layer}-new" for layer in range(num_layers)]
    keys_layer_major = [
        [old_keys[layer], new_keys[layer]] for layer in range(num_layers)
    ]
    missing_keys = [[key] for key in new_keys]
    allocated = [[_FakePinnedMemObj()] for _ in range(num_layers)]
    cached_memory_objs = [
        [_FakeTensorMemObj(torch.empty(1))] for _ in range(num_layers)
    ]
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = num_layers
    engine.config = SimpleNamespace(
        experimental_sampled_layerwise_lookup=False,
        shared_cpu_remote_layers_per_batch=1,
        extra_config={"mooncake_page_first_multi_buffer": True},
    )
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "RemoteBackend",
        [0, 1],
        [1, 2],
        keys_layer_major,
    )
    located_keys = []
    engine._find_shared_rank0_chunk_location = lambda key: (
        located_keys.append(key) or "RemoteBackend"
    )
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}
    windowed_calls = []

    def resolve_windowed(**kwargs):
        windowed_calls.append(kwargs)
        return allocated

    engine._resolve_shared_rank0_remote_layers_windowed = resolve_windowed
    engine._resolve_shared_rank0_layer_mem_objs = lambda **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("layerwise resolver must be bypassed"))
    engine._broadcast_shared_envelope = lambda _envelope: None

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[[key] for key in old_keys],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[["old-handle"] for _ in range(num_layers)],
        kv_group=0,
        req_id="req-incremental",
    )

    next(retriever)

    assert located_keys == new_keys
    assert len(windowed_calls) == 1
    assert windowed_calls[0]["keys_layer_major"] == missing_keys
    assert windowed_calls[0]["layers_per_batch"] == num_layers

    retriever.close()
    assert all(obj.unpin_count == 1 for layer in allocated for obj in layer)
    assert all(obj.release_count == 1 for layer in allocated for obj in layer)


def test_sampled_sparse_preflight_batches_local_misses_without_contains(
    monkeypatch,
):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    keys_layer_major = [
        ["layer0-chunk0", "layer0-chunk1", "layer0-chunk2"],
        ["layer1-chunk0", "layer1-chunk1", "layer1-chunk2"],
    ]
    local_objects = [
        [_FakePinnedMemObj()],
        [],
    ]

    class _BatchedLocalBackend:
        def __init__(self):
            self.calls = []

        def batched_get_prefix_with_misses(self, keys):
            layer_id = len(self.calls)
            self.calls.append(list(keys))
            objects = list(local_objects[layer_id])
            positions = list(range(len(objects), len(keys)))
            return LocalCPUPrefixGetResult(
                objects,
                positions,
                [keys[position] for position in positions],
            )

    local_backend = _BatchedLocalBackend()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.use_layerwise = True
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=True)
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "mixed",
        [0, 2, 4],
        [2, 4, 6],
        keys_layer_major,
    )
    engine._shared_local_cpu_backend = lambda: local_backend
    engine._find_shared_rank0_chunk_location = lambda _key: (_ for _ in ()).throw(
        AssertionError("sampled worker preflight must not call contains")
    )
    capacity_calls = []
    engine._shared_cpu_runtime_capacity_details = lambda **kwargs: (
        capacity_calls.append(kwargs) or {"fits": True}
    )
    resolver_calls = []
    remote_objects = []

    def resolve_layer(**kwargs):
        resolver_calls.append(kwargs)
        local_prefix = kwargs["local_prefix"]
        resolved = list(local_prefix.local_memory_objs)
        for _ in local_prefix.remote_positions:
            remote_obj = _FakePinnedMemObj()
            remote_objects.append(remote_obj)
            resolved.append(remote_obj)
        return resolved

    engine._resolve_shared_rank0_layer_mem_objs = resolve_layer
    engine._broadcast_shared_envelope = lambda _envelope: None

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2, 3, 4, 5, 6],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)

    assert local_backend.calls == keys_layer_major
    assert capacity_calls[0]["chunk_locations_layer_major"] == [
        ["LocalCPUBackend", "RemoteBackend", "RemoteBackend"],
        ["RemoteBackend", "RemoteBackend", "RemoteBackend"],
    ]
    assert [call["local_prefix"].remote_positions for call in resolver_calls] == [
        [1, 2],
        [0, 1, 2],
    ]
    assert [call["local_prefix"].remote_keys for call in resolver_calls] == [
        keys_layer_major[0][1:],
        keys_layer_major[1],
    ]

    retriever.close()
    all_objects = [
        mem_obj for layer in local_objects for mem_obj in layer if mem_obj is not None
    ] + remote_objects
    assert all(obj.unpin_count == 1 for obj in all_objects)
    assert all(obj.release_count == 1 for obj in all_objects)


def test_sparse_passive_public_path_accepts_ret_mask_kwarg(monkeypatch):
    """Passive sparse retrieve receives ret_mask once, not both ways."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    handle = object()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[handle],
    )
    ret_mask = torch.ones(1, dtype=torch.bool)

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        ret_mask=ret_mask,
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    yielded = next(retriever)

    assert yielded is ret_mask
    assert ret_mask.tolist() == [False]


def test_sparse_passive_metadata_warm_without_data_waits_for_envelope(monkeypatch):
    """Warm metadata alone is not enough for passive ranks to skip broadcast."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    handle = object()
    received = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )

    def receive_envelope():
        received.append(True)
        return SharedHandleEnvelope(
            request_id="req-1",
            phase="sparse_decode_bootstrap",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
            status="ok",
            generation=7,
            handles=[handle],
        )

    engine._receive_shared_envelope = receive_envelope
    ret_mask = torch.ones(1, dtype=torch.bool)
    cached_memory_objs = []
    cached_shared_handles = []

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        ret_mask=ret_mask,
        cached_keys=[[key.get_first_layer()]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
        _retrieve_metadata_warm=True,
    )

    next(retriever)
    yielded = retriever.send(([0], 0))

    assert received == [True]
    assert yielded is ret_mask
    assert ret_mask.tolist() == [True]
    assert cached_memory_objs == [[mem_obj]]
    assert cached_shared_handles == [[handle]]
    retriever.close()


def test_sparse_indexer_passive_metadata_warm_waits_without_storage(monkeypatch):
    """Indexer-only passive ranks must not probe storage with no storage manager."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key(kv_group=1)
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    handle = object()
    received = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = None
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._is_indexer_passive = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )

    def receive_envelope():
        received.append(True)
        return SharedHandleEnvelope(
            request_id="req-1",
            phase="sparse_decode_bootstrap",
            request_ordinal=0,
            layer_id=0,
            kv_group=1,
            status="ok",
            generation=7,
            handles=[handle],
        )

    engine._receive_shared_envelope = receive_envelope
    ret_mask = torch.ones(1, dtype=torch.bool)
    cached_memory_objs = []
    cached_shared_handles = []

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        ret_mask=ret_mask,
        cached_keys=[[key.get_first_layer()]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=1,
        req_id="req-1",
        _retrieve_metadata_warm=True,
    )

    next(retriever)
    yielded = retriever.send(([0], 0))

    assert received == [True]
    assert yielded is ret_mask
    assert cached_memory_objs == [[mem_obj]]
    assert cached_shared_handles == [[handle]]
    retriever.close()


def test_sparse_passive_populates_metadata_and_hands_off_views(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    handle = object()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[handle],
    )
    cached_keys = []
    cached_starts = []
    cached_ends = []
    cached_memory_objs = []
    cached_tensors = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []
    cached_shared_handles = []

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0], 0))
    retriever.close()

    assert cached_keys == [[key.get_first_layer()]]
    assert cached_starts == [0]
    assert cached_ends == [1]
    assert cached_memory_objs == [[mem_obj]]
    assert cached_shared_handles == [[handle]]
    assert mem_obj.release_count == 0
    assert mem_obj.is_valid()


def test_sparse_passive_appends_only_new_shared_view(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    old_view = _FakeTensorMemObj(torch.empty(1))
    new_view = _FakeTensorMemObj(torch.empty(1))
    old_handle, new_handle = object(), object()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase([key0, key1])
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(new_view)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[new_handle],
    )
    cached_keys = [[key0.get_first_layer()]]
    cached_starts = [0]
    cached_ends = [1]
    cached_memory_objs = [[old_view]]
    cached_tensors = [[old_view.tensor]]
    cached_shared_handles = [[old_handle]]

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1, 2],
        None,
        torch.zeros(2, dtype=torch.bool),
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0, 1], 0))
    retriever.close()

    assert cached_starts == [0, 1]
    assert cached_ends == [1, 2]
    assert cached_keys == [[key0.get_first_layer(), key1.get_first_layer()]]
    assert cached_memory_objs == [[old_view, new_view]]
    assert cached_shared_handles == [[old_handle, new_handle]]
    assert old_view.is_valid() and new_view.is_valid()


def test_sparse_passive_extension_failure_rolls_back_all_layers(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    key0_layers = key0.split_layers(2)
    old_views = [_FakeTensorMemObj(torch.empty(1)) for _ in range(2)]
    new_view = _FakeTensorMemObj(torch.empty(1))
    old_handles = [[object()], [object()]]
    envelopes = iter(
        SharedHandleEnvelope(
            request_id="req-1",
            phase="sparse_decode_bootstrap",
            request_ordinal=0,
            layer_id=layer_id,
            kv_group=0,
            status="ok",
            generation=7,
            handles=[object()],
        )
        for layer_id in range(2)
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase([key0, key1])
    engine.shared_cpu_cache_passive_allocator = _SequencePassiveAllocator(
        [new_view, ValueError("view failed")]
    )
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: next(envelopes)
    cached_keys = [[key] for key in key0_layers]
    cached_starts = [0]
    cached_ends = [1]
    cached_memory_objs = [[view] for view in old_views]
    cached_tensors = [[view.tensor] for view in old_views]

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1, 2],
        None,
        torch.zeros(2, dtype=torch.bool),
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=old_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0, 1], 0))
    with pytest.raises(ValueError, match="view failed"):
        retriever.send(([0, 1], 0))

    assert cached_starts == [0]
    assert cached_ends == [1]
    assert cached_keys == [[key] for key in key0_layers]
    assert cached_memory_objs == [[view] for view in old_views]
    assert all(len(layer) == 1 for layer in old_handles)
    assert new_view.release_count == 1


def test_sparse_passive_close_before_handoff_releases_views(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[object()],
    )

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0], 0))
    retriever.close()

    assert mem_obj.release_count == 1
    assert not mem_obj.is_valid()


def test_sparse_rank0_cached_request_objects_publish_handles(monkeypatch):
    """Rank0 must still broadcast handles when passive ranks are cold."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeClaimableMemObj()
    cached_memory_objs = [[mem_obj]]
    cached_shared_handles = []
    broadcasts = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine._shared_cpu_request_leases = {}
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[key]],
    )

    def make_handles(**kwargs):
        assert kwargs["mem_objs_layer"] == [mem_obj]
        assert mem_obj.ref_up_count == 1
        assert mem_obj.pin_count == 1
        assert mem_obj.is_pinned
        return ["handle"]

    engine._make_shared_handles_for_layer = make_handles
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[[key]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0], 0))
    retriever.close()

    assert len(broadcasts) == 1
    assert broadcasts[0].status == "ok"
    assert broadcasts[0].handles == ["handle"]
    assert cached_shared_handles == [["handle"]]
    assert mem_obj.ref_up_count == 1
    assert mem_obj.pin_count == 1
    assert mem_obj.ref_down_count == 0
    assert mem_obj.unpin_count == 0


def test_sparse_rank0_fences_store_before_first_handle_publication(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    events = []
    key = _make_key()
    mem_obj = _FakeClaimableMemObj()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = _OrderedSparseConsumer(events)
    engine.shared_cpu_cache_generation = 7
    engine._shared_cpu_request_leases = {}
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[key]],
    )

    def make_handles(**_kwargs):
        events.append("make_handles")
        return ["handle"]

    engine._make_shared_handles_for_layer = make_handles
    engine._broadcast_shared_envelope = lambda _envelope: events.append("broadcast")

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[[key]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=[[mem_obj]],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0], 0))
    retriever.close()

    assert events[:3] == ["store_fence", "make_handles", "broadcast"]


def test_sparse_rank0_cached_publication_failure_releases_claim(monkeypatch):
    """Failed cached-object publication must release its request pin/ref."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    mem_obj = _FakeClaimableMemObj()
    broadcasts = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine._shared_cpu_request_leases = {}
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[key]],
    )
    engine._make_shared_handles_for_layer = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("publish failed")
    )
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[[key]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=[[mem_obj]],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    with pytest.raises(ValueError, match="publish failed"):
        retriever.send(([0], 0))

    assert mem_obj.ref_up_count == 1
    assert mem_obj.pin_count == 1
    assert mem_obj.unpin_count == 1
    assert mem_obj.ref_down_count == 1
    assert broadcasts
    assert broadcasts[-1].status == "error"
    assert "handle publication failed" in broadcasts[-1].message


def test_sparse_rank0_republish_claims_only_new_cached_chunks(monkeypatch):
    """Decode-save republish must not re-pin/ref old shared backing chunks."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    old_mem_obj = _FakeClaimableMemObj()
    old_mem_obj.is_pinned = True
    old_mem_obj.pin_count = 1
    new_mem_obj = _FakeClaimableMemObj()
    cached_shared_handles = [["old-handle"]]
    broadcasts = []
    key0 = _make_key()
    key1 = _make_key()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0, 1],
        [1, 2],
        [[key0, key1]],
    )

    def make_handles(**kwargs):
        assert kwargs["mem_objs_layer"] == [new_mem_obj]
        assert kwargs["keys_layer"] == [key1]
        assert kwargs["chunk_index_base"] == 1
        assert old_mem_obj.ref_up_count == 0
        assert old_mem_obj.pin_count == 1
        assert new_mem_obj.ref_up_count == 1
        assert new_mem_obj.pin_count == 1
        return ["new-handle"]

    engine._make_shared_handles_for_layer = make_handles
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)

    def request_object_ids(req_id, kv_group):
        assert (req_id, kv_group) == ("req-1", 0)
        return {id(old_mem_obj)}

    engine.shared_cpu_rank0_request_object_ids = request_object_ids

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[[key0, key1]],
        cached_starts=[0, 1],
        cached_ends=[1, 2],
        cached_memory_objs=[[old_mem_obj, new_mem_obj]],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0, 1], 0))
    retriever.close()

    assert len(broadcasts) == 1
    assert broadcasts[0].status == "ok"
    assert broadcasts[0].handles == ["new-handle"]
    assert cached_shared_handles == [["old-handle", "new-handle"]]
    assert old_mem_obj.ref_up_count == 0
    assert old_mem_obj.unpin_count == 0
    assert new_mem_obj.ref_up_count == 1
    assert new_mem_obj.unpin_count == 0


def test_sparse_rank0_retrieves_and_publishes_only_missing_suffix(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    old_mem_obj = _FakeTensorMemObj(torch.empty(1))
    new_mem_obj = _FakePinnedMemObj()
    old_handle, new_handle = object(), object()
    resolved_keys = []
    broadcasts = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=False)
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine._shared_cpu_request_leases = {}
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0, 1],
        [1, 2],
        [[key0, key1]],
    )
    engine._find_shared_rank0_chunk_location = lambda key: (
        resolved_keys.append(key) or "LocalCPUBackend"
    )
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}

    def resolve_layer(**kwargs):
        assert kwargs["keys_layer"] == [key1]
        return [new_mem_obj]

    engine._resolve_shared_rank0_layer_mem_objs = resolve_layer

    def make_handles(**kwargs):
        assert kwargs["keys_layer"] == [key1]
        assert kwargs["mem_objs_layer"] == [new_mem_obj]
        assert kwargs["chunk_index_base"] == 1
        return [new_handle]

    engine._make_shared_handles_for_layer = make_handles
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)
    cached_memory_objs = [[old_mem_obj]]
    cached_tensors = [[old_mem_obj.tensor]]
    cached_shared_handles = [[old_handle]]

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[[key0]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0, 1], 0))
    retriever.close()

    assert resolved_keys == [key1]
    assert cached_memory_objs == [[old_mem_obj, new_mem_obj]]
    assert cached_shared_handles == [[old_handle, new_handle]]
    assert len(broadcasts) == 1
    assert broadcasts[0].handles == [new_handle]


def test_sparse_per_rank_retrieves_missing_suffix_without_shared_handles(
    monkeypatch,
):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    old_mem_obj = _FakeTensorMemObj(torch.empty(1))
    new_mem_obj = _FakeTensorMemObj(torch.empty(1))
    get_calls = []

    def layerwise_batched_get(keys, *, location):
        get_calls.append((keys, location))
        yield SimpleNamespace(result=lambda: [new_mem_obj])

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=False)
    engine.storage_manager = SimpleNamespace(
        layerwise_batched_get=layerwise_batched_get
    )
    engine.gpu_connector = _FakeSparseConsumer()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "MooncakeStore",
        [0, 1],
        [1, 2],
        [[key0, key1]],
    )
    cached_memory_objs = [[old_mem_obj]]
    cached_tensors = [[old_mem_obj.tensor]]
    cached_shared_handles = []

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[[key0, key1]],
        cached_starts=[0, 1],
        cached_ends=[1, 2],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0, 1], 0))
    retriever.close()

    assert get_calls == [([[key1]], "MooncakeStore")]
    assert cached_memory_objs == [[old_mem_obj, new_mem_obj]]
    assert cached_tensors == [[old_mem_obj.tensor, new_mem_obj.tensor]]
    assert cached_shared_handles == []


def test_sparse_shared_extension_still_requires_complete_handles():
    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    old_mem_obj = _FakeTensorMemObj(torch.empty(1))
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0, 1],
        [1, 2],
        [[key0, key1]],
    )

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[[key0, key1]],
        cached_starts=[0, 1],
        cached_ends=[1, 2],
        cached_memory_objs=[[old_mem_obj]],
        cached_tensors=[[old_mem_obj.tensor]],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    with pytest.raises(ValueError, match="requires complete cached handles"):
        next(retriever)


def test_sparse_rank0_hot_shared_handles_do_not_republish(monkeypatch):
    """Hot request-scoped shared handles do not create extra collectives."""

    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    key = _make_key()
    cached_shared_handles = [["existing-handle"]]
    broadcasts = []
    events = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.storage_manager = object()
    engine.gpu_connector = _OrderedSparseConsumer(events)
    engine.shared_cpu_cache_generation = 7
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[key]],
    )
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[[key]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=[[object()]],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )

    next(retriever)
    retriever.send(([0], 0))

    assert broadcasts == []
    assert "store_fence" not in events
    assert cached_shared_handles == [["existing-handle"]]


def test_sparse_retrieve_incomplete_request_cache_fails_loudly():
    key_layers = _make_key().split_layers(2)
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: False
    engine._is_passive = lambda: False
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[key_layers[0]], [key_layers[1]]],
    )

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[[key_layers[0]], [key_layers[1]]],
        cached_starts=[0],
        cached_ends=[1],
        cached_memory_objs=[],
        cached_tensors=[[torch.empty(1)], []],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-1",
    )

    with pytest.raises(ValueError, match="inconsistent per-layer chunk counts"):
        next(retriever)


def test_sparse_pointer_cache_reuse_rejects_invalid_dtype():
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    cached_ptrs = [torch.tensor([1234], dtype=torch.int32)]

    with pytest.raises(RuntimeError, match="invalid dtype"):
        connector._resolve_sparse_chunk_ptrs_npu(
            0,
            [torch.empty(1)],
            cached_ptrs,
        )


def test_sparse_pointer_cache_reuse_rejects_wrong_device():
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("meta")
    cached_ptrs = [torch.tensor([1234], dtype=torch.long)]

    with pytest.raises(RuntimeError, match="wrong device"):
        connector._resolve_sparse_chunk_ptrs_npu(
            0,
            [torch.empty(1)],
            cached_ptrs,
        )


def test_sparse_pointer_cache_reuse_does_not_read_pointer_values(monkeypatch):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    cached_ptrs = [torch.tensor([0], dtype=torch.long)]

    def fail_any(*_args, **_kwargs):
        raise AssertionError("hot path must not read cached pointer values")

    monkeypatch.setattr(npu_connectors.torch, "any", fail_any)

    assert connector._resolve_sparse_chunk_ptrs_npu(
        0,
        [torch.empty(1)],
        cached_ptrs,
    ) is cached_ptrs[0]


def test_append_retrieve_layer_cache_is_atomic_when_pointer_install_fails():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2

    def fail_pointer_install(*_args, **_kwargs):
        raise RuntimeError("pointer install failed")

    engine.gpu_connector = type(
        "FakeConnector",
        (),
        {"append_sparse_chunk_ptr_cache_for_layer": fail_pointer_install},
    )()
    cached_memory_objs = []
    cached_tensors = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    with pytest.raises(RuntimeError, match="pointer install failed"):
        engine._append_retrieve_layer_cache(
            0,
            [_FakeTensorMemObj(torch.empty(1))],
            cached_memory_objs,
            cached_tensors,
            cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu,
        )

    assert cached_memory_objs == []
    assert cached_tensors == []
    assert cached_chunk_dev_ptrs == []
    assert cached_chunk_ptrs_npu == []


def test_append_retrieve_layer_cache_rejects_missing_tensor_without_shift():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = object()
    cached_memory_objs = []
    cached_tensors = []

    with pytest.raises(ValueError, match="without a tensor"):
        engine._append_retrieve_layer_cache(
            0,
            [
                _FakeTensorMemObj(torch.empty(1)),
                _FakeTensorMemObj(None),
                _FakeTensorMemObj(torch.empty(1)),
            ],
            cached_memory_objs,
            cached_tensors,
            [],
            [],
        )

    assert cached_memory_objs == []
    assert cached_tensors == []


def test_sparse_pointer_cache_append_failure_is_atomic(monkeypatch):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    connector.kv_device = torch.device("cpu")
    monkeypatch.setattr(npu_connectors.lmc_ops, "get_device_ptr", lambda _ptr: None)
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    with pytest.raises(RuntimeError, match="pointer-cache install failed"):
        connector.append_sparse_chunk_ptr_cache_for_layer(
            0,
            [torch.empty(1)],
            cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu,
        )

    assert cached_chunk_dev_ptrs == []
    assert cached_chunk_ptrs_npu == []


def test_sparse_pointer_cache_tensor_build_failure_is_atomic(monkeypatch):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    connector.kv_device = torch.device("cpu")
    monkeypatch.setattr(npu_connectors.lmc_ops, "get_device_ptr", lambda _ptr: 1234)

    def fail_tensor(*_args, **_kwargs):
        raise RuntimeError("npu pointer tensor allocation failed")

    monkeypatch.setattr(npu_connectors.torch, "tensor", fail_tensor)
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    with pytest.raises(RuntimeError, match="pointer tensor allocation failed"):
        connector.append_sparse_chunk_ptr_cache_for_layer(
            0,
            [torch.empty(1)],
            cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu,
        )

    assert cached_chunk_dev_ptrs == []
    assert cached_chunk_ptrs_npu == []


def test_sparse_passive_pointer_failure_releases_unpublished_view(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    key = _make_key()
    mem_obj = _FakeTensorMemObj(torch.empty(1))
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _FakePassiveAllocator(mem_obj)
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[object()],
    )
    engine._append_retrieve_layer_cache = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(RuntimeError("pointer install failed"))
    )
    cached_memory_objs = []
    cached_tensors = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []
    cached_shared_handles = []

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-1",
    )
    next(retriever)

    with pytest.raises(RuntimeError, match="pointer install failed"):
        retriever.send(([0], 0))

    assert mem_obj.release_count == 1
    assert not mem_obj.is_valid()
    assert cached_memory_objs == []
    assert cached_tensors == []
    assert cached_chunk_dev_ptrs == []
    assert cached_chunk_ptrs_npu == []
    assert cached_shared_handles == []
