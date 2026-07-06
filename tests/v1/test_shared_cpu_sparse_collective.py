# SPDX-License-Identifier: Apache-2.0
"""Shared CPU sparse decode collective-order tests."""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.shared_cpu_cache import SharedHandleEnvelope
import lmcache_ascend.v1.cache_engine as ascend_cache_engine
import lmcache_ascend.v1.npu_connector.npu_connectors as npu_connectors
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


class _FakePinnedMemObj:
    def __init__(self, ref_count=1):
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


class _FakeTokenDatabase:
    def __init__(self, key):
        self.key = key

    def process_tokens(self, **_kwargs):
        yield 0, 1, self.key


def _make_key(kv_group=0):
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=1234,
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
    engine._min_layer_cache_chunks = lambda *_args: 0
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
    engine._min_layer_cache_chunks = lambda *_args: 0
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
    engine._min_layer_cache_chunks = lambda *_args: 0
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
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._min_layer_cache_chunks = AscendLMCacheEngine._min_layer_cache_chunks
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
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = AscendLMCacheEngine._has_retrieve_data_cache
    engine._retrieve_data_cache_covers = (
        AscendLMCacheEngine._retrieve_data_cache_covers
    )
    engine._min_layer_cache_chunks = AscendLMCacheEngine._min_layer_cache_chunks
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
    engine._min_layer_cache_chunks = AscendLMCacheEngine._min_layer_cache_chunks
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
    assert cached_shared_handles == [["existing-handle"]]


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
