# SPDX-License-Identifier: Apache-2.0
"""Sparse decode cache-state and shared collective-order tests."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import (
    _SHARED_SPARSE_DEFER_COMMIT,
    _SHARED_SPARSE_PREPARE_ONLY,
)
from lmcache.v1.memory_management import LayerPageMemoryObj, LayerPageSource
from lmcache.v1.shared_cpu_cache import SharedHandleBatch, SharedHandleEnvelope
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult
import lmcache_ascend.v1.cache_engine as ascend_cache_engine
import lmcache_ascend.v1.npu_connector.npu_connectors as npu_connectors
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


def test_direct_indexer_and_cpu_fallback_never_enter_shared_collectives() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine._should_use_shared_layerwise_retrieve = lambda kv_group: True
    engine._is_shared_retrieve_passive = lambda kv_group: True

    assert engine._cold_retrieve_collective_mode(1, True) == (False, False)
    assert engine._cold_retrieve_collective_mode(1, False) == (True, True)


def test_direct_indexer_missing_readiness_uses_cpu_fallback() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.gpu_connector = SimpleNamespace(
        plan_direct_page_destinations=lambda *args: ([[1]], [[1]], ())
    )
    engine.storage_manager = SimpleNamespace(
        batched_get_external_pages=lambda *args: pytest.fail(
            "direct I/O must not start without readiness publication"
        )
    )

    assert not engine._try_direct_indexer_page_load(
        req_id="request",
        keys_layer_major=[[object()]],
        starts=[0],
        ends=[1],
        retrieve_kwargs={
            "kvcaches": [object()],
            "slot_mapping": torch.tensor([0]),
        },
    )


def test_cold_direct_indexer_defers_readiness_to_request_event() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    calls = []
    engine.gpu_connector = SimpleNamespace(
        plan_direct_page_destinations=lambda *args: ([[7]], [[8]], (object(),)),
        record_dense_load_readiness=lambda: calls.append("record"),
        consume_dense_load_readiness=lambda event: calls.append(("consume", event)),
    )
    engine.storage_manager = SimpleNamespace(
        batched_get_external_pages=lambda *args: calls.append("get")
    )
    key = CacheEngineKey("model", 1, 0, 7, torch.float16).split_layers(1)[0]

    assert engine._try_direct_indexer_page_load(
        req_id="request",
        keys_layer_major=[[key]],
        starts=[0],
        ends=[1],
        retrieve_kwargs={
            "kvcaches": [object()],
            "slot_mapping": torch.tensor([0]),
            "_defer_direct_load_readiness": True,
        },
    )
    assert calls == ["get"]


def test_cold_direct_indexer_failure_fences_before_cpu_fallback() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    calls = []
    owner = object()
    event = object()
    engine.gpu_connector = SimpleNamespace(
        plan_direct_page_destinations=lambda *args: ([[7]], [[8]], (owner,)),
        record_dense_load_readiness=lambda: calls.append("record") or event,
        consume_dense_load_readiness=lambda value: calls.append(
            ("consume", value)
        ),
    )

    def fail_after_submit(*_args):
        calls.append("get")
        raise RuntimeError("partial direct submit")

    engine.storage_manager = SimpleNamespace(
        batched_get_external_pages=fail_after_submit
    )
    key = CacheEngineKey("model", 1, 0, 7, torch.float16).split_layers(1)[0]

    assert not engine._try_direct_indexer_page_load(
        req_id="request",
        keys_layer_major=[[key]],
        starts=[0],
        ends=[1],
        retrieve_kwargs={
            "kvcaches": [object()],
            "slot_mapping": torch.tensor([0]),
            "_defer_direct_load_readiness": True,
        },
    )
    assert calls == ["get", "record", ("consume", event)]


def test_cold_direct_indexer_cpu_fallback_reuses_layer_pages(monkeypatch) -> None:
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    monkeypatch.setattr(
        ascend_cache_engine,
        "mooncake_layer_pages_enabled",
        lambda _config: True,
    )
    monkeypatch.setattr(
        ascend_cache_engine,
        "mooncake_page_layout_enabled",
        lambda _config: True,
    )
    key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    page = _FakePinnedMemObj()
    resolved = [[page], [page]]
    resolve_calls = []
    cached_memory_objs = []

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=True)
    engine.metadata = SimpleNamespace(worker_id=0)
    engine.storage_manager = SimpleNamespace(
        layerwise_batched_get=lambda *_args, **_kwargs: pytest.fail(
            "merged-page fallback must not enter legacy layerwise retrieval"
        )
    )
    engine.gpu_connector = _FakeSparseConsumer()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    def ensure_metadata(**kwargs):
        kwargs["ret_mask"].fill_(True)
        return (
            "mixed",
            [0],
            [1],
            [[layer_key] for layer_key in key.split_layers(2)],
        )

    engine._ensure_retrieve_chunk_metadata = ensure_metadata
    engine._try_direct_indexer_page_load = lambda **_kwargs: False
    engine._get_shared_config_value = lambda _name, default: default
    engine._use_sampled_worker_retrieve = lambda _kv_group: True

    def resolve_pages(**kwargs):
        resolve_calls.append(kwargs)
        return resolved, 1

    engine._resolve_shared_rank0_layer_pages = resolve_pages

    def append_group(sources, owners, *_args):
        owners.extend([[*source.pages, *source.suffix] for source in sources])

    engine._append_retrieve_group_cache = append_group

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kvcaches=[object(), object()],
        slot_mapping=torch.tensor([0]),
        direct_external_pages=True,
        kv_group=1,
        req_id="request",
    )

    next(retriever)
    retriever.send(None)
    result = retriever.send(None)
    retriever.close()

    assert result.tolist() == [True]
    assert len(resolve_calls) == 1
    assert resolve_calls[0]["page_chunks"] == 1
    assert cached_memory_objs == resolved
    assert page.release_count == 0
    assert page.unpin_count == 0


def test_rank0_partial_page_remains_one_layer_page_source(monkeypatch) -> None:
    key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    keys = [[layer_key] for layer_key in key.split_layers(2)]
    page_key = CacheEngineKey(
        "model",
        1,
        0,
        7,
        torch.float16,
        {"lmcache.tag.internal.valid_tokens": 3},
    )
    requested_page_keys = []
    page = SimpleNamespace(
        valid_tokens=3,
        num_layers=2,
        get_shape=lambda: torch.Size([3, 4]),
    )
    local = SimpleNamespace(
        batched_get_layer_page_prefix=lambda base_keys: (
            requested_page_keys.extend(base_keys) or [page],
            1,
        ),
        contains_all_exact=lambda layer_keys: False,
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = SimpleNamespace(storage_backends={})
    engine._shared_local_cpu_backend = lambda: local
    engine._expected_shared_cpu_chunk_metadata = (
        lambda **kwargs: (torch.Size([kwargs["num_tokens"], 4]), torch.float16, None)
    )
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None
    monkeypatch.setattr(
        ascend_cache_engine.LayerPageMemoryObj,
        "pin_many",
        lambda pages: True,
    )

    resolved, page_chunks = engine._resolve_shared_rank0_layer_pages(
        req_id="request",
        phase="cold",
        kv_group=0,
        keys_layer_major=keys,
        page_chunks=1,
        base_page_keys=[page_key],
    )
    sources = [
        LayerPageSource(tuple(layer[:page_chunks]), layer_id)
        for layer_id, layer in enumerate(resolved)
    ]

    assert page_chunks == 1
    assert requested_page_keys == [page_key]
    assert all(source.pages == (page,) and not source.suffix for source in sources)


def test_rank0_page_miss_expands_base_key_for_legacy_fallback() -> None:
    page_key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    fallback_keys = []
    suffix = [[object()], [object()]]
    local = SimpleNamespace(
        batched_get_layer_page_prefix=lambda _keys: ([], 0),
        contains_all_exact=lambda _keys: False,
    )
    remote = SimpleNamespace(
        batched_contains_layer_pages=lambda _keys: 0,
        batched_get_layer_pages=lambda _keys: pytest.fail(
            "a missing page must use legacy retrieval"
        ),
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = SimpleNamespace(
        storage_backends={"RemoteBackend": remote}
    )
    engine._shared_local_cpu_backend = lambda: local

    def resolve_legacy(**kwargs):
        fallback_keys.extend(kwargs["keys_layer_major"])
        return suffix

    engine._resolve_shared_rank0_page_first_layers = resolve_legacy

    resolved, page_chunks = engine._resolve_shared_rank0_layer_pages(
        req_id="request",
        phase="dense_prefix",
        kv_group=0,
        keys_layer_major=[[page_key], [page_key]],
        page_chunks=1,
        base_page_keys=[page_key],
    )

    assert page_chunks == 0
    assert resolved == suffix
    assert fallback_keys == [[page_key.get_layer(0)], [page_key.get_layer(1)]]
    assert all(
        isinstance(key, ascend_cache_engine.LayerCacheEngineKey)
        for layer in fallback_keys
        for key in layer
    )


def test_partial_legacy_state_does_not_hide_remote_layer_page(monkeypatch) -> None:
    page_key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    page = SimpleNamespace(
        valid_tokens=4,
        num_layers=2,
        get_shape=lambda: torch.Size([4, 4]),
    )
    local = SimpleNamespace(
        batched_get_layer_page_prefix=lambda _keys: ([], 0),
        contains_all_exact=lambda _keys: False,
    )
    remote = SimpleNamespace(
        batched_contains_layer_pages=lambda keys: len(keys),
        batched_get_layer_pages=lambda _keys: [page],
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = SimpleNamespace(
        storage_backends={"RemoteBackend": remote}
    )
    engine._shared_local_cpu_backend = lambda: local
    engine._expected_shared_cpu_chunk_metadata = (
        lambda **kwargs: (torch.Size([kwargs["num_tokens"], 4]), torch.float16, None)
    )
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None
    engine._resolve_shared_rank0_page_first_layers = lambda **kwargs: pytest.fail(
        "a valid merged remote page must win over partial legacy state"
    )
    monkeypatch.setattr(
        ascend_cache_engine.LayerPageMemoryObj,
        "pin_many",
        lambda pages: True,
    )

    resolved, page_chunks = engine._resolve_shared_rank0_layer_pages(
        req_id="request",
        phase="dense_prefix",
        kv_group=0,
        keys_layer_major=[
            [page_key.get_layer(0)],
            [page_key.get_layer(1)],
        ],
        page_chunks=1,
        base_page_keys=[page_key],
    )

    assert page_chunks == 1
    assert resolved == [[page], [page]]


def test_live_import_admission_transfers_temporary_page_ownership() -> None:
    class _Page:
        def __init__(self) -> None:
            self.is_pinned = True
            self.unpins = 0
            self.releases = 0

        def unpin(self) -> None:
            self.is_pinned = False
            self.unpins += 1

        def is_valid(self) -> bool:
            return True

        def ref_count_down(self) -> None:
            self.releases += 1

    page = _Page()
    submissions = []
    key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    local = SimpleNamespace(
        batched_submit_layer_pages=lambda keys, pages: submissions.append(
            (list(keys), list(pages))
        ),
        contains_compatible_layer_pages_exact=lambda keys, pages: (
            list(keys) == [key] and list(pages) == [page]
        ),
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine._shared_local_cpu_backend = lambda: local
    context = {"keys": [key], "pages": [page]}

    engine.admit_live_split_pages(context)

    assert submissions == [([key], [page])]
    assert context["pages"] == []
    assert page.unpins == 1
    assert page.releases == 1
    with pytest.raises(RuntimeError, match="no latent pages"):
        engine.admit_live_split_pages(context)


def test_live_import_admission_releases_duplicate_import() -> None:
    class _Page:
        is_pinned = True

        def __init__(self) -> None:
            self.unpins = 0
            self.releases = 0

        def unpin(self) -> None:
            self.is_pinned = False
            self.unpins += 1

        def is_valid(self) -> bool:
            return True

        def ref_count_down(self) -> None:
            self.releases += 1

    page = _Page()
    key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    engine = object.__new__(AscendLMCacheEngine)
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        # A pre-existing exact key wins; LocalCPU deliberately takes no ref
        # from the duplicate page supplied here.
        batched_submit_layer_pages=lambda _keys, _pages: None,
        contains_compatible_layer_pages_exact=lambda _keys, _pages: True,
    )
    context = {"keys": [key], "pages": [page]}

    engine.admit_live_split_pages(context)

    assert context["pages"] == []
    assert (page.unpins, page.releases) == (1, 1)


@pytest.mark.parametrize("layer_scoped", [False, True])
def test_live_import_accepts_page_and_layer_keys(layer_scoped: bool) -> None:
    page_key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    key = page_key.get_layer(0) if layer_scoped else page_key
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine._dense_retrieve_token_results = lambda *_args: [(0, 4, key)]
    engine.gpu_connector = SimpleNamespace(
        plan_compact_page_layout=lambda *_args: (
            [{"layer_id": 0, "buffer_base": 100,
              "token_bytes": 4, "slot_capacity": 4}],
            [{"logical_token_start": 0, "physical_slot_start": 0,
              "token_count": 4}],
            ("owner",),
        )
    )

    plan, context = engine._prepare_live_split_import(
        tokens=[1, 2, 3, 4],
        indexer_slots=torch.arange(4),
        indexer_kvcaches=[object()],
        request_configs=None,
        tp_rank=0,
        dp_rank=0,
        handled_groups=(1,),
    )

    assert context["keys"] == [page_key]
    assert plan["group_byte_totals"] == (0, 16)
    assert plan["segments"] == []
    assert plan["format"] == "layer_slot_runs_v1"
    assert plan["compact_layout"]["group_id"] == 1
    assert "latent_pages" not in plan
    assert context["pages"] == []


def test_live_import_hybrid_plan_uses_rank0_cpu_pages() -> None:
    class _Page:
        data_ptr = 3000
        is_pinned = False
        valid_tokens = 3

        def get_size(self) -> int:
            return 96

    page_key = CacheEngineKey("model", 1, 0, 7, torch.float16)
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.num_layers = 2
    engine._is_passive = lambda: False
    engine._dense_retrieve_token_results = lambda *_args: [(0, 3, page_key)]
    engine._ensure_layerwise_connector_layout = MagicMock()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([3, 4]),
        torch.float16,
        object(),
    )
    local = SimpleNamespace(
        batched_allocate_layer_pages=MagicMock(return_value=[_Page()])
    )
    engine._shared_local_cpu_backend = lambda: local
    engine.gpu_connector = SimpleNamespace(
        direct_page_token_widths=lambda caches, group: (
            (8, 24) if caches == ["latent"] and group == 0 else None
        ),
        plan_compact_page_layout=lambda *_args: (
            [
                {
                    "layer_id": 0,
                    "buffer_base": 100,
                    "token_bytes": 4,
                    "slot_capacity": 4,
                }
            ],
            [
                {
                    "logical_token_start": 0,
                    "physical_slot_start": 0,
                    "token_count": 3,
                }
            ],
            ("owner",),
        ),
    )

    with patch.object(LayerPageMemoryObj, "pin_many", return_value=True):
        plan, context = engine._prepare_live_split_import(
            tokens=[1, 2, 3],
            latent_kvcaches=["latent"],
            indexer_slots=torch.arange(3),
            indexer_kvcaches=[object()],
            request_configs=None,
            tp_rank=0,
            dp_rank=0,
            handled_groups=(0, 1),
        )

    assert plan["format"] == "hybrid_compact_v1"
    assert plan["segments"] == []
    assert plan["group_byte_totals"] == (96, 12)
    assert plan["latent_pages"] == [
        {
            "logical_token_start": 0,
            "destination_address": 3000,
            "length": 96,
            "valid_tokens": 3,
        }
    ]
    assert plan["latent_token_bytes"] == [8, 24]
    assert context["pages"][0].data_ptr == 3000
    engine._ensure_layerwise_connector_layout.assert_called_once_with(
        kvcaches=["latent"], kv_group=0
    )
    assert (
        local.batched_allocate_layer_pages.call_args.kwargs["busy_loop"]
        is False
    )


def test_live_import_passive_rank_rejects_group0_before_allocation() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine._is_passive = lambda: True
    engine._dense_retrieve_token_results = lambda *_args: [
        (0, 1, CacheEngineKey("model", 1, 0, 7, torch.float16))
    ]
    engine._shared_local_cpu_backend = lambda: pytest.fail(
        "passive rank must not access rank-0 LocalCPU allocation"
    )

    with pytest.raises(RuntimeError, match="Passive ranks"):
        engine._prepare_live_split_import(
            tokens=[1],
            latent_kvcaches=[object()],
            indexer_slots=torch.arange(1),
            indexer_kvcaches=[object()],
            request_configs=None,
            tp_rank=1,
            dp_rank=0,
            handled_groups=(0, 1),
        )


def test_live_import_pin_failure_releases_allocated_page_owner() -> None:
    class _Page:
        is_pinned = False
        releases = 0

        def is_valid(self) -> bool:
            return True

        def ref_count_down(self) -> None:
            self.releases += 1

    page = _Page()
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.num_layers = 1
    engine._is_passive = lambda: False
    engine._dense_retrieve_token_results = lambda *_args: [
        (0, 1, CacheEngineKey("model", 1, 0, 7, torch.float16))
    ]
    engine._ensure_layerwise_connector_layout = MagicMock()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1, 1]), torch.float16, object()
    )
    engine._shared_local_cpu_backend = lambda: SimpleNamespace(
        batched_allocate_layer_pages=MagicMock(return_value=[page])
    )
    engine.gpu_connector = SimpleNamespace(
        direct_page_token_widths=lambda *_args: (2, 2)
    )

    with (
        patch.object(
            LayerPageMemoryObj,
            "pin_many",
            side_effect=RuntimeError("monitor failure"),
        ),
        pytest.raises(RuntimeError, match="monitor failure"),
    ):
        engine._prepare_live_split_import(
            tokens=[1],
            latent_kvcaches=[object()],
            indexer_slots=torch.arange(1),
            indexer_kvcaches=[object()],
            request_configs=None,
            tp_rank=0,
            dp_rank=0,
            handled_groups=(0, 1),
        )

    assert page.releases == 1


def test_live_source_record_is_cumulative_deduplicated_and_exact() -> None:
    class _Owner:
        def __init__(self, base: int, size: int) -> None:
            self.base = base
            self.size = size

        def data_ptr(self) -> int:
            return self.base

        def numel(self) -> int:
            return self.size // 2

        def element_size(self) -> int:
            return 2

    engine = object.__new__(AscendLMCacheEngine)
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request")
    owners = (_Owner(1000, 100), _Owner(2000, 100))

    for group, base in ((0, 1000), (1, 2000)):
        engine._record_live_source_pages(
            "request", group, [(0, 4)], [[base]], [[16]], owners
        )
        # Repeated final-layer/wait-for-save observation must not duplicate.
        engine._record_live_source_pages(
            "request", group, [(0, 4)], [[base]], [[16]], owners
        )
        engine._record_live_source_pages(
            "request", group, [(4, 6)], [[base + 16]], [[8]], owners
        )

    assert engine.finalize_live_source_descriptor("request", 6, 3, 1)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["group_byte_totals"] == [24, 24]
    assert (descriptor["tp_rank"], descriptor["dp_rank"]) == (3, 1)
    assert [segment["group_id"] for segment in descriptor["segments"]] == [
        0, 0, 1, 1
    ]
    assert [segment["source_offset"] for segment in descriptor["segments"]] == [
        0, 16, 0, 16
    ]
    assert [segment["source_buffer_base"] for segment in descriptor["segments"]] == [
        1000, 1000, 2000, 2000
    ]
    assert not engine.finalize_live_source_descriptor("missing", 6, 3, 1)


def test_compact_live_source_ignores_rank0_direct_store_observation() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request", (1,))
    layers = [
        {
            "layer_id": 0,
            "buffer_base": 1000,
            "token_bytes": 4,
            "slot_capacity": 16,
        }
    ]
    runs = [
        {
            "logical_token_start": 0,
            "physical_slot_start": 2,
            "token_count": 4,
        }
    ]

    engine._record_live_source_layout("request", 1, 0, 4, layers, runs)
    engine._record_live_source_pages(
        "request", 1, [(0, 4)], [[1008]], [[16]], ()
    )

    assert engine.finalize_live_source_descriptor("request", 4, 0, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["compact_layout"]["layers"] == layers
    assert descriptor["compact_layout"]["runs"] == runs
    assert descriptor["group_byte_totals"] == [0, 16]


def test_live_source_capture_defers_partial_tail_until_final_step() -> None:
    class _Owner:
        def data_ptr(self) -> int:
            return 1000

        def numel(self) -> int:
            return 100

        def element_size(self) -> int:
            return 1

    class _Tokens:
        @staticmethod
        def process_tokens(*, tokens=None, hashes=None, kv_group=0, **_kwargs):
            if hashes is not None:
                return iter(
                    (0, size, SimpleNamespace(chunk_hash=value))
                    for value, size in zip(hashes, _kwargs["offsets"], strict=True)
                )
            chunks = [(0, 4, SimpleNamespace(chunk_hash=11))]
            if len(tokens) > 4:
                chunks.append((4, 6, SimpleNamespace(chunk_hash=12)))
            return iter(chunks)

        @staticmethod
        def process_tokens_from_prefix(
            tokens, *, prefix_token_count, prefix_hash, **_kwargs
        ):
            assert len(tokens) == 6
            assert (prefix_token_count, prefix_hash) == (4, 11)
            return iter(((4, 6, SimpleNamespace(chunk_hash=12)),))

    def planner(_caches, _slots, starts, ends, _group, **_kwargs):
        return (
            [{"layer_id": 0, "buffer_base": 1000,
              "token_bytes": 1, "slot_capacity": 100}],
            [{"logical_token_start": starts[0],
              "physical_slot_start": starts[0],
              "token_count": ends[-1] - starts[0]}],
            (_Owner(),),
        )

    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.token_database = _Tokens()
    engine.gpu_connector = SimpleNamespace(plan_compact_page_layout=planner)
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    caches = {0: [object()], 1: [object()]}
    slots = {0: object(), 1: object()}

    engine.capture_live_source_step(
        "request", [1] * 6, caches, slots, None, 0, final=False
    )
    assert engine._live_source_builders["request"]["ends"] == {1: 4}
    engine.capture_live_source_step(
        "request", [1] * 6, caches, slots, None, 4, final=True
    )
    assert engine._live_source_builders["request"]["ends"] == {1: 6}
    assert engine.finalize_live_source_descriptor("request", 6, 0, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["group_byte_totals"] == [0, 6]
    assert descriptor["compact_layout"]["token_count"] == 6
    assert len(descriptor["compact_layout"]["runs"]) == 2


def test_hybrid_live_source_emits_compact_latent_pages() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request", (0, 1))
    latent_layers = [
        {
            "layer_id": 0,
            "buffer_base": 1000,
            "token_bytes": 8,
            "slot_capacity": 16,
        }
    ]
    latent_pages = [
        {
            "logical_token_start": 0,
            "token_count": 3,
            "runs": [
                {
                    "logical_token_start": 0,
                    "physical_slot_start": 2,
                    "token_count": 3,
                }
            ],
        },
        {
            "logical_token_start": 3,
            "token_count": 1,
            "runs": [
                {
                    "logical_token_start": 3,
                    "physical_slot_start": 9,
                    "token_count": 1,
                }
            ],
        },
    ]
    index_layers = [
        {
            "layer_id": 0,
            "buffer_base": 2000,
            "token_bytes": 4,
            "slot_capacity": 16,
        }
    ]
    index_runs = [
        {
            "logical_token_start": 0,
            "physical_slot_start": 4,
            "token_count": 4,
        }
    ]

    engine._record_live_latent_source_layout(
        "request", 0, 4, latent_layers, latent_pages
    )
    engine._record_live_source_layout(
        "request", 1, 0, 4, index_layers, index_runs
    )

    assert engine.finalize_live_source_descriptor("request", 4, 0, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["format"] == "layer_slot_runs_v1"
    assert descriptor["group_byte_totals"] == [0, 16]
    assert descriptor["latent_group_byte_total"] == 32
    assert descriptor["latent_layout"] == {
        "group_id": 0,
        "token_count": 4,
        "layers": latent_layers,
        "pages": latent_pages,
    }
    assert descriptor["compact_layout"]["group_id"] == 1


def test_latent_plan_failure_preserves_group1_live_source() -> None:
    class _Tokens:
        @staticmethod
        def process_tokens(*, tokens=None, hashes=None, **kwargs):
            if hashes is not None:
                return iter(
                    (0, size, SimpleNamespace(chunk_hash=value))
                    for value, size in zip(
                        hashes, kwargs["offsets"], strict=True
                    )
                )
            return iter(((0, len(tokens), SimpleNamespace(chunk_hash=11)),))

    def index_planner(_caches, _slots, starts, ends, _group, **_kwargs):
        return (
            [{"layer_id": 0, "buffer_base": 2000,
              "token_bytes": 4, "slot_capacity": 16}],
            [{"logical_token_start": starts[0],
              "physical_slot_start": 2,
              "token_count": ends[-1] - starts[0]}],
            (),
        )

    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.token_database = _Tokens()
    engine.gpu_connector = SimpleNamespace(
        plan_compact_page_layout=index_planner,
        plan_compact_latent_page_layout=lambda *_args, **_kwargs: None,
    )
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request", (0, 1))

    engine.capture_live_source_step(
        "request",
        [1, 2, 3, 4],
        {0: [object()], 1: [object()]},
        {0: object(), 1: object()},
        None,
        0,
        final=True,
    )

    assert engine.finalize_live_source_descriptor("request", 4, 0, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["format"] == "layer_slot_runs_v1"
    assert descriptor["group_byte_totals"] == [0, 16]
    assert "latent_layout" not in descriptor


def test_legacy_group1_source_disables_incompatible_latent_extension() -> None:
    class _Tokens:
        @staticmethod
        def process_tokens(*, tokens=None, hashes=None, **kwargs):
            if hashes is not None:
                return iter(
                    (0, size, SimpleNamespace(chunk_hash=value))
                    for value, size in zip(
                        hashes, kwargs["offsets"], strict=True
                    )
                )
            return iter(((0, len(tokens), SimpleNamespace(chunk_hash=11)),))

    owner = MagicMock()
    owner.data_ptr.return_value = 2000
    owner.numel.return_value = 16
    owner.element_size.return_value = 1
    latent_planner = MagicMock()

    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(chunk_size=4)
    engine.token_database = _Tokens()
    engine.gpu_connector = SimpleNamespace(
        plan_compact_page_layout=lambda *_args, **_kwargs: None,
        plan_compact_latent_page_layout=latent_planner,
        plan_direct_page_sources=lambda *_args, **_kwargs: (
            [[2000]], [[16]], (owner,)
        ),
    )
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request", (0, 1))

    engine.capture_live_source_step(
        "request",
        [1, 2, 3, 4],
        {0: [object()], 1: [object()]},
        {0: object(), 1: object()},
        None,
        0,
        final=True,
    )

    assert engine.finalize_live_source_descriptor("request", 4, 0, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["segments"] == [{
        "group_id": 1,
        "source_buffer_index": 0,
        "source_buffer_base": 2000,
        "source_offset": 0,
        "length": 16,
    }]
    assert descriptor["group_byte_totals"] == [0, 16]
    assert "latent_layout" not in descriptor
    latent_planner.assert_not_called()


def test_group1_only_live_source_keeps_original_wire_format() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine.begin_live_source_descriptor("request", (1,))
    layers = [
        {
            "layer_id": 0,
            "buffer_base": 2000,
            "token_bytes": 4,
            "slot_capacity": 16,
        }
    ]
    runs = [
        {
            "logical_token_start": 0,
            "physical_slot_start": 4,
            "token_count": 4,
        }
    ]
    engine._record_live_source_layout("request", 1, 0, 4, layers, runs)

    assert engine.finalize_live_source_descriptor("request", 4, 1, 0)
    descriptor = engine.drain_live_source_descriptors()["request"]
    assert descriptor["format"] == "layer_slot_runs_v1"
    assert "latent_layout" not in descriptor


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


class _CompactPassiveAllocator:
    shm_name = "test"
    slab_size = 2

    def __init__(self, views):
        self.views = iter(views)

    def create_batch_view(self, *_args, **_kwargs):
        return next(self.views)


class _FakeTokenDatabase:
    def __init__(self, keys):
        self.keys = keys if isinstance(keys, list) else [keys]

    def process_tokens(self, **_kwargs):
        for index, key in enumerate(self.keys):
            yield index, index + 1, key


class _IncrementalTokenDatabase(_FakeTokenDatabase):
    chunk_size = 1

    def __init__(self, keys):
        super().__init__(keys)
        self.full_calls = 0
        self.suffix_calls = 0

    def process_tokens(self, **kwargs):
        self.full_calls += 1
        yield from super().process_tokens(**kwargs)

    def process_tokens_from_prefix(
        self, _tokens, *, prefix_token_count, prefix_hash, **_kwargs
    ):
        self.suffix_calls += 1
        assert self.keys[prefix_token_count - 1].chunk_hash == prefix_hash
        for index, key in enumerate(
            self.keys[prefix_token_count:], start=prefix_token_count
        ):
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


def test_active_metadata_hashes_only_incremental_suffix():
    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    token_database = _IncrementalTokenDatabase([key0, key1])
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=True)
    engine.token_database = token_database
    engine._should_use_shared_layerwise_retrieve = lambda _group: True
    engine._is_shared_retrieve_passive = lambda _group: False
    cached_keys = [[key] for key in key0.split_layers(2)]
    kwargs = {
        "kv_group": 0,
        "cached_metadata_token_ids": [10],
        "shared_cpu_request_preflight_state": {},
    }

    _, starts, ends, keys = engine._ensure_retrieve_chunk_metadata(
        tokens=[10, 11],
        mask=None,
        request_configs=None,
        cached_keys=cached_keys,
        cached_starts=[0],
        cached_ends=[1],
        ret_mask=torch.zeros(2, dtype=torch.bool),
        retrieve_kwargs=kwargs,
    )

    assert starts == [0, 1]
    assert ends == [1, 2]
    assert keys == [
        [key0.split_layers(2)[layer], key1.split_layers(2)[layer]]
        for layer in range(2)
    ]
    assert token_database.suffix_calls == 1
    assert token_database.full_calls == 0
    assert kwargs["_retrieve_metadata_mode"] == "incremental"


@pytest.mark.parametrize(
    "malformation",
    [
        "snapshot_length",
        "mask_shape",
        "layer_id",
        "kv_group",
        "identity",
        "chunk_hash",
    ],
)
def test_incremental_metadata_rejects_malformed_cached_prefix(malformation):
    key0 = _make_key(chunk_hash=1)
    key1 = _make_key(chunk_hash=2)
    token_database = _IncrementalTokenDatabase([key0, key1])
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.token_database = token_database
    cached_keys = [[key] for key in key0.split_layers(2)]
    snapshot = [10]
    mask = torch.ones(2, dtype=torch.bool)
    if malformation == "snapshot_length":
        snapshot.append(11)
    elif malformation == "mask_shape":
        mask = mask.reshape(1, 2)
    elif malformation == "layer_id":
        cached_keys[1][0] = key0.get_layer(0)
    elif malformation == "kv_group":
        cached_keys[1][0] = _make_key(kv_group=1, chunk_hash=1).get_layer(1)
    elif malformation == "identity":
        cached_keys[1][0] = CacheEngineKey(
            model_name="other-model",
            world_size=8,
            worker_id=0,
            chunk_hash=1,
            dtype=torch.float16,
            kv_group=0,
        ).get_layer(1)
    elif malformation == "chunk_hash":
        cached_keys[1][0] = _make_key(chunk_hash=99).get_layer(1)

    extension = engine._build_retrieve_metadata_extension(
        tokens=[10, 11],
        mask=mask,
        request_configs=None,
        kv_group=0,
        cached_keys=cached_keys,
        cached_starts=[0],
        cached_ends=[1],
        cached_metadata_token_ids=snapshot,
    )

    assert extension is None
    assert token_database.suffix_calls == 0


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


def test_sparse_retrieve_allocated_ret_mask_preserves_metadata_warm_flag(monkeypatch):
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
        assert retrieve_kwargs["_retrieve_metadata_warm"] is True
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


def test_metadata_warm_data_hot_repopulates_stale_ret_mask():
    engine = object.__new__(AscendLMCacheEngine)
    engine._resolve_local_cpu_retrieve_location = lambda location: location
    ret_mask = torch.ones(3, dtype=torch.bool)
    retrieve_kwargs = {
        "_retrieve_metadata_warm": True,
        "_use_cached_retrieve": True,
        "cached_retrieve_location": "LocalCPUBackend",
    }

    AscendLMCacheEngine._ensure_retrieve_chunk_metadata(
        engine,
        tokens=[1, 2, 3],
        mask=None,
        request_configs=None,
        cached_keys=[["layer0-key"]],
        cached_starts=[1],
        cached_ends=[3],
        ret_mask=ret_mask,
        retrieve_kwargs=retrieve_kwargs,
    )

    assert ret_mask.tolist() == [False, True, True]


@pytest.mark.parametrize(
    ("direct_external_pages", "expected_location"),
    [(False, "mixed"), (True, "RemoteBackend")],
)
def test_sampled_metadata_build_skips_per_chunk_contains(
    direct_external_pages, expected_location
):
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
        retrieve_kwargs={
            "kv_group": 1,
            "direct_external_pages": direct_external_pages,
        },
    )

    assert location == expected_location
    assert starts == [0]
    assert ends == [1]
    assert keys == [[key.split_layers(2)[0]], [key.split_layers(2)[1]]]
    assert ret_mask.tolist() == [True]


def test_sampled_metadata_reuses_latent_chunk_hashes_for_indexer() -> None:
    tokens = [10, 11, 12, 13, 14]
    mask = torch.tensor([False, False, True, True, True])
    calls: list[dict[str, Any]] = []

    def process_tokens(**kwargs: Any) -> Any:
        calls.append(kwargs)
        kv_group = kwargs.get("kv_group", 0)
        if kwargs.get("hashes") is not None:
            for offset, chunk_hash in zip(
                kwargs["offsets"],
                kwargs["hashes"],
                strict=True,
            ):
                yield 0, offset, _make_key(kv_group, chunk_hash)
            return
        yield 2, 4, _make_key(kv_group, 101)
        yield 4, 5, _make_key(kv_group, 202)

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.use_layerwise = True
    engine.config = SimpleNamespace(experimental_sampled_layerwise_lookup=True)
    engine.token_database = SimpleNamespace(process_tokens=process_tokens)
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_shared_retrieve_passive = lambda _kv_group: False
    preflight_state = {}

    def build(kv_group: int) -> Any:
        return engine._ensure_retrieve_chunk_metadata(
            tokens=tokens,
            mask=mask,
            request_configs={"lmcache.tag.request": "same"},
            cached_keys=[],
            cached_starts=[],
            cached_ends=[],
            ret_mask=torch.zeros(len(tokens), dtype=torch.bool),
            retrieve_kwargs={
                "kv_group": kv_group,
                "shared_cpu_request_preflight_state": preflight_state,
            },
        )

    _, latent_starts, latent_ends, latent_keys = build(0)
    _, index_starts, index_ends, index_keys = build(1)

    assert calls[0]["tokens"] is tokens
    assert calls[1]["hashes"] == [101, 202]
    assert calls[1]["offsets"] == [2, 1]
    assert latent_starts == index_starts == [2, 4]
    assert latent_ends == index_ends == [4, 5]
    assert [key.chunk_hash for key in latent_keys[0]] == [101, 202]
    assert [key.chunk_hash for key in index_keys[0]] == [101, 202]
    assert all(key.kv_group == 0 for layer in latent_keys for key in layer)
    assert all(key.kv_group == 1 for layer in index_keys for key in layer)
    assert preflight_state == {}

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
    engine._resolve_shared_rank0_page_first_layers = resolve_windowed
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
    if page_first:
        assert "layers_per_batch" not in windowed_calls[0]
    else:
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
    engine._resolve_shared_rank0_page_first_layers = resolve_windowed
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
    assert "layers_per_batch" not in windowed_calls[0]

    retriever.close()
    assert all(obj.unpin_count == 1 for layer in allocated for obj in layer)
    assert all(obj.release_count == 1 for layer in allocated for obj in layer)


def test_sampled_page_first_preflight_batches_local_prefix_layers(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )

    num_layers = 3
    keys_layer_major = [
        [f"layer-{layer}-chunk-{chunk}" for chunk in range(2)]
        for layer in range(num_layers)
    ]
    allocated = [[_FakePinnedMemObj(), _FakePinnedMemObj()] for _ in range(num_layers)]
    local_objects = [
        [_FakePinnedMemObj(), _FakePinnedMemObj()],
        [_FakePinnedMemObj()],
        [_FakePinnedMemObj()],
    ]

    class _CrossLayerLocalBackend:
        def __init__(self):
            self.calls = []

        def batched_get_prefixes_with_misses(self, keys):
            self.calls.append(keys)
            return [
                LocalCPUPrefixGetResult(
                    list(local_objects[layer_id]),
                    list(range(len(local_objects[layer_id]), len(layer_keys))),
                    list(layer_keys[len(local_objects[layer_id]) :]),
                )
                for layer_id, layer_keys in enumerate(keys)
            ]

        def batched_get_prefix_with_misses(self, _keys):
            raise AssertionError("per-layer prefix lookup must be bypassed")

    local_backend = _CrossLayerLocalBackend()
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = num_layers
    engine.use_layerwise = True
    engine.config = SimpleNamespace(
        experimental_sampled_layerwise_lookup=True,
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
        "mixed",
        [0, 1],
        [1, 2],
        keys_layer_major,
    )
    engine._shared_local_cpu_backend = lambda: local_backend
    capacity_calls = []
    engine._shared_cpu_runtime_capacity_details = lambda **kwargs: (
        capacity_calls.append(kwargs) or {"fits": True}
    )
    windowed_calls = []

    def resolve_windowed(**kwargs):
        windowed_calls.append(kwargs)
        return allocated

    engine._resolve_shared_rank0_remote_layers_windowed = resolve_windowed
    engine._resolve_shared_rank0_page_first_layers = resolve_windowed
    engine._resolve_shared_rank0_layer_mem_objs = lambda **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("layerwise resolver must be bypassed"))
    engine._broadcast_shared_envelope = lambda _envelope: None

    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-batched-prefix",
    )

    next(retriever)

    assert local_backend.calls == [keys_layer_major]
    assert len(windowed_calls) == 1
    assert windowed_calls[0]["keys_layer_major"] == keys_layer_major
    assert windowed_calls[0]["local_prefix_layers"]
    assert "layers_per_batch" not in windowed_calls[0]
    assert capacity_calls[0]["chunk_locations_layer_major"] == [
        ["LocalCPUBackend", "RemoteBackend"] for _ in range(num_layers)
    ]

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
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm passive metadata must not be rebuilt")
        )
    )
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
    cached_memory_objs = []
    cached_shared_handles = []

    retriever = engine.retrieve_layer_head_token_wise(
        [1],
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

    ret_mask = next(retriever)
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


def test_sparse_passive_materialize_only_skips_npu_consumer(monkeypatch):
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
        request_id="req-materialize",
        phase="dsa_cold_compact_latent",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[handle],
    )
    cached_memory_objs = []
    cached_tensors = []

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-materialize",
        shared_cpu_phase="dsa_cold_compact_latent",
        materialize_only=True,
    )

    next(retriever)
    retriever.send(None)
    retriever.close()

    assert cached_memory_objs == [[mem_obj]]
    assert cached_tensors == [[mem_obj.tensor]]


@pytest.mark.parametrize("materialize_only", [True, False])
@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("prepare_early", [True, False])
@pytest.mark.parametrize("valid_final", [True, False])
def test_sparse_passive_reuses_one_merged_page(
    monkeypatch, complete, materialize_only, prepare_early, valid_final
):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    perf_events = []
    monkeypatch.setattr(ascend_cache_engine, "cold_start_perf_enabled", lambda: True)
    monkeypatch.setattr(
        ascend_cache_engine,
        "cold_start_perf_log",
        lambda _logger, event, **fields: perf_events.append((event, fields)),
    )
    key = _make_key()
    page = _FakeTensorMemObj(torch.empty(1))
    batch = SharedHandleBatch(
        shm_name="test",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[1],
        chunk_hashes=[int(key.chunk_hash)],
        offsets=[],
        page_offsets=[0],
        page_physical_sizes=[2],
    )
    envelopes = iter(
        [
            SharedHandleEnvelope(
                request_id="req-page",
                phase="dsa_cold_compact_latent",
                request_ordinal=0,
                layer_id=0,
                kv_group=0,
                status="ok",
                generation=7,
                handles=[],
                batch=batch,
            ),
            SharedHandleEnvelope(
                request_id="req-page",
                phase="dsa_cold_compact_latent",
                request_ordinal=0,
                layer_id=2,
                kv_group=0,
                status="skipped" if valid_final else "error",
                generation=7,
                handles=[],
                message=(
                    "compact shared batch committed"
                    if valid_final
                    else "failed before compact commit"
                ),
            ),
        ]
    )

    class Connector(_FakeSparseConsumer):
        def __init__(self):
            self.pointer_snapshots = []
            self.final_appends = 0
            self.events = []

        def batched_to_gpu_head_token_wise(self, **kwargs):
            host_ptrs = kwargs.get("cached_chunk_dev_ptrs")
            npu_ptrs = kwargs.get("cached_chunk_ptrs_npu")
            for _layer_id in range(2):
                payload = yield
                self.events.append("send")
                self.pointer_snapshots.append(
                    (
                        [list(row) for row in host_ptrs],
                        [row.tolist() for row in npu_ptrs],
                        payload,
                    )
                )
            yield
            yield

        def prepare_sparse_page_ptr_cache_for_layers(
            self, sources, host_ptrs, npu_ptrs
        ):
            if not prepare_early:
                return False
            self.events.append("prepare")
            self.sources = sources
            host_ptrs.extend(([11], [22]))
            npu_ptrs.extend((torch.tensor([11]), torch.tensor([22])))
            return True

        def append_sparse_chunk_ptr_cache_for_layers(
            self, sources, host_ptrs, npu_ptrs
        ):
            self.final_appends += 1
            self.sources = sources
            host_ptrs.extend(([11], [22]))
            npu_ptrs.extend((torch.tensor([11]), torch.tensor([22])))

    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.gpu_connector = Connector()
    engine.token_database = _FakeTokenDatabase(key)
    engine.shared_cpu_cache_passive_allocator = _CompactPassiveAllocator([])
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0, "worker_id": 0})()
    engine._receive_shared_envelope = lambda: next(envelopes)
    engine._make_passive_layer_page_views = lambda *_args, **_kwargs: (page,)
    cached_memory_objs = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []
    cached_shared_handles = []

    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-page",
        shared_cpu_phase="dsa_cold_compact_latent",
        materialize_only=materialize_only,
    )

    next(retriever)
    retriever.send(None)
    if complete:
        if valid_final:
            retriever.send(None)
        else:
            with pytest.raises(ValueError, match="rank0 error envelope"):
                retriever.send(None)
    retriever.close()

    early_active = prepare_early
    expected_transient = (
        ([[11], [22]], [[11], [22]]) if early_active else ([], [])
    )
    if complete and valid_final:
        assert all(
            isinstance(source, LayerPageSource)
            for source in engine.gpu_connector.sources
        )
        assert cached_memory_objs == [[page], [page]]
        assert cached_chunk_dev_ptrs == [[11], [22]]
        assert [row.tolist() for row in cached_chunk_ptrs_npu] == [[11], [22]]
        assert cached_shared_handles == [[None], [None]]
        assert page.release_count == 0
        assert engine.gpu_connector.final_appends == (0 if early_active else 1)
        assert len(engine.gpu_connector.pointer_snapshots) == 2
        assert engine.gpu_connector.events == (
            ["prepare", "send", "send"]
            if early_active
            else ["send", "send"]
        )
        assert all(
            (host, npu) == expected_transient and payload is not None
            for host, npu, payload in engine.gpu_connector.pointer_snapshots
        )
        aggregate = dict(perf_events)["passive_compact_materialize"]
        assert aggregate["req_id"] == "req-page"
        assert aggregate["rank"] == 0
        assert aggregate["kv_group"] == 0
        assert aggregate["physical_pages"] == 1
        assert aggregate["logical_entries"] == 2
        assert aggregate["legacy_tail_objects"] == 0
        assert aggregate["page_view_build_ms"] >= 0
        assert aggregate["pointer_seal_ms"] >= 0
    else:
        assert cached_memory_objs == []
        assert cached_chunk_dev_ptrs == []
        assert cached_chunk_ptrs_npu == []
        assert cached_shared_handles == []
        assert page.release_count == 1
        assert engine.gpu_connector.final_appends == 0
        assert engine.gpu_connector.pointer_snapshots[0][:2] == (
            expected_transient
        )
        assert "passive_compact_materialize" not in dict(perf_events)


def test_materialize_only_npu_consumer_never_initializes_transfer_state() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    consumer = connector.batched_to_gpu_head_token_wise(
        materialize_only=True
    )

    next(consumer)
    consumer.send(object())
    consumer.send(object())
    next(consumer)
    consumer.close()


def test_dense_load_completion_api_synchronizes_the_load_stream() -> None:
    calls = []
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.load_stream = SimpleNamespace(synchronize=lambda: calls.append(True))

    connector.synchronize_dense_load_stream()

    assert calls == [True]


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


def test_sparse_passive_partial_view_failure_releases_created_views(monkeypatch):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    keys = [_make_key(chunk_hash=1), _make_key(chunk_hash=2)]
    created = _FakeTensorMemObj(torch.empty(1))
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(keys)
    engine.shared_cpu_cache_passive_allocator = _SequencePassiveAllocator(
        [created, ValueError("bad second view")]
    )
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )
    engine._receive_shared_envelope = lambda: SharedHandleEnvelope(
        request_id="req-partial",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=7,
        handles=[object(), object()],
    )
    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1, 2],
        None,
        torch.zeros(2, dtype=torch.bool),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        kv_group=0,
        req_id="req-partial",
    )

    next(retriever)
    with pytest.raises(ValueError, match="bad second view"):
        retriever.send(([0, 1], 0))

    assert created.release_count == 1
    assert not created.is_valid()


@pytest.mark.parametrize(
    "final_status", ["skipped", "deferred", "error", "close", "view_error"]
)
def test_sparse_passive_compact_waits_for_final_status(monkeypatch, final_status):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    key = _make_key()
    views = [
        _FakeTensorMemObj(torch.empty(1)),
        _FakeTensorMemObj(torch.empty(1)),
    ]
    batch = SharedHandleBatch(
        shm_name="test",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[1],
        chunk_hashes=[int(key.chunk_hash)],
        offsets=[0, 1],
    )
    envelopes = iter(
        [
            SharedHandleEnvelope(
                request_id="req-passive-compact",
                phase="sparse_decode_bootstrap",
                request_ordinal=0,
                layer_id=0,
                kv_group=0,
                status="ok",
                generation=7,
                handles=[],
                batch=batch,
            ),
            SharedHandleEnvelope(
                request_id="req-passive-compact",
                phase="sparse_decode_bootstrap",
                request_ordinal=0,
                layer_id=2,
                kv_group=0,
                status=(
                    "skipped"
                    if final_status in ("skipped", "deferred")
                    else "error"
                ),
                generation=7,
                handles=[],
                message=(
                    "compact shared batch committed"
                    if final_status in ("skipped", "deferred")
                    else "compact failed"
                ),
                error_details=(
                    None if final_status == "skipped" else {"error": "failed"}
                ),
            ),
        ]
    )
    receives = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.gpu_connector = _FakeSparseConsumer()
    engine.token_database = _FakeTokenDatabase(key)
    allocator_views = (
        [views[0], ValueError("bad compact view")]
        if final_status == "view_error"
        else views
    )
    engine.shared_cpu_cache_passive_allocator = _CompactPassiveAllocator(
        allocator_views
    )
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"first_rank": 0})()
    engine._expected_shared_cpu_chunk_metadata = lambda **_kwargs: (
        torch.Size([1]),
        torch.float16,
        object(),
    )

    def receive_envelope():
        receives.append(True)
        return next(envelopes)

    engine._receive_shared_envelope = receive_envelope
    cached_memory_objs = []
    cached_shared_handles = []
    retriever = engine._retrieve_layer_head_token_wise_shared_passive(
        [1],
        None,
        torch.zeros(1, dtype=torch.bool),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=cached_memory_objs,
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        kv_group=0,
        req_id="req-passive-compact",
    )

    next(retriever)
    if final_status == "deferred":
        retriever.send({_SHARED_SPARSE_PREPARE_ONLY: True})
    retriever.send(([0], 0))
    if final_status == "close":
        retriever.close()
    elif final_status == "view_error":
        with pytest.raises(ValueError, match="bad compact view"):
            retriever.send(([0], 0))
    elif final_status == "error":
        with pytest.raises(ValueError, match="rank0 error envelope"):
            retriever.send(([0], 0))
    elif final_status == "deferred":
        retriever.send(
            {
                "selected_token_ids": [0],
                "token_start_index": 0,
                _SHARED_SPARSE_DEFER_COMMIT: True,
            }
        )
        assert receives == [True]
        next(retriever)
    else:
        retriever.send(([0], 0))

    assert receives == [True, True]
    if final_status not in ("skipped", "deferred"):
        assert cached_memory_objs == []
        assert cached_shared_handles == []
        expected_releases = (
            [1, 0]
            if final_status in ("close", "view_error")
            else [1, 1]
        )
        assert [view.release_count for view in views] == expected_releases
    else:
        assert cached_memory_objs == [[views[0]], [views[1]]]
        assert cached_shared_handles == [[None], [None]]
        assert all(view.release_count == 0 for view in views)


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
        assert kwargs["validate_memory_objs"] is True
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


@pytest.mark.parametrize(
    "exit_mode", ["success", "deferred", "exception", "close"]
)
def test_compact_batch_uses_exactly_one_final_status(
    monkeypatch, exit_mode
):
    monkeypatch.setattr(
        ascend_cache_engine,
        "assert_layerwise_gpu_connector",
        lambda _connector: None,
    )
    key = _make_key()
    layer_keys = key.split_layers(2)
    allocated = [_FakePinnedMemObj(), _FakePinnedMemObj()]
    preflight_state = {}
    broadcasts = []
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.config = SimpleNamespace(
        experimental_sampled_layerwise_lookup=False,
        extra_config={},
    )
    engine.storage_manager = object()
    engine.gpu_connector = _FakeSparseConsumer()
    engine.shared_cpu_cache_generation = 7
    engine.metadata = type("Meta", (), {"worker_id": 0})()
    engine.is_healthy = lambda: True
    engine._should_use_shared_layerwise_retrieve = lambda _group: True
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._has_retrieve_data_cache = lambda *_args: False
    engine._retrieve_data_cache_covers = lambda *_args: False
    engine._ensure_retrieve_chunk_metadata = lambda **_kwargs: (
        "LocalCPUBackend",
        [0],
        [1],
        [[layer_keys[0]], [layer_keys[1]]],
    )
    engine._find_shared_rank0_chunk_location = lambda _key: "LocalCPUBackend"
    engine._get_shared_config_value = lambda _name, default: default
    engine._shared_cpu_runtime_capacity_details = lambda **_kwargs: {"fits": True}
    engine._resolve_shared_rank0_layer_mem_objs = lambda **kwargs: [
        allocated[kwargs["layer_id"]]
    ]
    batch = SharedHandleBatch(
        shm_name="test",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[1],
        chunk_hashes=[int(key.chunk_hash)],
        offsets=[0, 1],
    )
    engine._make_shared_handle_batch = lambda *_args: batch
    engine._broadcast_shared_envelope = lambda envelope: broadcasts.append(envelope)
    cached_shared_handles = []
    retriever = engine.retrieve_layer_head_token_wise(
        [1],
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=cached_shared_handles,
        shared_cpu_request_preflight_state=preflight_state,
        kv_group=0,
        req_id="req-compact",
    )

    next(retriever)
    if exit_mode == "deferred":
        retriever.send({_SHARED_SPARSE_PREPARE_ONLY: True})
    retriever.send(([0], 0))
    assert len(broadcasts) == 1
    assert broadcasts[0].batch is batch
    assert cached_shared_handles == [[None], [None]]

    if exit_mode == "exception":
        preflight_state.update(
            source_kv_group=1,
            source_layer_id=0,
            message="indexer failed",
            details={},
            error=ValueError("indexer failed"),
        )
        with pytest.raises(ValueError, match="request preflight failed"):
            retriever.send(([0], 0))
    elif exit_mode == "close":
        retriever.close()
    elif exit_mode == "deferred":
        retriever.send(
            {
                "selected_token_ids": [0],
                "token_start_index": 0,
                _SHARED_SPARSE_DEFER_COMMIT: True,
            }
        )
        assert len(broadcasts) == 1
        next(retriever)
    else:
        retriever.send(([0], 0))

    assert len(broadcasts) == 2
    assert broadcasts[1].layer_id == engine.num_layers
    assert broadcasts[1].handles == []
    assert broadcasts[1].batch is None
    if exit_mode != "success":
        assert broadcasts[1].status == "error"
        assert cached_shared_handles == []
    else:
        assert broadcasts[1].status == "skipped"
        assert broadcasts[1].message == "compact shared batch committed"
        assert cached_shared_handles == [[None], [None]]


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
        assert kwargs["validate_memory_objs"] is False
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
    pointer_sources = []
    engine.gpu_connector = _FakeSparseConsumer()
    engine.gpu_connector.append_sparse_chunk_ptr_cache_for_layer = (
        lambda _layer_id, sources, *_caches: pointer_sources.extend(sources)
    )
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
    cached_tensors = []
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
    assert cached_tensors == []
    assert pointer_sources == [new_mem_obj]
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


def test_sparse_pointer_first_accepts_one_layout_tensor_for_full_table():
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    cached_ptrs = [torch.tensor([101, 102], dtype=torch.long)]

    assert connector._resolve_sparse_chunk_ptrs_npu(
        0,
        [torch.empty(1)],
        cached_ptrs,
        expected_num_chunks=2,
    ) is cached_ptrs[0]


def test_dense_pointer_resolution_retains_host_row_without_extra_work(monkeypatch):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.kv_device = torch.device("cpu")
    tensors = [torch.empty(1), torch.empty(1)]
    resolved = []
    tensor_calls = 0
    original_tensor = npu_connectors.torch.tensor

    def resolve(host_ptr):
        resolved.append(host_ptr)
        return host_ptr + 1000

    def build_tensor(*args, **kwargs):
        nonlocal tensor_calls
        tensor_calls += 1
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(npu_connectors.lmc_ops, "get_device_ptr", resolve)
    monkeypatch.setattr(npu_connectors.torch, "tensor", build_tensor)
    cached_host_ptrs = []
    cached_npu_ptrs = []

    first = connector._resolve_sparse_chunk_ptrs_npu(
        0,
        tensors,
        cached_npu_ptrs,
        cached_chunk_dev_ptrs=cached_host_ptrs,
    )
    second = connector._resolve_sparse_chunk_ptrs_npu(
        0,
        tensors,
        cached_npu_ptrs,
        cached_chunk_dev_ptrs=cached_host_ptrs,
    )

    assert len(resolved) == 2
    assert tensor_calls == 1
    assert cached_host_ptrs == [[ptr + 1000 for ptr in resolved]]
    assert cached_npu_ptrs[0] is first
    assert second is first


def test_append_retrieve_group_accepts_empty_prefix():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    calls = []

    def append(sources, host_ptrs, npu_ptrs):
        calls.append(sources)
        host_ptrs.extend(([11], [22]))
        npu_ptrs.extend(
            (
                torch.tensor([11], dtype=torch.long),
                torch.tensor([22], dtype=torch.long),
            )
        )

    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layers=append
    )
    new_objs = [
        [_FakeTensorMemObj(torch.empty(1))],
        [_FakeTensorMemObj(torch.empty(1))],
    ]
    cached_memory_objs = []
    cached_tensors = []
    cached_host_ptrs = []
    cached_npu_ptrs = []

    engine._append_retrieve_group_cache(
        new_objs,
        cached_memory_objs,
        cached_tensors,
        cached_host_ptrs,
        cached_npu_ptrs,
    )

    assert len(calls) == 1
    assert cached_memory_objs == new_objs
    assert cached_host_ptrs == [[11], [22]]
    assert [row.tolist() for row in cached_npu_ptrs] == [[11], [22]]


def test_append_retrieve_group_preserves_layer_page_sources():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    sources = []

    def append(new_sources, host_ptrs, npu_ptrs):
        sources.extend(new_sources)
        host_ptrs[:] = [[11], [22]]
        npu_ptrs[:] = [
            torch.tensor([11], dtype=torch.long),
            torch.tensor([22], dtype=torch.long),
        ]

    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layers=append
    )
    page = _FakeTensorMemObj(torch.empty(1))
    tails = [_FakeTensorMemObj(torch.empty(1)) for _ in range(2)]
    page_sources = [
        LayerPageSource((page,), layer_id, (tails[layer_id],))
        for layer_id in range(2)
    ]
    cached_memory_objs = []

    engine._append_retrieve_group_cache(
        page_sources,
        cached_memory_objs,
        [],
        [],
        [None, None],
    )

    assert sources == page_sources
    assert cached_memory_objs == [[page, tails[0]], [page, tails[1]]]


def test_append_retrieve_group_rejects_incomplete_prefix_before_mutation():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    calls = []
    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layers=lambda *args: calls.append(args)
    )
    old_objs = [
        _FakeTensorMemObj(torch.empty(1)),
        _FakeTensorMemObj(torch.empty(1)),
    ]
    new_objs = [
        [_FakeTensorMemObj(torch.empty(1))],
        [_FakeTensorMemObj(torch.empty(1))],
    ]
    cached_memory_objs = [[old_objs[0]], [old_objs[1]]]
    cached_tensors = []
    cached_host_ptrs = [[11], []]
    cached_npu_ptrs = [
        torch.tensor([11], dtype=torch.long),
        torch.tensor([22], dtype=torch.long),
    ]

    with pytest.raises(ValueError, match="prefix coverage mismatch"):
        engine._append_retrieve_group_cache(
            new_objs,
            cached_memory_objs,
            cached_tensors,
            cached_host_ptrs,
            cached_npu_ptrs,
        )

    assert calls == []
    assert cached_memory_objs == [[old_objs[0]], [old_objs[1]]]
    assert cached_tensors == []
    assert cached_host_ptrs == [[11], []]
    assert [row.tolist() for row in cached_npu_ptrs] == [[11], [22]]


@pytest.mark.parametrize("layer_count", [1, 3])
def test_append_retrieve_group_rejects_wrong_outer_layer_count(layer_count):
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    calls = []
    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layers=lambda *args: calls.append(args)
    )
    cached_memory_objs = [
        [_FakeTensorMemObj(torch.empty(1))] for _ in range(layer_count)
    ]
    cached_tensors = []
    cached_host_ptrs = [[11] for _ in range(layer_count)]
    cached_npu_ptrs = [
        torch.tensor([11], dtype=torch.long) for _ in range(layer_count)
    ]
    original_objs = [list(layer) for layer in cached_memory_objs]

    with pytest.raises(ValueError, match="layer coverage mismatch"):
        engine._append_retrieve_group_cache(
            [
                [_FakeTensorMemObj(torch.empty(1))],
                [_FakeTensorMemObj(torch.empty(1))],
            ],
            cached_memory_objs,
            cached_tensors,
            cached_host_ptrs,
            cached_npu_ptrs,
        )

    assert calls == []
    assert cached_memory_objs == original_objs
    assert cached_tensors == []
    assert cached_host_ptrs == [[11] for _ in range(layer_count)]
    assert [row.tolist() for row in cached_npu_ptrs] == [
        [11] for _ in range(layer_count)
    ]


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


def test_append_retrieve_layer_cache_uses_memory_obj_pointer_path():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    calls = []
    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layer=lambda *args: calls.append(args)
    )
    mem_obj: Any = SimpleNamespace(is_valid=lambda: True, data_ptr=123)
    cached_memory_objs = []
    cached_tensors = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    engine._append_retrieve_layer_cache(
        0,
        [mem_obj],
        cached_memory_objs,
        cached_tensors,
        cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu,
    )

    assert calls[0][1] == [mem_obj]
    assert cached_memory_objs == [[mem_obj]]
    assert cached_tensors == []


def test_append_retrieve_layer_cache_preserves_tensor_backed_prefix():
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 1
    calls = []
    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layer=lambda *args: calls.append(args)
    )
    old_tensor, new_tensor = torch.empty(1), torch.empty(1)
    old_obj = _FakeTensorMemObj(old_tensor)
    new_obj = _FakeTensorMemObj(new_tensor)
    cached_memory_objs = [[old_obj]]
    cached_tensors = [[old_tensor]]

    engine._append_retrieve_layer_cache(
        0,
        [new_obj],
        cached_memory_objs,
        cached_tensors,
        [[]],
        [torch.tensor([123], dtype=torch.long)],
    )

    assert calls[0][1] == [new_tensor]
    assert cached_memory_objs == [[old_obj, new_obj]]
    assert cached_tensors == [[old_tensor, new_tensor]]


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


def test_sparse_pointer_cache_accepts_memory_obj_sources(monkeypatch):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 1
    connector.kv_device = torch.device("cpu")
    monkeypatch.setattr(
        npu_connectors.lmc_ops,
        "get_device_ptr",
        lambda host_ptr: host_ptr + 1000,
    )
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    sources: list[Any] = [
        SimpleNamespace(data_ptr=23),
        SimpleNamespace(data_ptr=42),
    ]
    connector.append_sparse_chunk_ptr_cache_for_layer(
        0,
        sources,
        cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu,
    )

    assert cached_chunk_dev_ptrs == [[1023, 1042]]
    assert cached_chunk_ptrs_npu[0].tolist() == [1023, 1042]


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
