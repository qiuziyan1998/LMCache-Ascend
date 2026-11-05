# SPDX-License-Identifier: Apache-2.0
"""Per-group cardinality tests for GLM-5.2 (79 LATENT / 22 INDEXER) on the
Ascend two-group layerwise paths.

Covers the design contract:
- _GroupLayout.num_layers/layer_indices established from the registered group
  caches at layout initialization (79 latent / 22 indexer).
- Connector get_num_layers/get_layer_indices/_expected_group_layers with
  fail-closed DSA and the legacy single-group fallback.
- AscendLMCacheEngine._num_layers_for_kv_group: connector layout preferred,
  engine-level resolution fallback, disagreement fail-closed.
- _num_transfer_layers_for_call fail-closes against the per-group kvcaches
  list.
- store_layer transfers exactly the group's layer rows: split_layers(22),
  batched_allocate(batch_size=22), 23-yield generator cadence, and the
  no-key yield cadence.
- _append_retrieve_layer_cache initializes cached rows at the group count.
"""
# Standard
from types import SimpleNamespace
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache_ascend.v1 import cache_engine as ascend_engine_module
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector import npu_connectors
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    _GroupLayout,
    VLLMPagedMemLayerwiseNPUConnector,
)

LATENT_LAYERS = 79
INDEXER_LAYERS = 22


def _latent_caches(layers: int):
    return [
        (
            torch.zeros(4, 128, 1, 512, dtype=torch.bfloat16),
            torch.zeros(4, 128, 1, 64, dtype=torch.bfloat16),
        )
        for _ in range(layers)
    ]


def _indexer_caches(layers: int):
    return [
        (torch.zeros(4, 128, 1, 128, dtype=torch.bfloat16),)
        for _ in range(layers)
    ]


def _layout_for_format(kv_format: KVCacheFormat, num_layers: int) -> _GroupLayout:
    layout = _GroupLayout()
    layout.kv_format = kv_format
    layout.num_layers = num_layers
    layout.layer_indices = tuple(range(num_layers))
    return layout


def _connector_with_layouts(
    layouts: dict[int, _GroupLayout], *, dsa_two_groups: bool = True
):
    connector = SimpleNamespace()
    connector._group_layouts = layouts
    connector._current_kv_group = 0
    connector.num_layers = LATENT_LAYERS
    connector.dsa_two_groups = dsa_two_groups
    connector.get_num_layers = (
        lambda kv_group=0: VLLMPagedMemLayerwiseNPUConnector.get_num_layers(
            connector, kv_group
        )
    )
    connector.get_layer_indices = (
        lambda kv_group=0: VLLMPagedMemLayerwiseNPUConnector.get_layer_indices(
            connector, kv_group
        )
    )
    connector._expected_group_layers = (
        lambda kv_group=None: (
            VLLMPagedMemLayerwiseNPUConnector._expected_group_layers(
                connector, kv_group
            )
        )
    )
    return connector


class TestGroupLayoutCardinality:
    def test_layout_captures_group_layer_count(self):
        layout = _layout_for_format(KVCacheFormat.DSA_INDEX, INDEXER_LAYERS)
        assert layout.num_layers == 22
        assert layout.layer_indices == tuple(range(22))

    def test_get_num_layers_uses_registered_layout(self):
        connector = _connector_with_layouts(
            {
                0: _layout_for_format(
                    KVCacheFormat.MLA_LATENT, LATENT_LAYERS
                ),
                1: _layout_for_format(KVCacheFormat.DSA_INDEX, INDEXER_LAYERS),
            }
        )
        assert connector.get_num_layers(0) == 79
        assert connector.get_num_layers(1) == 22

    def test_get_layer_indices_per_group(self):
        connector = _connector_with_layouts(
            {
                0: _layout_for_format(
                    KVCacheFormat.MLA_LATENT, LATENT_LAYERS
                ),
                1: _layout_for_format(KVCacheFormat.DSA_INDEX, INDEXER_LAYERS),
            }
        )
        assert connector.get_layer_indices(1) == tuple(range(22))

    def test_expected_group_layers_falls_back_for_single_group(self):
        connector = _connector_with_layouts({}, dsa_two_groups=False)
        assert connector._expected_group_layers(1) == LATENT_LAYERS

    def test_expected_group_layers_rejects_uninitialized_dsa_group(self):
        connector = _connector_with_layouts({})
        with pytest.raises(RuntimeError, match="layout is initialized"):
            connector._expected_group_layers(1)

    def test_uninitialized_group_returns_none(self):
        connector = _connector_with_layouts({})
        assert connector.get_num_layers(1) is None
        assert connector.get_layer_indices(1) is None


def _engine_for_cardinality(
    connector_layers: Optional[dict[int, int]],
    engine_group_layers: dict[int, int],
) -> AscendLMCacheEngine:
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.num_layers = LATENT_LAYERS
    if connector_layers is None:
        engine.gpu_connector = SimpleNamespace()
    else:
        layouts = {
            group: _layout_for_format(
                KVCacheFormat.DSA_INDEX
                if group == 1
                else KVCacheFormat.MLA_LATENT,
                layers,
            )
            for group, layers in connector_layers.items()
        }
        engine.gpu_connector = _connector_with_layouts(layouts)

    def num_layers_for_group(kv_group: int) -> int:
        return engine_group_layers.get(kv_group, LATENT_LAYERS)

    engine.num_layers_for_group = num_layers_for_group
    return engine


class TestEngineNumLayersForKvGroup:
    def test_connector_layout_preferred(self):
        engine = _engine_for_cardinality(
            {0: LATENT_LAYERS, 1: INDEXER_LAYERS}, {0: 79, 1: 22}
        )
        assert engine._num_layers_for_kv_group(0) == 79
        assert engine._num_layers_for_kv_group(1) == 22

    def test_engine_resolution_when_layout_missing(self):
        engine = _engine_for_cardinality(None, {0: 79, 1: 22})
        assert engine._num_layers_for_kv_group(1) == 22

    def test_disagreement_fails_closed(self):
        engine = _engine_for_cardinality(
            {0: LATENT_LAYERS, 1: INDEXER_LAYERS}, {0: 79, 1: 79}
        )
        with pytest.raises(ValueError, match="disagrees"):
            engine._num_layers_for_kv_group(1)

    def test_transfer_call_validates_kvcaches(self):
        engine = _engine_for_cardinality(
            {0: LATENT_LAYERS, 1: INDEXER_LAYERS}, {0: 79, 1: 22}
        )
        assert (
            engine._num_transfer_layers_for_call(
                1, {"kvcaches": [object()] * 22}
            )
            == 22
        )
        with pytest.raises(ValueError, match="cardinality mismatch"):
            engine._num_transfer_layers_for_call(
                1, {"kvcaches": [object()] * 79}
            )


class _FakeLayerKey:
    """Chunk key recording the cardinality each split_layers call used."""

    def __init__(
        self,
        chunk_id: int,
        layer_id: Optional[int],
        deps=None,
        kv_group: int = 0,
    ):
        self.chunk_id = chunk_id
        self.layer_id = layer_id
        self.deps = deps
        self.kv_group = kv_group
        self.chunk_hash = chunk_id

    def split_layers(self, num_layers: int):
        if self.deps is not None:
            self.deps.split_sizes.setdefault(self.kv_group, []).append(
                num_layers
            )
        return [
            _FakeLayerKey(
                self.chunk_id, i, self.deps, kv_group=self.kv_group
            )
            for i in range(num_layers)
        ]


class _StoreDeps:
    def __init__(self, group_layers: int):
        self.group_layers = group_layers
        self.allocate_batch_sizes: list[int] = []
        self.split_sizes: dict[int, list[int]] = {}
        self.put_layers: list[int] = []

    def make_key(self, chunk_id: int) -> _FakeLayerKey:
        return _FakeLayerKey(chunk_id, None, deps=self, kv_group=1)


def _store_layer_engine(group_layers: int):
    """Build a minimal AscendLMCacheEngine for an indexer store_layer run."""
    deps = _StoreDeps(group_layers)
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.num_layers = LATENT_LAYERS
    engine.gpu_connector = _connector_with_layouts(
        {1: _layout_for_format(KVCacheFormat.DSA_INDEX, group_layers)}
    )
    engine.num_layers_for_group = lambda kv_group: (
        group_layers if kv_group == 1 else LATENT_LAYERS
    )
    engine.kv_events_enabled = False
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine.is_frozen = lambda: False
    engine._get_req_id = lambda _kwargs: "req"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine._ensure_layerwise_connector_layout = lambda **_kwargs: None
    engine._layerwise_chunk_fully_stored = lambda *a, **k: False
    engine._shared_cpu_dtype_for_kv_group = lambda _g: torch.bfloat16
    engine._memory_format_for_kv_group = lambda _g: None
    engine._should_use_shared_layerwise_retrieve = lambda _g: False
    engine._track_sync_store_futures = lambda futures, **kwargs: None
    engine.store_location = "LocalCPUBackend"
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_kwargs: iter([(0, 4, deps.make_key(0))])
    )
    engine.stats_monitor = SimpleNamespace(
        on_store_request=lambda _n: "monitor-id",
        on_store_finished=lambda _m, _n: None,
        on_retrieve_request=lambda _n: "monitor-id",
        on_retrieve_finished=lambda _m, _n: None,
    )
    engine.config = SimpleNamespace(
        get_extra_config_value=lambda _k, default=None: default,
        chunk_size=256,
        dsa_two_groups=True,
    )

    mem_obj = SimpleNamespace(
        get_size=lambda: 8,
        is_valid=lambda: True,
        ref_count_down=lambda: None,
        metadata=SimpleNamespace(fmt=None),
        tensor=torch.zeros(1),
    )

    class FakeStorageManager:
        @staticmethod
        def batched_allocate(_shape, _dtype, batch_size=None, **_kwargs):
            deps.allocate_batch_sizes.append(batch_size)
            return [mem_obj for _ in range(batch_size)]

        @staticmethod
        def batched_put(keys, _objs, location=None):
            deps.put_layers.extend(key.layer_id for key in keys)

    class _GeneratorConnector:
        """Layerwise generator connector serving the group's rows."""

        def __init__(self):
            self.engine_ref = engine

        def get_shape(self, num_tokens: int, kv_group: Optional[int] = None):
            layers = engine._num_layers_for_kv_group(kv_group or 0)
            return torch.Size([num_tokens * layers])

        def batched_from_gpu(self, _objs, _starts, _ends, **_kwargs):
            def gen():
                yield
                for _ in range(deps.group_layers):
                    yield

            return gen()

    connector = _GeneratorConnector()
    connector._group_layouts = engine.gpu_connector._group_layouts
    connector._current_kv_group = 1
    connector.num_layers = LATENT_LAYERS
    engine.gpu_connector = connector
    engine.storage_manager = FakeStorageManager()
    return engine, deps


class TestStoreLayerIndexerGroup:
    def test_indexer_group_transfers_22_rows(self, monkeypatch):
        monkeypatch.setattr(
            ascend_engine_module, "CacheEngineKey", _FakeLayerKey
        )
        monkeypatch.setattr(
            ascend_engine_module,
            "assert_layerwise_gpu_connector",
            lambda _c: None,
        )
        engine, deps = _store_layer_engine(INDEXER_LAYERS)
        results = list(
            engine.store_layer(
                [1, 2, 3, 4],
                kv_group=1,
                req_id="req",
                kvcaches=[object()] * INDEXER_LAYERS,
            )
        )
        # 22 layer yields + 1 final store-result yield.
        assert len(results) == INDEXER_LAYERS + 1
        assert results[-1] is not None
        assert deps.allocate_batch_sizes == [INDEXER_LAYERS]
        assert deps.split_sizes == {1: [INDEXER_LAYERS]}
        assert sorted(deps.put_layers) == list(range(INDEXER_LAYERS))

    def test_kvcaches_mismatch_fails_closed(self, monkeypatch):
        monkeypatch.setattr(
            ascend_engine_module, "CacheEngineKey", _FakeLayerKey
        )
        engine, _deps = _store_layer_engine(INDEXER_LAYERS)
        with pytest.raises(ValueError, match="cardinality mismatch"):
            list(
                engine.store_layer(
                    [1, 2, 3, 4],
                    kv_group=1,
                    req_id="req",
                    kvcaches=[object()] * LATENT_LAYERS,
                )
            )

    def test_unhealthy_cadence_uses_group_count(self):
        engine, _deps = _store_layer_engine(INDEXER_LAYERS)
        engine.is_healthy = lambda: False
        results = list(
            engine.store_layer([1, 2, 3, 4], kv_group=1, req_id="req")
        )
        assert len(results) == INDEXER_LAYERS + 1


class TestAppendRetrieveLayerCache:
    def test_initializes_rows_at_group_count(self):
        engine = _engine_for_cardinality({0: 79, 1: 22}, {0: 79, 1: 22})
        engine.gpu_connector = SimpleNamespace()
        cached_memory_objs: list = []
        engine._append_retrieve_layer_cache(
            5,
            [],
            cached_memory_objs,
            None,
            None,
            None,
            num_layers=INDEXER_LAYERS,
            kv_group=1,
        )
        assert len(cached_memory_objs) == INDEXER_LAYERS


class TestAppendSparseChunkPtrCacheForLayer:
    def test_explicit_group_ignores_stale_current_group(self, monkeypatch):
        connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
        connector._group_layouts = {
            0: _layout_for_format(KVCacheFormat.MLA_LATENT, LATENT_LAYERS),
            1: _layout_for_format(KVCacheFormat.DSA_INDEX, INDEXER_LAYERS),
        }
        connector._current_kv_group = 0
        connector.num_layers = LATENT_LAYERS
        connector.kv_device = torch.device("cpu")
        monkeypatch.setattr(
            npu_connectors.lmc_ops,
            "get_device_ptr",
            lambda host_ptr: host_ptr + 1000,
        )
        cached_chunk_dev_ptrs: list[list[int]] = []
        cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = []

        connector.append_sparse_chunk_ptr_cache_for_layer(
            5,
            [SimpleNamespace(data_ptr=23)],
            cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu,
            kv_group=1,
        )

        assert connector._current_kv_group == 0
        assert len(cached_chunk_dev_ptrs) == INDEXER_LAYERS
        assert len(cached_chunk_ptrs_npu) == INDEXER_LAYERS
        assert cached_chunk_dev_ptrs[5] == [1023]
        assert cached_chunk_ptrs_npu[5].tolist() == [1023]


def test_key_split_never_exceeds_group_cardinality():
    key = CacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=0,
        chunk_hash=123,
        dtype=torch.bfloat16,
    )
    split = key.split_layers(INDEXER_LAYERS)
    assert [k.layer_id for k in split] == list(range(INDEXER_LAYERS))
    assert max(k.layer_id for k in split) < INDEXER_LAYERS
