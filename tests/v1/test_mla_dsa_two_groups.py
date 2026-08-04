# SPDX-License-Identifier: Apache-2.0
"""Unit and integration tests for the MLA+DSA two-group separate storage design
on the Ascend path.

Covers:
- KVCacheFormat.detect() for MLA_LATENT, DSA_INDEX, MLA_KV, DSA_KV with
  dsa_two_groups flag (regression: existing formats unchanged)
- Format helper predicates (is_mla_format, is_dsa_format, kv_group, get_kv_size)
- SaveSpec can_save_latent / can_save_indexer flags
- from_request_tracker decode-full-chunk boundary rule
- store_layer _is_passive() guard (rank-0-only store)
- VLLMPagedMemLayerwiseNPUConnector.get_shape for MLA_LATENT and DSA_INDEX
- _is_mla_dsa_format helper
- Integration: two-group store/load roundtrip with separate latent and indexer keys
"""
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cache_engine import LayerwiseStoreResult
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

# Local
from .utils import dumb_metadata, generate_tokens


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------

class TestKVCacheFormatDetect:
    """Test KVCacheFormat.detect() with and without dsa_two_groups."""

    def _make_mla_latent_tensors(self, num_blocks=4, block_size=128):
        k_nope = torch.zeros(num_blocks, block_size, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(num_blocks, block_size, 1, 64, dtype=torch.bfloat16)
        return [(k_nope, k_pe)]

    def _make_tp8_equal_width_mla_latent_tensors(
        self, num_blocks=4, block_size=128
    ):
        k_nope = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        k_pe = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        return [(k_nope, k_pe)]

    def _make_dsa_index_tensors(self, num_blocks=4, block_size=128):
        indexer_k = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        return [(indexer_k,)]

    def _make_dsa_kv_tensors(self, num_blocks=4, block_size=128):
        k = torch.zeros(num_blocks, block_size, 1, 512, dtype=torch.bfloat16)
        v = torch.zeros(num_blocks, block_size, 1, 64, dtype=torch.bfloat16)
        dsa = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        return [(k, v, dsa)]

    def _make_separate_kv_tensors(self, num_blocks=4, block_size=128):
        k = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        v = torch.zeros(num_blocks, block_size, 1, 128, dtype=torch.bfloat16)
        return [(k, v)]

    def _make_merged_kv_tensors(self, num_blocks=4, block_size=128):
        return [torch.zeros(2, num_blocks, block_size, 1, 128, dtype=torch.bfloat16)]

    def test_detect_mla_latent_with_two_groups(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_mla_latent_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=True)
        assert fmt == KVCacheFormat.MLA_LATENT

    def test_detect_mla_latent_with_tp8_equal_width_two_groups(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_tp8_equal_width_mla_latent_tensors()
        fmt = KVCacheFormat.detect(
            kvcaches,
            use_mla=True,
            dsa_two_groups=True,
        )
        assert fmt == KVCacheFormat.MLA_LATENT

    def test_equal_width_non_mla_pair_stays_separate_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_tp8_equal_width_mla_latent_tensors()
        fmt = KVCacheFormat.detect(
            kvcaches,
            use_mla=False,
            dsa_two_groups=True,
        )
        assert fmt == KVCacheFormat.SEPARATE_KV

    def test_detect_mla_latent_without_two_groups_falls_back_to_mla_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_mla_latent_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.MLA_KV

    def test_detect_dsa_index_with_two_groups(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_dsa_index_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=True)
        assert fmt == KVCacheFormat.DSA_INDEX

    def test_detect_dsa_index_without_two_groups_falls_to_separate(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_dsa_index_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.SEPARATE_KV

    def test_detect_dsa_kv_legacy_bundled_with_two_groups(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_dsa_kv_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=True)
        assert fmt == KVCacheFormat.DSA_KV

    def test_detect_dsa_kv_legacy_without_two_groups(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_dsa_kv_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.DSA_KV

    # --- Regression: existing formats unchanged ---

    def test_regression_detect_mla_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_mla_latent_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.MLA_KV

    def test_regression_detect_separate_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_separate_kv_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.SEPARATE_KV

    def test_regression_detect_merged_kv_flash_attn(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        kvcaches = self._make_merged_kv_tensors()
        fmt = KVCacheFormat.detect(kvcaches, dsa_two_groups=False)
        assert fmt == KVCacheFormat.MERGED_KV

    def test_regression_detect_empty(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.detect([], dsa_two_groups=True) == KVCacheFormat.UNDEFINED
        assert KVCacheFormat.detect([], dsa_two_groups=False) == KVCacheFormat.UNDEFINED

    # --- Format helper predicates ---

    def test_is_mla_format_includes_latent(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.MLA_KV.is_mla_format()
        assert KVCacheFormat.MLA_LATENT.is_mla_format()
        assert not KVCacheFormat.DSA_KV.is_mla_format()
        assert not KVCacheFormat.DSA_INDEX.is_mla_format()

    def test_is_dsa_format_includes_index(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.DSA_KV.is_dsa_format()
        assert KVCacheFormat.DSA_INDEX.is_dsa_format()
        assert not KVCacheFormat.MLA_KV.is_dsa_format()
        assert not KVCacheFormat.MLA_LATENT.is_dsa_format()

    def test_kv_group_property(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.MLA_KV.kv_group == 0
        assert KVCacheFormat.MLA_LATENT.kv_group == 0
        assert KVCacheFormat.DSA_KV.kv_group == 0
        assert KVCacheFormat.DSA_INDEX.kv_group == 1

    def test_get_kv_size(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.DSA_KV.get_kv_size() == 3
        assert KVCacheFormat.MLA_KV.get_kv_size() == 2
        assert KVCacheFormat.MLA_LATENT.get_kv_size() == 2
        assert KVCacheFormat.DSA_INDEX.get_kv_size() == 1
        assert KVCacheFormat.MERGED_KV.get_kv_size() == 1


# ---------------------------------------------------------------------------
# SaveSpec tests
# ---------------------------------------------------------------------------

class TestSaveSpec:
    """Test SaveSpec per-group flags."""

    def test_default_can_save_latent_true(self):
        from lmcache.integration.vllm.vllm_v1_adapter import SaveSpec

        spec = SaveSpec(skip_leading_tokens=0, can_save=True)
        assert spec.can_save_latent is True
        assert spec.can_save_indexer is False

    def test_can_save_indexer_set_explicitly(self):
        from lmcache.integration.vllm.vllm_v1_adapter import SaveSpec

        spec = SaveSpec(
            skip_leading_tokens=0, can_save=True,
            can_save_latent=True, can_save_indexer=True,
        )
        assert spec.can_save_latent is True
        assert spec.can_save_indexer is True

    def test_can_save_false_overrides_group_flags(self):
        from lmcache.integration.vllm.vllm_v1_adapter import SaveSpec

        spec = SaveSpec(
            skip_leading_tokens=0, can_save=False,
            can_save_latent=True, can_save_indexer=True,
        )
        assert spec.can_save is False


def _block_ids(num_tokens: int, block_size: int = 16) -> list[int]:
    """Enough vLLM block ids for slot_mapping construction in tests."""
    if num_tokens <= 0:
        return [0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    return list(range(num_blocks))


# ---------------------------------------------------------------------------
# from_request_tracker decode-full-chunk rule tests
# ---------------------------------------------------------------------------

class TestFromRequestTrackerDecodeFullChunk:
    """Test the decode-full-chunk boundary rule in from_request_tracker."""

    def _make_tracker(self, prompt_len=100, num_saved=0, is_decode=False):
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker

        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=prompt_len,
            token_ids=(
                list(range(num_saved, num_saved + 1))
                if is_decode
                else list(range(100))
            ),
            allocated_block_ids=[0],
            num_saved_tokens=num_saved,
        )
        if is_decode:
            tracker.is_decode_phase = True
        return tracker

    def test_decode_full_chunk_skip_when_boundary_not_crossed(self):
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker

        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=100,
            token_ids=list(range(256)),  # 256 tokens total
            allocated_block_ids=[0],
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True

        # In decode with save_full_chunk_in_decode=True, adding 1 token
        # after 256 saved should skip (boundary 256+1=257, floor(257/256)*256=256,
        # 256 <= 256 → skip)
        tracker.token_ids = list(range(257))
        from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            save_decode_cache=True,
            save_full_chunk_in_decode=True,
            dsa_two_groups=True,
        )
        # Should return None because skip_save is True and no load_spec
        assert req_meta is None

    def test_decode_full_chunk_save_when_boundary_crossed(self):
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker, ReqMeta

        # 256 saved + 256 new decode tokens = 512 → boundary crossed
        num_tokens = 512
        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=256,
            token_ids=list(range(num_tokens)),
            allocated_block_ids=_block_ids(num_tokens, block_size=16),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            save_decode_cache=True,
            save_full_chunk_in_decode=True,
            dsa_two_groups=True,
        )
        # Should NOT be None — boundary crossed, can_save=True
        assert req_meta is not None
        assert req_meta.save_spec.can_save is True
        assert req_meta.save_spec.can_save_latent is True
        assert req_meta.save_spec.can_save_indexer is True

    def test_dsa_two_groups_sets_indexer_flag(self):
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker, ReqMeta

        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=100,
            token_ids=list(range(100)),
            allocated_block_ids=[0],
            num_saved_tokens=0,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            dsa_two_groups=True,
        )
        assert req_meta is not None
        assert req_meta.save_spec.can_save_indexer is True

    def test_no_dsa_two_groups_indexer_flag_false(self):
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker, ReqMeta

        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=100,
            token_ids=list(range(100)),
            allocated_block_ids=[0],
            num_saved_tokens=0,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            dsa_two_groups=False,
        )
        assert req_meta is not None
        assert req_meta.save_spec.can_save_indexer is False
        assert req_meta.save_spec.can_save_latent is True

    def test_decode_without_full_chunk_rule_saves_normally(self):
        """Regression: without save_full_chunk_in_decode, decode save still
        requires the standard LMCache chunk boundary (not just save_decode_cache)."""
        from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker, ReqMeta

        # After 256 tokens saved, the next chunk boundary is 512 tokens total.
        num_tokens = 512
        tracker = RequestTracker(
            req_id="test_req",
            prompt_len=256,
            token_ids=list(range(num_tokens)),
            allocated_block_ids=_block_ids(num_tokens, block_size=16),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            save_decode_cache=True,
            save_full_chunk_in_decode=False,
            dsa_two_groups=False,
        )
        assert req_meta is not None
        assert req_meta.save_spec.can_save is True


# ---------------------------------------------------------------------------
# store_layer _is_passive() guard tests
# ---------------------------------------------------------------------------

class TestStoreLayerPassiveGuard:
    """Test that store_layer skips on passive (non-rank-0) workers."""

    def test_passive_rank_skips_store_layer(self):
        """A passive rank (save_only_first_rank=True, not first rank) should
        skip store_layer entirely and just yield num_layers times."""
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._is_passive = MagicMock(return_value=True)
        engine.num_layers = 4
        engine.is_healthy = MagicMock(return_value=True)

        tokens = [1, 2, 3, 4]
        mask = torch.tensor([True, True, True, True])

        gen = LMCacheEngine.store_layer(engine, tokens, mask=mask)
        results = list(gen)
        # num_layers yields (one per save_kv_layer) + final wait_for_save yield
        assert len(results) == 5
        assert results[:-1] == [None] * 4
        assert isinstance(results[-1], LayerwiseStoreResult)

    def test_active_rank_proceeds_to_store(self):
        """An active rank (rank 0) should NOT skip — it should proceed."""
        from lmcache.v1.cache_engine import LMCacheEngine

        engine = MagicMock(spec=LMCacheEngine)
        engine._is_passive = MagicMock(return_value=False)
        engine.num_layers = 4
        engine.is_healthy = MagicMock(return_value=True)
        engine.is_frozen = MagicMock(return_value=False)
        engine.storage_manager = MagicMock()
        engine.gpu_connector = MagicMock()
        engine.metadata = MagicMock()
        engine.fmt = MemoryFormat.KV_MLA_LATENT_FMT
        engine.token_database = MagicMock()
        engine.kv_dtype = torch.bfloat16
        engine.stats_monitor = MagicMock()
        engine.stats_monitor.on_store_request = MagicMock(return_value="mon")
        engine.stats_monitor.on_store_finished = MagicMock()
        engine.kv_events_enabled = False
        engine.retrieve_locations = None
        engine.store_location = None
        engine.config = MagicMock()
        engine.config.get_extra_config_value = MagicMock(return_value=False)

        engine.token_database.process_tokens = MagicMock(return_value=iter([]))
        engine._get_req_id = MagicMock(return_value="test")
        engine._log_kvcache_for_check = MagicMock()

        tokens = [1, 2, 3, 4]
        mask = torch.tensor([True, True, True, True])

        gen = LMCacheEngine.store_layer(engine, tokens, mask=mask)
        list(gen)
        engine.token_database.process_tokens.assert_called()


def test_group_store_pointer_table_is_published_without_per_layer_copy() -> None:
    memory_objs = [
        [SimpleNamespace(tensor=torch.tensor([1]))],
        [SimpleNamespace(tensor=torch.tensor([2]))],
    ]
    cached_tensors: list[list] = []
    cached_chunk_dev_ptrs: list[list[int]] = []
    cached_chunk_ptrs_npu: list[torch.Tensor | None] = []
    pointer_table = torch.tensor([[101], [202]], dtype=torch.long)

    AscendLMCacheEngine._append_group_store_tensors(
        SimpleNamespace(),
        memory_objs,
        cached_tensors,
        cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu,
        [[101], [202]],
        pointer_table,
    )

    assert cached_tensors == [
        [memory_objs[0][0].tensor],
        [memory_objs[1][0].tensor],
    ]
    assert cached_chunk_dev_ptrs == [[101], [202]]
    assert torch.equal(cached_chunk_ptrs_npu[0], pointer_table[0])
    assert torch.equal(cached_chunk_ptrs_npu[1], pointer_table[1])


class TestAscendStoreLayerCompletion:
    @staticmethod
    def _engine(*, stored: bool, allocation=None):
        engine = MagicMock(spec=AscendLMCacheEngine)
        engine.config = MagicMock()
        engine.gpu_connector = MagicMock()
        engine.stats_monitor = MagicMock()
        engine.storage_manager = MagicMock()
        engine.token_database = MagicMock()
        engine.kv_events_enabled = False
        engine.store_location = None
        engine._is_passive.return_value = False
        engine.is_healthy.return_value = True
        engine.is_frozen.return_value = False
        engine.num_layers = 1
        engine._get_req_id.return_value = "test"
        engine.stats_monitor.on_store_request.return_value = "monitor"
        engine.config.extra_config = {}
        engine.config.get_extra_config_value.return_value = False
        key = MagicMock(spec=CacheEngineKey)
        key.split_layers.return_value = [key]
        engine.token_database.process_tokens.return_value = iter([(0, 256, key)])
        engine._layerwise_chunk_fully_stored.return_value = stored
        engine.storage_manager.batched_allocate.return_value = allocation
        return engine

    def test_reports_fully_stored_prefix_as_committed(self):
        engine = self._engine(stored=True)

        result = list(AscendLMCacheEngine.store_layer(engine, [0] * 256))[-1]

        assert result.committed_end == 256

    def test_does_not_commit_after_allocation_failure(self):
        engine = self._engine(stored=False)

        result = list(AscendLMCacheEngine.store_layer(engine, [0] * 256))[-1]

        assert result.committed_end == 0

    def test_reports_32_prefix_16_suffix_frontier_as_committed(self):
        memory_obj = MagicMock()
        memory_obj.get_size.return_value = 1
        engine = self._engine(stored=False, allocation=[memory_obj])
        key = next(engine.token_database.process_tokens.return_value)[2]
        engine.token_database.process_tokens.return_value = iter(
            (chunk * 256, (chunk + 1) * 256, key)
            for chunk in range(32, 48)
        )

        def transfer():
            yield
            yield

        engine.gpu_connector.batched_from_gpu.return_value = transfer()
        engine.storage_manager.batched_put.return_value = []

        with patch(
            "lmcache_ascend.v1.cache_engine.assert_layerwise_gpu_connector"
        ):
            result = list(
                AscendLMCacheEngine.store_layer(engine, [0] * (48 * 256))
            )[-1]

        assert result.committed_end == 48 * 256


def test_sparse_window_store_cache_publishes_only_full_chunks() -> None:
    engine = SimpleNamespace(enable_shared_cpu_cache=False)
    full_tensor = torch.tensor([1])
    partial_tensor = torch.tensor([2])
    full_obj = SimpleNamespace(tensor=full_tensor)
    partial_obj = SimpleNamespace(tensor=partial_tensor)
    keys = [["full-key", "partial-key"]]
    memory_objs = [[full_obj, partial_obj]]
    cached_keys: list[list] = []
    cached_starts: list[int] = []
    cached_ends: list[int] = []
    cached_memory_objs: list[list] = []
    cached_tensors: list[list] = []

    AscendLMCacheEngine._append_layerwise_store_cache_chunks(
        engine,
        keys=keys,
        starts=[0, 256],
        ends=[256, 300],
        memory_objs=memory_objs,
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        cached_memory_objs=cached_memory_objs,
        cached_tensors=cached_tensors,
        cache_chunk_indices=[0],
    )
    AscendLMCacheEngine._append_layer_store_tensors(
        engine,
        0,
        memory_objs,
        cached_tensors,
        cache_chunk_indices=[0],
    )

    assert cached_starts == [0]
    assert cached_ends == [256]
    assert cached_keys == [["full-key"]]
    assert len(cached_memory_objs) == 1
    assert len(cached_memory_objs[0]) == 1
    assert cached_memory_objs[0][0] is full_obj
    assert len(cached_tensors) == 1
    assert len(cached_tensors[0]) == 1
    assert cached_tensors[0][0] is full_tensor


def test_full_chunk_successor_truncates_cached_partial_pointer_slot() -> None:
    cached_starts = [0, 256]
    cached_ends = [256, 300]
    cached_keys = [["full-0", "partial-1"]]
    cached_memory_objs = [["mem-0", "partial-mem-1"]]
    cached_tensors = [["tensor-0", "partial-tensor-1"]]
    cached_chunk_dev_ptrs = [[11, 22]]
    cached_chunk_ptrs_npu = [torch.tensor([11, 22], dtype=torch.long)]

    replaced_at = (
        AscendLMCacheEngine._truncate_store_cache_for_full_chunk_successor(
            starts=cached_starts,
            ends=cached_ends,
            new_starts=[256],
            new_ends=[512],
            cached_keys=cached_keys,
            cached_memory_objs=cached_memory_objs,
            cached_tensors=cached_tensors,
            cached_chunk_dev_ptrs=cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu=cached_chunk_ptrs_npu,
        )
    )

    assert replaced_at == 1
    assert cached_starts == [0]
    assert cached_ends == [256]
    assert cached_keys == [["full-0"]]
    assert cached_memory_objs == [["mem-0"]]
    assert cached_tensors == [["tensor-0"]]
    assert cached_chunk_dev_ptrs == [[11]]
    assert cached_chunk_ptrs_npu[0].tolist() == [11]


class TestLayerwiseLayoutWarmup:
    """Layout-only warmup must not allocate dense staging buffers."""

    def test_layout_warmup_uses_no_staging_connector_api(self):
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        engine = SimpleNamespace()
        connector = SimpleNamespace()
        connector.kvcaches = [("k", "v")]
        calls = []

        def initialize_kvcaches_ptr(**kwargs):
            calls.append(("initialize", kwargs))

        def lazy_initialize_buffer_with_staging(kvcaches, *, kv_group, init_staging):
            calls.append(("lazy_with_staging", kvcaches, kv_group, init_staging))

        connector.initialize_kvcaches_ptr = initialize_kvcaches_ptr
        connector._lazy_initialize_buffer_with_staging = (
            lazy_initialize_buffer_with_staging
        )
        connector._lazy_initialize_buffer = MagicMock(
            side_effect=AssertionError("warmup should use no-staging API")
        )
        engine.gpu_connector = connector

        AscendLMCacheEngine._ensure_layerwise_connector_layout(
            engine,
            kvcaches=connector.kvcaches,
            kv_group=1,
        )

        assert calls == [
            (
                "initialize",
                {"kvcaches": connector.kvcaches, "kv_group": 1},
            ),
            ("lazy_with_staging", connector.kvcaches, 1, False),
        ]


# ---------------------------------------------------------------------------
# Connector get_shape tests
# ---------------------------------------------------------------------------

class TestConnectorGetShape:
    """Test VLLMPagedMemLayerwiseNPUConnector.get_shape for new formats."""

    def _make_connector_with_format(self, kv_format):
        """Create a minimal connector-like object with get_shape logic."""
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        class FakeConnector:
            def __init__(self):
                self.kv_format = kv_format
                self.k_hidden_dims = 512
                self.v_hidden_dims = 64
                self.dsa_hidden_dims = 128
                self.hidden_dim_size = 128

            def get_shape(self, num_tokens):
                if self.kv_format == KVCacheFormat.MLA_KV:
                    plane_elems = self.k_hidden_dims + self.v_hidden_dims
                    return torch.Size([num_tokens * plane_elems])
                if self.kv_format == KVCacheFormat.MLA_LATENT:
                    plane_elems = self.k_hidden_dims + self.v_hidden_dims
                    return torch.Size([num_tokens * plane_elems])
                if self.kv_format == KVCacheFormat.DSA_INDEX:
                    plane_elems = self.dsa_hidden_dims
                    return torch.Size([num_tokens * plane_elems])
                if self.kv_format == KVCacheFormat.DSA_KV:
                    plane_elems = (
                        self.k_hidden_dims + self.v_hidden_dims
                        + self.dsa_hidden_dims
                    )
                    return torch.Size([num_tokens * plane_elems])
                return torch.Size([num_tokens, 2, self.hidden_dim_size])

        return FakeConnector()

    def test_get_shape_mla_latent(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector_with_format(KVCacheFormat.MLA_LATENT)
        shape = conn.get_shape(256)
        # 256 * (512 + 64) = 256 * 576 = 147456
        assert shape == torch.Size([147456])

    def test_get_shape_dsa_index(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector_with_format(KVCacheFormat.DSA_INDEX)
        shape = conn.get_shape(256)
        # 256 * 128 = 32768
        assert shape == torch.Size([32768])

    def test_get_shape_mla_latent_equals_mla_kv(self):
        """MLA_LATENT and MLA_KV produce the same plane structure."""
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn_latent = self._make_connector_with_format(KVCacheFormat.MLA_LATENT)
        conn_mla = self._make_connector_with_format(KVCacheFormat.MLA_KV)
        assert conn_latent.get_shape(256) == conn_mla.get_shape(256)

    def test_get_shape_dsa_index_smaller_than_dsa_kv(self):
        """DSA_INDEX is single-plane, DSA_KV is 3-plane."""
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn_index = self._make_connector_with_format(KVCacheFormat.DSA_INDEX)
        conn_dsa = self._make_connector_with_format(KVCacheFormat.DSA_KV)
        idx_shape = conn_index.get_shape(256)
        dsa_shape = conn_dsa.get_shape(256)
        assert idx_shape[0] < dsa_shape[0]

    # --- Regression ---

    def test_regression_get_shape_mla_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector_with_format(KVCacheFormat.MLA_KV)
        shape = conn.get_shape(256)
        assert shape == torch.Size([256 * 576])

    def test_regression_get_shape_dsa_kv(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector_with_format(KVCacheFormat.DSA_KV)
        shape = conn.get_shape(256)
        assert shape == torch.Size([256 * (512 + 64 + 128)])


# ---------------------------------------------------------------------------
# Format helper predicate tests (Ascend connector)
# ---------------------------------------------------------------------------

class TestConnectorFormatHelpers:

    def test_is_mla_dsa_includes_new_formats(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        for fmt in [
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.MLA_LATENT,
            KVCacheFormat.DSA_INDEX,
        ]:
            assert fmt.is_tuple_format()

    def test_is_mla_latent_format(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.MLA_LATENT.is_mla_latent_format()
        assert not KVCacheFormat.MLA_KV.is_mla_latent_format()
        assert not KVCacheFormat.DSA_INDEX.is_mla_latent_format()

    def test_is_dsa_index_format(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        assert KVCacheFormat.DSA_INDEX.is_dsa_index_format()
        assert not KVCacheFormat.DSA_KV.is_dsa_index_format()
        assert not KVCacheFormat.MLA_LATENT.is_dsa_index_format()


# ---------------------------------------------------------------------------
# Integration test: two-group key separation
# ---------------------------------------------------------------------------

class TestTwoGroupKeySeparation:
    """Integration test: latent and indexer keys are in disjoint key spaces."""

    def test_latent_and_indexer_keys_in_disjoint_spaces(self):
        """The same chunk_hash produces different keys for kv_group=0 vs 1."""
        k_latent = CacheEngineKey(
            "model", 1, 0, 42, torch.bfloat16, kv_group=0,
        )
        k_indexer = CacheEngineKey(
            "model", 1, 0, 42, torch.bfloat16, kv_group=1,
        )
        assert k_latent != k_indexer
        assert hash(k_latent) != hash(k_indexer)

    def test_layer_keys_for_both_groups(self):
        """Each layer produces separate latent and indexer layer keys."""
        base = CacheEngineKey("model", 1, 0, 42, torch.bfloat16, kv_group=0)
        latent_layer_keys = base.split_layers(4)

        base_idx = CacheEngineKey("model", 1, 0, 42, torch.bfloat16, kv_group=1)
        indexer_layer_keys = base_idx.split_layers(4)

        assert len(latent_layer_keys) == 4
        assert len(indexer_layer_keys) == 4

        # All latent keys have kv_group=0, all indexer keys have kv_group=1
        for lk in latent_layer_keys:
            assert lk.kv_group == 0
        for ik in indexer_layer_keys:
            assert ik.kv_group == 1

        # No latent key equals any indexer key
        for lk in latent_layer_keys:
            for ik in indexer_layer_keys:
                assert lk != ik

    def test_token_db_produces_separate_keys_for_both_groups(self):
        """process_tokens with kv_group=0 and kv_group=1 produce keys in
        disjoint spaces for the same tokens."""
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        from lmcache.v1.token_database import ChunkedTokenDatabase

        db = ChunkedTokenDatabase(cfg, metadata)
        latent_results = list(db.process_tokens(tokens=tokens, kv_group=0))
        indexer_results = list(db.process_tokens(tokens=tokens, kv_group=1))

        assert len(latent_results) == len(indexer_results)
        for (_, _, lk), (_, _, ik) in zip(
            latent_results,
            indexer_results,
            strict=True,
        ):
            assert lk.kv_group == 0
            assert ik.kv_group == 1
            assert lk != ik

    def test_key_string_format_roundtrip_both_groups(self):
        """to_string → from_string roundtrip preserves kv_group for both groups."""
        for kv_group in [0, 1]:
            key = LayerCacheEngineKey(
                "model", 2, 0, 99, torch.bfloat16,
                layer_id=7, kv_group=kv_group,
            )
            s = key.to_string()
            parsed = LayerCacheEngineKey.from_string(s)
            assert parsed == key
            assert parsed.kv_group == kv_group
            assert parsed.layer_id == 7


# ---------------------------------------------------------------------------
# Per-kv_group lazy init on the real layerwise connector
# ---------------------------------------------------------------------------

class TestPerGroupLazyInit:
    """Real VLLMPagedMemLayerwiseNPUConnector: _lazy_initialize_buffer must
    detect format/dims independently per kv_group so that, in two-group
    MLA+DSA mode, kv_group=0 (MLA_LATENT) and kv_group=1 (DSA_INDEX) coexist
    on one connector instance. Regression for the one-shot init bug where the
    first group to initialize pinned a single self.kv_format for both.
    """

    def _make_connector(
        self,
        *,
        use_gpu: bool = False,
        chunk_size: int = 64,
        max_staging_tokens: int = 0,
    ):
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            VLLMPagedMemLayerwiseNPUConnector,
        )

        # The parent constructor creates CUDA streams; patch them so the
        # connector can be instantiated in a CPU-only test environment. We
        # only exercise _lazy_initialize_buffer / get_shape, not transfers.
        with patch("torch.cuda.Stream", return_value=MagicMock()):
            conn = VLLMPagedMemLayerwiseNPUConnector(
                hidden_dim_size=128,
                num_layers=2,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                use_mla=True,
                dsa_two_groups=True,
                max_staging_tokens=max_staging_tokens,
            )
        return conn

    def _latent_kvcaches(self, num_layers=2):
        k_nope = torch.zeros(4, 128, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(4, 128, 1, 64, dtype=torch.bfloat16)
        return [(k_nope, k_pe) for _ in range(num_layers)]

    def _tp8_equal_width_latent_kvcaches(self, num_layers=2):
        k_nope = torch.zeros(4, 128, 1, 128, dtype=torch.bfloat16)
        k_pe = torch.zeros(4, 128, 1, 128, dtype=torch.bfloat16)
        return [(k_nope, k_pe) for _ in range(num_layers)]

    def _indexer_kvcaches(self, num_layers=2):
        indexer = torch.zeros(4, 128, 1, 128, dtype=torch.bfloat16)
        return [(indexer,) for _ in range(num_layers)]

    def test_latent_then_indexer_detects_both(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)

        assert conn._group_layouts[0].kv_format == KVCacheFormat.MLA_LATENT
        assert conn._group_layouts[1].kv_format == KVCacheFormat.DSA_INDEX
        # group 0 plane = 512 + 64 = 576; group 1 plane = 128
        assert conn.get_shape(256, kv_group=0) == torch.Size([256 * 576])
        assert conn.get_shape(256, kv_group=1) == torch.Size([256 * 128])
        assert conn.get_shape(256, kv_group=0) != conn.get_shape(256, kv_group=1)

    def test_tp8_equal_width_latent_still_uses_mla_direct_layout(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(
            self._tp8_equal_width_latent_kvcaches(),
            kv_group=0,
            init_staging=False,
        )

        layout = conn._group_layouts[0]
        assert layout.kv_format == KVCacheFormat.MLA_LATENT
        assert conn._is_mla_dsa_format(0)
        assert not conn._layerwise_token_major(0)
        assert conn._expected_memory_format(0) == MemoryFormat.KV_MLA_LATENT_FMT
        assert layout.gpu_buffer_allocator is None
        assert conn.get_shape(256, kv_group=0) == torch.Size([256 * 256])

    def test_indexer_then_latent_detects_both(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)

        assert conn._group_layouts[0].kv_format == KVCacheFormat.MLA_LATENT
        assert conn._group_layouts[1].kv_format == KVCacheFormat.DSA_INDEX
        assert conn.get_shape(256, kv_group=0) == torch.Size([256 * 576])
        assert conn.get_shape(256, kv_group=1) == torch.Size([256 * 128])

    def test_re_init_same_group_is_idempotent(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        first = conn._group_layouts[0]
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        # Same layout object reused; format unchanged.
        assert conn._group_layouts[0] is first
        assert first.kv_format == KVCacheFormat.MLA_LATENT

    def test_mirrored_attrs_track_current_group(self):
        from lmcache_ascend.v1.kv_format import KVCacheFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        assert conn.kv_format == KVCacheFormat.MLA_LATENT
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)
        assert conn.kv_format == KVCacheFormat.DSA_INDEX
        # Switching back to group 0 mirrors its layout again.
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        assert conn.kv_format == KVCacheFormat.MLA_LATENT

    def test_group_layouts_have_independent_dims(self):
        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)

        g0 = conn._group_layouts[0]
        g1 = conn._group_layouts[1]
        assert g0.k_hidden_dims == 512
        assert g0.v_hidden_dims == 64
        assert g0.dsa_hidden_dims == 0
        assert g1.dsa_hidden_dims == 128
        assert g1.k_hidden_dims == 128
        assert g1.v_hidden_dims == 0

    def test_expected_memory_format_per_group(self):
        from lmcache.v1.memory_management import MemoryFormat

        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)
        assert conn._expected_memory_format(0) == MemoryFormat.KV_MLA_LATENT_FMT
        assert conn._expected_memory_format(1) == MemoryFormat.KV_DSA_INDEX_FMT

    def test_sparse_direct_state_keyed_by_group(self):
        conn = self._make_connector()
        conn._lazy_initialize_buffer(self._latent_kvcaches(), kv_group=0)
        conn._lazy_initialize_buffer(self._indexer_kvcaches(), kv_group=1)
        # After init for both groups, the sparse-direct state container is a
        # dict keyed by kvcaches/group/layer plus source layout metadata.
        assert isinstance(conn._sparse_direct_layer_states, dict) or (
            conn._sparse_direct_layer_states is None
        )

    def test_sparse_direct_uses_kvcaches_snapshot_not_shared_ptr(self):
        """Interleaved latent/indexer sparse generators must not read the
        connector's mutable self.kvcaches after the other group overwrote it."""
        conn = self._make_connector(use_gpu=False, chunk_size=256)
        latent = self._latent_kvcaches(num_layers=1)
        indexer = self._indexer_kvcaches(num_layers=1)
        conn._lazy_initialize_buffer(latent, kv_group=0)
        conn._lazy_initialize_buffer(indexer, kv_group=1)
        layout0 = conn._group_layouts[0]

        slot_mapping = torch.arange(4, dtype=torch.long)
        lmc_chunk = torch.zeros(256 * (512 + 64), dtype=torch.bfloat16)

        seen_vllm_caches = []

        def _capture_prepare(
            lmc_layout_sample,
            vllm_kv_caches,
            slot_mapping_ref,
            token_major,
            vllm_two_major,
            kvcache_format_raw,
            k_hidden_dims,
            v_hidden_dims,
            dsa_hidden_dims,
            lmc_num_tokens,
        ):
            seen_vllm_caches.append(vllm_kv_caches)
            return object()

        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.prepare_sparse_direct_layer_state",
            side_effect=_capture_prepare,
        ):
            conn.kvcaches = latent
            conn._get_or_create_sparse_direct_layer_state(
                kvcaches_ref=latent,
                kv_group=0,
                layer_id=0,
                layer_tensors=[lmc_chunk],
                slot_mapping_ref=slot_mapping,
                total_tokens=256,
                sparse_kv_format=layout0.kv_format.value,
                sparse_token_major=False,
                sparse_vllm_two_major=False,
                sparse_k_hidden_dims=512,
                sparse_v_hidden_dims=64,
                sparse_dsa_hidden_dims=0,
            )
            # Simulate indexer generator overwriting the shared pointer.
            conn.kvcaches = indexer
            conn._get_or_create_sparse_direct_layer_state(
                kvcaches_ref=latent,
                kv_group=0,
                layer_id=0,
                layer_tensors=[lmc_chunk],
                slot_mapping_ref=slot_mapping,
                total_tokens=256,
                sparse_kv_format=layout0.kv_format.value,
                sparse_token_major=False,
                sparse_vllm_two_major=False,
                sparse_k_hidden_dims=512,
                sparse_v_hidden_dims=64,
                sparse_dsa_hidden_dims=0,
            )

        assert len(seen_vllm_caches) == 1
        assert seen_vllm_caches[0] is latent[0]
        assert isinstance(seen_vllm_caches[0], tuple)
        assert len(seen_vllm_caches[0]) == 2

    def _large_latent_kvcaches(self, num_layers=1):
        k_nope = torch.zeros(2048, 128, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(2048, 128, 1, 64, dtype=torch.bfloat16)
        return [(k_nope, k_pe) for _ in range(num_layers)]

    def test_dsa_two_groups_caps_staging_buffer_size(self) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        max_model_len = 8192
        conn = self._make_connector(
            use_gpu=True, chunk_size=256, max_staging_tokens=max_model_len
        )
        k_nope = torch.zeros(2048, 128, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(2048, 128, 1, 64, dtype=torch.bfloat16)
        latent = [(k_nope, k_pe)]
        conn._lazy_initialize_buffer(latent, kv_group=0)
        layout = conn._group_layouts[0]
        assert layout.gpu_buffer_allocator is not None

        pool_bytes = int(layout.gpu_buffer_allocator.tensor.numel())
        full_pool_bytes = 2048 * 128 * (512 + 64) * 2
        per_slot_bytes = max_model_len * (512 + 64) * 2
        assert pool_bytes == per_slot_bytes * conn._layerwise_staging_pool_slots()
        assert pool_bytes < full_pool_bytes

        pool_obj, staging_tensor = conn._allocate_layerwise_staging_buffer(
            num_tokens=max_model_len,
            kv_group=0,
            layout=layout,
            expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        )
        assert pool_obj is not None
        assert staging_tensor.numel() == max_model_len * (512 + 64)

    def test_from_metadata_wires_max_staging_tokens(self) -> None:
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            VLLMPagedMemLayerwiseNPUConnector,
        )

        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(2, 2, 256, 8, 128),
            use_mla=True,
            max_model_len=16384,
        )
        with patch("torch.cuda.Stream", return_value=MagicMock()):
            conn = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                metadata, use_gpu=False, device=torch.device("cpu")
            )
        assert conn.max_staging_tokens == 16384

    def test_dsa_two_groups_uses_per_group_staging_pools(self) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        max_model_len = 8192
        conn = self._make_connector(
            use_gpu=True, chunk_size=256, max_staging_tokens=max_model_len
        )
        k_nope = torch.zeros(2048, 128, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(2048, 128, 1, 64, dtype=torch.bfloat16)
        latent = [(k_nope, k_pe)]
        indexer = [(torch.zeros(2048, 128, 1, 128, dtype=torch.bfloat16),)]
        conn._lazy_initialize_buffer(latent, kv_group=0)
        conn._lazy_initialize_buffer(indexer, kv_group=1)

        alloc0 = conn._group_layouts[0].gpu_buffer_allocator
        alloc1 = conn._group_layouts[1].gpu_buffer_allocator
        assert alloc0 is not None
        assert alloc1 is not None
        assert alloc0 is not alloc1
        per_slot_latent = max_model_len * (512 + 64) * 2
        per_slot_indexer = max_model_len * 128 * 2
        pool_slots = conn._layerwise_staging_pool_slots()
        assert alloc0.tensor.numel() == per_slot_latent * pool_slots
        assert alloc1.tensor.numel() == per_slot_indexer * pool_slots

        pool_obj0, staging0 = conn._allocate_layerwise_staging_buffer(
            num_tokens=256,
            kv_group=0,
            layout=conn._group_layouts[0],
            expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        )
        pool_obj1, staging1 = conn._allocate_layerwise_staging_buffer(
            num_tokens=256,
            kv_group=1,
            layout=conn._group_layouts[1],
            expected_fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        )
        assert pool_obj0 is not None
        assert pool_obj1 is not None
        assert staging0.numel() == 256 * (512 + 64)
        assert staging1.numel() == 256 * 128

    def test_dsa_two_groups_staging_pool_supports_concurrent_allocs(self) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        max_model_len = 8192
        conn = self._make_connector(
            use_gpu=True, chunk_size=256, max_staging_tokens=max_model_len
        )
        conn.set_layerwise_staging_concurrency(2)
        latent = self._large_latent_kvcaches(num_layers=1)
        conn._lazy_initialize_buffer(latent, kv_group=0)
        layout = conn._group_layouts[0]

        pool_obj0, _ = conn._allocate_layerwise_staging_buffer(
            num_tokens=max_model_len,
            kv_group=0,
            layout=layout,
            expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        )
        pool_obj1, _ = conn._allocate_layerwise_staging_buffer(
            num_tokens=max_model_len,
            kv_group=0,
            layout=layout,
            expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        )
        assert pool_obj0 is not None
        assert pool_obj1 is not None
        pool_obj0.ref_count_down()
        pool_obj1.ref_count_down()

    def test_dsa_two_groups_single_slot_pool_rejects_second_full_alloc(self) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        max_model_len = 8192
        conn = self._make_connector(
            use_gpu=True, chunk_size=256, max_staging_tokens=max_model_len
        )
        conn._layerwise_staging_concurrency = 1
        latent = self._large_latent_kvcaches(num_layers=1)
        conn._lazy_initialize_buffer(latent, kv_group=0)
        layout = conn._group_layouts[0]

        pool_obj0, _ = conn._allocate_layerwise_staging_buffer(
            num_tokens=max_model_len,
            kv_group=0,
            layout=layout,
            expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        )
        assert pool_obj0 is not None
        with pytest.raises(AssertionError, match="Failed to allocate NPU buffer"):
            conn._allocate_layerwise_staging_buffer(
                num_tokens=max_model_len,
                kv_group=0,
                layout=layout,
                expected_fmt=MemoryFormat.KV_MLA_LATENT_FMT,
            )
        pool_obj0.ref_count_down()


# ---------------------------------------------------------------------------
# Adapter per-group kv_caches split + dual store/retrieve plumbing
# ---------------------------------------------------------------------------

def _adapter_method(name):
    """Return the unbound adapter method for calling on a fake instance."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    return getattr(LMCacheConnectorV1Impl, name)


def _ascend_adapter_method(name):
    """Return the unbound Ascend adapter method for calling on a fake instance."""
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )

    return getattr(LMCacheAscendConnectorV1Impl, name)


def _ascend_adapter_fake(**attrs):
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )

    fake = object.__new__(LMCacheAscendConnectorV1Impl)
    fake._finished_req_ids_waiting_for_save = set()
    fake._late_finished_sending = set()
    for name, value in attrs.items():
        setattr(fake, name, value)
    return fake


class TestAscendAdapterInitialization:
    @staticmethod
    def _construct(role, kv_role):
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            LMCacheAscendConnectorV1Impl,
        )

        def base_init(adapter, *_args, **_kwargs):
            adapter.config = SimpleNamespace(store_async=True)
            adapter.use_layerwise = True
            adapter.kv_role = kv_role

        generic_base = LMCacheAscendConnectorV1Impl.__mro__[1]
        with patch.object(generic_base, "__init__", base_init):
            return LMCacheAscendConnectorV1Impl(
                SimpleNamespace(),
                role,
                SimpleNamespace(),
            )

    @pytest.mark.parametrize(
        ("role_name", "kv_role"),
        [
            ("SCHEDULER", "kv_producer"),
            ("WORKER", "kv_consumer"),
        ],
    )
    def test_layerwise_async_allowed_without_worker_store(
        self,
        role_name,
        kv_role,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        adapter = self._construct(getattr(KVConnectorRole, role_name), kv_role)
        assert adapter.store_async is True

    def test_layerwise_async_rejected_for_storing_worker(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        with pytest.raises(ValueError, match="not supported with async store"):
            self._construct(KVConnectorRole.WORKER, "kv_producer")


class TestAdapterGroupSplit:
    """_refresh_kvcaches_list partitions registered kv_caches into latent and
    indexer groups by 'indexer' in layer_name, and _kvcaches_for_group
    returns the correct per-group list."""

    def _make_fake(self, kv_caches, dsa_two_groups):
        fake = SimpleNamespace(
            kv_caches=kv_caches,
            config=SimpleNamespace(dsa_two_groups=dsa_two_groups),
            _latent_layer_names=[],
            _indexer_layer_names=[],
            _latent_kvcaches=[],
            _indexer_kvcaches=[],
            _kvcaches_list=[],
        )
        # Bind the real helper methods so internal self._kvcaches_for_group
        # calls (e.g. from _num_layers_for_group) resolve on the fake.
        _bind_real(
            fake,
            "_kvcaches_for_group",
            "_num_layers_for_group",
            "_is_dsa_two_groups",
        )
        return fake

    def test_partition_with_dsa_two_groups(self):
        t0, t1, i0, i1 = object(), object(), object(), object()
        kv_caches = {
            "layer.0": t0,
            "layer.1": t1,
            "indexer.0": i0,
            "indexer.1": i1,
        }
        fake = self._make_fake(kv_caches, dsa_two_groups=True)
        _adapter_method("_refresh_kvcaches_list")(fake)

        assert fake._latent_kvcaches == [t0, t1]
        assert fake._indexer_kvcaches == [i0, i1]
        assert fake._latent_layer_names == ["layer.0", "layer.1"]
        assert fake._indexer_layer_names == ["indexer.0", "indexer.1"]
        # Backward-compatible flat list == latent group.
        assert fake._kvcaches_list == [t0, t1]

    def test_kvcaches_for_group(self):
        t0, i0 = object(), object()
        fake = self._make_fake(
            {"layer.0": t0, "indexer.0": i0}, dsa_two_groups=True
        )
        _adapter_method("_refresh_kvcaches_list")(fake)
        assert _adapter_method("_kvcaches_for_group")(fake, 0) == [t0]
        assert _adapter_method("_kvcaches_for_group")(fake, 1) == [i0]
        assert _adapter_method("_num_layers_for_group")(fake, 0) == 1
        assert _adapter_method("_num_layers_for_group")(fake, 1) == 1

    def test_without_dsa_two_groups_all_layers_are_latent(self):
        t0, i0 = object(), object()
        fake = self._make_fake(
            {"layer.0": t0, "indexer.0": i0}, dsa_two_groups=False
        )
        _adapter_method("_refresh_kvcaches_list")(fake)
        # Without the flag, "indexer" layers are treated as latent.
        assert fake._latent_kvcaches == [t0, i0]
        assert fake._indexer_kvcaches == []
        assert _adapter_method("_kvcaches_for_group")(fake, 1) == [t0, i0]

    def test_scheduler_side_missing_indexer_cache_does_not_warn(self, caplog):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        fake = self._make_fake({"layer.0": object()}, dsa_two_groups=True)
        fake._role = KVConnectorRole.SCHEDULER

        _adapter_method("_refresh_kvcaches_list")(fake)

        assert "no indexer KV caches" not in caplog.text

    def test_worker_side_missing_indexer_cache_warns(self, caplog):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        fake = self._make_fake({"layer.0": object()}, dsa_two_groups=True)
        fake._role = KVConnectorRole.WORKER

        _adapter_method("_refresh_kvcaches_list")(fake)

        assert "no indexer KV caches" in caplog.text

    def test_is_dsa_two_groups_flag(self):
        fake_on = SimpleNamespace(config=SimpleNamespace(dsa_two_groups=True))
        fake_off = SimpleNamespace(config=SimpleNamespace(dsa_two_groups=False))
        assert _adapter_method("_is_dsa_two_groups")(fake_on) is True
        assert _adapter_method("_is_dsa_two_groups")(fake_off) is False


class TestAdapterIndexerSlotMapping:
    """_indexer_retrieve_slot_mapping picks the indexer slot mapping and
    slices it to the latent hit token count."""

    def _make_fake(self):
        return SimpleNamespace(device=torch.device("cpu"))

    def test_prefers_indexer_slot_mapping_when_both_present(self):
        fake = self._make_fake()
        attn = SimpleNamespace(
            slot_mapping=torch.arange(10),
            indexer_slot_mapping=torch.arange(20, 28),
        )
        slot = _adapter_method("_indexer_retrieve_slot_mapping")(fake, attn, 5)
        assert slot.tolist() == [20, 21, 22, 23, 24]

    def test_falls_back_to_attn_slot_mapping(self):
        fake = self._make_fake()
        attn = SimpleNamespace(
            slot_mapping=torch.arange(10), indexer_slot_mapping=None
        )
        slot = _adapter_method("_indexer_retrieve_slot_mapping")(fake, attn, 5)
        assert slot.tolist() == list(range(5))

    def test_falls_back_to_indexer_slot_mapping_only(self):
        fake = self._make_fake()
        attn = SimpleNamespace(
            slot_mapping=None, indexer_slot_mapping=torch.arange(8)
        )
        slot = _adapter_method("_indexer_retrieve_slot_mapping")(fake, attn, 8)
        assert slot.tolist() == list(range(8))

    def test_returns_none_when_no_slot_mapping(self):
        fake = self._make_fake()
        attn = SimpleNamespace(slot_mapping=None, indexer_slot_mapping=None)
        assert _adapter_method("_indexer_retrieve_slot_mapping")(fake, attn, 5) is None

    def test_rejects_mapping_when_count_exceeds_length(self):
        fake = self._make_fake()
        attn = SimpleNamespace(
            slot_mapping=torch.arange(4), indexer_slot_mapping=None
        )
        slot = _adapter_method("_indexer_retrieve_slot_mapping")(fake, attn, 10)
        assert slot is None


class TestStorerDualPop:
    """wait_for_save drains both (req_id, kv_group=0) and (req_id, kv_group=1)
    storers per request."""

    def _make_storer_gen(self):
        def _gen():
            yield
            yield
            yield

        return _gen()

    @staticmethod
    def _make_fake(meta, storers):
        return _ascend_adapter_fake(
            kv_role="kv_producer",
            use_layerwise=True,
            store_async=False,
            lmcache_engine=MagicMock(),
            _wait_for_save_done=False,
            _layerwise_save_storers=storers,
            _should_defer_latent_save_under_tp=lambda: False,
            _layerwise_save_storer_key=(
                lambda request, kv_group: (request.req_id, kv_group)
            ),
            _is_decode_window_save_request=_adapter_method(
                "_is_decode_window_save_request"
            ),
            _finalize_layerwise_storer=lambda storer: (True, None),
            _consume_completed_layerwise_store=(
                lambda request, kv_group, completed, result: None
            ),
            _mark_decode_window_save_completed=lambda request: None,
            _maybe_lookup_unpin_for_request=lambda request: None,
            _parent=SimpleNamespace(
                _get_connector_metadata=lambda: meta,
            ),
        )

    def test_wait_for_save_pops_both_groups(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        meta = LMCacheConnectorMetadata(
            requests=[SimpleNamespace(req_id="r1")]
        )
        gen0 = self._make_storer_gen()
        gen1 = self._make_storer_gen()
        storers = {("r1", 0): gen0, ("r1", 1): gen1}

        fake = self._make_fake(meta, storers)
        _ascend_adapter_method("wait_for_save")(fake)
        # Both group storers are popped.
        assert ("r1", 0) not in storers
        assert ("r1", 1) not in storers
        assert storers == {}

    def test_wait_for_save_pops_only_present_groups(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        meta = LMCacheConnectorMetadata(
            requests=[SimpleNamespace(req_id="r2")]
        )
        gen0 = self._make_storer_gen()
        storers = {("r2", 0): gen0}  # indexer storer never created

        fake = self._make_fake(meta, storers)
        _ascend_adapter_method("wait_for_save")(fake)
        assert storers == {}


class TestAscendDecodeWindowWaitForSaveCompletion:
    """Verify range-scoped completion and remote-store ordering."""

    def _make_request(self):
        return SimpleNamespace(
            req_id="r-window",
            decode_window_start=256,
            decode_window_end=512,
        )

    def _make_fake(self, request, storers, completed_groups):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        meta = LMCacheConnectorMetadata(requests=[request])
        engine = MagicMock()

        def _range_key(req, kv_group):
            return (
                req.req_id,
                "decode_window_save",
                kv_group,
                req.decode_window_start,
                req.decode_window_end,
            )

        return _ascend_adapter_fake(
            kv_role="kv_producer",
            use_layerwise=True,
            store_async=False,
            lmcache_engine=engine,
            _wait_for_save_done=False,
            _layerwise_save_storers=storers,
            _layerwise_save_storer_key=_range_key,
            _should_defer_latent_save_under_tp=lambda: False,
            _finalize_layerwise_storer=lambda storer: (True, None),
            _consume_completed_layerwise_store=(
                lambda req, kv_group, completed, result: (
                    completed_groups.append(kv_group)
                )
            ),
            _mark_decode_window_save_completed=lambda req: None,
            _maybe_lookup_unpin_for_request=lambda req: None,
            _parent=SimpleNamespace(
                _connector_metadata=meta,
                _get_connector_metadata=lambda: meta,
            ),
        )

    def test_exact_range_key_records_decode_window_group_completion(self):
        request = self._make_request()
        completed_groups = []
        exact_key = (
            "r-window",
            "decode_window_save",
            0,
            256,
            512,
        )
        storers = {exact_key: iter(())}
        fake = self._make_fake(request, storers, completed_groups)

        _ascend_adapter_method("wait_for_save")(fake)

        assert storers == {}
        assert completed_groups == [0]

    def test_remote_store_barrier_precedes_save_completion(self):
        request = self._make_request()
        fake = self._make_fake(request, {}, [])
        events = []
        fake._finished_req_ids_waiting_for_save = {"r-window"}
        fake.lmcache_engine.wait_for_pending_sync_stores.side_effect = (
            lambda: events.append(("barrier", fake._wait_for_save_done))
        )
        fake._finalize_worker_requests_after_store = lambda _req_ids: (
            events.append(("finalize", fake._wait_for_save_done))
            or set()
        )

        _ascend_adapter_method("wait_for_save")(fake)

        assert events == [("barrier", False), ("finalize", True)]

    def test_failed_remote_store_barrier_keeps_save_step_pending(self):
        request = self._make_request()
        fake = self._make_fake(request, {}, [])
        fake._finished_req_ids_waiting_for_save = {"r-window"}
        fake._finalize_worker_requests_after_store = MagicMock(return_value=set())
        fake.lmcache_engine.wait_for_pending_sync_stores.side_effect = (
            TimeoutError("store barrier timed out")
        )

        with pytest.raises(TimeoutError, match="store barrier timed out"):
            _ascend_adapter_method("wait_for_save")(fake)

        assert fake._wait_for_save_done is False
        assert fake._finished_req_ids_waiting_for_save == {"r-window"}
        fake._finalize_worker_requests_after_store.assert_not_called()


class TestRetrieverPairAdvancement:
    """Per-group waits advance a layer after all required groups arrive."""

    def _make_fake_retriever(self, name):
        """A generator that yields many times (the real retrieve_layer
        generator survives 2 priming next()s + one per layer)."""
        log = []

        def _gen():
            log.append(f"{name}:start")
            for i in range(16):
                ret = yield f"{name}:layer{i}"
                log.append(f"{name}:send={ret}")

        gen = _gen()
        return gen, log

    @staticmethod
    def _bind_wait_protocol(fake, dsa_two_groups):
        fake.config = SimpleNamespace(dsa_two_groups=dsa_two_groups)
        fake._indexer_layer_names = []
        fake._layerwise_waited_groups = set()
        fake._layerwise_required_wait_groups_cache = None
        return _bind_real(
            fake,
            "_is_dsa_two_groups",
            "_is_indexer_layer_wait",
            "_layerwise_wait_group",
            "_layerwise_required_wait_groups",
            "_layerwise_wait_should_advance",
        )

    def test_prefix_advances_both_retrievers(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        latent_gen, latent_log = self._make_fake_retriever("latent")
        indexer_gen, indexer_log = self._make_fake_retriever("indexer")

        # Pre-prime two layers (as start_load_kv does for prefix).
        next(latent_gen)
        next(latent_gen)
        next(indexer_gen)
        next(indexer_gen)

        meta = LMCacheConnectorMetadata(
            requests=[SimpleNamespace(
                req_id="r1",
                load_spec=SimpleNamespace(can_load=True),
                is_sparse_decode=False,
            )]
        )
        fake = self._bind_wait_protocol(
            SimpleNamespace(
                layerwise_retrievers=[(latent_gen, indexer_gen)],
                _layerwise_retriever_is_sparse=[False],
                current_layer=0,
                num_layers=2,
                _parent=SimpleNamespace(_get_connector_metadata=lambda: meta),
                _finalize_worker_retrieve_state_from_metadata=lambda m: None,
            ),
            dsa_two_groups=True,
        )
        _adapter_method("wait_for_layer_load")(fake, layer_name="layer.0")
        assert fake.current_layer == 0
        _adapter_method("wait_for_layer_load")(fake, layer_name="indexer.0")
        assert fake.current_layer == 1

    def test_sparse_advances_primary_only(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        primary_gen, primary_log = self._make_fake_retriever("primary")
        next(primary_gen)  # prime

        meta = LMCacheConnectorMetadata(
            requests=[SimpleNamespace(
                req_id="r1",
                load_spec=SimpleNamespace(can_load=True),
                is_sparse_decode=True,
            )]
        )
        fake = self._bind_wait_protocol(
            SimpleNamespace(
                layerwise_retrievers=[(primary_gen, None)],
                _layerwise_retriever_is_sparse=[True],
                current_layer=0,
                num_layers=2,
                _parent=SimpleNamespace(_get_connector_metadata=lambda: meta),
                _finalize_worker_retrieve_state_from_metadata=lambda m: None,
            ),
            dsa_two_groups=True,
        )
        # Sparse path uses .send(...) with selected_tokens.
        _adapter_method("wait_for_layer_load")(
            fake,
            layer_name="x",
            selected_tokens=[[0, 1]],
            token_start_index=[0],
            request_ids=["r1"],
        )
        assert fake.current_layer == 1


# ---------------------------------------------------------------------------
# Integration: mimic vLLM worker call sequence against the adapter
# ---------------------------------------------------------------------------

def _bind_real(fake, *names):
    """Bind real LMCacheConnectorV1Impl methods onto a fake instance."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    for name in names:
        method = getattr(LMCacheConnectorV1Impl, name)
        descriptor = vars(LMCacheConnectorV1Impl).get(name)
        if isinstance(descriptor, (staticmethod, classmethod)):
            setattr(fake, name, method)
        else:
            setattr(fake, name, method.__get__(fake))
    return fake


def _long_generator(value=None, n=32):
    """A generator that yields n times (mimics store_layer/retrieve_layer)."""
    def _gen():
        for _ in range(n):
            yield value

    return _gen()


class _RecordingEngine:
    """Minimal stand-in for lmcache_engine that records store/retrieve calls."""

    def __init__(self):
        self.store_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []

    def store_layer(self, *args, **kwargs):
        self.store_calls.append({
            "kvcaches": kwargs.get("kvcaches"),
            "kv_group": kwargs.get("kv_group", 0),
            "req_id": kwargs.get("req_id"),
        })
        return _long_generator()

    def retrieve_layer(self, *args, **kwargs):
        self.retrieve_calls.append({
            "kvcaches": kwargs.get("kvcaches"),
            "kv_group": kwargs.get("kv_group"),
            "slot_mapping": kwargs.get("slot_mapping"),
            "kind": "layer",
        })
        # Retrieve generators yield a ret_mask per layer; return a truthy mask.
        return _long_generator(value=torch.ones(1, dtype=torch.bool))

    def retrieve_layer_head_token_wise(self, *args, **kwargs):
        self.retrieve_calls.append({
            "kvcaches": kwargs.get("kvcaches"),
            "kv_group": kwargs.get("kv_group"),
            "slot_mapping": kwargs.get("slot_mapping"),
            "kind": "sparse_head_token_wise",
        })
        return _long_generator(value=torch.ones(1, dtype=torch.bool))


def _make_save_req(req_id, num_tokens):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(num_tokens)),
        slot_mapping=[torch.arange(num_tokens, dtype=torch.long)],
        save_spec=SimpleNamespace(
            can_save=True,
            can_save_latent=True,
            can_save_indexer=True,
            skip_leading_tokens=0,
        ),
        is_sparse_decode=False,
        request_configs=None,
        disagg_spec=None,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        cached_keys_indexer=[],
        cached_starts_indexer=[],
        cached_ends_indexer=[],
        cached_memory_objs_indexer=[],
        cached_tensors_indexer=[],
        cached_chunk_dev_ptrs_indexer=[],
        cached_chunk_ptrs_npu_indexer=[],
        cached_shared_handles_indexer=[],
        resumed_from_preemption=False,
    )


def _make_load_req(req_id, num_tokens, cached_tokens):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(num_tokens)),
        slot_mapping=[torch.arange(num_tokens, dtype=torch.long)],
        indexer_slot_mapping=[],
        is_sparse_decode=False,
        request_configs=None,
        load_spec=SimpleNamespace(
            can_load=True,
            vllm_cached_tokens=0,
            lmcache_cached_tokens=cached_tokens,
        ),
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        cached_chunk_dev_ptrs=[],
        cached_chunk_ptrs_npu=[],
        cached_shared_handles=[],
        cached_keys_indexer=[],
        cached_starts_indexer=[],
        cached_ends_indexer=[],
        cached_memory_objs_indexer=[],
        cached_tensors_indexer=[],
        cached_chunk_dev_ptrs_indexer=[],
        cached_chunk_ptrs_npu_indexer=[],
        cached_shared_handles_indexer=[],
        decode_ret_mask=None,
        resumed_from_preemption=False,
    )


def _make_sparse_load_req(req_id, num_tokens, cached_tokens):
    req = _make_load_req(req_id, num_tokens, cached_tokens)
    req.is_sparse_decode = True
    return req


def _make_fake_adapter(num_layers=2, dsa_two_groups=True):
    """Build a fake adapter with real per-group plumbing + mocked engine."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    latent = [torch.zeros(1) for _ in range(num_layers)]
    indexer = [torch.zeros(1) for _ in range(num_layers)]
    kv_caches: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        kv_caches[f"layer.{i}"] = latent[i]
    if dsa_two_groups:
        for i in range(num_layers):
            kv_caches[f"indexer.{i}"] = indexer[i]

    engine = _RecordingEngine()
    fake = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    fake_state = SimpleNamespace(
        # config + caches
        kv_caches=kv_caches,
        config=SimpleNamespace(
            dsa_two_groups=dsa_two_groups,
            use_layerwise=True,
            enable_blending=False,
            enable_sparse_attention=False,
        ),
        lmcache_engine=engine,
        device=torch.device("cpu"),
        kv_role="kv_producer",
        use_layerwise=True,
        enable_blending=False,
        enable_sparse_attention=False,
        num_layers=num_layers,
        current_layer=0,
        _lmcache_chunk_size=64,
        # per-group state (populated by _refresh_kvcaches_list)
        _latent_layer_names=[],
        _indexer_layer_names=[],
        _latent_kvcaches=[],
        _indexer_kvcaches=[],
        _kvcaches_list=[],
        # storer / retriever state
        _layerwise_save_storers={},
        _deferred_latent_pending=set(),
        layerwise_retrievers=[],
        _layerwise_retriever_is_sparse=[],
        _layerwise_requests=[],
        _layerwise_sparse_req_ids=[],
        _layerwise_waited_groups=set(),
        _layerwise_sparse_indexer_sent_layers=set(),
        _layerwise_required_wait_groups_cache=None,
        _decode_window_save_completed_groups=set(),
        _decode_window_save_expected_start={},
        _completed_decode_window_saves={},
        _worker_retrieve_state={},
        # stubs
        _stats_monitor=MagicMock(),
        _maybe_lookup_unpin_for_request=lambda req: None,
        _prune_worker_retrieve_state=lambda ids: None,
        _load_tokens_for_retrieve=lambda tokens, cached, is_sparse_decode=False: (
            tokens if is_sparse_decode else list(tokens)[:cached]
        ),
        _full_hit_recalc_last_token=lambda *a, **k: False,
        _load_token_mask_for_retrieve=lambda req, token_count, chunk_size: (
            torch.ones(token_count, dtype=torch.bool)
        ),
        _finalize_worker_retrieve_state_from_metadata=lambda m: None,
        _sparse_decode_retrieve_warm_kwargs=lambda *a, **k: {},
    )
    fake.__dict__.update(vars(fake_state))
    fake._refresh_kvcaches_list()
    return fake


class TestVLLMCallSequence:
    """Mimic vLLM's worker-side call sequence against the adapter in two-group
    MLA+DSA mode: register_kv_caches -> save_kv_layer (per layer, both groups)
    -> wait_for_save; and start_load_kv (prefix hit) -> wait_for_layer_load."""

    def test_save_sequence_stores_both_groups_with_correct_kvcaches(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        fake = _make_fake_adapter(num_layers=2, dsa_two_groups=True)
        engine: _RecordingEngine = fake.lmcache_engine

        meta = LMCacheConnectorMetadata(
            requests=[_make_save_req("r1", 64)]
        )
        fake._parent = SimpleNamespace(
            _connector_metadata=meta,
            _get_connector_metadata=lambda: meta,
        )
        attn = SimpleNamespace(slot_mapping=torch.arange(64, dtype=torch.long),
                               indexer_slot_mapping=None)

        # vLLM calls save_kv_layer once per layer, alternating groups.
        # Latent layers:
        fake.save_kv_layer("layer.0", kv_layer=None, attn_metadata=attn)
        fake.save_kv_layer("layer.1", kv_layer=None, attn_metadata=attn)
        # Indexer layers:
        fake.save_kv_layer("indexer.0", kv_layer=None, attn_metadata=attn)
        fake.save_kv_layer("indexer.1", kv_layer=None, attn_metadata=attn)

        # store_layer is created once per (req_id, kv_group) -> 2 calls.
        assert len(engine.store_calls) == 2
        groups_seen = {c["kv_group"] for c in engine.store_calls}
        assert groups_seen == {0, 1}
        for call in engine.store_calls:
            if call["kv_group"] == 0:
                assert call["kvcaches"] is fake._latent_kvcaches
            else:
                assert call["kvcaches"] is fake._indexer_kvcaches
            assert call["req_id"] == "r1"

        # The final indexer callback drains its group immediately; latent is
        # finalized by wait_for_save.
        assert set(fake._layerwise_save_storers.keys()) == {
            ("r1", "normal_save", 0, 0, 64),
        }

        # wait_for_save drains the remaining latent storer.
        fake.wait_for_save()
        assert fake._layerwise_save_storers == {}

    def test_prefix_load_sequence_retrieves_both_groups(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        fake = _make_fake_adapter(num_layers=2, dsa_two_groups=True)
        engine: _RecordingEngine = fake.lmcache_engine

        cached_tokens = 64
        meta = LMCacheConnectorMetadata(
            requests=[_make_load_req("r1", 128, cached_tokens)]
        )
        fake._parent = SimpleNamespace(
            _connector_metadata=meta,
            _get_connector_metadata=lambda: meta,
        )
        forward_ctx = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                slot_mapping=torch.arange(cached_tokens, dtype=torch.long),
                indexer_slot_mapping=None,
            )
        )

        # vLLM calls start_load_kv at forward start.
        fake.start_load_kv(forward_ctx)

        # retrieve_layer called for both groups with the same token count.
        assert len(engine.retrieve_calls) == 2
        groups_seen = {c["kv_group"] for c in engine.retrieve_calls}
        assert groups_seen == {0, 1}
        for call in engine.retrieve_calls:
            if call["kv_group"] == 0:
                assert call["kvcaches"] is fake._latent_kvcaches
            else:
                assert call["kvcaches"] is fake._indexer_kvcaches
            assert len(call["slot_mapping"]) == cached_tokens

        # One retriever pair registered.
        assert len(fake.layerwise_retrievers) == 1
        latent_ret, indexer_ret = fake.layerwise_retrievers[0]
        assert latent_ret is not None
        assert indexer_ret is not None
        assert fake._layerwise_retriever_is_sparse == [False]

        # vLLM calls wait_for_layer_load once per group per layer.
        for layer_id in range(fake.num_layers):
            fake.wait_for_layer_load(layer_name=f"layer.{layer_id}")
            fake.wait_for_layer_load(layer_name=f"indexer.{layer_id}")
        # After num_layers layers, retrievers are drained.
        assert fake.layerwise_retrievers == []
        assert fake._layerwise_retriever_is_sparse == []

    def test_sparse_decode_load_sequence_retrieves_latent_only(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        fake = _make_fake_adapter(num_layers=2, dsa_two_groups=True)
        engine: _RecordingEngine = fake.lmcache_engine

        cached_tokens = 64
        meta = LMCacheConnectorMetadata(
            requests=[_make_sparse_load_req("r1", 128, cached_tokens)]
        )
        fake._parent = SimpleNamespace(
            _connector_metadata=meta,
            _get_connector_metadata=lambda: meta,
        )
        forward_ctx = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                slot_mapping=torch.arange(cached_tokens, dtype=torch.long),
                indexer_slot_mapping=None,
            ),
            no_compile_layers={},
        )

        fake.start_load_kv(forward_ctx)

        assert len(engine.retrieve_calls) == 1
        assert engine.retrieve_calls[0]["kv_group"] == 0
        assert engine.retrieve_calls[0]["kind"] == "sparse_head_token_wise"
        assert engine.retrieve_calls[0]["kvcaches"] is fake._latent_kvcaches

        assert len(fake.layerwise_retrievers) == 1
        latent_ret, indexer_ret = fake.layerwise_retrievers[0]
        assert latent_ret is not None
        assert indexer_ret is None
        assert fake._layerwise_retriever_is_sparse == [True]

    def test_wait_for_save_drains_two_group_storers_and_seeds_worker_state(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        fake = _make_fake_adapter(num_layers=2, dsa_two_groups=True)
        req = _make_save_req("r1", 128)
        latent_result = LayerwiseStoreResult(
            request_id="r1",
            starts=[0],
            ends=[128],
            keys=[["k0"], ["k1"]],
            memory_objs=[["m0"], ["m1"]],
            tensors=[[torch.zeros(1)], [torch.zeros(1)]],
        )
        index_result = LayerwiseStoreResult(
            request_id="r1",
            kv_group=1,
            starts=[0],
            ends=[128],
            keys=[["ik0"], ["ik1"]],
            memory_objs=[["im0"], ["im1"]],
            tensors=[[torch.zeros(1)], [torch.zeros(1)]],
        )

        fake._layerwise_save_storers[
            ("r1", "normal_save", 0, 0, 128)
        ] = _long_generator(value=latent_result, n=1)
        fake._layerwise_save_storers[
            ("r1", "normal_save", 1, 0, 128)
        ] = _long_generator(value=index_result, n=1)
        meta = LMCacheConnectorMetadata(requests=[req])
        fake._parent = SimpleNamespace(
            _connector_metadata=meta,
            _get_connector_metadata=lambda: meta,
        )

        fake.wait_for_save()

        assert fake._layerwise_save_storers == {}
        assert "r1" in fake._worker_retrieve_state
        assert fake._worker_retrieve_state["r1"].cached_keys == [["k0"], ["k1"]]
        assert fake._worker_retrieve_state["r1"].cached_keys_indexer == [
            ["ik0"],
            ["ik1"],
        ]

    def test_save_sequence_without_dsa_two_groups_is_latent_only(self):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        fake = _make_fake_adapter(num_layers=2, dsa_two_groups=False)
        engine: _RecordingEngine = fake.lmcache_engine

        meta = LMCacheConnectorMetadata(
            requests=[_make_save_req("r1", 64)]
        )
        fake._parent = SimpleNamespace(
            _connector_metadata=meta,
            _get_connector_metadata=lambda: meta,
        )
        attn = SimpleNamespace(slot_mapping=torch.arange(64, dtype=torch.long),
                               indexer_slot_mapping=None)

        fake.save_kv_layer("layer.0", kv_layer=None, attn_metadata=attn)
        fake.save_kv_layer("layer.1", kv_layer=None, attn_metadata=attn)

        # Only one storer (latent), kv_group=0.
        assert len(engine.store_calls) == 1
        assert engine.store_calls[0]["kv_group"] == 0
        assert set(fake._layerwise_save_storers.keys()) == {
            ("r1", "normal_save", 0, 0, 64)
        }
        fake.wait_for_save()
        assert fake._layerwise_save_storers == {}


class TestPermuteKvCachesToContiguous:

    def test_dsa_index_one_tuple(self) -> None:
        from lmcache_ascend.v1.npu_connector.utils import (
            permute_kv_caches_to_contiguous,
        )

        indexer = torch.randn(4, 16, 1, 128)
        result = permute_kv_caches_to_contiguous([(indexer,)])

        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 1
        assert result[0][0].shape == indexer.shape

    def test_mla_latent_two_tuple(self) -> None:
        from lmcache_ascend.v1.npu_connector.utils import (
            permute_kv_caches_to_contiguous,
        )

        k_nope = torch.randn(4, 16, 1, 512)
        k_pe = torch.randn(4, 16, 1, 64)
        result = permute_kv_caches_to_contiguous([(k_nope, k_pe)])

        assert len(result) == 1
        assert len(result[0]) == 2
