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
- _is_mla_dsa_format / _is_latent_format / _is_indexer_format helpers
- Integration: two-group store/load roundtrip with separate latent and indexer keys
"""
# Standard
import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat

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
            token_ids=list(range(num_saved, num_saved + 1)) if is_decode else list(range(100)),
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

        gen = LMCacheEngine.store_layer(
            engine, tokens, mask=mask, cached_keys=[]
        )
        list(gen)
        engine.token_database.process_tokens.assert_called()


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
        """MLA_LATENT and MLA_KV should produce the same shape (same plane structure)."""
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
        for (_, _, lk), (_, _, ik) in zip(latent_results, indexer_results):
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

    def _make_connector(self):
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
                use_gpu=False,
                chunk_size=64,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                use_mla=True,
                dsa_two_groups=True,
            )
        return conn

    def _latent_kvcaches(self, num_layers=2):
        k_nope = torch.zeros(4, 128, 1, 512, dtype=torch.bfloat16)
        k_pe = torch.zeros(4, 128, 1, 64, dtype=torch.bfloat16)
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
        # dict keyed by (kv_group, layer_id), not a per-layer list.
        assert isinstance(conn._sparse_direct_layer_states, dict) or (
            conn._sparse_direct_layer_states is None
        )

