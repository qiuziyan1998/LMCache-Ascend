# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prepare_layerwise_prefill_group0_skip."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import unittest

# Third Party
import lmcache.v1.memory_management as _mm
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


class TestPrepareLayerwisePrefillGroup0Skip(unittest.TestCase):
    def test_prepare_layerwise_prefill_group0_skip_avoids_staging(self) -> None:
        conn = VLLMPagedMemLayerwiseNPUConnector.__new__(
            VLLMPagedMemLayerwiseNPUConnector
        )
        conn.kvcaches = {"group0": ["k0"], "group1": ["k1"]}  # type: ignore[assignment]
        conn._layerwise_prefill_dma = True

        init_ptr_calls: list[dict] = []
        lazy_init_calls: list[tuple[object, int, bool]] = []
        bank_counts: dict[int, int] = {1: 4}
        group1_staging = object()
        staging_by_group: dict[int, object] = {1: group1_staging}

        conn.initialize_kvcaches_ptr = (  # type: ignore[method-assign]
            lambda **kw: init_ptr_calls.append(kw)
        )

        def _fake_lazy_init(
            kvcaches: object, *, kv_group: int, init_staging: bool
        ) -> SimpleNamespace:
            lazy_init_calls.append((kvcaches, kv_group, init_staging))
            if init_staging:
                staging_by_group[kv_group] = object()
            return SimpleNamespace(kv_group=kv_group, init_staging=init_staging)

        conn._lazy_initialize_buffer_with_staging = _fake_lazy_init  # type: ignore[method-assign]
        conn._layerwise_prefill_transfer_state = lambda: (  # type: ignore[method-assign]
            bank_counts,
            {},
            {},
            {},
        )

        conn.prepare_layerwise_prefill_group0_skip(
            layerwise_prefill_bank_count=3,
            marker="forwarded",
        )

        self.assertEqual(len(init_ptr_calls), 1)
        self.assertEqual(init_ptr_calls[0]["layerwise_prefill_bank_count"], 3)
        self.assertEqual(lazy_init_calls, [(conn.kvcaches, 0, False)])
        self.assertNotIn(0, staging_by_group)
        self.assertIs(staging_by_group[1], group1_staging)
        self.assertEqual(bank_counts[0], 3)
        self.assertEqual(bank_counts[1], 4)

        # Fallback bank count = 2 when _layerwise_prefill_dma is False
        conn._layerwise_prefill_dma = False
        bank_counts.pop(0)
        snap = {"custom": ["kv"]}
        conn.prepare_layerwise_prefill_group0_skip(
            kvcaches=snap,
            layerwise_prefill_bank_count=4,
        )
        self.assertEqual(lazy_init_calls[-1], (snap, 0, False))
        self.assertEqual(bank_counts[0], 2)
        self.assertNotIn(0, staging_by_group)

    def test_ascend_engine_close_skips_page_release_with_unsafe_sources(
        self,
    ) -> None:
        released: list[str] = []
        eng: Any = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        eng._checkpoint_quarantined_owners = None
        eng.checkpoint_worker = None
        eng._failed_sparse_loads = None
        eng._remote_fill_fatal_transfers = set()
        eng._remote_fill_runtime = None
        eng._group1_external_page_reader = None
        eng.close_remote_fill_producer = lambda: None
        eng.wait_for_direct_stores = lambda _s: None
        eng._direct_store_states = {}
        eng._live_source_builders = {}
        eng._completed_live_sources = {}
        eng._store_queue = None
        eng._force_layerwise_prefill_store = True
        eng._retain_unfenced_external_dma = False
        eng._layerwise_prefill_page_owners = {"req-unsafe": {0: []}}
        eng.release_layerwise_prefill_pages = lambda rid: released.append(rid)

        with patch("lmcache.v1.cache_engine.LMCacheEngine.close", lambda _s: None):
            eng._unsafe_layerwise_retrieve_sources = [object()]
            eng.close()
            self.assertEqual(released, [])

            eng._unsafe_layerwise_retrieve_sources = []
            eng.close()
            self.assertEqual(released, ["req-unsafe"])

    def _build_omit_repair_harness(
        self,
        *,
        req_id: str,
        total_tokens: int,
        omit_skip: int,
        hbm_caches: list[torch.Tensor],
    ) -> dict[str, Any]:
        # Standard
        from concurrent.futures import Future
        import threading

        # Third Party
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
            WorkerRetrieveState,
        )
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache.v1.prefill_metadata import PrefillMetadataCache
        from lmcache.v1.token_database import ChunkedTokenDatabase

        cs, nl = 256, 2
        cfg = LMCacheEngineConfig(
            chunk_size=cs,
            local_cpu=True,
            max_local_cpu_size=0.1,
            use_layerwise=True,
            save_unfull_chunk=True,
            dsa_two_groups=True,
            remote_url="mooncakestore://test",
            extra_config={
                "enable_shared_cpu_cache": True,
                "save_only_first_rank": True,
                "mooncake_page_first_multi_buffer": True,
                "mooncake_layer_merged_page_objects": True,
            },
        )
        g0 = KVLayerGroupInfo(
            ["latent_0", "latent_1"],
            [0, 1],
            torch.Size([2, 2, cs, 8]),
            torch.float32,  # type: ignore[arg-type]
        )
        g1 = KVLayerGroupInfo(
            ["indexer_0"],
            [2],
            torch.Size([2, 1, cs, 4]),
            torch.float32,  # type: ignore[arg-type]
        )
        meta = LMCacheMetadata(
            model_name="ascend-test-model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            chunk_size=cs,
            kv_dtype=torch.float32,
            kv_shape=(3, 1, cs, 1, 8),
            use_mla=True,
            engine_id="eng0",
            kv_layer_groups_manager=KVLayerGroupsManager([g0, g1]),
            runtime_kv_group_layer_counts=(2, 1),
            kv_connector_extra_config={"lmcache_rpc_port": 29111},
            runtime_kv_group_layer_names=(("latent_0", "latent_1"), ("indexer_0",)),
        )
        db = ChunkedTokenDatabase(cfg, meta)
        tokens = list(range(5000, 5000 + total_tokens))
        raw_chunks = list(db.process_tokens(tokens=tokens, make_key=True, kv_group=0))
        base_keys: list[CacheEngineKey] = []
        for _, _, k in raw_chunks:
            assert isinstance(k, CacheEngineKey)
            base_keys.append(k)

        layer_shape = torch.Size([1, 1, cs, 8])
        layer_bytes = cs * 8 * 4
        prefix_chunks = omit_skip // cs
        legacy_buf = torch.zeros(prefix_chunks * nl * layer_bytes, dtype=torch.uint8)
        legacy_alloc = _mm.PagedTensorMemoryAllocator(
            legacy_buf, [layer_shape], [torch.float32], _mm.MemoryFormat.KV_MLA_FMT
        )
        alloc_pages: list[_mm.LayerPageMemoryObj] = []
        legacy_objs: dict[CacheEngineKey, _mm.TensorMemoryObj] = {}

        def _cleanup_mem_objs() -> None:
            for obj in legacy_objs.values():
                obj.invalidate()
            for pg in alloc_pages:
                pg.invalidate()
            for blk in list(legacy_alloc.free_blocks):
                blk.invalidate()

        self.addCleanup(_cleanup_mem_objs)
        for bk in base_keys[:prefix_chunks]:
            for lkey in bk.split_layers(nl):
                obj = legacy_alloc.allocate(
                    [layer_shape], [torch.float32], _mm.MemoryFormat.KV_MLA_FMT
                )
                assert isinstance(obj, _mm.TensorMemoryObj)
                assert obj.tensor is not None
                obj.tensor.fill_(3.25)
                legacy_objs[lkey] = obj

        local_get_calls: list[CacheEngineKey] = []
        page_data: dict[CacheEngineKey, torch.Tensor] = {}
        remote_data: dict[CacheEngineKey, torch.Tensor] = {}

        def _done_future() -> Future[None]:
            fut: Future[None] = Future()
            fut.set_result(None)
            return fut

        def _alloc_layer_pages(
            _shapes: Any,
            _dtypes: Any,
            count: int,
            num_layers: int,
            fmt: Any,
            **kw: Any,
        ) -> list[_mm.LayerPageMemoryObj]:
            self.assertEqual(
                (count, num_layers, fmt), (1, nl, _mm.MemoryFormat.KV_MLA_FMT)
            )
            p_buf = torch.zeros(nl * layer_bytes, dtype=torch.uint8)
            p_meta = _mm.MemoryObjMetadata(
                shape=layer_shape,
                dtype=torch.float32,
                address=0,
                phy_size=layer_bytes,
                ref_count=1,
                pin_count=0,
                fmt=_mm.MemoryFormat.KV_MLA_FMT,
                shapes=[layer_shape] * nl,
                dtypes=[torch.float32] * nl,
            )
            pg = _mm.LayerPageMemoryObj(
                p_buf,
                p_meta,
                None,
                num_layers=nl,
                group_prefix_sum=(0, layer_bytes, 2 * layer_bytes),
                valid_tokens=kw.get("valid_tokens", [cs])[0],
            )
            alloc_pages.append(pg)
            return [pg]

        def _get_blocking(lkey: CacheEngineKey) -> Any:
            local_get_calls.append(lkey)
            obj = legacy_objs.get(lkey)
            if obj is not None:
                obj.ref_count_up()
            return obj

        local_backend = SimpleNamespace(
            batched_get_layer_page_prefix=lambda _ks: ([], []),
            get_blocking=_get_blocking,
            batched_allocate_layer_pages=_alloc_layer_pages,
        )

        def _remote_contains(flat_keys: list[CacheEngineKey]) -> int:
            hit = 0
            for k in flat_keys:
                if k not in remote_data:
                    break
                hit += 1
            return hit

        def _remote_submit(
            layer_keys: list[CacheEngineKey], owned: list[Any]
        ) -> list[Any]:
            for k, obj in zip(layer_keys, owned, strict=True):
                assert obj.tensor is not None
                remote_data[k] = obj.tensor.detach().clone()
            return [_done_future()]

        remote_backend = SimpleNamespace(
            batched_contains=_remote_contains,
            batched_submit_put_task=_remote_submit,
        )

        def _put_layer_pages(
            chunk_keys: list[CacheEngineKey], layer_pages: list[Any], **_kw: Any
        ) -> list[Any]:
            for k, pg in zip(chunk_keys, layer_pages, strict=True):
                page_data[k] = pg.raw_data.detach().clone()
            return [_done_future()]

        sm: Any = SimpleNamespace(
            local_cpu_backend=local_backend,
            storage_backends={
                "LocalCPUBackend": local_backend,
                "RemoteBackend": remote_backend,
            },
            supports_batched_put_layer_pages=lambda _loc: True,
            batched_contains_layer_pages=lambda ks, *_a, **_k: (
                sum(1 for k in ks if k.without_layer() in page_data),
                {k: "RemoteBackend" for k in ks if k.without_layer() in page_data},
            ),
            batched_contains=lambda keys, **_k: sum(
                1 for k in keys if k in remote_data
            ),
            batched_put_layer_pages=_put_layer_pages,
            batched_external_pages_exist=lambda ks: [k in page_data for k in ks],
        )

        dma_ranges: list[tuple[list[int], list[int]]] = []

        def _batched_from_gpu(
            memory_objs: list[list[Any]],
            starts: list[int],
            ends: list[int],
            **kw: Any,
        ) -> Any:
            dma_ranges.append((list(starts), list(ends)))
            slot_mapping = kw["slot_mapping"]
            yield None
            for lid in range(nl):
                s0, e0 = starts[0], ends[0]
                src = hbm_caches[lid][slot_mapping[s0:e0]].reshape(-1)
                self.assertFalse(bool((src == -999.0).any()))
                pg = memory_objs[lid][0]
                pg.raw_data[lid * layer_bytes : (lid + 1) * layer_bytes].view(
                    torch.float32
                ).copy_(src)
                yield lid

        gpu_conn: Any = VLLMPagedMemLayerwiseNPUConnector.__new__(
            VLLMPagedMemLayerwiseNPUConnector
        )
        gpu_conn.get_shape = lambda nt, kv_group=0: torch.Size([1, 1, nt, 8])  # type: ignore[method-assign]
        gpu_conn.batched_from_gpu = _batched_from_gpu  # type: ignore[method-assign]
        gpu_conn.release_sparse_chunk_ptr_cache = lambda *_a: None  # type: ignore[method-assign]

        eng: Any = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        eng.config = cfg
        eng.metadata = meta
        eng.token_database = db
        eng.storage_manager = sm
        eng.gpu_connector = gpu_conn
        eng.lmcache_worker = None
        eng.num_layers = nl
        eng.retrieve_locations = ("LocalCPUBackend", "RemoteBackend")
        eng.store_location = "LocalCPUBackend"
        eng._force_layerwise_prefill_store = True
        eng._layerwise_prefill_store_frontiers = {}
        eng._layerwise_prefill_store_unverified_ends = {}
        eng._layerwise_prefill_page_owners = {}
        eng._direct_store_states = {}
        eng._direct_retry_args = {}
        eng._engine_state_lock = threading.RLock()
        eng._layerwise_put_queue = None
        eng._store_queue_maxsize = 2
        eng.kv_events_enabled = False
        eng._init_failed = False
        eng._health_monitor = None
        eng._is_passive = lambda: False
        eng.is_healthy = lambda: True
        eng.is_frozen = lambda: False
        eng._get_req_id = lambda kw: str(kw["req_id"])
        eng._log_kvcache_for_check = lambda *_a, **_k: None
        eng._remote_fill_prepare_request = lambda _r, _c, st: setattr(
            st,
            "remote_fill",
            SimpleNamespace(
                handoff=object(), last_future=None, session=None, disabled_reason=None
            ),
        )
        eng._finish_remote_fill = lambda *_a, **_k: None
        eng._remote_fill_direct_groups = lambda: ()
        eng.config.enable_remote_lmcache_store = True
        eng.stats_monitor = SimpleNamespace(
            on_store_request=lambda *_a, **_k: 1,
            on_store_finished=lambda *_a, **_k: None,
        )
        eng._num_transfer_layers_for_call = lambda _g, _k: nl
        eng._num_layers_for_kv_group = lambda g: nl if g == 0 else 1
        eng._ensure_layerwise_connector_layout = lambda **_k: None
        eng._shared_cpu_dtype_for_kv_group = lambda _g: torch.float32
        eng._memory_format_for_kv_group = lambda _g: _mm.MemoryFormat.KV_MLA_FMT
        eng._layerwise_prefill_async_store_supported = lambda **_k: True
        eng.submit_layerwise_prefill_fills = lambda *_a, **_k: None
        eng.poll_layerwise_prefill_puts = lambda *_a, **_k: None
        eng.wait_for_pending_sync_stores = lambda: None
        eng._append_layer_store_tensors = lambda *_a, **_k: None
        eng._append_layerwise_store_cache_chunks = lambda *_a, **_k: None
        eng._remove_pending_layerwise_store_objs = lambda *_a, **_k: None
        eng._track_sync_store_futures = lambda *_a, **_k: None
        self.addCleanup(lambda: setattr(eng, "_layerwise_put_queue", None))

        impl: Any = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        impl.kv_role = "kv_producer"
        impl.device = torch.device("cpu")
        impl._lmcache_chunk_size = cs
        impl.enable_sparse_attention = True
        impl._force_layerwise_prefill_store = True
        impl.lmcache_engine = eng
        impl._is_dsa_two_groups = lambda: True
        impl._is_decode_window_save_request = lambda r: bool(
            getattr(r, "is_decode_window_save", False)
        )
        impl._is_recompute_fallback_request = lambda _rid: False
        impl._is_fallback_req_id = lambda _rid: False
        impl._prefill_metadata_cache_enabled = lambda *_a, **_k: True
        impl._prefill_metadata_scope_changed = lambda *_a, **_k: False
        impl._worker_retrieve_state = {}
        impl._set_worker_retrieve_state = lambda rid, st: (
            impl._worker_retrieve_state.__setitem__(rid, st)
        )

        def _pmc_for_req(_r: Any, st: Any) -> PrefillMetadataCache:
            if st.prefill_metadata_cache is None:
                st.prefill_metadata_cache = PrefillMetadataCache()
            return st.prefill_metadata_cache

        impl._prefill_metadata_cache_for_request = _pmc_for_req
        impl._windowed_sparse_save_mapping = lambda *_a, **_k: None
        impl._layerwise_store_kwargs = lambda r, g: (
            LMCacheConnectorV1Impl._layerwise_store_kwargs(impl, r, g)
        )
        impl._selected_group0_omit_skip_tokens = lambda r: (
            LMCacheConnectorV1Impl._selected_group0_omit_skip_tokens(impl, r)
        )
        impl._selected_group0_omit_end_tokens = {req_id: omit_skip}
        impl._worker_retrieve_state[req_id] = WorkerRetrieveState(
            req_id=req_id, prefill_metadata_cache=PrefillMetadataCache()
        )
        impl._completed_layerwise_stores = {}
        impl._layerwise_local_store_results = {}
        meta_box: list[Any] = [SimpleNamespace(requests=[])]
        impl._parent = SimpleNamespace(_get_connector_metadata=lambda: meta_box[0])

        return {
            "nl": nl,
            "tokens": tokens,
            "base_keys": base_keys,
            "local_get_calls": local_get_calls,
            "page_data": page_data,
            "remote_data": remote_data,
            "dma_ranges": dma_ranges,
            "eng": eng,
            "impl": impl,
            "meta_box": meta_box,
        }

    def test_kv_producer_selected_group0_omit_repairs_legacy_prefix_from_cpu_without_npu_d2h(  # noqa: E501
        self,
    ) -> None:
        # Third Party
        from lmcache.v1.cache_engine import LayerwiseStoreResult

        hbm_caches = [torch.empty((768, 8), dtype=torch.float32) for _ in range(2)]
        for hc in hbm_caches:
            hc[:512].fill_(-999.0)
            hc[512:].fill_(7.5)
        h = self._build_omit_repair_harness(
            req_id="req-p-omit", total_tokens=768, omit_skip=512, hbm_caches=hbm_caches
        )
        nl, tokens = h["nl"], h["tokens"]
        eng, impl = h["eng"], h["impl"]

        req: Any = SimpleNamespace(
            req_id="req-p-omit",
            token_ids=tokens,
            slot_mapping=[torch.arange(768, dtype=torch.long)],
            is_sparse_decode=False,
            is_decode_window_save=False,
            request_configs=None,
            decode_window_start=None,
            decode_window_end=None,
            decode_window_size=None,
        )
        prep = impl._prepare_layerwise_store_inputs(req, None, 0)
        self.assertIsNotNone(prep)
        assert prep is not None
        toks_out, smap_out, smask_out, skip_out, skw_out, _ = prep
        self.assertEqual(skip_out, 512)
        self.assertFalse(bool(smask_out[:512].any()))
        self.assertTrue(bool(smask_out[512:].all()))
        self.assertEqual(skw_out["_unverified_prefix_skip_tokens"], 512)

        gen = eng.store_layer(
            toks_out,
            mask=smask_out,
            req_id="req-p-omit",
            kv_group=0,
            slot_mapping=smap_out,
            deferred_layerwise_put=True,
            layerwise_prefill_incremental=True,
            **skw_out,
        )
        next(gen)
        for _ in range(nl):
            next(gen)
            next(gen)
        store_res = next(gen)
        self.assertEqual(store_res.unverified_prefix_end, 512)
        self.assertEqual(store_res.committed_end, 768)

        eng.adopt_completed_layerwise_store(store_res)
        state = eng._direct_store_states["req-p-omit"]
        self.assertEqual(state.committed_end.get(0, 0), 0)
        eng.adopt_completed_layerwise_store(
            LayerwiseStoreResult(request_id="req-p-omit", kv_group=1, committed_end=768)
        )
        eng.finish_layerwise_prefill_store(
            "req-p-omit",
            None,
            required_store_end=768,
            persistence_fenced=True,
            tokens=toks_out,
        )

        self.assertEqual(h["dma_ranges"], [([512], [768])])
        self.assertEqual(state.committed_end[0], 768)
        base_keys = h["base_keys"]
        self.assertIn(base_keys[2], h["page_data"])
        pg_floats = h["page_data"][base_keys[2]].view(torch.float32)
        self.assertTrue(bool(torch.all(pg_floats == 7.5)))
        self.assertFalse(bool((pg_floats == -999.0).any()))

        expected_legacy_keys = base_keys[0].split_layers(nl) + base_keys[
            1
        ].split_layers(nl)
        self.assertEqual(set(h["local_get_calls"]), set(expected_legacy_keys))
        self.assertEqual(set(h["remote_data"]), set(expected_legacy_keys))
        for lk in expected_legacy_keys:
            self.assertTrue(bool(torch.all(h["remote_data"][lk] == 3.25)))
            self.assertFalse(bool((h["remote_data"][lk] == -999.0).any()))

    def test_kv_producer_chunked_prefill_preserves_unverified_prefix_across_steps(
        self,
    ) -> None:
        # Third Party
        from lmcache.v1.cache_engine import LayerwiseStoreResult

        req_id, total_tokens, omit_skip = "req-chunk-unv", 1024, 512
        hbm_caches = [
            torch.empty((total_tokens, 8), dtype=torch.float32) for _ in range(2)
        ]
        for hc in hbm_caches:
            hc[:512].fill_(-999.0)
            hc[512:768].fill_(7.5)
            hc[768:].fill_(8.25)

        h = self._build_omit_repair_harness(
            req_id=req_id,
            total_tokens=total_tokens,
            omit_skip=omit_skip,
            hbm_caches=hbm_caches,
        )
        nl, tokens, eng, impl = h["nl"], h["tokens"], h["eng"], h["impl"]
        base_keys = h["base_keys"]

        # Step 1 (is_last_prefill=False): Selected-KV active, saves [512, 768)
        req_s1: Any = SimpleNamespace(
            req_id=req_id,
            token_ids=tokens[:768],
            slot_mapping=[torch.arange(768, dtype=torch.long)],
            is_sparse_decode=False,
            is_decode_window_save=False,
            is_last_prefill=False,
            request_configs=None,
            decode_window_start=None,
            decode_window_end=None,
            decode_window_size=None,
        )
        prep1 = impl._prepare_layerwise_store_inputs(req_s1, None, 0)
        assert prep1 is not None
        t1, s1, m1, skip1, kw1, _ = prep1
        self.assertEqual((skip1, kw1["_unverified_prefix_skip_tokens"]), (512, 512))
        gen1 = eng.store_layer(
            t1,
            mask=m1,
            req_id=req_id,
            kv_group=0,
            slot_mapping=s1,
            deferred_layerwise_put=True,
            layerwise_prefill_incremental=True,
            **kw1,
        )
        next(gen1)
        for _ in range(nl):
            next(gen1)
            next(gen1)
        res1 = next(gen1)
        impl._completed_layerwise_stores[(req_id, 0)] = res1
        h["meta_box"][0] = SimpleNamespace(requests=[req_s1])
        impl._finish_layerwise_prefill_batch()
        self.assertEqual(
            impl._layerwise_local_store_results[(req_id, 0)].unverified_prefix_end, 512
        )
        self.assertEqual(eng._layerwise_prefill_store_frontiers[req_id][0][0], 768)
        self.assertEqual(eng._layerwise_prefill_store_unverified_ends[req_id][0], 512)

        # Synthetic unverified_prefix_end=0 merge must retain prior 512 upper bound
        impl._completed_layerwise_stores[(req_id, 0)] = LayerwiseStoreResult(
            request_id=req_id, kv_group=0, committed_end=768, unverified_prefix_end=0
        )
        impl._finish_layerwise_prefill_batch()
        self.assertEqual(
            impl._layerwise_local_store_results[(req_id, 0)].unverified_prefix_end, 512
        )

        # Step 2 (is_last_prefill=True): falls back to selected=False (no omit skip)
        impl._selected_group0_omit_end_tokens = {}
        req_s2: Any = SimpleNamespace(
            req_id=req_id,
            token_ids=tokens,
            slot_mapping=[torch.arange(total_tokens, dtype=torch.long)],
            is_sparse_decode=False,
            is_decode_window_save=False,
            is_last_prefill=True,
            request_configs=None,
            decode_window_start=None,
            decode_window_end=None,
            decode_window_size=None,
        )
        prep2 = impl._prepare_layerwise_store_inputs(req_s2, None, 0)
        assert prep2 is not None
        t2, s2, m2, skip2, kw2, _ = prep2
        self.assertEqual(skip2, 0)
        self.assertNotIn("_unverified_prefix_skip_tokens", kw2)
        gen2 = eng.store_layer(
            t2,
            mask=m2,
            req_id=req_id,
            kv_group=0,
            slot_mapping=s2,
            deferred_layerwise_put=True,
            layerwise_prefill_incremental=True,
            **kw2,
        )
        next(gen2)
        for _ in range(nl):
            next(gen2)
            next(gen2)
        res2 = next(gen2)
        self.assertEqual(res2.unverified_prefix_end, 512)
        self.assertEqual(res2.committed_end, 1024)

        impl._completed_layerwise_stores[(req_id, 0)] = res2
        impl._completed_layerwise_stores[(req_id, 1)] = LayerwiseStoreResult(
            request_id=req_id, kv_group=1, committed_end=total_tokens
        )
        h["meta_box"][0] = SimpleNamespace(requests=[req_s2])
        impl._finish_layerwise_prefill_batch()

        self.assertEqual(h["dma_ranges"], [([512], [768]), ([768], [1024])])
        self.assertTrue(
            bool(torch.all(h["page_data"][base_keys[2]].view(torch.float32) == 7.5))
        )
        self.assertTrue(
            bool(torch.all(h["page_data"][base_keys[3]].view(torch.float32) == 8.25))
        )

        expected_legacy_keys = base_keys[0].split_layers(nl) + base_keys[
            1
        ].split_layers(nl)
        self.assertEqual(set(h["local_get_calls"]), set(expected_legacy_keys))
        self.assertEqual(set(h["remote_data"]), set(expected_legacy_keys))
        for lk in expected_legacy_keys:
            self.assertTrue(bool(torch.all(h["remote_data"][lk] == 3.25)))
            self.assertFalse(bool((h["remote_data"][lk] == -999.0).any()))

        state = eng._direct_store_states[req_id]
        self.assertEqual(state.committed_end[0], total_tokens)
        self.assertEqual(state.unverified_prefix_end, {})
        self.assertEqual(
            eng._layerwise_prefill_store_unverified_ends.get(req_id, {}), {}
        )


if __name__ == "__main__":
    unittest.main()
