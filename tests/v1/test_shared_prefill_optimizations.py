# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU unit tests for LMCache-Ascend shared prefill optimizations."""

# Standard
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import ast
import sys
import threading
import types
import unittest
import warnings

# Third Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import ChunkedTokenDatabase
import lmcache.v1.memory_management as _mm
import numpy as np
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine, _DirectStoreRequestState
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.layerwise_cpu_fill import LayerwiseCPUFillLease
from lmcache_ascend.v1.npu_connector.layerwise_dma import (
    DmaPlan,
    bind_incremental_copy_addresses,
    prepare_source_addresses,
    ptrs_to_int64_tensor,
)
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


def _opt(mod: str, *syms: str) -> Any:
    try:
        m = __import__(mod, fromlist=list(syms))
        return getattr(m, syms[0]) if len(syms) == 1 else [getattr(m, s) for s in syms]
    except (ImportError, AttributeError):
        return None if len(syms) == 1 else [None] * len(syms)


_MM = "lmcache.v1.memory_management"
_MOM, _TMO = _mm.MemoryObjMetadata, _mm.TensorMemoryObj
PagedTensorMemoryAllocator = _mm.PagedTensorMemoryAllocator
PinMemoryAllocator = _mm.PinMemoryAllocator
_LPM = getattr(_mm, "LayerPageMemoryObj", None)
_LPS = getattr(_mm, "LayerPageSource", None)
MixedMemoryAllocator = getattr(_mm, "MixedMemoryAllocator", None)
_V1A, _PM = "lmcache.integration.vllm.vllm_v1_adapter", "lmcache.v1.prefill_metadata"
_OmittedPrefixTokens = _opt(_V1A, "_OmittedPrefixTokens")
_split_cache_engine_key_layers = _opt(_PM, "_split_cache_engine_key_layers")
SharedHandleBatch, SharedSlabMapping = _opt(
    "lmcache.v1.shared_cpu_cache", "SharedHandleBatch", "SharedSlabMapping"
)

_N, _SZ, _U8, _BF = SimpleNamespace, torch.Size, torch.uint8, torch.bfloat16
_FMT, _F2D = _mm.MemoryFormat.KV_MLA_FMT, _mm.MemoryFormat.KV_2TD
_SH4 = _SZ([1, 2, 4, 4])
_NPU_OPS = "lmcache_ascend.v1.npu_connector.npu_connectors.lmc_ops"
_C8_FAIL_PTN = "exceeds logical block table"


def _rec(bucket: list[Any], item: Any, ret: Any) -> Any:
    bucket.append(item)
    return ret


def _npu_stream(name: str = "s") -> Any:
    return _N(name=name, wait_stream=lambda _o: None, wait_event=lambda _e: None)


class _StubEvent:
    def record(self, _stream: Any) -> None:
        pass


def _cfg(cs: int = 128) -> LMCacheEngineConfig:
    kw: dict[str, Any] = dict(chunk_size=cs, local_cpu=True, max_local_cpu_size=0.1)
    kw.update(use_layerwise=True, enable_blending=False, save_unfull_chunk=True)
    return LMCacheEngineConfig(**kw, dsa_two_groups=True)


def _meta(cs: int = 128) -> LMCacheMetadata:
    g0 = KVLayerGroupInfo(["latent_0", "latent_1"], [0, 1], _SZ([2, 2, cs, 64]), _BF)
    g1 = KVLayerGroupInfo(["indexer_0"], [2], _SZ([1, 1, cs, 32]), _BF)
    mgr = KVLayerGroupsManager(kv_layer_groups=[g0, g1])
    kw: dict[str, Any] = dict(model_name="ascend-test-model", world_size=1)
    kw.update(local_world_size=1, worker_id=0, local_worker_id=0, chunk_size=cs)
    kw.update(kv_dtype=_BF, kv_shape=(3, 2, cs, 1, 64), use_mla=True, engine_id="eng0")
    kw.update(kv_layer_groups_manager=mgr, runtime_kv_group_layer_counts=(2, 1))
    kw["kv_connector_extra_config"] = {"lmcache_rpc_port": 29111}
    kw["runtime_kv_group_layer_names"] = (("latent_0", "latent_1"), ("indexer_0",))
    return LMCacheMetadata(**kw)


def _conn(dev: str | None = "npu:0") -> Any:
    c_cls = VLLMPagedMemLayerwiseNPUConnector
    c: Any = c_cls.__new__(c_cls)
    d = c.__dict__
    if dev is not None:
        d["kv_device"] = dev
    d.update(num_layers=2, enable_npu_transfer_validation=True, _current_kv_group=0)
    d.update(kv_format=KVCacheFormat.MLA_LATENT, k_hidden_dims=8, v_hidden_dims=8)
    d.update(dsa_hidden_dims=4, hidden_dim_size=16, _group_layouts={})
    d.update(_layerwise_prefill_dma=True, _expected_group_layers=lambda _g: 2)
    return c


def _pin_ctx(sz: int) -> ExitStack:
    unreg = lambda *a, **k: None  # noqa: E731
    nop = _N(register_cpu_memory=unreg, unregister_cpu_memory=unreg)
    st, buf = ExitStack(), torch.zeros(sz, dtype=_U8)
    st.enter_context(patch(f"{_MM}._allocate_cpu_memory", return_value=buf))
    st.enter_context(patch(f"{_MM}.lmc_ops", nop))
    return st


def _mixed(sz: int = 2 * 1024 * 1024) -> Any:
    with _pin_ctx(sz):
        return MixedMemoryAllocator(sz)


def _slab(name: str, sz: int = 4096) -> tuple[torch.Tensor, Any, Any]:
    buf = torch.zeros(sz, dtype=_U8)
    kw: dict[str, Any] = dict(shm_name=name, size=sz, ptr=int(buf.data_ptr()))
    m = SharedSlabMapping(**kw, tensor=buf, generation=1, owner=True)
    return buf, m, m.passive_allocator(reuse_prefill=True)


def _batch(
    name: str, hs: list[int], offs: list[int], psz: int, po: list[int], ps: list[int]
) -> Any:
    kw: dict[str, Any] = dict(shm_name=name, producer_rank=0, num_layers=2)
    kw.update(num_chunks=len(hs), physical_sizes=[psz] * len(hs), chunk_hashes=hs)
    kw.update(offsets=offs, page_offsets=po, page_physical_sizes=ps)
    return SharedHandleBatch(**kw)


def _pm(sh: Any, dt: Any, phy: int, fmt: Any = _FMT, n: int = 2, addr: int = 0) -> Any:
    s_l, d_l = ([sh] * n if n > 1 else None), ([dt] * n if n > 1 else None)
    return _MOM(sh, dt, addr, phy, 1, fmt=fmt, shapes=s_l, dtypes=d_l)


def _lps(pages: Any, nl: int = 2) -> list[Any]:
    seq = tuple(pages)
    return [_LPS(seq, layer_id=lid, suffix=()) for lid in range(nl)]


def _mk_eng(cs: int = 128) -> Any:
    cfg, meta = _cfg(cs), _meta(cs)
    e: Any = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    d = e.__dict__
    d.update(config=cfg, metadata=meta, token_database=ChunkedTokenDatabase(cfg, meta))
    d.update(num_layers=2, _init_failed=False, _health_monitor=None)
    d.update(_layerwise_prefill_store_frontiers={}, _force_layerwise_prefill_store=True)
    d.update(kvcache_check_log_enabled=False, store_location="LocalCPUBackend")
    d.update(_direct_store_states={}, _layerwise_put_queue=None)
    d.update(_is_passive=lambda: False, is_healthy=lambda: True)
    sr, sf = (lambda *_a, **_k: 1), (lambda *_a, **_k: None)
    d["stats_monitor"] = _N(on_store_request=sr, on_store_finished=sf)
    return e


def _c8_caches(nl: int, cap: int = 128) -> tuple[Any, list[tuple[torch.Tensor, ...]]]:
    ltb, lsf = (256,) + (130,) * (nl - 1), (1,) + (2,) * (nl - 1)
    layout = _N(layer_token_bytes=ltb, layer_slot_factors=lsf, kv_device="cpu")
    backing = torch.zeros((nl, cap, 1, 128), dtype=_BF)
    k_c8 = torch.zeros((nl * 2, cap, 1, 128), dtype=torch.int8)
    s_c8 = torch.zeros((nl * 2, cap, 1, 1), dtype=torch.float16)
    return layout, [(backing,)] + [(k_c8, s_c8)] * (nl - 1)


def _npu_ctx(ops: Any, stream: Any = None) -> ExitStack:
    s = stream or _npu_stream()
    fnpu = _N(current_stream=lambda: s, Event=_StubEvent)
    st = ExitStack()
    st.enter_context(patch.object(torch, "npu", fnpu, create=True))
    st.enter_context(patch(_NPU_OPS, ops))
    return st


def _inv_free(alloc: Any) -> None:
    for b in list(alloc.free_blocks):
        b.invalidate()


def _tmo_pair(n: int = 2) -> tuple[Any, Any, Any]:
    buf = torch.zeros(n * 32, dtype=_U8)
    alloc = PagedTensorMemoryAllocator(buf, [_SH4], [_U8], _FMT)
    oa, ob = alloc.allocate([_SH4], [_U8], _FMT), alloc.allocate([_SH4], [_U8], _FMT)
    assert isinstance(oa, _TMO) and isinstance(ob, _TMO)
    return alloc, oa, ob


def _append_sparse_pair(
    conn: Any, dva_off: int
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[Any]]:
    t0, t1 = torch.zeros(16, dtype=_U8), torch.zeros(16, dtype=_U8)
    c_dev, c_npu = [], []
    with patch(_NPU_OPS, _N(get_device_ptr=lambda p, _s=0: int(p) + dva_off)):
        conn.append_sparse_chunk_ptr_cache_for_layer(0, [t0], c_dev, c_npu, kv_group=0)
        conn.append_sparse_chunk_ptr_cache_for_layer(1, [t1], c_dev, c_npu, kv_group=0)
    return t0, t1, c_dev, c_npu


class TestAscendSharedPrefillOptimizations(unittest.TestCase):
    """CPU tests for Ascend DVA, liveness, token count, omitted prefix."""

    def _chk_freed(self, alloc: Any, *objs: Any) -> None:
        for obj in objs:
            self.assertTrue(any(x is obj for x in alloc.free_blocks))
            self.assertEqual(obj.get_ref_count(), 0)

    def test_slab_dva_resolution_and_layer_page_pointer_rows(self) -> None:
        conn, fb_calls, eq = _conn(), [], self.assertEqual
        fb = lambda o, **k: _rec(fb_calls, int(o.data_ptr), int(o.data_ptr) + 0x5000)  # noqa: E731
        conn._resolve_registered_cpu_source_device_ptr = fb
        slab_a, _m, pa = _slab("shm-ascend-dva")
        ba = _batch("shm-ascend-dva", [1, 2], [64, 96], 32, [0], [64])
        kw_ra: dict[str, Any] = dict(
            start_chunk=1, end_chunk=2, starts=[0, 4], ends=[4, 8]
        )
        ra = pa.create_batch_views_for_group(
            ba, **kw_ra, meta_fn=lambda _l, _n: (_SH4, _U8, _FMT)
        )
        pg = PagedTensorMemoryAllocator(torch.zeros(64, dtype=_U8), [_SH4], [_U8], _FMT)  # noqa: E501
        self.addCleanup(_inv_free, pg)
        sl0, sl1 = pg.allocate([_SH4], [_U8], _FMT), pg.allocate([_SH4], [_U8], _FMT)
        assert sl0 is not None and sl1 is not None
        probes: list[tuple[int, int]] = []
        gdp = lambda p, sz: _rec(probes, (int(p), int(sz)), int(p) + 0x2000_0000)  # noqa: E731
        fops = _N(get_device_ptr=gdp, unlink_shm=lambda _n: None)
        kwr: dict[str, Any] = dict(
            layer_id=0, connector_id=id(conn), device_key="npu:0"
        )
        kwr["validate_span"] = True
        with patch(_NPU_OPS, fops):
            memo: dict[int, Any] = {}
            fn = conn._resolve_source_device_ptr_row
            got_a = fn([ra[0][0], ra[1][0]], source="addr", slab_memo=memo, **kwr)
            got_s = fn([sl0, sl1], source="slice", slab_memo=memo, **kwr)
            fn([ra[0][0], ra[1][0]], source="addr2", slab_memo={}, **kwr)
        ba_p = int(slab_a.data_ptr()) + 0x2000_0000
        eq(got_a, [ba_p + 64, ba_p + 96])
        eq(got_s, [sl0.data_ptr + 0x2000_0000, sl1.data_ptr + 0x2000_0000])
        eq((len(probes), fb_calls), (2, []))
        pa.clear_recycled_views()
        self.assertIsNone(pa._cached_npu_slab_dva)
        self.assertFalse(ra[0][0].is_valid())
        sl0.ref_count_down()
        sl1.ref_count_down()

        with _pin_ctx(4096):
            pin_a = PinMemoryAllocator(4096)
        pc, c2 = [], _conn()
        c2._resolve_registered_cpu_source_device_ptr = lambda o, **k: _rec(  # noqa: E731
            pc, int(o._base_data_ptr), int(o._base_data_ptr) + 0x3000_0000
        )
        pg_a = _LPM(
            pin_a.buffer[0:64],
            _pm(_SH4, _U8, 64),
            pin_a,
            num_layers=2,
            group_prefix_sum=(0, 32, 64),
            valid_tokens=4,
        )  # noqa: E501
        rows, nop, z0 = _lps([pg_a]), (lambda *a, **k: None), (lambda *a, **k: 0)
        f0 = _N(
            get_device_ptr=z0,
            alloc_pinned_ptr=z0,
            free_pinned_ptr=nop,
            unregister_cpu_memory=nop,
        )  # noqa: E501
        with patch(_NPU_OPS, f0), patch(f"{_MM}.lmc_ops", f0):
            pt1 = c2._layer_page_pointer_rows(rows)
            pt2 = c2._layer_page_pointer_rows(rows)
            eq((len(pc), pt1), (1, pt2))
            pin_a.close()
            self.assertFalse(c2._is_parent_alloc_live(pin_a))
            c2._layer_page_pointer_rows(rows)
            eq(len(pc), 2)
        pg_a.invalidate()

    def test_fail_closed_parent_alloc_liveness_and_obj_dva(self) -> None:
        conn, fb_calls = _conn(), []
        conn._resolve_registered_cpu_source_device_ptr = lambda obj, **kw: _rec(
            fb_calls, int(obj.data_ptr), 0x9000
        )
        self.assertFalse(conn._is_parent_alloc_live(None))
        stock = _N(buffer=torch.zeros(64, dtype=_U8))
        self.assertFalse(conn._is_parent_alloc_live(stock))
        minimal = _N(data_ptr=0x1234)
        args = (0x1234, 32, 0, 0, "s", id(conn), "npu:0", True)
        self.assertEqual(
            conn._resolve_cached_obj_dva(minimal, stock, False, *args), 0x9000
        )
        self.assertFalse(hasattr(minimal, "_cached_npu_base_dva"))
        buf = torch.zeros(128, dtype=_U8)
        no_par = _TMO(buf[:32], _pm(_SZ([1, 2, 2, 4]), _U8, 32, n=1), None)
        conn._resolve_cached_obj_dva(
            no_par, None, False, int(buf.data_ptr()), *args[1:]
        )
        self.assertIsNone(no_par._cached_npu_base_dva)
        self.assertEqual(len(fb_calls), 2)

    def test_zero_tensor_token_counting_fast_path_and_fallback(self) -> None:
        conn, eq = _conn(), self.assertEqual
        count_fn = conn._lmc_layer_num_tokens_fn(kv_group=0, layer_id=0)
        buf, sh = torch.zeros(4096, dtype=_U8), _SZ([4, 16])
        meta = _pm(sh, _BF, 128, n=1)
        slice_obj = _TMO(buf[:128], meta, None)

        def _no_t(_s: Any) -> Any:
            raise AssertionError("tensor property must not be constructed")

        with patch.object(_TMO, "tensor", new_callable=property, fget=_no_t):
            eq(count_fn(slice_obj), 4)
        _buf, _map, passive = _slab("shm-tok-count")
        batch = _batch("shm-tok-count", [1, 2], [256, 384], 128, [0], [256])
        pk = dict(chunk_index=0, shape=sh, dtype=_BF, fmt=_FMT)
        pg_o = passive.create_page_view(batch, **pk, cached_positions=range(0, 4))
        rk = dict(start_chunk=1, end_chunk=2, starts=[0, 4], ends=[4, 8])
        rows = passive.create_batch_views_for_group(
            batch, **rk, meta_fn=lambda _l, _n: (sh, _BF, _FMT)
        )
        addr_o = rows[0][0]
        assert isinstance(pg_o, _LPM) and isinstance(addr_o, _TMO)
        p1 = patch.object(_TMO, "tensor", new_callable=property, fget=_no_t)
        p2 = patch.object(_LPM, "layer_tensor", side_effect=AssertionError("forbidden"))
        with p1, p2:
            eq((count_fn(addr_o), conn._lmc_obj_plane_num_tokens(pg_o, 0, 0)), (4, 4))
        pg_o.invalidate()
        passive.clear_recycled_views()
        eq(count_fn(_TMO(torch.zeros(sh, dtype=_BF), meta, None)), 4)

    def test_omitted_prefix_tokens_all_modes_and_ascend_engine_paths(self) -> None:
        impl: Any = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        self.assertTrue(impl._engine_supports_omitted_prefix_tokens_all_modes())
        eng, toks = _mk_eng(128), list(range(300))
        eng.config.enable_remote_lmcache_store = True
        pre = list(eng.token_database.process_tokens(tokens=toks, make_key=False))
        ph, po = tuple(h for _, _, h in pre), tuple(e - s for s, e, _ in pre)
        om = _OmittedPrefixTokens(256, tuple(toks[256:]))
        eng.is_frozen = lambda: True
        eng._num_layers_for_kv_group = lambda g: 2 if g == 0 else 1
        rb = _N(batched_contains=lambda keys: len(keys))
        bep = lambda ks: [True for _ in ks]  # noqa: E731
        sm = _N(
            storage_backends={"RemoteBackend": rb}, batched_external_pages_exist=bep
        )
        eng.storage_manager = sm
        eng.gpu_connector = _N(batched_to_cpu_layerwise=lambda *a, **k: None)
        eng.wait_for_pending_sync_stores = eng.poll_layerwise_prefill_puts = (
            lambda **k: None
        )
        skw = {"prefill_chunk_hashes": ph[:2], "prefill_chunk_offsets": po[:2]}
        pk = dict(req_id="req-op", tokens=om, mask=None, request_configs=None)
        p_it, p_end, prev = eng._layerwise_prefill_store_plan(
            **pk, kv_group=0, incremental=True, store_kwargs=skw
        )
        got_rng = [(s, e) for s, e, _ in list(p_it)]
        self.assertEqual(
            (p_end, prev, got_rng), (0, None, [(0, 128), (128, 256), (256, 300)])
        )
        fin: list[int] = []
        ho = _N(handoff=object(), last_future=None, session=None, disabled_reason=None)
        eng._remote_fill_prepare_request = lambda _r, _c, st: setattr(
            st, "remote_fill", ho
        )
        eng._finish_remote_fill = lambda _r, _st, end: fin.append(end)
        fkw: dict[str, Any] = dict(
            tokens=om, prefill_chunk_hashes=ph, prefill_chunk_offsets=po
        )
        eng.finish_layerwise_prefill_store(
            "req-of", None, required_store_end=300, persistence_fenced=True, **fkw
        )
        self.assertEqual(fin, [300])
        gkw: dict[str, Any] = dict(
            deferred_layerwise_put=True, _prefill_skip_tokens=256
        )
        gen = eng.store_layer(
            om,
            mask=None,
            req_id="req-os",
            kv_group=0,
            **gkw,
            _prefill_metadata_cache=True,
        )
        next(gen)
        gen.close()

    def test_layer_page_pointer_rows_homogeneous_and_fail_closed(self) -> None:
        conn, dva, mixed, eq = _conn(), 0x4000_0000, _mixed(), self.assertEqual
        pin_a, alloc = mixed.pin_allocator, mixed.batched_allocate_layer_pages
        sg0, sbf, sc8 = _SZ([1, 2, 128, 64]), _SZ([128, 2, 128]), _SZ([128, 1, 130])
        pg0 = alloc([sg0, sg0], [_U8] * 2, 16, 2, fmt=_FMT, valid_tokens=128)
        pmx = alloc([sbf, sc8], [_U8] * 2, 16, 2, fmt=_F2D, valid_tokens=128)
        assert pg0 is not None and pmx is not None
        try:
            rg0, rmx, probes = _lps(pg0), _lps(pmx), []
            gdp = lambda p, sz: _rec(probes, (int(p), int(sz)), int(p) + dva)  # noqa: E731
            with patch(_NPU_OPS, _N(get_device_ptr=gdp)):
                g0, gmx = (
                    conn._layer_page_pointer_rows(rg0),
                    conn._layer_page_pointer_rows(rmx),
                )  # noqa: E501
                assert g0 is not None and gmx is not None
                eq(len(probes), 1)
                for lid in range(2):
                    exp0 = [
                        int(p.layer_data_ptr(0))
                        - p.group_prefix_sum[0]
                        + dva
                        + p.group_prefix_sum[lid]
                        for p in pg0
                    ]  # noqa: E501
                    eq(g0[lid], exp0)
                    eq(gmx[lid], [int(p.layer_data_ptr(lid)) + dva for p in pmx])
                tok_sh = [_SZ([64, 2, 128]), _SZ([64, 1, 130])]
                tm_ok = _MOM(
                    _SZ([64, 1, 386]),
                    _U8,
                    pmx[-1].meta.address,
                    64 * 386,
                    1,
                    fmt=_F2D,
                    shapes=tok_sh,
                    dtypes=[_U8, _U8],
                )  # noqa: E501
                t_ok = _LPM(
                    pmx[-1].raw_data[: 64 * 386],
                    tm_ok,
                    pin_a,
                    num_layers=2,
                    group_prefix_sum=(0, 64 * 256, 64 * 386),
                    valid_tokens=64,
                )  # noqa: E501
                try:
                    self.assertIsNotNone(
                        conn._layer_page_pointer_rows(_lps(list(pmx[:-1]) + [t_ok]))
                    )  # noqa: E501
                finally:
                    t_ok.invalidate()
                pg0[7].invalidate()
                self.assertIsNone(conn._layer_page_pointer_rows(rg0))
                pg0[7].valid = True
                tp = pg0[-1]
                raw, tls, ad = tp.raw_data, tp.layer_size, tp.meta.address
                psz, gps = tp.meta.phy_size, tp.group_prefix_sum
                assert tls is not None
                sfm, sw = _SZ([128, 2, 64]), _SZ([1, 2, 128, 32])
                bad_cases = [
                    (raw, tp.meta, (0, 128, 2 * tls)),
                    (raw, _pm(sg0, torch.int8, psz, addr=ad), gps),
                    (raw, _pm(sfm, _U8, psz, _F2D, addr=ad), gps),
                    (
                        raw[: 2 * 128 * 64],
                        _pm(sw, _U8, 2 * 128 * 64, addr=ad),
                        (0, 128 * 64, 2 * 128 * 64),
                    ),  # noqa: E501
                ]
                for braw, bm, bgps in bad_cases:
                    bp_k = dict(num_layers=2, group_prefix_sum=bgps, valid_tokens=128)
                    bp = _LPM(braw, bm, pin_a, **bp_k)
                    try:
                        self.assertIsNone(
                            conn._layer_page_pointer_rows(_lps(list(pg0[:-1]) + [bp]))
                        )  # noqa: E501
                    finally:
                        bp.invalidate()
        finally:
            for obj in (*pg0, *pmx):
                obj.ref_count_down()

    def test_prepare_prefill_c8_packets_slab_dva_and_fail_closed(self) -> None:
        conn, eq = _conn("cpu"), self.assertEqual
        prep, banks = _npu_stream("prep"), [_npu_stream("b0"), _npu_stream("b1")]
        conn.indexer_c8_layout = _N(mixed=True, is_c8=lambda lyr: lyr >= 1)
        conn._stream_context_or_null = lambda _s: nullcontext()
        conn._layerwise_prefill_dma_stream = lambda _g, b: banks[b]
        layout, kvcaches = _c8_caches(4)
        mixed = _mixed()
        pin_a = mixed.pin_allocator
        slab_ptr, dva = int(pin_a.buffer.data_ptr()), 0x5000_0000
        c8_sh = [_SZ([128, 2, 128])] + [_SZ([128, 1, 130])] * 3
        pages = mixed.batched_allocate_layer_pages(
            c8_sh, [_U8] * 4, 2, 4, fmt=_F2D, valid_tokens=[128, 80]
        )
        assert pages is not None and len(pages) == 2
        probes: list[tuple[int, int]] = []
        gdp = lambda p, sz: _rec(probes, (int(p), int(sz)), int(p) + dva)  # noqa: E731
        fops = _N(IndexerC8State=lambda *a: a, get_device_ptr=gdp)
        prows, bmaps = _lps(pages, 4), ((0, 1), (2, 3))

        def _prep(objs: Any, bb: Any = bmaps, *rng: Any) -> Any:
            st, en = rng or ((0, 128), (128, 192))
            pk = dict(layout=layout, kvcaches=kvcaches, block_size=128)
            sk = dict(memory_objs=objs, block_ids_by_bank=bb)
            kw_s = dict(starts=st, ends=en, transfer_stream=prep)
            return conn._prepare_prefill_c8_packets(**pk, **sk, **kw_s)

        with _npu_ctx(fops, _npu_stream("compute")):
            p1, p2 = _prep(prows), _prep(prows)
            eq(len(probes), 1)
            bases = [int(p.layer_data_ptr(0)) - p.group_prefix_sum[0] for p in pages]
            for lid in range(4):
                _st, ptrs, offs, cnts, maps, fcap = p1[lid]
                eq((cnts.tolist(), offs.tolist(), fcap), ([128, 80], [0, 128], 128))
                exp_p = [
                    b + dva + p.group_prefix_sum[lid]
                    for b, p in zip(bases, pages, strict=True)
                ]  # noqa: E501
                self.assertEqual((ptrs.tolist(), p2[lid][1].tolist()), (exp_p, exp_p))
                self.assertGreaterEqual(bases[0], slab_ptr)
                for bi, bids in enumerate(bmaps):
                    b0 = list(range(bids[0] * 128, (bids[0] + 1) * 128))
                    b1 = list(range(bids[1] * 128, bids[1] * 128 + 64)) + [-1] * 16
                    eq(maps[bi].tolist(), b0 + b1)
            for idx in (2, 3, 4):
                self.assertIs(p1[1][idx], p1[2][idx])
                self.assertIs(p1[1][idx], p1[3][idx])
            self.assertIsNot(p1[1][1], p1[2][1])

            del pin_a._cached_npu_slab_dva
            probes.clear()
            ab = lambda tw, c: mixed.batched_allocate_address_backed(  # noqa: E731
                _SZ([c, 1, tw]), _U8, 1, fmt=_F2D
            )[0]  # type: ignore[index]
            a_objs = [[ab(tw, c) for c in (128, 80)] for tw in (256, 130, 130, 130)]
            for row in a_objs:
                for o in row:
                    self.assertNotIn("raw_data", o.__dict__)
            pa = _prep(a_objs)
            eq(len(probes), 1)
            for row, (_, ptrs, *_r) in zip(a_objs, pa, strict=True):
                for o in row:
                    self.assertNotIn("raw_data", o.__dict__)
                eq(ptrs.tolist(), [int(o.data_ptr) + dva for o in row])
            bad_rng = [
                ((-1, 128), (128, 192), bmaps, _C8_FAIL_PTN),
                ((10, 128), (5, 192), bmaps, _C8_FAIL_PTN),
                ((0, 128), (128, 300), bmaps, _C8_FAIL_PTN),
                ((0, 128), (128, 192), ((0, 9), (2, 3)), "exceeds registered capacity"),
            ]
            for bs, be, bb, msg in bad_rng:
                with self.assertRaisesRegex(ValueError, msg):
                    _prep(prows, bb, bs, be)
        for p in pages:
            p.ref_count_down()
        for row in a_objs:
            for o in row:
                o.ref_count_down()

    def test_ascend_store_layer_uses_split_layers_cached(self) -> None:
        eng = _mk_eng(4)
        eng.is_frozen = lambda: False
        eng._num_transfer_layers_for_call = lambda _g, _k: 2
        eng._get_req_id = lambda kw: kw.get("req_id", "req-split")
        eng._log_kvcache_for_check = eng._ensure_layerwise_connector_layout = (
            lambda **_k: None
        )
        eng._shared_cpu_dtype_for_kv_group = lambda _g: _BF
        eng._memory_format_for_kv_group = lambda _g: _FMT
        eng._layerwise_chunk_fully_stored = lambda *a, **kw: True
        eng.storage_manager = _N(supports_batched_put_layer_pages=lambda _l: False)
        eng.gpu_connector = _N(batched_from_gpu=lambda *a, **kw: iter(()))
        toks = torch.tensor([11, 12, 13, 14], dtype=torch.long)
        _split_cache_engine_key_layers.cache_clear()
        list(eng.store_layer(tokens=toks, req_id="req-split-1", kv_group=0))
        i1 = _split_cache_engine_key_layers.cache_info()
        self.assertGreaterEqual(i1.misses, 1)
        list(eng.store_layer(tokens=toks, req_id="req-split-2", kv_group=0))
        self.assertGreater(_split_cache_engine_key_layers.cache_info().hits, i1.hits)

    def test_no_kv_device_layer_page_rows_and_append_sparse_ptr_cache(self) -> None:
        conn, mixed, eq = _conn(None), _mixed(), self.assertEqual
        self.assertNotIn("kv_device", conn.__dict__)
        slab_ptr, dva = int(mixed.pin_allocator.buffer.data_ptr()), 0x7000_0000
        sh = _SZ([1, 2, 16, 8])
        pages = mixed.batched_allocate_layer_pages(
            [sh, sh], [_U8, _U8], 2, 2, fmt=_FMT, valid_tokens=16
        )
        assert pages is not None and len(pages) == 2
        srcs, c_dev, fb_dev = _lps(pages), [], []
        raw_suf = [[torch.zeros(16, dtype=_U8) for _ in range(2)] for _ in range(2)]
        with patch(_NPU_OPS, _N(get_device_ptr=lambda p, _s=0: int(p) + dva)):
            rows = conn._layer_page_pointer_rows(srcs)
            self.assertIsNotNone(rows)
            conn.append_sparse_chunk_ptr_cache_for_layers(srcs, c_dev, None, kv_group=0)
            ap = conn.append_sparse_chunk_ptr_cache_for_layers
            ap(raw_suf, fb_dev, None, kv_group=0)
        eq((c_dev, c_dev[0]), (rows, [int(p.layer_data_ptr(0)) + dva for p in pages]))
        p0_ptr = int(pages[0].layer_data_ptr(0))
        self.assertTrue(p0_ptr >= slab_ptr and pages[0].meta.address >= 0)
        eq(fb_dev, [[t.data_ptr() + dva for t in r] for r in raw_suf])
        for p in pages:
            p.ref_count_down()

    def test_incremental_dma_tuple_and_zero_begin_chunk_bit_identical(self) -> None:
        hp, ht = (lambda o: o.ptr), (lambda o: o.tok)
        pw, npt = (16, 8), (0x8000_0000, 0x9000_0000)

        def _dp(c: list[int], s: list[int], ct: list[int], t: list[int]) -> DmaPlan:
            return DmaPlan(*(torch.tensor(x, dtype=torch.long) for x in (c, s, ct, t)))

        p3 = _dp([0, 0, 1, 2], [0, 2, 4, 6], [0, 2, 0, 0], [2, 2, 4, 2])
        p4 = _dp([0, 0, 1, 2, 3], [0, 2, 4, 6, 8], [0, 2, 0, 0, 0], [2, 2, 4, 4, 4])
        p5 = _dp(
            [0, 0, 1, 2, 3, 4],
            [0, 2, 4, 6, 8, 10],
            [0, 2, 0, 0, 0, 0],
            [2, 2, 4, 4, 4, 4],
        )  # noqa: E501
        objs = [_N(ptr=0x1000 * (i + 1), tok=4) for i in range(3)]
        prev_s: Any = None
        prev_st: Any = None
        bl: Any = None
        bt: Any = None
        eq = self.assertEqual
        o4 = objs + [_N(ptr=0x4000, tok=4)]
        o5 = o4 + [_N(ptr=0x5000, tok=4)]
        steps = [
            (objs, [0, 4, 8], [4, 8, 10], p3, 0, True),
            (o4, [0, 4, 8, 12], [4, 8, 12, 16], p4, 2, True),
            (o5, [0, 4, 8, 12, 16], [4, 8, 12, 16, 20], p5, 4, False),
        ]

        def _bnd(pl: Any, co: Any, cs: Any, ce: Any, pr: Any) -> Any:
            return bind_incremental_copy_addresses(
                pl, co, cs, ce, npt, pw, 2, hp, ht, pr, slot_prefix_unchanged=True
            )

        for cobjs, cs, ce, plan, exp_r, chk_src in steps:
            tc, ts, tp_e = tuple(cobjs), tuple(cs), tuple(ce)
            if chk_src:
                sl = prepare_source_addresses(cobjs, cs, ce, hp, ht, previous=prev_s)
                st = prepare_source_addresses(tc, ts, tp_e, hp, ht, previous=prev_st)
                eq((sl.reused_chunks, st.reused_chunks), (exp_r, exp_r))
                eq((sl.owners, sl.starts, sl.ends), (st.owners, st.starts, st.ends))
                tl, tt = sl.table[:, : len(cobjs)], st.table[:, : len(cobjs)]
                eq(tl.tolist(), tt.tolist())
                prev_s, prev_st = sl, st
            bl, bt = _bnd(plan, cobjs, cs, ce, bl), _bnd(plan, tc, ts, tp_e, bt)
            eq((bl.rows, bl.reused_chunks, bt.reused_chunks), (bt.rows, exp_r, exp_r))
            eq((bl.tail_segment_start, bl.owners), (bt.tail_segment_start, bt.owners))

    def test_sparse_pointer_table_cleanup_after_list_clear(self) -> None:
        conn = _conn("cpu")
        conn.enable_npu_transfer_validation = False
        _, _, _, c_npu = _append_sparse_pair(conn, 0x3000_0000)
        lid, tables = id(c_npu), conn._layerwise_pointer_tables
        self.assertEqual(tables.get((lid, None)), {0, 1})
        self.assertTrue((lid, 0) in tables and (lid, 1) in tables)
        c_npu.clear()
        conn.release_sparse_chunk_ptr_cache(c_npu)
        self.assertEqual(
            [k for k in tables if k == lid or (isinstance(k, tuple) and k[0] == lid)],
            [],
        )

    def test_device_fallback_and_prepare_c8_override_or_2d_fallback(self) -> None:
        cdc, eq = _conn(None), self.assertEqual
        cdc.device, cdc.enable_npu_transfer_validation = torch.device("cpu"), False
        self.assertFalse(hasattr(cdc, "kv_device"))
        t0, t1, cd, cn = _append_sparse_pair(cdc, 0x4000_0000)
        cdc._append_sparse_chunk_ptr_rows(
            [[int(t0.data_ptr()) + 0x10], [int(t1.data_ptr()) + 0x20]], cd, cn
        )  # noqa: E501
        eq(cd[0], [int(t0.data_ptr()) + 0x4000_0000, int(t0.data_ptr()) + 0x10])
        eq(cd[1], [int(t1.data_ptr()) + 0x4000_0000, int(t1.data_ptr()) + 0x20])
        cdc.release_sparse_chunk_ptr_cache(cn)

        ov_calls: list[tuple[int, int, str]] = []

        class _OvConn(VLLMPagedMemLayerwiseNPUConnector):
            def _resolve_registered_cpu_source_device_ptr(
                self,
                source_obj: Any,
                *,
                layer_id: int,
                chunk_index: int,
                source: str,
                required_bytes: int | None = None,
            ) -> int:
                del source_obj, required_bytes
                ov_calls.append((layer_id, chunk_index, source))
                return 0x6000_0000 + layer_id * 0x100 + chunk_index

        layout, kvcaches = _c8_caches(2, cap=2)

        def _non1d() -> list[list[Any]]:
            tm0 = _pm(_SZ([64, 256]), _U8, 64 * 256, _F2D, n=1)
            tm1 = _pm(_SZ([64, 65]), torch.int16, 64 * 130, _F2D, n=1)
            t_a = _TMO(torch.zeros((64, 256), dtype=_U8), tm0, None)
            t_b = _TMO(torch.zeros((64, 65), dtype=torch.int16), tm1, None)
            return [[t_a], [t_b]]

        def _wire(cls: Any) -> Any:
            c = object.__new__(cls)
            c.device = torch.device("cpu")
            c.indexer_c8_layout = _N(mixed=True, is_c8=lambda lyr: lyr >= 1)
            c.indexer_hbm_block_map, c.load_stream = None, _npu_stream()
            c._stream_context_or_null = lambda _s: nullcontext()
            return c

        sub_c, fb_seen = _wire(_OvConn), []
        base_c = _wire(VLLMPagedMemLayerwiseNPUConnector)
        rfn = lambda _o, *, layer_id, chunk_index, source: _rec(
            fb_seen,
            (layer_id, chunk_index),
            0x7000_0000 + layer_id * 0x10 + chunk_index,
        )  # noqa: E501, E731
        base_c._resolve_registered_cpu_source_device_ptr = rfn
        gdp_fail = lambda *_a: (_ for _ in ()).throw(AssertionError("no fast DVA"))  # noqa: E731
        ops_c8 = _N(IndexerC8State=lambda *a: a, get_device_ptr=gdp_fail)
        lmt = _NPU_OPS.replace("lmc_ops", "_layer_memory_tensor")
        kw_c8: dict[str, Any] = dict(
            layout=layout, kvcaches=kvcaches, block_ids_by_bank=((0,), (1,))
        )
        kw_c8.update(block_size=128, starts=(0,), ends=(64,))
        with _npu_ctx(ops_c8):
            fn_p = lambda c: c._prepare_prefill_c8_packets(
                **kw_c8, memory_objs=_non1d(), transfer_stream=_npu_stream()
            )  # noqa: E731
            ps = fn_p(sub_c)
            with patch(lmt, side_effect=lambda obj, _l: obj.raw_data):
                pb = fn_p(base_c)
        exp_ov = [(0, 0, "_prepare_prefill_c8_packets")] * 2
        exp_ov[1] = (1, 0, "_prepare_prefill_c8_packets")
        eq(
            (ov_calls, ps[0][1].tolist(), ps[1][1].tolist()),
            (exp_ov, [0x6000_0000], [0x6000_0100]),
        )  # noqa: E501
        eq(
            (fb_seen, pb[0][1].tolist(), pb[1][1].tolist()),
            ([(0, 0), (1, 0)], [0x7000_0000], [0x7000_0010]),
        )  # noqa: E501

    def test_ascend_release_layerwise_prefill_dma_cache_batch_releases_owners(
        self,
    ) -> None:
        conn, (alloc, oa, ob), syncs = _conn(), _tmo_pair(3), []
        self.addCleanup(_inv_free, alloc)
        conn.wait_for_layerwise_prefill_stores = lambda rids: syncs.append(
            f"store:{rids[0]}"
        )  # noqa: E501
        conn._layerwise_prefill_load_request_events = {
            "req-rel": [_N(synchronize=lambda: syncs.append("ev"))]
        }  # noqa: E501
        conn._layerwise_prefill_load_owners = {"req-rel": {id(oa): oa, id(ob): ob}}
        conn._prefill_dma_bound_loads = {"req-rel": []}
        conn._prefill_dma_plan_cache, conn._prefill_dma_source_cache = (
            {"req-rel": {}},
            {"req-rel": {}},
        )  # noqa: E501
        conn.release_layerwise_prefill_dma_cache("req-rel")
        self.assertEqual(syncs, ["store:req-rel", "ev"])
        cs = (
            conn._layerwise_prefill_load_owners,
            conn._layerwise_prefill_load_request_events,
            conn._prefill_dma_bound_loads,
            conn._prefill_dma_plan_cache,
            conn._prefill_dma_source_cache,
        )
        self.assertTrue(all("req-rel" not in c for c in cs))
        self._chk_freed(alloc, oa, ob)

        t_err = alloc.allocate([_SH4], [_U8], _FMT)
        assert isinstance(t_err, _TMO)
        sfail = lambda: (_ for _ in ()).throw(RuntimeError("sync fail"))  # noqa: E731
        conn._layerwise_prefill_load_request_events = {
            "req-rel-err": [_N(synchronize=sfail)]
        }
        conn._layerwise_prefill_load_owners = {"req-rel-err": {id(t_err): t_err}}
        rel_fn = conn.release_layerwise_prefill_dma_cache
        self.assertRaisesRegex(RuntimeError, "sync fail", rel_fn, "req-rel-err")
        self.assertIn("req-rel-err", conn._layerwise_prefill_load_owners)
        self.assertIn("req-rel-err", conn._layerwise_prefill_load_request_events)
        busy = t_err.is_valid() and t_err.get_ref_count() == 1
        self.assertTrue(busy and not any(x is t_err for x in alloc.free_blocks))
        t_err.ref_count_down()

    def test_stock_lmcache_missing_optional_symbols_fallback(self) -> None:
        root = Path(__file__).resolve().parents[2]

        def _exec_try(rel: str, sym: str) -> Any:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
            nds = [
                n for n in tree.body if isinstance(n, ast.Try) and sym in ast.dump(n)
            ]
            self.assertTrue(nds, f"missing optional import block in {rel}")
            ns: dict[str, Any] = {}
            exec(
                compile(ast.fix_missing_locations(ast.Module(nds, [])), rel, "exec"), ns
            )
            return ns[sym]

        stubs = {
            k: types.ModuleType(k)
            for k in ("lmcache.v1.shared_cpu_cache", "lmcache.v1.prefill_metadata")
        }
        nc_p = "lmcache_ascend/v1/npu_connector/npu_connectors.py"
        ce_p = "lmcache_ascend/v1/cache_engine.py"
        with patch.dict(sys.modules, stubs):
            self.assertIsNone(_exec_try(nc_p, "release_passive_views_batch"))
            self.assertIsNone(_exec_try(ce_p, "split_layers_cached"))
        conn, (alloc, oa, ob) = _conn(), _tmo_pair()
        self.addCleanup(_inv_free, alloc)
        conn.wait_for_layerwise_prefill_stores = lambda _i: None
        conn._layerwise_prefill_load_request_events, conn._prefill_dma_bound_loads = (
            {},
            {},
        )  # noqa: E501
        conn._layerwise_prefill_load_owners = {"req-fb": {id(oa): oa, id(ob): ob}}
        with patch(f"{_NPU_OPS[:-8]}.release_passive_views_batch", None):
            conn.release_layerwise_prefill_dma_cache("req-fb")
        self.assertNotIn("req-fb", conn._layerwise_prefill_load_owners)
        self._chk_freed(alloc, oa, ob)

    def test_direct_store_plans_with_omitted_prefix_tokens(self) -> None:
        eng, toks, eq = _mk_eng(256), list(range(800)), self.assertEqual
        eng._remote_fill_direct_groups = lambda: (0, 1)
        pre = list(eng.token_database.process_tokens(tokens=toks, make_key=False))
        eq([(s, e) for s, e, _ in pre[:3]], [(0, 256), (256, 512), (512, 768)])
        ph, po = tuple(h for _, _, h in pre[:2]), tuple(e - s for s, e, _ in pre[:2])
        om: Any = _OmittedPrefixTokens(512, tuple(range(512, 800)))
        dsp, rfp = eng._direct_suffix_plans, eng._remote_fill_prefix_plans
        st_k: dict[str, Any] = dict(prefill_chunk_hashes=ph, prefill_chunk_offsets=po)
        om0 = dsp(_DirectStoreRequestState(**st_k), om, (0, 1), None)
        eq(om0, dsp(_DirectStoreRequestState(), toks, (0, 1), None))
        eq([(s, e) for s, e, _ in om0[0]], [(0, 256), (256, 512), (512, 768)])
        k256: dict[str, Any] = dict(
            planned_end=256, planned_hash=ph[0], submitted_end={0: 256, 1: 256}
        )
        om256 = dsp(_DirectStoreRequestState(**k256, **st_k), om, (0, 1), None)
        eq(om256, dsp(_DirectStoreRequestState(**k256), toks, (0, 1), None))
        eq([(s, e) for s, e, _ in om256[1]], [(256, 512), (512, 768)])
        rf_om = rfp(om, None, 512, **st_k)
        eq(rf_om, rfp(toks, None, 512))
        eq([(s, e) for s, e, _ in rf_om[1]], [(0, 256), (256, 512)])
        eng._direct_store_enabled = True
        eng._direct_store_states = {"req-sdp": _DirectStoreRequestState(finalized=True)}
        eng.storage_manager, eng.gpu_connector = _N(), _N()
        sdp, emp = eng.store_direct_prefill, torch.empty(0)
        smap = {0: torch.empty(0, dtype=torch.long)}
        ok_sdp = sdp(
            "req-sdp",
            om,
            {0: [emp]},
            smap,
            None,
            final=True,
            slot_mapping_base=512,
            verified_prefix_end=512,
            **st_k,
        )
        self.assertTrue(ok_sdp)
        st_sdp = eng._direct_store_states["req-sdp"]
        eq((st_sdp.prefill_chunk_hashes, st_sdp.prefill_chunk_offsets), (ph, po))

    def test_paged_allocator_raw_data_none_guard_and_skew_c_probe(self) -> None:
        conn = _conn()
        p_alloc = PagedTensorMemoryAllocator(
            torch.zeros(64, dtype=_U8), [_SH4], [_U8], _FMT
        )
        self.addCleanup(_inv_free, p_alloc)
        sl = p_alloc.allocate([_SH4], [_U8], _FMT)
        assert isinstance(sl, _TMO)
        sl.meta.address = 1
        sl.__dict__.pop("raw_data", None)
        dva_calls: list[tuple[int, int]] = []

        def _fake_dva(p: int, sz: int = 0) -> int:
            dva_calls.append((int(p), int(sz)))
            return int(p) + 0x2000_0000

        fops = _N(get_device_ptr=_fake_dva)
        try:
            with (
                patch(_NPU_OPS, fops),
                self.assertRaisesRegex(RuntimeError, "PagedTensorMemoryAllocator"),
            ):
                conn._resolve_source_device_ptr_row(
                    [sl],
                    layer_id=0,
                    source="pg",
                    connector_id=id(conn),
                    device_key="npu:0",
                    validate_span=True,
                    slab_memo={},
                )
            self.assertNotIn(int(sl.meta.address), [ptr for ptr, _ in dva_calls])
        finally:
            sl.invalidate()

        impl: Any = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        fake_ce_no = _N(PrefillSequenceView=object)
        with patch(
            "lmcache_ascend.integration.vllm.vllm_v1_adapter.LMCacheEngine", fake_ce_no
        ):
            self.assertFalse(impl._engine_supports_omitted_prefix_tokens_all_modes())
        fake_ce_yes = _N(
            PrefillSequenceView=object,
            _process_tokens_with_precomputed_hashes=lambda *_a, **_k: None,
        )
        with patch(
            "lmcache_ascend.integration.vllm.vllm_v1_adapter.LMCacheEngine", fake_ce_yes
        ):
            self.assertTrue(impl._engine_supports_omitted_prefix_tokens_all_modes())

        conn_dva = _conn("cpu")
        conn_dva._layerwise_prefill_dma = True
        high_dva = (1 << 63) | 0x5000_0000
        dev_ptrs = [[high_dva, 0x2000], [high_dva + 0x100, 0x3000]]
        npu_ptrs: list[torch.Tensor | None] = [None, None]
        conn_dva._deferred_sparse_pointer_caches = {id(npu_ptrs): npu_ptrs}
        conn_dva.materialize_sparse_chunk_ptr_cache(dev_ptrs, npu_ptrs)
        self.assertNotIn(id(npu_ptrs), conn_dva._deferred_sparse_pointer_caches)
        self.assertEqual(len(npu_ptrs), 2)
        assert npu_ptrs[0] is not None and npu_ptrs[1] is not None
        self.assertEqual(npu_ptrs[0].dtype, torch.int64)
        self.assertEqual(npu_ptrs[0].view(torch.uint64).tolist(), [high_dva, 0x2000])
        self.assertEqual(
            npu_ptrs[1].view(torch.uint64).tolist(), [high_dva + 0x100, 0x3000]
        )

    def test_finish_save_batch_recompute_fallback_and_lease_release(self) -> None:
        class _CpuPage:
            def __init__(self) -> None:
                self.raw_data = torch.empty(1, device="cpu")
                self.ref_count = 0

            def is_valid(self) -> bool:
                return True

            def ref_count_up(self) -> None:
                self.ref_count += 1

            def ref_count_down(self) -> None:
                self.ref_count -= 1

        req_bad = _N(
            req_id="req-bad",
            is_last_prefill=True,
            is_sparse_decode=False,
            is_decode_window_save=False,
            live_source_requested=False,
            load_spec=None,
            save_spec=_N(can_save=True, can_save_latent=True, can_save_indexer=True),
            slot_mapping=[torch.arange(4, dtype=torch.long)],
            indexer_slot_mapping=[torch.arange(4, dtype=torch.long)],
            request_configs={},
            token_ids=[1, 2, 3, 4],
        )
        req_ok = _N(
            req_id="req-ok",
            is_last_prefill=True,
            is_sparse_decode=False,
            is_decode_window_save=False,
            live_source_requested=False,
            load_spec=None,
            save_spec=_N(can_save=True, can_save_latent=True, can_save_indexer=True),
            slot_mapping=[torch.arange(4, dtype=torch.long)],
            indexer_slot_mapping=[torch.arange(4, dtype=torch.long)],
            request_configs={"k": "v"},
            token_ids=[5, 6, 7, 8],
        )
        bad_page, ok_page = _CpuPage(), _CpuPage()
        bad_lease = LayerwiseCPUFillLease((bad_page,))
        ok_lease = LayerwiseCPUFillLease((ok_page,))
        self.assertEqual((bad_page.ref_count, ok_page.ref_count), (1, 1))

        fill_reqs: list[list[str]] = []
        poll_calls: list[dict[str, object]] = []
        dropped_states: list[tuple[str, ...]] = []
        released_pages: list[str] = []
        adopted: list[object] = []
        finished: list[tuple[str, object]] = []

        impl: Any = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        impl.kv_role = "kv_producer"
        impl._step_recompute_fallback_req_ids = {"req-bad"}
        impl._completed_layerwise_stores = {
            ("req-bad", 0): "bad-res",
            ("req-ok", 0): "ok-res",
        }
        impl._layerwise_local_store_results = {}
        impl._parent = _N(
            _get_connector_metadata=lambda: _N(requests=[req_bad, req_ok])
        )
        fill_srcs = {
            "req-bad": {1: ("bad-batch", bad_lease)},
            "req-ok": {1: ("ok-batch", ok_lease)},
        }
        impl.lmcache_engine = _N(
            _recompute_fallback_req_ids=set(),
            _layerwise_cpu_fill_sources=fill_srcs,
            submit_layerwise_prefill_fills=lambda rids: fill_reqs.append(list(rids)),
            poll_layerwise_prefill_puts=lambda **kw: poll_calls.append(kw),
            wait_for_pending_sync_stores=lambda: None,
            drop_direct_store_states=lambda rids: dropped_states.append(tuple(rids)),
            release_layerwise_prefill_pages=lambda rid: released_pages.append(rid),
            adopt_completed_layerwise_store=lambda res: adopted.append(res),
            finish_layerwise_prefill_store=lambda rid, *_a, **kw: finished.append(
                (rid, kw.get("required_store_end"))
            ),
        )
        impl._finish_layerwise_prefill_batch()
        self.assertEqual(fill_reqs, [["req-ok"]])
        self.assertEqual(poll_calls[0]["req_ids"], ("req-ok",))
        self.assertIn({"final": True, "req_ids": ("req-bad",)}, poll_calls)
        self.assertEqual(dropped_states, [("req-bad",)])
        self.assertEqual(adopted, ["ok-res"])
        self.assertEqual(finished, [("req-ok", 4)])
        self.assertEqual(released_pages, ["req-bad"])
        self.assertEqual((bad_page.ref_count, ok_page.ref_count), (0, 1))
        ok_lease.release()

        discarded: list[str] = []
        adopted_direct: list[object] = []
        stored_direct: list[str] = []
        waited_direct: list[set[str]] = []
        rec_groups: list[tuple[str, int, Any]] = []
        committed: list[str] = []
        events: list[tuple[str, tuple[str, ...]]] = []

        meta2 = _N(requests=[req_bad, req_ok])
        impl2: Any = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        impl2.kv_role = "kv_producer"
        impl2.config = _N(
            dsa_two_groups=True, remote_fill_submission_mode="final_deferred"
        )
        impl2._vllm_config = _N(
            parallel_config=_N(
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_index=0,
            )
        )
        impl2._latent_layer_names = ["latent_0"]
        impl2._indexer_layer_names = ["indexer_0"]
        impl2._direct_store_step_supported = None
        impl2._direct_store_observed_layers = set()
        impl2._latest_live_source_ready_event = None
        impl2._latest_live_source_ready_event_source = "missing"
        impl2._latest_direct_source_ready_events = {}
        impl2._step_recompute_fallback_req_ids = {"req-bad"}
        impl2._force_layerwise_prefill_store = False
        impl2._direct_store_requested = True
        impl2._remote_store_requested = False
        impl2._unfenced_live_stores = {}
        impl2._live_source_ready_fences = {}
        impl2._finalized_live_source_submissions = set()
        impl2._layerwise_local_store_results = {}
        impl2._parent = _N(
            _connector_metadata=meta2, _get_connector_metadata=lambda: meta2
        )
        impl2._direct_group_caches = lambda: {0: [torch.empty(0)]}
        impl2._direct_request_inputs = lambda _req, _gc: (
            {0: [torch.empty(0)]},
            {0: torch.arange(4, dtype=torch.long)},
            0,
        )
        impl2._record_prefill_save_group_completed = lambda req, grp, res: (
            rec_groups.append((req.req_id, grp, res.committed_end))
        )
        impl2._mark_prefill_committed = lambda req: committed.append(req.req_id)

        eng2 = _N(
            _recompute_fallback_req_ids=set(),
            _direct_store_states={},
            wait_for_pending_sync_stores=lambda: None,
            adopt_completed_layerwise_store=lambda res: adopted_direct.append(res),
            discard_live_source_descriptor=lambda rid: discarded.append(rid),
            wait_for_pending_stores=lambda rids: events.append(("wait", tuple(rids))),
            drop_direct_store_states=lambda rids: events.append(("drop", tuple(rids))),
            direct_prefill_store_enabled=lambda: True,
            store_direct_prefill=lambda rid, *_a, **_k: stored_direct.append(rid),
            wait_for_direct_stores=lambda rids: waited_direct.append(set(rids)),
            direct_store_committed_ends=lambda _rid: {0: 4},
        )
        impl2.lmcache_engine = eng2
        impl2._completed_layerwise_stores = {
            ("req-bad", 0): "bad-res",
            ("req-ok", 0): "ok-res",
        }
        impl2._finish_save_batch({})
        self.assertEqual(adopted_direct, ["ok-res"])
        self.assertEqual(stored_direct, ["req-ok"])
        self.assertEqual(waited_direct, [{"req-ok"}])
        self.assertEqual(rec_groups, [("req-ok", 0, 4)])
        self.assertEqual(committed, ["req-ok"])
        self.assertEqual(discarded, ["req-bad"])
        self.assertEqual(events, [("drop", ("req-bad",))])

        events.clear()
        eng2._direct_store_states["req-bad"] = _N(futures={object()})
        impl2._completed_layerwise_stores = {
            ("req-bad", 0): "bad-res",
            ("req-ok", 0): "ok-res",
        }
        impl2._finish_save_batch({})
        self.assertEqual(events, [("wait", ("req-bad",)), ("drop", ("req-bad",))])

    def test_high_bit_64bit_dva_across_connector_paths(self) -> None:
        high_dva = (1 << 63) | 0x1234_5600
        ops_mod = "lmcache_ascend.v1.npu_connector.npu_connectors"
        fake_ops = _N(
            get_device_ptr=lambda _p, _s=0: high_dva, IndexerC8State=lambda *a: a
        )

        def _mk_c(*, dma: bool = True) -> VLLMPagedMemLayerwiseNPUConnector:
            c = _conn("cpu")
            c._layerwise_prefill_dma = dma
            return c

        with _npu_ctx(fake_ops, _npu_stream("c")) as st:
            src_t = torch.zeros(16, dtype=_U8)

            c_dma = _mk_c(dma=True)
            dev_dma: list[list[int]] = []
            npu_dma: list[torch.Tensor | None] = []
            c_dma.append_sparse_chunk_ptr_cache_for_layer(0, [src_t], dev_dma, npu_dma)
            assert npu_dma[0] is not None
            self.assertEqual(npu_dma[0].view(torch.uint64).tolist(), [high_dva])

            c_leg = _mk_c(dma=False)
            dev_leg: list[list[int]] = []
            npu_leg: list[torch.Tensor | None] = []
            c_leg.append_sparse_chunk_ptr_cache_for_layer(0, [src_t], dev_leg, npu_leg)
            assert npu_leg[0] is not None
            self.assertEqual(npu_leg[0].view(torch.uint64).tolist(), [high_dva])

            dev_rows: list[list[int]] = []
            npu_rows: list[torch.Tensor | None] = []
            c_dma.append_sparse_chunk_ptr_cache_for_layers(
                [[src_t], [src_t]], dev_rows, npu_rows, defer_copy=False, kv_group=0
            )
            self.assertEqual(
                [r.view(torch.uint64).tolist() for r in npu_rows if r is not None],
                [[high_dva], [high_dva]],
            )

            mixed = _mixed()
            pages = mixed.batched_allocate_layer_pages(
                [_SH4, _SH4], [_U8, _U8], 1, 2, fmt=_FMT
            )
            assert pages is not None
            prows = _lps(pages, 2)
            dev_pg: list[list[int]] = []
            npu_pg: list[torch.Tensor | None] = []
            self.assertTrue(
                c_dma.prepare_sparse_page_ptr_cache_for_layers(
                    prows, dev_pg, npu_pg, defer_copy=False
                )
            )
            self.assertEqual(
                [r.view(torch.uint64).tolist() for r in npu_pg if r is not None],
                [[high_dva], [high_dva + pages[0].group_prefix_sum[1]]],
            )
            for p in pages:
                p.ref_count_down()

            c_c8 = _mk_c(dma=True)
            c_c8.indexer_c8_layout = _N(mixed=True, is_c8=lambda i: i >= 1)
            c_c8._stream_context_or_null = lambda _s: nullcontext()
            c_c8._layerwise_prefill_dma_stream = lambda _g, _b: _npu_stream("b")
            layout_c8, kv_c8 = _c8_caches(2)

            def _c8_obj(tw: int) -> Any:
                buf = torch.zeros((128, tw), dtype=_U8)
                return _N(
                    tensor=buf,
                    data_ptr=int(buf.data_ptr()),
                    get_size=lambda b=buf: b.numel(),
                    get_physical_size=lambda b=buf: b.numel(),
                    is_valid=lambda: True,
                    parent_allocator=None,
                    valid=True,
                )

            c8_objs = [[_c8_obj(256)], [_c8_obj(130)]]
            prep_c8 = c_c8._prepare_prefill_c8_packets(
                layout=layout_c8,
                kvcaches=kv_c8,
                memory_objs=c8_objs,
                block_ids_by_bank=((0,),),
                block_size=128,
                starts=[0],
                ends=[128],
                transfer_stream=_npu_stream("p"),
            )
            for _st_c8, ptrs_c8, *_rest in prep_c8:
                self.assertEqual(ptrs_c8.dtype, torch.int64)
                self.assertEqual(ptrs_c8.view(torch.uint64).tolist(), [high_dva])

            c_cap = _mk_c(dma=True)
            c_cap.store_stream = None
            c_cap.kv_device = torch.device("cpu")
            c_cap.kvcaches = [
                torch.zeros((4, 1, 128), dtype=_BF),
                torch.zeros((4, 1, 128), dtype=_BF),
            ]
            c_cap.initialize_kvcaches_ptr = lambda **_k: None
            c_cap._lazy_initialize_buffer_with_staging = lambda _c, **_k: _N(
                indexer_c8=False,
                kv_format=KVCacheFormat.MLA_LATENT,
                k_hidden_dims=8,
                v_hidden_dims=8,
                dsa_hidden_dims=4,
                vllm_two_major=False,
                kv_device=torch.device("cpu"),
            )
            c_cap.supports_batched_from_gpu_group = lambda _g: True
            c_cap._check_layerwise_transfer_invariants = lambda **_k: None
            c_cap._expected_memory_format = lambda _g: _FMT
            c_cap._slot_mapping_on_kv_device = lambda slots, _s: slots
            c_cap._layerwise_token_major = lambda _g: True
            c_cap._sparse_lmc_host_interleaved = lambda _g: False
            c_cap._stream_context_or_null = lambda _s: nullcontext()
            orig_tensor = torch.tensor

            def _cpu_tensor_no_pin(*a: Any, **kw: Any) -> torch.Tensor:
                kw.pop("pin_memory", None)
                return orig_tensor(*a, **kw)

            st.enter_context(patch("torch.tensor", side_effect=_cpu_tensor_no_pin))
            st.enter_context(
                patch(f"{ops_mod}.prepare_sparse_direct_layer_state", lambda *a: a)
            )
            cap_obj0: Any = _N(
                metadata=_N(fmt=_FMT), tensor=torch.zeros((4,), dtype=_U8)
            )
            cap_obj1: Any = _N(
                metadata=_N(fmt=_FMT), tensor=torch.zeros((4,), dtype=_U8)
            )
            cap = c_cap.prepare_group_capture(
                [[cap_obj0], [cap_obj1]],
                [0],
                [4],
                kv_group=0,
                slot_mapping=torch.arange(4, dtype=torch.long),
                slot_mapping_base=0,
            )
            self.assertEqual(cap.pointers.dtype, torch.int64)
            self.assertEqual(
                cap.pointers.view(torch.uint64).tolist(), [[high_dva], [high_dva]]
            )

    def test_ordinary_dense_direct_store_abort_fence_and_high_bit_dma_addresses(
        self,
    ) -> None:
        # Third Party
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            LMCacheAscendConnectorV1Impl,
        )
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
        from lmcache_ascend.v1.npu_connector import npu_connectors as nc_mod
        from lmcache_ascend.v1.npu_connector.layerwise_dma import (
            DmaPlan,
            bind_copy_addresses,
            bind_incremental_copy_addresses,
            prepare_source_addresses,
        )
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            VLLMPagedMemLayerwiseNPUConnector,
        )

        # Ordinary-store dense_direct mid-.close() fence
        def _mk_mem(events: list[str], tag: str) -> Any:
            obj: Any = _N(
                metadata=_N(fmt=_FMT),
                tensor=torch.zeros((4, 8), dtype=_BF),
                ref_count=1,
            )

            def _is_valid() -> bool:
                return bool(obj.ref_count > 0)

            def _ref_up() -> None:
                obj.ref_count += 1
                events.append(f"{tag}_up")

            def _ref_down() -> None:
                obj.ref_count -= 1
                events.append(f"{tag}_down")

            obj.is_valid = _is_valid
            obj.ref_count_up = _ref_up
            obj.ref_count_down = _ref_down
            return obj

        def _mk_ordinary_conn(sync_fn: Any) -> Any:
            c: Any = VLLMPagedMemLayerwiseNPUConnector.__new__(
                VLLMPagedMemLayerwiseNPUConnector
            )
            c.kv_device = torch.device("cpu")
            c.kvcaches = None
            c.use_gpu = False
            c._layerwise_prefill_dma = False
            c.store_stream = _N(synchronize=sync_fn)
            c.initialize_kvcaches_ptr = lambda **_k: None
            c._lazy_initialize_buffer_with_staging = lambda _c, **_k: _N(
                indexer_c8=False,
                kv_format=KVCacheFormat.MLA_LATENT,
                k_hidden_dims=8,
                v_hidden_dims=8,
                dsa_hidden_dims=0,
                vllm_two_major=False,
                kv_device=torch.device("cpu"),
                gpu_buffer_allocator=None,
                layer_token_bytes=(16, 16),
            )
            c._is_mla_dsa_format = lambda _g: True
            c._check_layerwise_transfer_invariants = lambda **_k: None
            c._slot_mapping_on_kv_device = lambda slots, _s: slots
            c._layerwise_token_major = lambda _g: True
            c._expected_memory_format = lambda _g: _FMT
            c._sparse_lmc_host_interleaved = lambda _g: False
            c._prepare_dense_direct_chunk_metadata = lambda *a, **_k: (
                4,
                torch.tensor([0], dtype=torch.long),
                torch.tensor([4], dtype=torch.long),
            )
            c._expected_group_layers = lambda _g=0: 2
            c._resolve_sparse_chunk_ptrs_npu = lambda _lid, _ct: torch.tensor(
                [1], dtype=torch.int64
            )
            c._run_dense_direct_kv_transfer_layer = lambda **_k: None
            c.release_sparse_chunk_ptr_cache = lambda *_a: None
            return c

        kvcaches = [torch.zeros((4, 2, 8), dtype=_BF) for _ in range(2)]
        slot_map = torch.arange(4, dtype=torch.long)
        fake_npu = _N(
            current_stream=lambda: _N(wait_stream=lambda _s: None),
            stream=lambda _s: nullcontext(),
        )

        ev_ok: list[str] = []
        m0_ok, m1_ok = _mk_mem(ev_ok, "m0"), _mk_mem(ev_ok, "m1")
        c_ok = _mk_ordinary_conn(lambda: ev_ok.append("store_sync"))
        with ExitStack() as st:
            st.enter_context(patch.object(nc_mod, "_DENSE_DIRECT_STORE_DISABLE", False))
            st.enter_context(patch.object(torch, "npu", fake_npu, create=True))
            g_ok = c_ok.batched_from_gpu(
                [[m0_ok], [m1_ok]],
                [0],
                [4],
                kvcaches=kvcaches,
                slot_mapping=slot_map,
                sync=True,
                kv_group=0,
                req_id="req-dense-direct-ok",
            )
            next(g_ok)
            g_ok.close()
        m0_ok.ref_count_down()
        self.assertLess(ev_ok.index("store_sync"), ev_ok.index("m0_down"))
        self.assertIsNone(getattr(c_ok, "_layerwise_prefill_unfenced_stores", None))

        ev_fail: list[str] = []
        m0_fail, m1_fail = _mk_mem(ev_fail, "m0"), _mk_mem(ev_fail, "m1")

        def _hung_npu_sync() -> None:
            ev_fail.append("store_sync_fail")
            raise RuntimeError("hung npu store")

        c_fail = _mk_ordinary_conn(_hung_npu_sync)
        with ExitStack() as st:
            st.enter_context(patch.object(nc_mod, "_DENSE_DIRECT_STORE_DISABLE", False))
            st.enter_context(patch.object(torch, "npu", fake_npu, create=True))
            g_fail = c_fail.batched_from_gpu(
                [[m0_fail], [m1_fail]],
                [0],
                [4],
                kvcaches=kvcaches,
                slot_mapping=slot_map,
                sync=True,
                kv_group=0,
                req_id="req-dense-direct-hung",
            )
            next(g_fail)
            with self.assertRaisesRegex(RuntimeError, "hung npu store"):
                g_fail.close()

        self.assertEqual((m0_fail.ref_count, m1_fail.ref_count), (2, 2))
        self.assertNotIn("m0_down", ev_fail)
        self.assertIn(
            "req-dense-direct-hung", c_fail._layerwise_prefill_unfenced_stores
        )

        released: list[str] = []
        eng: Any = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        eng.gpu_connector = c_fail
        eng._store_worker_thread = None
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
        eng._unsafe_layerwise_retrieve_sources = []
        eng._layerwise_prefill_page_owners = {"req-dense-direct-hung": {0: []}}
        eng.release_layerwise_prefill_pages = lambda rid: released.append(rid)
        with patch("lmcache.v1.cache_engine.LMCacheEngine.close", lambda _s: None):
            eng.close()
            self.assertEqual(released, [])
            c_fail._layerwise_prefill_unfenced_stores.clear()
            eng.close()
            self.assertEqual(released, ["req-dense-direct-hung"])

        # High >= 1<<63 & boundary-crossing addresses
        high_host = (1 << 63) | 0x2000_0000
        high_npu = (1 << 63) | 0x3000_0000
        bnd_host = (1 << 63) - 64
        bnd_npu = (1 << 63) - 16
        plan = DmaPlan(
            chunk=torch.tensor([0, 1], dtype=torch.long),
            slot=torch.tensor([0, 4], dtype=torch.long),
            chunk_token=torch.tensor([0, 2], dtype=torch.long),
            tokens=torch.tensor([2, 2], dtype=torch.long),
        )
        rows = bind_copy_addresses(
            plan,
            [high_host, bnd_host],
            [high_npu, bnd_npu],
            [4, 4],
            [8, 4],
            2,
            device_to_host=False,
        )
        exp_rows = [
            [high_npu, high_host, 32],
            [bnd_npu, high_host + 64, 16],
            [high_npu + 64, bnd_host + 32, 32],
            [bnd_npu + 32, bnd_host + 80, 16],
        ]
        self.assertEqual(rows, exp_rows)
        self.assertTrue(all((1 << 63) <= v < (1 << 64) for v in rows[3][:2]))
        self.assertLess(rows[1][0], 1 << 63)
        self.assertLess(rows[2][1], 1 << 63)
        self.assertEqual(rows[1][0], bnd_npu)

        src_objs = [_N(id=0), _N(id=1)]
        src_ptrs = {id(src_objs[0]): high_host, id(src_objs[1]): bnd_host}
        src_cache = prepare_source_addresses(
            src_objs,
            [0, 4],
            [4, 8],
            lambda o: src_ptrs[id(o)],
            lambda _o: 4,
        )
        self.assertEqual(
            src_cache.table[0, : len(src_objs)].view(torch.uint64).tolist(),
            [high_host, bnd_host],
        )
        bound_inc = bind_incremental_copy_addresses(
            plan,
            src_objs,
            [0, 4],
            [4, 8],
            [high_npu, bnd_npu],
            [8, 4],
            2,
            lambda o: src_ptrs[id(o)],
            lambda _o: 4,
            None,
            slot_prefix_unchanged=True,
            source_metadata=src_cache,
        )
        self.assertEqual(bound_inc.rows, exp_rows)

        high_v2_0 = (1 << 63) | 0x4444_0000
        high_v2_1 = (1 << 63) | 0x5555_0000
        c_v2 = nc_mod.VLLMPagedMemNPUConnectorV2.__new__(
            nc_mod.VLLMPagedMemNPUConnectorV2
        )
        c_v2.num_layers = 2
        c_v2.use_mla = False
        c_v2.dsa_two_groups = False
        c_v2.kv_cache_pointers_on_gpu = {}

        def _fake_kv_t(ptr: int):
            return _N(
                device=_N(type="npu", index=0),
                data_ptr=lambda p=ptr: p,
                shape=(2, 4, 2, 8),
                dim=lambda: 4,
            )

        kv_v2 = [
            (_fake_kv_t(high_v2_0), _fake_kv_t(high_v2_0 + 0x100)),
            (_fake_kv_t(high_v2_1), _fake_kv_t(high_v2_1 + 0x100)),
        ]
        orig_empty = torch.empty

        def _empty_cpu_for_npu(*args: Any, **kw: Any) -> torch.Tensor:
            dev = kw.get("device")
            if dev is not None and getattr(dev, "type", None) == "npu":
                kw["device"] = "cpu"
            return orig_empty(*args, **kw)

        with (
            patch.object(
                nc_mod.KVCacheFormat, "detect", return_value=KVCacheFormat.SEPARATE_KV
            ),
            patch.object(nc_mod, "is_310p", lambda: False),
            patch.object(torch, "empty", side_effect=_empty_cpu_for_npu),
        ):
            out_ptrs = c_v2._initialize_pointers(kv_v2)
        self.assertEqual(
            out_ptrs.view(torch.uint64).tolist(),
            [high_v2_0, high_v2_0 + 0x100, high_v2_1, high_v2_1 + 0x100],
        )

        impl_sk = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        impl_sk._step_recompute_fallback_req_ids = {"req-fb-skew"}
        impl_sk.lmcache_engine = _N(_recompute_fallback_req_ids={"req-eng-skew"})
        impl_sk._direct_store_requested = False
        impl_sk._remote_store_requested = False
        with patch.object(
            LMCacheConnectorV1Impl,
            "_is_recompute_fallback_request",
            None,
            create=True,
        ):
            self.assertTrue(impl_sk._is_fallback_req_id("req-fb-skew"))
            self.assertTrue(impl_sk._is_fallback_req_id("req-eng-skew"))
            self.assertFalse(impl_sk._is_fallback_req_id("req-ok"))
        with patch.object(LMCacheConnectorV1Impl, "wait_for_save", lambda _s: None):
            impl_sk.wait_for_save()
        self.assertEqual(impl_sk._step_recompute_fallback_req_ids, set())

    def test_ptrs_to_int64_tensor_high_bit_negative_and_pin_memory(self) -> None:
        eq = self.assertEqual
        hi_a = 0x8000_0000_0000_0000
        hi_b = 0xFFFF_FFFF_FFFF_FFE0
        u64_1d = [0x1000, hi_a, hi_b]
        u64_2d = [[0x1000, hi_a], [hi_b, 0x2000]]
        neg_1d = [-32, 0x1000, hi_a]
        neg_2d = [[-32, 64], [0, -1]]
        t_u64 = torch.tensor(u64_1d, dtype=torch.uint64)

        for fn in (
            ptrs_to_int64_tensor,
            VLLMPagedMemLayerwiseNPUConnector._dva_to_int64_tensor,
        ):
            r1 = fn(u64_1d)
            eq((r1.dtype, r1.view(torch.uint64).tolist()), (torch.int64, u64_1d))
            r2 = fn(u64_2d)
            eq((r2.dtype, r2.view(torch.uint64).tolist()), (torch.int64, u64_2d))
            rn1 = fn(neg_1d)
            eq(
                (rn1.dtype, rn1.tolist()),
                (torch.int64, [-32, 0x1000, hi_a - (1 << 64)]),
            )
            rn2 = fn(neg_2d)
            eq((rn2.dtype, rn2.tolist()), (torch.int64, neg_2d))
            e1 = fn([])
            eq((e1.dtype, tuple(e1.shape)), (torch.int64, (0,)))
            e2 = fn([[], []])
            eq((e2.dtype, tuple(e2.shape)), (torch.int64, (2, 0)))
            rt = fn(t_u64)
            eq((rt.dtype, rt.view(torch.uint64).tolist()), (torch.int64, u64_1d))
            eq(rt.data_ptr(), t_u64.data_ptr())

        # Verify pin_memory=True path on both uint64 Tensor view and list inputs.
        pin_calls: list[int] = []
        orig_tensor = torch.tensor

        def _fake_pin(self_t: torch.Tensor) -> torch.Tensor:
            pin_calls.append(int(self_t.numel()))
            return self_t

        def _tensor_spy(*a: Any, **kw: Any) -> torch.Tensor:
            if kw.pop("pin_memory", False):
                pin_calls.append(-int(len(a[0]) if a else 0))
            return orig_tensor(*a, **kw)

        dma_mod = "lmcache_ascend.v1.npu_connector.layerwise_dma"
        with (
            patch.object(torch.Tensor, "pin_memory", _fake_pin),
            patch(f"{dma_mod}.torch.tensor", _tensor_spy),
        ):
            p_t = ptrs_to_int64_tensor(t_u64, pin_memory=True)
            p_l = ptrs_to_int64_tensor(u64_1d, pin_memory=True)
        eq(p_t.view(torch.uint64).tolist(), u64_1d)
        eq(p_l.view(torch.uint64).tolist(), u64_1d)
        eq(pin_calls, [3, -3])

        # Generator, np.ndarray (both pin_memory modes), int32 tensor, CPU pin_memory.
        np_u64 = np.array(u64_1d, dtype=np.uint64)
        rg = ptrs_to_int64_tensor(v for v in u64_1d)
        rn_f = ptrs_to_int64_tensor(np_u64, pin_memory=False)
        ri32 = ptrs_to_int64_tensor(torch.tensor([1, 2], dtype=torch.int32))
        eq((rg.dtype, rg.view(torch.uint64).tolist()), (torch.int64, u64_1d))
        eq((rn_f.dtype, rn_f.view(torch.uint64).tolist()), (torch.int64, u64_1d))
        eq((ri32.dtype, ri32.tolist()), (torch.int64, [1, 2]))

        cpu_pins: list[int] = []

        def _fake_pin_cnt(self_t: torch.Tensor) -> torch.Tensor:
            cpu_pins.append(int(self_t.numel()))
            return self_t

        with (
            patch.object(torch.Tensor, "pin_memory", _fake_pin_cnt),
            patch(f"{dma_mod}.torch.tensor", _tensor_spy),
        ):
            rn_t = ptrs_to_int64_tensor(np_u64, pin_memory=True)
            rcpu = ptrs_to_int64_tensor(t_u64, device="cpu", pin_memory=True)
            rcpu_l = ptrs_to_int64_tensor(u64_1d, device="cpu", pin_memory=True)
        eq(rn_t.view(torch.uint64).tolist(), u64_1d)
        eq(rcpu.view(torch.uint64).tolist(), u64_1d)
        eq(rcpu_l.view(torch.uint64).tolist(), u64_1d)
        eq(cpu_pins, [3, 3])
        eq(pin_calls, [3, -3, -3])

        # Multi-threaded concurrency while companion thread emits warnings.
        orig_filters = list(warnings.filters)
        errs: list[BaseException] = []

        def _worker() -> None:
            try:
                for _ in range(200):
                    r2 = ptrs_to_int64_tensor(u64_2d)
                    eq(r2.view(torch.uint64).tolist(), u64_2d)
                    rn = ptrs_to_int64_tensor(neg_1d)
                    eq(rn.tolist(), [-32, 0x1000, hi_a - (1 << 64)])
                    rg_w = ptrs_to_int64_tensor(v for v in u64_1d)
                    eq(rg_w.view(torch.uint64).tolist(), u64_1d)
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)

        def _warn_loop() -> None:
            try:
                for _ in range(200):
                    warnings.warn("bg", DeprecationWarning, stacklevel=1)
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        threads.append(threading.Thread(target=_warn_loop))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        eq(errs, [])
        eq(list(warnings.filters), orig_filters)

    def test_prepare_prefill_c8_packets_nonaligned_disjoint_and_empty(self) -> None:
        conn, eq = _conn("cpu"), self.assertEqual
        prep, banks = _npu_stream("prep"), [_npu_stream("b0"), _npu_stream("b1")]
        conn.indexer_c8_layout = _N(mixed=True, is_c8=lambda lyr: lyr >= 1)
        conn._stream_context_or_null = lambda _s: nullcontext()
        conn._layerwise_prefill_dma_stream = lambda _g, b: banks[b]
        layout, kvcaches = _c8_caches(4)
        mixed = _mixed()
        dva = 0x8000_0000_5000_0000
        c8_sh = [_SZ([256, 2, 128])] + [_SZ([256, 1, 130])] * 3
        pages = mixed.batched_allocate_layer_pages(
            c8_sh, [_U8] * 4, 2, 4, fmt=_F2D, valid_tokens=[256, 256]
        )
        assert pages is not None and len(pages) == 2
        for p in pages:
            self.addCleanup(p.ref_count_down)
        dva_calls = 0

        def _get_dva(p: Any, _s: Any) -> int:
            nonlocal dva_calls
            dva_calls += 1
            return int(p) + dva

        fops = _N(
            IndexerC8State=lambda *a: a,
            get_device_ptr=_get_dva,
        )
        prows = _lps(pages, 4)

        def _ref_slots(
            bmap: tuple[int, ...], st: tuple[int, ...], en: tuple[int, ...]
        ) -> list[int]:
            return [
                bmap[pos // 128] * 128 + (pos % 128)
                for s, e in zip(st, en, strict=True)
                for pos in range(s, e)
            ]

        def _prep(bb: Any, st: Any, en: Any, mem: Any = prows) -> Any:
            return conn._prepare_prefill_c8_packets(
                layout=layout,
                kvcaches=kvcaches,
                memory_objs=mem,
                block_ids_by_bank=bb,
                block_size=128,
                starts=st,
                ends=en,
                transfer_stream=prep,
            )

        with _npu_ctx(fops, _npu_stream("compute")):
            bmaps = ((0, 1, 2), (2, 3, 0))
            # 1) Contiguous unaligned start (first_off=64) + 2) Non-contiguous ranges.
            for st, en in (((64, 256), (256, 320)), ((0, 256), (64, 320))):
                prep_res = _prep(bmaps, st, en)
                bases = [
                    int(p.layer_data_ptr(0)) - p.group_prefix_sum[0] for p in pages
                ]
                seen_ptrs: set[int] = set()
                for lid, (_st, ptrs, offs, cnts, maps, fcap) in enumerate(prep_res):
                    self.assertTrue(ptrs.is_contiguous())
                    self.assertTrue(offs.is_contiguous())
                    self.assertTrue(cnts.is_contiguous())
                    self.assertTrue(all(m.is_contiguous() for m in maps))
                    self.assertNotIn(ptrs.data_ptr(), seen_ptrs)
                    seen_ptrs.add(ptrs.data_ptr())
                    exp_p = [
                        b + dva + p.group_prefix_sum[lid]
                        for b, p in zip(bases, pages, strict=True)
                    ]
                    eq(ptrs.view(torch.uint64).tolist(), exp_p)
                    eq(
                        (offs.tolist(), cnts.tolist(), fcap),
                        ([0, 256], [256, 256], 256),
                    )
                    pad0 = 256 - (en[0] - st[0])
                    pad1 = 256 - (en[1] - st[1])
                    for bi, bids in enumerate(bmaps):
                        r0 = _ref_slots(bids, (st[0],), (en[0],)) + [-1] * pad0
                        r1 = _ref_slots(bids, (st[1],), (en[1],)) + [-1] * pad1
                        eq(maps[bi].tolist(), r0 + r1)

            # 3) Empty ranges.
            p_empty = _prep(bmaps, (), (), [[] for _ in range(4)])
            eq(len(p_empty), 4)
            for _st, ptrs, offs, cnts, maps, fcap in p_empty:
                eq((ptrs.numel(), offs.numel(), cnts.numel(), fcap), (0, 0, 0, 0))
                eq([m.numel() for m in maps], [0, 0])

            # 4) Negative & out-of-range block IDs on contiguous and disjoint paths.
            for st, en in (((64, 256), (256, 320)), ((0, 256), (64, 320))):
                for bad_bb in (((-1, 1, 2), (2, 3, 0)), ((0, 1, 99), (2, 3, 0))):
                    with self.assertRaisesRegex(
                        ValueError, "Banked C8 logical slot exceeds registered capacity"
                    ):
                        _prep(bad_bb, st, en)
            eq(dva_calls, 1)

    def test_direct_prefill_skewed_engine_omits_empty_hash_kwargs(self) -> None:
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        class _StrictEngine:
            def direct_prefill_store_enabled(self) -> bool:
                return True

            def store_direct_prefill(
                self,
                req_id: Any,
                token_ids: Any,
                selected: Any,
                slot_mappings: Any,
                request_configs: Any,
                **kw: Any,
            ) -> None:
                calls.append(
                    (
                        (
                            req_id,
                            token_ids,
                            selected,
                            slot_mappings,
                            request_configs,
                        ),
                        dict(kw),
                    )
                )

            def direct_store_committed_ends(self, _req_id: str) -> dict[int, int]:
                return {}

            def get_finished_stores(self, req_ids: set[str]) -> set[str]:
                return set(req_ids)

            def drop_direct_store_states(self, _req_ids: Any) -> None:
                return None

        impl = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
        impl.lmcache_engine = _StrictEngine()
        impl._force_layerwise_prefill_store = False
        impl._unfenced_live_stores = {}
        impl._remote_store_requested = False
        impl.store_async = False
        impl._is_fallback_req_id = lambda _rid: False
        impl._direct_group_caches = lambda: {0: [None]}
        impl._direct_request_inputs = lambda r, gc: (
            gc,
            {0: torch.arange(len(r.token_ids))},
            0,
        )
        impl._vllm_config = _N(
            parallel_config=_N(
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            )
        )
        impl._mark_prefill_committed = lambda _r: None
        impl._release_finished_worker_requests = lambda _ids: None

        req_none = _N(
            req_id="r-none",
            token_ids=[1, 2, 3, 4],
            request_configs={"k": "v"},
            is_last_prefill=True,
            is_sparse_decode=False,
            prefill_chunk_hashes=None,
            prefill_chunk_offsets=None,
        )
        req_with = _N(
            req_id="r-with",
            token_ids=[5, 6, 7, 8],
            request_configs={"k": "v2"},
            is_last_prefill=True,
            is_sparse_decode=False,
            prefill_chunk_hashes=["h0"],
            prefill_chunk_offsets=[0],
        )
        impl._submit_direct_prefill_requests([req_none, req_with])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("prefill_chunk_hashes", calls[0][1])
        self.assertNotIn("prefill_chunk_offsets", calls[0][1])
        self.assertEqual(calls[1][1]["prefill_chunk_hashes"], ["h0"])
        self.assertEqual(calls[1][1]["prefill_chunk_offsets"], [0])

        req_fin_none = _N(
            req_id="r-fin-none",
            token_ids=[9, 10],
            request_configs=None,
            prefill_chunk_hashes=None,
            prefill_chunk_offsets=None,
        )
        req_fin_with = _N(
            req_id="r-fin-with",
            token_ids=[11, 12],
            request_configs=None,
            prefill_chunk_hashes=["h1"],
            prefill_chunk_offsets=[2],
        )
        impl._unfenced_live_stores = {
            "r-fin-none": req_fin_none,
            "r-fin-with": req_fin_with,
        }
        calls.clear()
        impl._finalize_worker_requests_after_store({"r-fin-none", "r-fin-with"})
        by_id = {c[0][0]: c[1] for c in calls}
        self.assertEqual(set(by_id), {"r-fin-none", "r-fin-with"})
        self.assertNotIn("prefill_chunk_hashes", by_id["r-fin-none"])
        self.assertNotIn("prefill_chunk_offsets", by_id["r-fin-none"])
        self.assertEqual(by_id["r-fin-with"]["prefill_chunk_hashes"], ["h1"])
        self.assertEqual(by_id["r-fin-with"]["prefill_chunk_offsets"], [2])


if __name__ == "__main__":
    unittest.main()
