# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the production Ascend store and persistence dependency hook."""

import ast
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metadata_module():
    path = ROOT.parent / "LMCache/lmcache/v1/prefill_metadata.py"
    if not path.exists():
        spec = importlib.util.find_spec("lmcache.v1.prefill_metadata")
        path = spec.origin
    return _module("prefill_metadata_store_test", path)


@dataclass(frozen=True)
class Key:
    chunk_hash: tuple
    group: int = 0
    config: str = ""
    splits = 0

    def split_layers(self, count):
        type(self).splits += 1
        return tuple((self, layer) for layer in range(count))


class Database:
    chunk_size = 4
    config = NS(save_unfull_chunk=True)

    def __init__(self):
        self.hashed = 0

    def process_tokens(self, tokens=None, hashes=None, offsets=None, mask=None,
                       request_configs=None, kv_group=0, make_key=True):
        if hashes is not None:
            start = 0
            for value, offset in zip(hashes, offsets, strict=True):
                yield start, start + offset, Key(value, kv_group, repr(request_configs))
                start += offset
            return
        skip = int(mask.numel() - mask.sum()) if mask is not None else 0
        for start, end, value in self.process_tokens_from_prefix(
            tokens, prefix_token_count=0, prefix_hash=(), make_key=make_key,
            request_configs=request_configs, kv_group=kv_group,
        ):
            if start >= skip:
                yield start, end, value

    def process_tokens_from_prefix(self, tokens, *, prefix_token_count, prefix_hash,
                                   make_key=True, request_configs=None, kv_group=0):
        value = prefix_hash
        for start in range(prefix_token_count, len(tokens), self.chunk_size):
            end = min(start + self.chunk_size, len(tokens))
            value = value + tuple(tokens[start:end])
            self.hashed += 1
            yield start, end, (
                Key(value, kv_group, repr(request_configs)) if make_key else value
            )


@dataclass
class StoreResult:
    request_id: str
    kv_group: int
    keys: list = field(default_factory=list)
    starts: list = field(default_factory=list)
    ends: list = field(default_factory=list)
    memory_objs: list = field(default_factory=list)
    tensors: list = field(default_factory=list)
    chunk_dev_ptrs: list = field(default_factory=list)
    chunk_ptrs: list = field(default_factory=list)
    committed_end: int = 0


def _engine():
    metadata = _metadata_module()
    path = ROOT / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AscendLMCacheEngine")
    names = {"store_layer", "_layerwise_prefill_store_plan",
             "track_prefill_retrieve_keys", "_dense_retrieve_token_results"}
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []

    class Base:
        def _dense_retrieve_token_results(self, tokens, mask, configs, group, kwargs):
            return self.token_database.process_tokens(
                tokens=tokens, mask=mask, request_configs=configs, kv_group=group
            )

    runtime = dict(
        torch=torch, deepcopy=deepcopy, CacheEngineKey=Key,
        PrefillMetadataPlan=metadata.PrefillMetadataPlan,
        LayerwiseStoreResult=StoreResult, Base=Base,
        mooncake_layer_pages_enabled=lambda config: False,
        _mtp_dw_diag_enabled=lambda: False,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    isolated = ast.ClassDef(name="Engine", bases=[ast.Name(id="Base", ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, isolated], type_ignores=[]
    )), str(path), "exec"), runtime)
    engine = runtime["Engine"]()
    engine.token_database = Database()
    engine.config = NS(chunk_size=4, get_extra_config_value=lambda *args: False)
    engine._layerwise_prefill_store_frontiers = {}
    engine._force_layerwise_prefill_store = True
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine.is_frozen = lambda: False
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine._get_req_id = lambda kwargs: kwargs["req_id"]
    engine._log_kvcache_for_check = lambda **kwargs: None
    engine._num_layers_for_kv_group = lambda group: 2 if group == 0 else 1
    engine._ensure_layerwise_connector_layout = lambda **kwargs: None
    engine._num_transfer_layers_for_call = lambda group, kwargs: 2 if group == 0 else 1
    engine._shared_cpu_dtype_for_kv_group = lambda group: None
    engine._memory_format_for_kv_group = lambda group: None
    engine.stats_monitor = NS(on_store_request=lambda count: 1,
                              on_store_finished=lambda *args: None)
    engine.store_location = "cpu"
    engine._layerwise_put_queue = None
    engine.checked = []
    engine._layerwise_chunk_fully_stored = lambda keys, **kwargs: (
        engine.checked.append((keys, kwargs)) or True
    )
    return engine, metadata.PrefillMetadataCache


def _store(engine, cache, tokens, *, group=0, skip=0, configs=None):
    return engine.store_layer(
        tokens, req_id="req", kv_group=group, deferred_layerwise_put=True,
        layerwise_prefill_incremental=True, _prefill_metadata_cache=cache,
        _prefill_skip_tokens=skip, request_configs=configs,
    )


def test_public_store_reuses_retrieve_and_group_keys_without_advancing_on_prime():
    engine, Cache = _engine()
    cache = Cache()
    tokens = list(range(8))
    plan = cache.prepare(engine.token_database, tokens, kv_group=0, num_layers=2)
    hashed, splits = engine.token_database.hashed, Key.splits
    storer = _store(engine, cache, tokens)
    next(storer)
    assert engine._layerwise_prefill_store_frontiers["req"] == {}
    assert engine.token_database.hashed == hashed
    assert Key.splits == splits
    assert engine.checked[0][0] is plan.keys_chunk_major[0]
    storer.close()
    assert engine._layerwise_prefill_store_frontiers["req"] == {}
    result = list(_store(engine, cache, tokens))[-1]
    assert result.committed_end == 8
    assert engine._layerwise_prefill_store_frontiers["req"][0][0] == 8
    list(_store(engine, cache, tokens, group=1))
    assert engine.token_database.hashed == hashed
    assert Key.splits == splits + 2
    engine.checked.clear()
    list(_store(engine, cache, list(range(12))))
    assert engine.token_database.hashed == hashed + 1
    assert [(row[1]["start"], row[1]["end"]) for row in engine.checked] == [(8, 12)]


def test_store_plan_uses_maximum_mask_and_committed_frontier_without_committing():
    engine, Cache = _engine()
    cache = Cache()
    list(_store(engine, cache, list(range(8))))
    plan, base, prior = engine._layerwise_prefill_store_plan(
        req_id="req", tokens=list(range(16)), mask=None, request_configs=None,
        kv_group=0, incremental=True, metadata_cache=cache, num_layers=2,
        skip_tokens=12,
    )
    assert list(plan.starts) == [12] and base == 8 and prior[0] == 8
    assert engine._layerwise_prefill_store_frontiers["req"][0][0] == 8
    engine._layerwise_prefill_store_frontiers["req"][0] = (20, ())
    plan, base, prior = engine._layerwise_prefill_store_plan(
        req_id="req", tokens=list(range(16)), mask=None, request_configs=None,
        kv_group=0, incremental=True, metadata_cache=cache, num_layers=2, skip_tokens=0,
    )
    assert list(plan.starts) == [0, 4, 8, 12] and base == 0 and prior is None


def test_failed_public_store_does_not_commit_prepared_suffix():
    engine, Cache = _engine()
    cache = Cache()
    engine._layerwise_chunk_fully_stored = lambda *args, **kwargs: False
    engine.gpu_connector = NS(get_shape=lambda *args, **kwargs: (4,))
    engine.storage_manager = NS(batched_allocate=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="CPU cache is full"):
        next(_store(engine, cache, list(range(8))))
    assert len(cache.hashes) == 2
    assert engine._layerwise_prefill_store_frontiers["req"] == {}


def test_default_store_planner_keeps_legacy_iterable_and_frontier_contract():
    engine, _ = _engine()
    first, base, prior = engine._layerwise_prefill_store_plan(
        req_id="req", tokens=list(range(8)), mask=None, request_configs=None,
        kv_group=0, incremental=True,
    )
    rows = list(first)
    assert [(start, end) for start, end, _ in rows] == [(0, 4), (4, 8)]
    assert base == 0 and prior is None
    engine._layerwise_prefill_store_frontiers["req"][0] = (8, rows[-1][2].chunk_hash)
    following, base, prior = engine._layerwise_prefill_store_plan(
        req_id="req", tokens=list(range(12)), mask=None, request_configs=None,
        kv_group=0, incremental=True,
    )
    assert [(start, end) for start, end, _ in following] == [(8, 12)]
    assert base == 8 and prior[0] == 8


@pytest.mark.parametrize("change", ["scope", "config"])
def test_same_length_new_scope_or_config_does_not_inherit_committed_frontier(change):
    engine, Cache = _engine()
    cache = Cache()
    configs = {"mm": "first"}
    list(_store(engine, cache, list(range(8)), configs=configs))
    engine.checked.clear()
    if change == "scope":
        cache = Cache()
    else:
        configs["mm"] = "second"
    storer = _store(engine, cache, list(range(8)), configs=configs)
    next(storer)
    assert [row[1]["start"] for row in engine.checked] == [0, 4]
    assert engine._layerwise_prefill_store_frontiers["req"] == {}
    storer.close()


def test_prepared_retrieve_hook_and_fallback_both_retain_put_dependencies():
    engine, Cache = _engine()
    queue_module = _module(
        "layerwise_put_store_test", ROOT / "lmcache_ascend/v1/layerwise_cpu_fill.py"
    )
    queue = queue_module.LayerwisePutQueue(1024, 8, 0)
    engine._layerwise_put_queue = queue
    plan = Cache().prepare(engine.token_database, list(range(8)), num_layers=2)
    future = Future()
    queue.add(1, [future], req_id="producer", keys=plan.base_keys)
    hashed, splits = engine.token_database.hashed, Key.splits
    engine.track_prefill_retrieve_keys("consumer", iter(plan.base_keys))
    assert queue._request_futures["consumer"] == {future}
    assert engine.token_database.hashed == hashed and Key.splits == splits
    list(engine._dense_retrieve_token_results(
        list(range(8)), None, None, 0, {"req_id": "legacy"}
    ))
    assert queue._request_futures["legacy"] == {future}
    future.set_result(None)
    queue.drain_requests(["consumer", "legacy"])
    engine._layerwise_put_queue = None
    engine.track_prefill_retrieve_keys("none", ())
