# SPDX-License-Identifier: Apache-2.0
"""Exercise dense page planning with actual physical/layer key identities."""

from threading import Lock
from types import SimpleNamespace as NS

import pytest
import torch

from test_checkpoint_page_keys import key_types as production_key_types, use_real_keys
from test_local_checkpoint_restore import implementation
from test_preemption_checkpoint import Page, fake_engine

key_types = production_key_types


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("count", [3, 4, 7, 8])
@pytest.mark.parametrize("skip", [0, 8])
@pytest.mark.parametrize("tail_source", ["page", "remote", "legacy", "missing"])
def test_dense_loader_plans_full_and_partial_local_pages(
    key_types, group, count, skip, tail_source
):
    engine = fake_engine()
    use_real_keys(engine, key_types)
    candidates = [
        entry
        for entry in engine.token_database.process_tokens(
            tokens=list(range(skip + count)), kv_group=group
        )
        if entry[0] >= skip
    ]
    for start, end, key in candidates:
        engine.backend.pages[key] = Page(2, end - start, (2, 1) if group == 0 else (1,))
    last_key = candidates[-1][2]
    if tail_source != "page":
        engine.backend.pages.pop(last_key)
    if tail_source == "legacy":
        engine.backend.pages.update({key: object() for key in last_key.split_layers(2)})
    engine.backend.cpu_lock = Lock()
    engine.backend.hot_cache = engine.backend.pages
    base_key, layer_key = key_types
    cls = implementation(
        "../LMCache-NPU/lmcache/v1/cache_engine.py",
        "LMCacheEngine",
        {"retrieve_layer", "_shared_page_first_location_plan"},
        object,
        _lmcache_nvtx_annotate=lambda fn: fn,
        serving_perf_enabled=lambda: False,
        mooncake_page_layout_enabled=lambda config: True,
        mooncake_layer_pages_enabled=lambda config: True,
        CacheEngineKey=base_key,
        LayerCacheEngineKey=layer_key,
        LayerPageMemoryObj=Page,
        _RemoteFillMaterializationError=type("RemoteFillError", (RuntimeError,), {}),
    )
    obj = cls()
    obj.num_layers, obj.config = 2, engine.config
    obj.is_healthy = lambda: True
    obj._is_passive = lambda: False
    obj._should_use_shared_layerwise_retrieve = lambda group: True
    obj._get_req_id = lambda kw: kw["req_id"]
    obj._dense_retrieve_token_results = lambda *a: candidates
    obj._remote_fill_retrieve_plan = lambda *a: None
    obj._shared_local_cpu_backend = lambda: engine.backend
    obj.gpu_connector = object()
    obj.stats_monitor = NS(on_retrieve_request=lambda count: "monitor")
    obj.shared_cpu_cache_strict = True
    obj.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
    legacy_probes = []

    def legacy_lookup(keys, locations):
        legacy_probes.append(keys)
        # Real storage lookup can recognize a page through a layer alias.
        return len(keys), {"LocalCPUBackend": keys}

    def remote_lookup(keys, locations):
        count = int(tail_source == "remote" and keys == [last_key])
        return count, {"RemoteBackend": keys} if count else {}

    remote = NS(connection=NS(batched_contains_layer_pages=lambda keys: None))
    obj.storage_manager = NS(
        batched_contains=legacy_lookup,
        batched_contains_layer_pages=remote_lookup,
        get_active_storage_backends=lambda **kw: [
            ("LocalCPUBackend", engine.backend),
            ("RemoteBackend", remote),
        ],
    )

    def find(key):
        base = key.without_layer() if isinstance(key, layer_key) else key
        if base in engine.backend.pages or key in engine.backend.pages:
            return "LocalCPUBackend"
        return "RemoteBackend" if base == last_key and tail_source == "remote" else None

    obj._find_shared_rank0_chunk_location = find
    selected = []

    def materialize(**kw):
        selected.append(kw)
        yield kw["ret_mask"]

    obj._retrieve_layer_shared_rank0 = materialize
    mask = torch.arange(skip + count) >= skip
    result = list(
        obj.retrieve_layer(
            list(range(skip + count)),
            mask,
            req_id="checkpoint",
            kv_group=group,
            shared_cpu_phase="dsa_cold_compact_indexer",
        )
    )[-1]
    expected = mask.clone()
    if tail_source == "missing":
        expected[candidates[-1][0] :] = False
    assert torch.equal(result, expected)
    page_count = len(candidates) - int(tail_source in {"legacy", "missing"})
    assert selected[0]["planned_page_chunks"] == page_count, (
        "partial merged page was routed to the legacy per-layer loader"
    )
    assert not legacy_probes
    kept = candidates[:-1] if tail_source == "missing" else candidates
    expected_keys = (
        [
            [
                key if i < page_count else key.get_layer(layer)
                for i, (_, _, key) in enumerate(kept)
            ]
            for layer in range(2)
        ]
        if kept
        else []
    )
    assert selected[0]["keys_layer_major"] == expected_keys
