# SPDX-License-Identifier: Apache-2.0
"""Repeated local checkpoints use real chunk hashing and counted CPU owners."""

from dataclasses import replace
from types import SimpleNamespace as NS
import ast
import gc
import threading

from test_checkpoint_page_keys import key_types as key_types_fixture
from test_preemption_checkpoint import (
    ROOT,
    Page,
    fake_engine,
    fill,
    finish,
    publish,
    start_capture,
)
from test_preemption_checkpoint import (
    api as checkpoint_api,
)
import pytest
import torch

api = checkpoint_api
key_types = key_types_fixture


def token_database(key_cls):
    """Run production chunk/key methods without importing the device runtime."""
    path = ROOT.parent / "LMCache-NPU/lmcache/v1/token_database.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_make_key_by_hash",
        "_canonicalize_hash_inputs",
        "_hash_tokens",
        "_get_init_hash",
        "_chunk_tokens",
        "_prefix_hash",
        "process_tokens",
        "process_tokens_from_prefix",
    }
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {
            "TokenDatabase",
            "ChunkedTokenDatabase",
        }:
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    node.decorator_list = []
                    methods[node.name] = node
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods.values(),
        ],
        type_ignores=[],
    )
    ns = dict(
        torch=torch,
        CacheEngineKey=key_cls,
        NONE_HASH=0,
        MOONCAKE_PAYLOAD_LAYOUT_TAG="lmcache.tag.payload_v3",
        MOONCAKE_VALID_TOKENS_TAG="lmcache.tag.internal.valid_tokens",
        DSA_INDEX_CACHE_SCHEMA_TAG="lmcache.tag.dsa_idx",
        DSA_INDEX_CACHE_SCHEMA="v2",
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    db = type("ChunkDB", (), {name: ns[name] for name in methods})()
    db.chunk_size = 4
    db.config = NS(chunk_size=4, save_unfull_chunk=True, dsa_two_groups=True)
    db.metadata = NS(
        model_name="model",
        world_size=4,
        worker_id=0,
        get_dtypes=lambda: [torch.uint8, torch.uint8],
    )
    db.mooncake_payload_layout, db.save_only_first_rank = "test", True
    db.hashed = []
    db.hash_func = lambda value: db.hashed.append(value) or hash(value)
    return db


def active_state(engine, end):
    plans = list(engine.token_database.process_tokens(tokens=list(range(end))))
    engine.shared_cpu_cache_generation = 7
    return NS(
        req_id="r",
        shared_request_active=True,
        shared_generation=7,
        pointer_cache_generation=7,
        indexer_npu_resident=True,
        indexer_npu_materialization_pending=False,
        token_count=end,
        prepared_sparse_sources={0: NS(total_tokens=end)},
        cached_starts=[a for a, b, k in plans],
        cached_ends=[b for a, b, k in plans],
        cached_keys=[[k.split_layers(2)[0] for a, b, k in plans]],
    )


def setup_saved(api, monkeypatch, key_types, prefix=6, restored=13):
    engine = fake_engine()
    engine.token_database = token_database(key_types[0])
    engine.touches = []
    engine.backend.try_touch_layer_pages = (
        lambda keys: engine.touches.append(tuple(keys)) or True
    )
    store, spec, _ = start_capture(
        api, monkeypatch, engine, end=restored + 1, prefix=prefix
    )
    store.poll()
    publish(api, store, restored)
    _, owners = store.local.normalize("r", 1, list(range(restored)), None)
    for page in owners:
        page.ref_count_down()
    return engine, store, spec


def capture_again(api, store, spec, state, end=19):
    store.cancel(spec.req_id, spec.generation)
    spec = replace(
        spec,
        generation=spec.generation + 1,
        end=end,
        resident_start=state.token_count,
        blocks=(tuple(range(1, (end + 3) // 4 + 1)),) * 2,
    )
    store.capture(spec, {0: [1], 1: [2]}, 4, state)
    result = store.poll()
    assert result[0].status == "captured", result
    return spec, result[0].end


def assert_bytes(engine, end, base=4):
    for group in (0, 1):
        for start, stop, key in engine.token_database.process_tokens(
            tokens=list(range(end)), kv_group=group
        ):
            if start < base:
                continue
            expected = Page(2, stop - start, (2, 1) if group == 0 else (1,))
            fill(expected, start, group)
            assert torch.equal(engine.backend.pages[key].raw_data, expected.raw_data)


@pytest.mark.parametrize(
    "missing_start,expected_start", [(None, 12), (4, 4), (8, 8), (12, 12)]
)
def test_reuses_closed_index_pages_and_repairs_first_hole(
    api, monkeypatch, key_types, missing_start, expected_start
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    keys = list(
        engine.token_database.process_tokens(tokens=list(range(13)), kv_group=1)
    )
    if missing_start is not None:
        engine.backend.remove(next(k for a, b, k in keys if a == missing_start))
    spec, end = capture_again(api, store, spec, state)
    job = store.jobs["r", 2]
    assert [(a, b) for a, b, p, w in job.fragments[0]] == [(13, 16), (16, 19)]
    assert job.fragments[1][0][0] == expected_start
    store.seal(api[0].SealSpec("r", 2, tuple(range(18))))
    assert finish(store)[0].end == 18
    _, owners = store.local.normalize("r", 2, list(range(18)), None)
    assert_bytes(engine, 18)
    for page in owners:
        page.ref_count_down()
    store.close()
    assert all(p.refs <= 1 for p in engine.allocated)


@pytest.mark.parametrize("refuse", [False, True])
def test_whole_reuse_or_refused_extension_keeps_old_partial_without_dma(
    api, monkeypatch, key_types, refuse
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    before = len(engine.calls)
    if refuse:
        engine.allocate_checkpoint_fragment = lambda *args: (_ for _ in ()).throw(
            MemoryError("pressure")
        )
    spec, end = capture_again(api, store, spec, state, 19 if refuse else 13)
    assert end == 13 and engine.calls[before:] == []
    engine.backend.before_put = lambda: pytest.fail("redundant CPU page admission")
    store.seal(api[0].SealSpec("r", 2, tuple(range(13))))
    assert finish(store)[0].end == 13
    assert store.local.available("r", 2, 13) == 13
    store.close()
    assert all(p.refs <= 1 for p in engine.allocated)


@pytest.mark.parametrize("seal_end", [7, 11, 13, 18])
def test_incremental_hashes_match_full_history_even_for_shorter_seal(
    api, monkeypatch, key_types, seal_end
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    spec, _ = capture_again(api, store, spec, active_state(engine, 13))
    store.seal(api[0].SealSpec("r", 2, tuple(range(seal_end))))
    assert finish(store)[0].end == seal_end
    engine.token_database.hashed.clear()
    _, owners = store.local.normalize("r", 2, list(range(seal_end)), None)
    hashed = sum(len(value[1]) for value in engine.token_database.hashed)
    assert hashed == seal_end - min(seal_end // 4 * 4, 12), engine.token_database.hashed
    assert_bytes(engine, seal_end)
    for page in owners:
        page.ref_count_down()
    store.close()


def test_three_generations_advance_frontier_without_reading_released_slots(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    for restored, next_end in [(13, 18), (18, 22)]:
        spec, _ = capture_again(
            api, store, spec, active_state(engine, restored), next_end
        )
        job = store.jobs["r", spec.generation]
        assert job.fragments[0][0][0] == restored
        assert job.fragments[1][0][0] == restored // 4 * 4
        store.seal(api[0].SealSpec("r", spec.generation, tuple(range(next_end))))
        assert finish(store)[0].end == next_end
        _, owners = store.local.normalize(
            "r", spec.generation, list(range(next_end)), None
        )
        assert_bytes(engine, next_end)
        for page in owners:
            page.ref_count_down()
    store.close()


def test_reused_sources_are_owned_until_cancel_without_gc(api, monkeypatch, key_types):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    spec, _ = capture_again(api, store, spec, active_state(engine, 13))
    keys = [
        k
        for a, b, k in engine.token_database.process_tokens(
            tokens=list(range(13)), kv_group=1
        )
        if 4 <= a < 12
    ]
    assert all(not engine.backend.evict(k) for k in keys)
    enabled = gc.isenabled()
    gc.disable()
    try:
        store.cancel("r", spec.generation)
    finally:
        if enabled:
            gc.enable()
    assert all(engine.backend.evict(k) for k in keys)
    store.close()


@pytest.mark.parametrize("missing_start,expected_end", [(4, 6), (8, 8), (12, 12)])
def test_missing_nonresident_latent_caps_checkpoint_without_hbm_read(
    api, monkeypatch, key_types, missing_start, expected_end
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    key = next(
        k
        for a, b, k in engine.token_database.process_tokens(tokens=list(range(13)))
        if a == missing_start
    )
    engine.backend.remove(key)
    before = len(engine.calls), len(engine.allocated)
    # All old latent slots are deliberately invalid; no repair may read them.
    spec = replace(
        spec,
        generation=2,
        end=19,
        resident_start=13,
        blocks=((0,) * 5, tuple(range(1, 6))),
    )
    store.capture(spec, {0: [1], 1: [2]}, 4, state)
    result = store.poll()[0]
    assert (len(engine.calls), len(engine.allocated)) == before
    if expected_end == 6:
        assert result.status == "failed"
    else:
        assert result.status == "captured" and result.end == expected_end
        store.seal(api[0].SealSpec("r", 2, tuple(range(expected_end))))
        assert finish(store)[0].end == expected_end
        assert store.local.available("r", 2, expected_end) == expected_end
    store.close()
    assert all(p.refs <= 1 for p in engine.allocated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("req_id", "another"),
        ("shared_generation", 8),
        ("pointer_cache_generation", 8),
        ("shared_request_active", False),
        ("indexer_npu_resident", False),
        ("indexer_npu_materialization_pending", True),
    ],
)
def test_invalid_reuse_state_fails_before_capture(
    api, monkeypatch, key_types, field, value
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    setattr(state, field, value)
    before = len(engine.calls), len(engine.allocated)
    store.capture(
        replace(spec, generation=2, resident_start=13), {0: [1], 1: [2]}, 4, state
    )
    assert store.poll()[0].status == "failed"
    assert (len(engine.calls), len(engine.allocated)) == before
    store.close()


def test_checkpoint_recency_includes_reused_pages_and_tail_is_oldest(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    spec, _ = capture_again(api, store, spec, state)
    store.seal(api[0].SealSpec("r", 2, tuple(range(18))))
    finish(store)
    manifest = store.local.manifest("r", 2)
    by_key = {p.key: p.start for group in manifest.groups for p in group}
    assert [by_key[k] for k in engine.touches[-1]] == sorted(
        by_key.values(), reverse=True
    )
    assert any(not isinstance(k, tuple) for k in engine.touches[-1])
    _, owners = store.local.normalize("r", 2, list(range(18)), None)
    expected = [
        p.key
        for p in sorted(
            (p for g in store.local.manifest("r", 2).groups for p in g),
            key=lambda p: p.start,
            reverse=True,
        )
    ]
    assert list(engine.touches[-1]) == expected
    for page in owners:
        page.ref_count_down()
    store.close()


@pytest.mark.parametrize(
    "prefix,restored,end", [(0, 3, 5), (3, 6, 9), (4, 8, 12), (6, 13, 19), (9, 17, 18)]
)
@pytest.mark.parametrize("pressure", [False, True])
def test_boundary_and_fragmentation_matrix(
    api, monkeypatch, key_types, prefix, restored, end, pressure
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types, prefix, restored)
    state = active_state(engine, restored)
    if pressure:
        original = engine.allocate_checkpoint_fragment
        engine.allocate_checkpoint_fragment = (
            lambda group, count, caches=None: original(group, count, caches)
            if count <= 2
            else (_ for _ in ()).throw(MemoryError("fragmented"))
        )
    spec, captured = capture_again(api, store, spec, state, end)
    assert restored <= captured <= end
    store.seal(api[0].SealSpec("r", 2, tuple(range(captured))))
    assert finish(store)[0].end == captured
    # Canonical assembly can use normal workspace after fragmented capture.
    if pressure:
        engine.allocate_checkpoint_fragment = original
    _, owners = store.local.normalize("r", 2, list(range(captured)), None)
    assert_bytes(engine, captured, prefix // 4 * 4)
    for page in owners:
        page.ref_count_down()
    store.close()
    assert all(p.refs <= 1 for p in engine.allocated)


def test_missing_seed_falls_back_to_one_full_hash_and_window_save_keeps_capture(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    spec = replace(
        spec, generation=2, end=18, resident_start=13, blocks=(tuple(range(1, 6)),) * 2
    )
    store.capture(spec, {0: [1], 1: [2]}, 4, state, reuse_prefix=False)
    assert store.poll()[0].status == "captured"
    job = store.jobs["r", 2]
    assert job.fragments[1][0][0] == spec.base
    assert job.key_seed is None and not job.source_owners
    store.seal(api[0].SealSpec("r", 2, tuple(range(18))))
    finish(store)
    engine.token_database.hashed.clear()
    _, owners = store.local.normalize("r", 2, list(range(18)), None)
    assert sum(len(value[1]) for value in engine.token_database.hashed) == 18
    assert_bytes(engine, 18)
    for page in owners:
        page.ref_count_down()
    store.close()


def test_cancel_during_publication_keeps_borrowed_pages_until_writer_finishes(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    spec, _ = capture_again(api, store, spec, state)
    # The adapter clears these mutable lists after capture returns.
    state.cached_starts.clear()
    state.cached_ends.clear()
    state.cached_keys.clear()
    job = store.jobs["r", 2]
    owners = list(job.source_owners)
    entered, release = threading.Event(), threading.Event()

    def wait():
        entered.set()
        assert release.wait(3)

    engine.backend.before_put = wait
    store.seal(api[0].SealSpec("r", 2, tuple(range(18))))
    try:
        assert entered.wait(3)
        store.cancel("r", 2)
        assert all(page.refs > 1 for page in owners)
    finally:
        release.set()
        finish(store)
        store.close()
    assert store.local.manifest("r", 2) is None
    assert all(page.refs <= 1 for page in engine.allocated)


def test_missing_hash_seed_and_bad_reuse_page_never_fabricate_coverage(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    spec, _ = capture_again(api, store, spec, state)
    job = store.jobs["r", 2]
    # Force the documented no-seed fallback independently of capture reuse.
    job.key_seed = None
    store.seal(api[0].SealSpec("r", 2, tuple(range(18))))
    finish(store)
    engine.token_database.hashed.clear()
    _, owners = store.local.normalize("r", 2, list(range(18)), None)
    assert sum(len(value[1]) for value in engine.token_database.hashed) == 18
    assert_bytes(engine, 18)
    for page in owners:
        page.ref_count_down()
    state = active_state(engine, 18)
    key = next(
        k
        for a, b, k in engine.token_database.process_tokens(
            tokens=list(range(18)), kv_group=1
        )
        if a == 8
    )
    engine.backend.pages[key].valid_tokens -= 1
    before = len(engine.calls)
    store.capture(
        replace(spec, generation=3, resident_start=18), {0: [1], 1: [2]}, 4, state
    )
    assert store.poll()[0].status == "failed" and len(engine.calls) == before
    store.close()
    assert all(page.refs <= 1 for page in engine.allocated)


def test_first_preemption_uses_hash_seed_without_empty_cache_lock_probes(
    api, monkeypatch, key_types
):
    engine = fake_engine()
    engine.token_database = token_database(key_types[0])
    state = active_state(engine, 6)
    engine.backend.batched_get_layer_page_prefix = lambda keys: pytest.fail(
        "first preemption has no generated CPU prefix to probe"
    )
    store, _, _ = start_capture(api, monkeypatch, engine, end=10, prefix=6, state=state)
    assert store.poll()[0].status == "captured"
    assert store.jobs["r", 1].key_seed[0] == 4
    store.close()
