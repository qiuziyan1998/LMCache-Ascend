# SPDX-License-Identifier: Apache-2.0
"""Use production key identities at the checkpoint/page-loader boundary."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace as NS

import pytest
import torch

from test_local_checkpoint_restore import ROOT, implementation
from test_preemption_checkpoint import (
    Page,
    api as checkpoint_api,
    fake_engine,
    fill,
    finish,
    publish,
    start_capture,
)

api = checkpoint_api


@pytest.fixture
def key_types():
    source = ROOT.parent / "LMCache-NPU/lmcache/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"CacheEngineKey", "LayerCacheEngineKey"}
    ]
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        torch=torch,
        TORCH_DTYPE_TO_STR_DTYPE={torch.uint8: "uint8"},
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, *nodes], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        ns,
    )
    return ns["CacheEngineKey"], ns["LayerCacheEngineKey"]


def use_real_keys(engine, key_types):
    key_cls, _ = key_types

    def tokens(*, tokens, request_configs=None, kv_group=0):
        for start in range(0, len(tokens), 4):
            end = min(start + 4, len(tokens))
            configs = dict(request_configs or {}, **{"lmcache.tag.payload_v3": "test"})
            if end - start < 4:
                configs["lmcache.tag.internal.valid_tokens"] = end - start
            yield (
                start,
                end,
                key_cls(
                    "model",
                    1,
                    0,
                    bytes(tokens[:end]),
                    torch.uint8,
                    configs,
                    kv_group=kv_group,
                ),
            )

    engine.token_database.process_tokens = tokens


@pytest.mark.parametrize("group", [0, 1])
def test_normalized_checkpoint_uses_the_ordinary_page_key(
    api, monkeypatch, key_types, group
):
    engine = fake_engine()
    use_real_keys(engine, key_types)
    store, _, _ = start_capture(api, monkeypatch, engine=engine)
    store.poll()
    publish(api, store, 11)
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    try:
        for start, end, key in engine.token_database.process_tokens(
            tokens=list(range(11)), kv_group=group
        ):
            reader_key = key.split_layers(engine.num_layers)[0].without_layer()
            assert key != key.split_layers(engine.num_layers)[0]
            pages, count = engine.backend.batched_get_layer_page_prefix([reader_key])
            assert count == 1, (
                "checkpoint page is invisible to the ordinary page loader"
            )
            expected = Page(2, end - start, (2, 1) if group == 0 else (1,))
            fill(expected, start, group)
            assert torch.equal(pages[0].raw_data, expected.raw_data)
            pages[0].ref_count_down()
            expected.ref_count_down()
        assert store.local.available("r", 1, 11) == 11
    finally:
        for page in owners:
            page.ref_count_down()
        store.close()


@pytest.mark.parametrize("group", [0, 1])
def test_cached_original_boundary_needs_no_allocation(key_types, group):
    engine = fake_engine()
    use_real_keys(engine, key_types)
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"get_checkpoint_prefix", "load_checkpoint_prefix"},
        object,
    )
    obj = cls()
    obj.num_layers = engine.num_layers
    obj.token_database = engine.token_database
    obj._shared_local_cpu_backend = lambda: engine.backend
    obj._shared_cpu_dtype_for_kv_group = lambda group: torch.uint8
    obj._memory_format_for_kv_group = lambda group: "test"
    obj.gpu_connector = NS(
        checkpoint_plane_widths=lambda group: (2, 1) if group == 0 else (1,)
    )
    obj.allocate_checkpoint_fragment = lambda *a: pytest.fail(
        "unnecessary staging allocation"
    )
    key = list(engine.token_database.process_tokens(tokens=[0, 1, 2], kv_group=group))[
        -1
    ][2]
    page = Page(2, 3, (2, 1) if group == 0 else (1,))
    engine.backend.pages[key] = page
    result = obj.load_checkpoint_prefix((0, 1, 2), group, None)
    assert result is page and page.refs == 2
    result.ref_count_down()
    assert page.refs == 1


@pytest.mark.parametrize("backend", ["ascend", "base"])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("placement", ["LLL", "RRR", "LRL", "RRL", "RRM"])
def test_page_source_selection_and_failure_cleanup(
    key_types, group, placement, backend
):
    engine = fake_engine()
    use_real_keys(engine, key_types)

    class MergedPage(Page):
        def get_shape(self):
            return (self.valid_tokens,)

        def pin(self):
            self.pins += 1

        def unpin(self):
            self.pins -= 1

        @staticmethod
        def pin_many(pages):
            for page in pages:
                page.pin()
            return True

    _, layer_key = key_types
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py"
        if backend == "ascend"
        else "../LMCache-NPU/lmcache/v1/cache_engine.py",
        "AscendLMCacheEngine" if backend == "ascend" else "LMCacheEngine",
        {"_resolve_shared_rank0_layer_pages"},
        object,
        serving_perf_enabled=lambda: False,
        LayerCacheEngineKey=layer_key,
        LayerPageMemoryObj=MergedPage,
        nullcontext=nullcontext,
        mooncake_valid_tokens=lambda key, chunk: (key.request_configs or {}).get(
            "lmcache.tag.internal.valid_tokens", chunk
        ),
    )
    obj = cls()
    obj.num_layers = engine.num_layers
    obj.config = engine.config
    keys = [
        entry[2]
        for entry in engine.token_database.process_tokens(
            tokens=list(range(11)), kv_group=group
        )
    ]
    pages = [
        MergedPage(2, count, (2, 1) if group == 0 else (1,)) for count in (4, 4, 3)
    ]
    for page in pages:
        page.pins = 0
    for key, page, location in zip(keys, pages, placement):
        if location == "L":
            engine.backend.pages[key] = page
    engine.backend.contains_all_exact = lambda keys: False
    obj._shared_local_cpu_backend = lambda: engine.backend
    fetched_keys = []
    local_calls = []
    local_get = engine.backend.batched_get_layer_page_prefix

    def tracked_local_get(requested):
        local_calls.append(requested)
        return local_get(requested)

    engine.backend.batched_get_layer_page_prefix = tracked_local_get

    def remote_contains(requested):
        return next(
            (i for i, key in enumerate(requested) if placement[keys.index(key)] != "R"),
            len(requested),
        )

    def remote_get(requested):
        fetched_keys.extend(requested)
        assert all(placement[keys.index(key)] == "R" for key in requested), (
            "local generated KV must never be read from Mooncake"
        )
        return [pages[keys.index(key)] for key in requested]

    def legacy(**kw):
        assert placement == "RRM", (
            "local checkpoint tail reached legacy remote fallback"
        )
        raise ValueError("missing persistent page")

    obj.storage_manager = NS(
        storage_backends={
            "RemoteBackend": NS(
                batched_contains_layer_pages=remote_contains,
                batched_get_layer_pages=remote_get,
            )
        }
    )
    obj._resolve_shared_rank0_page_first_layers = legacy
    obj._expected_shared_cpu_chunk_metadata = lambda **kw: (
        (kw["num_tokens"],),
        None,
        None,
    )
    obj._validate_rank0_shared_mem_obj = lambda *a, **kw: None
    obj._release_shared_retrieve_objs = lambda objects, **kw: [
        p.ref_count_down() for p in objects
    ]
    args = dict(
        req_id="r",
        phase="dsa_cold_compact_latent",
        kv_group=group,
        keys_layer_major=[[key.get_layer(layer) for key in keys] for layer in range(2)],
        page_chunks=3,
    )
    if placement == "RRM":
        with pytest.raises(ValueError, match="missing persistent page"):
            obj._resolve_shared_rank0_layer_pages(**args)
        assert [p.refs for p in pages] == [0, 0, 1]
        assert all(p.pins == 0 for p in pages)
        return
    result, count = obj._resolve_shared_rank0_layer_pages(**args)
    assert count == 3 and result == [pages, pages]
    assert fetched_keys == [key for key, place in zip(keys, placement) if place == "R"]
    # Complete local/remote page hits retain the original single local lookup.
    assert len(local_calls) == (1 if placement in {"LLL", "RRR"} else 2)
    for page in result[0]:
        page.unpin()
        page.ref_count_down()
    assert [p.refs for p in pages] == [int(place == "L") for place in placement]
    assert all(p.pins == 0 for p in pages)


def test_boundary_allocation_refusal_is_bounded_and_keeps_checkpoint_data(
    api, monkeypatch, key_types
):
    engine = fake_engine()
    use_real_keys(engine, key_types)
    store, _, _ = start_capture(api, monkeypatch, engine=engine)
    store.poll()
    publish(api, store, 11)
    retained = [(page, page.raw_data.clone()) for page in engine.backend.pages.values()]
    calls = []

    def refuse(*args):
        calls.append("allocate")
        raise MemoryError("Checkpoint CPU staging allocation refused")

    # The original boundary is available; the new assembled page is not.
    original_prefix = engine.load_checkpoint_prefix((0, 1, 2), 0, None)
    engine.load_checkpoint_prefix = lambda *args: (
        original_prefix.ref_count_up(),
        original_prefix,
    )[1]
    engine.allocate_checkpoint_fragment = refuse
    engine.reclaim_checkpoint_capacity = lambda *args: calls.append("reclaim") or True
    with pytest.raises(
        api[0].CheckpointRestoreMiss, match="workspace is unavailable"
    ) as caught:
        store.local.normalize("r", 1, list(range(11)), None)
    assert caught.value.available_end == 0
    assert calls == ["allocate", "reclaim", "allocate"]
    assert original_prefix.refs == 1
    original_prefix.ref_count_down()
    assert all(
        page.refs == 1 and torch.equal(page.raw_data, before)
        for page, before in retained
    )
    assert store.local.available("r", 1, 11) == 11
    store.close()


def test_second_preemption_reuses_layer_key_metadata_as_physical_pages(
    api, monkeypatch, key_types
):
    control, module = api
    engine = fake_engine()
    use_real_keys(engine, key_types)
    first, _, _ = start_capture(api, monkeypatch, engine, end=14)
    first.poll()
    publish(api, first, 11)
    _, held = first.local.normalize("r", 1, list(range(11)), None)
    sources = list(engine.token_database.process_tokens(tokens=list(range(11))))
    state = NS(
        cached_starts=[a for a, b, key in sources],
        cached_ends=[b for a, b, key in sources],
        cached_keys=[
            [key.get_layer(layer) for a, b, key in sources] for layer in range(2)
        ],
    )
    second = module.CheckpointWorker(engine)
    restored = []
    try:
        second.capture(
            control.CaptureSpec(
                "r", 2, 0, 15, 11, (tuple(range(1, 5)),) * 2, prefix_end=3
            ),
            {0: [1], 1: [2]},
            4,
            state,
        )
        assert second.poll()[0].status == "captured"
        second.seal(control.SealSpec("r", 2, tuple(range(14))))
        assert [(result.status, result.end) for result in finish(second)] == [
            ("ready", 14)
        ]
        assert second.local.available("r", 2, 14) == 14
        _, restored = second.local.normalize("r", 2, list(range(14)), None)
        for group in (0, 1):
            for a, b, key in engine.token_database.process_tokens(
                tokens=list(range(14)), kv_group=group
            ):
                expected = Page(2, b - a, (2, 1) if group == 0 else (1,))
                fill(expected, a, group)
                assert torch.equal(
                    engine.backend.pages[key].raw_data, expected.raw_data
                )
                expected.ref_count_down()
    finally:
        for page in restored + held:
            page.ref_count_down()
        first.close()
        second.close()


def test_existing_normalized_boundaries_do_not_require_duplicate_workspace(
    api, monkeypatch, key_types
):
    engine = fake_engine()
    use_real_keys(engine, key_types)
    store, _, _ = start_capture(api, monkeypatch, engine)
    store.poll()
    publish(api, store, 11)
    # Another restore has already produced the exact canonical boundaries.
    for group in (0, 1):
        for a, b, key in engine.token_database.process_tokens(
            tokens=list(range(11)), kv_group=group
        ):
            page = Page(2, b - a, (2, 1) if group == 0 else (1,))
            fill(page, a, group)
            engine.backend.pages[key] = page

    def refuse(*args):
        raise MemoryError("no assembly workspace")

    engine.allocate_checkpoint_fragment = refuse
    engine.load_checkpoint_prefix = lambda *a: pytest.fail(
        "unused original boundary read"
    )
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    try:
        assert store.local.available("r", 1, 11) == 11
    finally:
        for page in owners:
            page.ref_count_down()
        assert all(page.refs == 1 for page in engine.backend.pages.values())
        store.close()
