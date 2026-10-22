# SPDX-License-Identifier: Apache-2.0
"""Checkpoint preparation work counts and boundary-preserving CPU probes."""

import ast
import abc
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_checkpoint_writeback import deployment as checkpoint_deployment
from test_checkpoint_page_keys import key_types as checkpoint_key_types
from test_local_checkpoint_restore import implementation
from test_checkpoint_allocation_budget import engine_fixture
from test_preemption_checkpoint import (
    Page,
    api as checkpoint_api,
    fake_engine,
    start_capture,
)

api = checkpoint_api
deployment = checkpoint_deployment
key_types = checkpoint_key_types


def test_capture_validates_each_touched_block_once(api):
    class Blocks(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    engine = fake_engine()

    def unavailable(*args):
        raise MemoryError("stop before device preparation")

    engine.allocate_checkpoint_fragment = unavailable
    worker = api[1].CheckpointWorker(engine)
    blocks = Blocks(range(1, 783))
    try:
        worker.capture(
            api[0].CaptureSpec("r", 1, 0, 100000, 0, (blocks, blocks)),
            {0: [1], 1: [2]},
            128,
        )
        assert worker.poll()[0].status == "failed"
        assert blocks.reads == 2 * 782
        assert not engine.calls
    finally:
        worker.close()


def test_block_validation_matches_original_token_predicate_at_boundaries():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/preemption_checkpoint.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    check = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "Checkpoint source includes released/null blocks" in ast.unparse(n)
        and "len(blocks)" in ast.unparse(n.test)
    )
    predicate = compile(ast.Expression(check.test), str(path), "eval")
    for end in range(17):
        for start in range(end + 1):
            for size in range(6):
                for null in range(-1, size):
                    blocks = [0 if i == null else i + 1 for i in range(size)]
                    stop = (end + 3) // 4
                    expected = stop > len(blocks) or any(
                        blocks[i // 4] <= 0 for i in range(start, end)
                    )
                    assert (
                        eval(
                            predicate,
                            dict(
                                stop=stop,
                                start=start,
                                spec=NS(end=end),
                                block_size=4,
                                blocks=blocks,
                            ),
                        )
                        == expected
                    )


def test_ordered_source_selector_preserves_spanning_fragments_and_holes(api):
    local = api[1].LocalCheckpointStore(fake_engine())
    sources = [(NS(start=a, end=b), object()) for a, b in ((0, 3), (3, 10), (12, 14))]
    select = local._source_selector(sources)
    for start, end in ((0, 4), (4, 8), (8, 12), (12, 14), (14, 16)):
        assert select(start, end) == [
            (s, p) for s, p in sources if s.start < end and s.end > start
        ]


def test_capture_passes_physical_pages_without_layer_wrappers(api, monkeypatch):
    engine = fake_engine((2, 3))
    prepare = engine.gpu_connector.prepare_group_capture

    def pages_only(rows, *args, **kwargs):
        assert all(isinstance(page, Page) for row in rows for page in row)
        return prepare(rows, *args, **kwargs)

    engine.gpu_connector.prepare_group_capture = pages_only
    worker, _, _ = start_capture(api, monkeypatch, engine)
    try:
        assert worker.poll()[0].status == "captured"
    finally:
        worker.close()


def test_capture_uses_one_real_allocator_preparation_per_group(api, monkeypatch):
    engine = fake_engine()
    allocator, local, _ = engine_fixture(False)
    preparations, widths = [], []
    allocator._ensure_layerwise_connector_layout = lambda **kw: preparations.append(
        kw["kv_group"]
    )
    allocator.gpu_connector.checkpoint_plane_widths = lambda g: (
        widths.append(g),
        (3,) if g == 0 else (1,),
    )[1]
    local.batched_allocate_layer_pages = lambda *a, **kw: [
        Page(a[3], kw["valid_tokens"], (3,) if a[4] == "group0" else (1,))
    ]
    engine.prepare_checkpoint_allocator = allocator.prepare_checkpoint_allocator
    worker, _, _ = start_capture(api, monkeypatch, engine, end=256, prefix=0)
    try:
        assert worker.poll()[0].status == "captured"
        assert preparations == widths == [0, 1]
        assert (
            sum(len(pages) for pages in worker.jobs["r", 1].fragments.values()) == 128
        )
    finally:
        worker.close()


@pytest.mark.parametrize("operation", ["spill", "normalize"])
def test_page_building_receives_only_overlapping_sources(
    deployment, monkeypatch, operation
):
    _, create, _ = deployment
    chunks = 64
    worker = create(end=chunks * 4 + 1, prefix=0, accepted=chunks * 4)
    local = worker.local
    build_page = local.build_page
    source_counts = []

    def counted(*args):
        source_counts.append(len(args[1]))
        return build_page(*args)

    monkeypatch.setattr(local, "build_page", counted)
    if operation == "spill":
        record = local.records["r", 1]
        assert local._spill(record, record.groups)
    else:
        _, owners = local.normalize("r", 1, list(range(chunks * 4)), None)
        for page in owners:
            page.ref_count_down()
    assert len(source_counts) == 2 * chunks
    assert sum(source_counts) == 2 * chunks


@pytest.mark.parametrize("byte_hash", [False, True])
@pytest.mark.parametrize("length", [1, 4, 5, 11])
def test_checkpoint_hash_reuse_preserves_exact_group_keys(key_types, byte_hash, length):
    key_cls = key_types[0]
    key_cls.__post_init__.__globals__["TORCH_DTYPE_TO_STR_DTYPE"][torch.bfloat16] = (
        "bf16"
    )
    token_base = implementation(
        "../LMCache-NPU/lmcache/v1/token_database.py",
        "TokenDatabase",
        {"_make_key_by_hash", "_hash_tokens", "_canonicalize_hash_inputs"},
        object,
        abc=abc,
        CacheEngineKey=key_cls,
        NONE_HASH=0,
        MOONCAKE_PAYLOAD_LAYOUT_TAG="lmcache.tag.payload_v3",
        MOONCAKE_VALID_TOKENS_TAG="lmcache.tag.internal.valid_tokens",
        DSA_INDEX_CACHE_SCHEMA_TAG="lmcache.tag.dsa_idx",
        DSA_INDEX_CACHE_SCHEMA="v2",
    )
    database = implementation(
        "../LMCache-NPU/lmcache/v1/token_database.py",
        "ChunkedTokenDatabase",
        {"process_tokens", "_prefix_hash", "_chunk_tokens", "_get_init_hash"},
        token_base,
        NONE_HASH=0,
        _lmcache_nvtx_annotate=lambda f: f,
    )()
    database.chunk_size = 4
    database.config = NS(chunk_size=4, save_unfull_chunk=True, dsa_two_groups=True)
    database.metadata = NS(
        model_name="model",
        worker_id=0,
        world_size=4,
        get_dtypes=lambda: [torch.bfloat16, torch.uint8],
    )
    database.mooncake_payload_layout = "test-layout"
    database.save_only_first_rank = True
    hashed = []

    def digest(value):
        hashed.append(value)
        raw = hashlib.sha256(repr(value).encode()).digest()
        return raw if byte_hash else int.from_bytes(raw, "big")

    database.hash_func = digest
    base = implementation(
        "../LMCache-NPU/lmcache/v1/cache_engine.py",
        "LMCacheEngine",
        {"_dense_retrieve_token_results"},
        object,
        _SHARED_CPU_CHUNK_PLAN_KEY="plan",
        _PREFILL_REUSE_DEBUG_KEY="debug",
    )
    engine = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"checkpoint_token_results"},
        base,
    )()
    engine.token_database = database
    # New operation/history must not consume the previous operation's hashes.
    for offset in (0, 100):
        tokens = list(range(offset, offset + length))
        configs = {"lmcache.tag.tenant": str(offset)}
        hashed.clear()
        expected = [
            list(
                database.process_tokens(
                    tokens=tokens, request_configs=configs, kv_group=g
                )
            )
            for g in (0, 1)
        ]
        assert len(hashed) == 2 * ((length + 3) // 4)
        hashed.clear()
        state = {}
        actual = [
            list(engine.checkpoint_token_results(tokens, configs, g, state))
            for g in (0, 1)
        ]
        assert actual == expected
        assert len(hashed) == (length + 3) // 4
        assert actual[0][-1][2].dtype == torch.bfloat16
        assert actual[1][-1][2].dtype == torch.uint8
        assert dict(actual[1][-1][2].tags)["dsa_idx"] == "v2"
