# SPDX-License-Identifier: Apache-2.0
"""Group-1 reused hash plans must match independent passive token metadata."""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


@pytest.fixture
def engine() -> AscendLMCacheEngine:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4,
        save_unfull_chunk=True,
        experimental_sampled_layerwise_lookup=True,
    )
    metadata = LMCacheMetadata(
        model_name="model",
        world_size=2,
        local_world_size=2,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 1, 4, 1, 2),
        use_mla=True,
    )
    result = object.__new__(AscendLMCacheEngine)
    result.config, result.metadata = config, metadata
    result.num_layers = 2
    result.token_database = ChunkedTokenDatabase(config, metadata)
    result._should_use_shared_layerwise_retrieve = lambda _group: True
    result._is_shared_retrieve_passive = lambda _group: False
    return result


def resolve_groups(engine, prefix: int, length: int, snapshot: bool):
    tokens = list(range(length))
    cached = list(engine.token_database.process_tokens(tokens=tokens[:prefix]))
    keys = [
        list(row)
        for row in zip(
            *(key.split_layers(engine.num_layers) for _, _, key in cached),
            strict=True,
        )
    ]
    starts, ends = [s for s, _, _ in cached], [e for _, e, _ in cached]
    state = {}
    outputs = []
    for group in (0, 1):
        kwargs = {"kv_group": group, "shared_cpu_request_preflight_state": state}
        if snapshot:
            kwargs["cached_metadata_token_ids"] = tokens[:prefix]
        mask = torch.zeros(length, dtype=torch.bool)
        _, resolved_starts, resolved_ends, resolved_keys = (
            engine._ensure_retrieve_chunk_metadata(
                tokens=tokens,
                mask=None,
                request_configs=None,
                cached_keys=keys if group == 0 else [],
                cached_starts=starts if group == 0 else [],
                cached_ends=ends if group == 0 else [],
                ret_mask=mask,
                retrieve_kwargs=kwargs,
            )
        )
        outputs.append((resolved_starts, resolved_ends, resolved_keys))
        assert mask.all()
    return outputs


@pytest.mark.parametrize("prefix", [0, 4, 8])
@pytest.mark.parametrize("length", [12, 13])
@pytest.mark.parametrize("snapshot", [False, True])
def test_group_plan_matches_independent_token_hashing(
    engine,
    prefix: int,
    length: int,
    snapshot: bool,
) -> None:
    with patch.object(
        engine.token_database,
        "process_tokens_from_prefix",
        wraps=engine.token_database.process_tokens_from_prefix,
    ) as suffix:
        outputs = resolve_groups(engine, prefix, length, snapshot)
    assert suffix.call_count == int(prefix > 0 and snapshot and length % 4 == 0)
    for group, (starts, ends, keys) in enumerate(outputs):
        expected = list(
            engine.token_database.process_tokens(
                tokens=list(range(length)),
                kv_group=group,
            )
        )
        assert starts == [s for s, _, _ in expected]
        assert ends == [e for _, e, _ in expected]
        assert keys == [
            list(row)
            for row in zip(
                *(key.split_layers(engine.num_layers) for _, _, key in expected),
                strict=True,
            )
        ]


def test_rebuilt_plan_compact_pages_fit_passive_metadata(engine) -> None:
    # Group 0 retains one chunk, but no token snapshot for incremental hashing.
    # Group 1 is cold and consumes Group 0's newly published hash plan.
    _, (starts, ends, keys) = resolve_groups(engine, 4, 8, False)
    slab = torch.empty(65536, dtype=torch.uint8)
    owner = TensorMemoryAllocator(slab)
    pages = []
    views = ()
    fmt, dtype = MemoryFormat.KV_DSA_INDEX_FMT, torch.float16
    engine.shared_cpu_cache_name = "test"
    engine.shared_cpu_cache_passive_allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="test",
        generation=1,
    )
    engine._expected_shared_cpu_chunk_metadata = lambda *, kv_group, num_tokens: (
        torch.Size([num_tokens * 2]),
        dtype,
        fmt,
    )
    try:
        for chunk, (start, end) in enumerate(zip(starts, ends, strict=True)):
            allocated = owner.batched_allocate_layer_pages(
                [torch.Size([(end - start) * 2])],
                [dtype],
                1,
                engine.num_layers,
                fmt,
                valid_tokens=end - start,
                full_tokens=end - start,
            )
            assert allocated is not None
            page = allocated[0]
            pages.append(page)
            for layer in range(engine.num_layers):
                page.layer_tensor(layer).fill_(chunk * 10 + layer)
        batch = engine._make_shared_handle_batch(
            [list(pages) for _ in range(engine.num_layers)],
            keys,
        )
        assert batch is not None
        passive = list(
            engine.token_database.process_tokens(
                tokens=list(range(8)),
                kv_group=1,
            )
        )
        views = engine._make_passive_layer_page_views(
            batch,
            starts=[s for s, _, _ in passive],
            ends=[e for _, e, _ in passive],
            keys_layer_major=[
                list(row)
                for row in zip(
                    *(key.split_layers(engine.num_layers) for _, _, key in passive),
                    strict=True,
                )
            ],
            kv_group=1,
        )
        assert len(views) == len(passive) == 2
        for chunk, view in enumerate(views):
            for layer in range(engine.num_layers):
                assert view.layer_data_ptr(layer) == pages[chunk].layer_data_ptr(layer)
                assert torch.all(view.layer_tensor(layer) == chunk * 10 + layer)
    finally:
        for obj in (*views, *pages):
            obj.ref_count_down()
    assert owner.total_allocated_size == 0
