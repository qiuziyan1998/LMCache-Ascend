# SPDX-License-Identifier: Apache-2.0
"""The real Ascend registration override recognizes a composite C8 group."""

from dataclasses import replace

import pytest
import torch

from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache_ascend.v1.kv_layer_groups import (
    _get_kv_cache_group_key_and_info,
    build_kv_layer_groups,
)


def test_registered_c8_group_matches_scheduler_storage_metadata():
    caches = {
        f"layer.{i}.attn": (
            torch.empty(2, 128, 1, 512, dtype=torch.bfloat16),
            torch.empty(2, 128, 1, 64, dtype=torch.bfloat16),
        )
        for i in range(3)
    }
    caches["layer.0.indexer"] = (
        torch.empty(18, 128, 1, 128, dtype=torch.int8),
        torch.empty(18, 128, 1, 1, dtype=torch.float16),
    )
    manager = KVLayerGroupsManager()
    build_kv_layer_groups(manager, caches)
    assert [group.num_layers for group in manager.kv_layer_groups] == [3, 1]
    assert [group.dtype for group in manager.kv_layer_groups] == [
        torch.bfloat16,
        torch.uint8,
    ]
    assert [group.hidden_dim_size for group in manager.kv_layer_groups] == [576, 130]
    scheduler = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(3, 1, 1024, 1, 576),
        use_mla=True,
        runtime_kv_group_layer_counts=(3, 1),
        indexer_c8_layout=IndexerC8Layout(),
    )
    worker = replace(scheduler, kv_layer_groups_manager=manager)
    assert worker.get_dtypes() == scheduler.get_dtypes()
    assert worker.get_shapes(7) == scheduler.get_shapes(7)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((18, 128, 1, 1), torch.bfloat16),
        ((17, 128, 1, 1), torch.float16),
        ((18, 128, 1, 2), torch.float16),
    ],
)
def test_other_mixed_or_misaligned_pairs_are_rejected(shape, dtype):
    with pytest.raises(ValueError):
        _get_kv_cache_group_key_and_info(
            (
                torch.empty(18, 128, 1, 128, dtype=torch.int8),
                torch.empty(shape, dtype=dtype),
            )
        )


def test_mixed_registration_preserves_two_groups_and_owner_order():
    policy = IndexerC8Layout(c8_layers=(False, True, False))
    caches = {
        "layer.0.attn": (
            torch.empty(2, 128, 1, 512, dtype=torch.bfloat16),
            torch.empty(2, 128, 1, 64, dtype=torch.bfloat16),
        )
    }
    for i, quantized in enumerate(policy.c8_layers):
        caches[f"layer.{i}.indexer"] = (
            (
                torch.empty(18, 128, 1, 128, dtype=torch.int8),
                torch.empty(18, 128, 1, 1, dtype=torch.float16),
            )
            if quantized
            else (torch.empty(9, 128, 1, 128, dtype=torch.bfloat16),)
        )
    manager = KVLayerGroupsManager()
    build_kv_layer_groups(manager, caches, indexer_c8_layout=policy)
    assert [g.num_layers for g in manager.kv_layer_groups] == [1, 3]
    assert manager.kv_layer_groups[1].layer_names == [
        f"layer.{i}.indexer" for i in range(3)
    ]
    assert manager.kv_layer_groups[1].dtype == torch.uint8
    wrong = IndexerC8Layout(c8_layers=(True, False, False))
    with pytest.raises(ValueError, match="dtype disagrees"):
        build_kv_layer_groups(KVLayerGroupsManager(), caches, indexer_c8_layout=wrong)


def test_mixed_remote_fill_reservation_and_checkpoint_tail():
    from lmcache.v1.memory_management import TensorMemoryAllocator, MemoryFormat
    from lmcache_ascend.v1.remote_fill import RemoteFillGroupLayout
    from lmcache_ascend.v1.local_checkpoint import LocalCheckpointStore
    from types import SimpleNamespace

    layout = RemoteFillGroupLayout(
        1, 256, torch.uint8, MemoryFormat.KV_DSA_INDEX_FMT, (256, 130, 256)
    )
    allocator = TensorMemoryAllocator(torch.empty(65536, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        layout.page_shapes(8),
        [torch.uint8] * 3,
        3,
        3,
        layout.fmt,
        valid_tokens=[3, 4, 5],
        full_tokens=8,
    )
    assert pages is not None
    widths = ((256,), (128, 2), (256,))
    try:
        # Separate key/scale planes ensure a partial-tail splice cannot silently
        # treat C8 as token-interleaved or copy a BF16 page using C8 widths.
        for page, start in zip(pages[:2], (0, 3), strict=True):
            for layer, planes in enumerate(widths):
                offset = 0
                for plane, width in enumerate(planes):
                    data = page.layer_tensor(layer)[
                        offset : offset + page.valid_tokens * width
                    ].view(page.valid_tokens, width)
                    data.copy_(
                        (
                            torch.arange(start, start + page.valid_tokens)
                            + layer * 20
                            + plane * 10
                        ).to(torch.uint8)[:, None]
                    )
                    offset += page.valid_tokens * width
        sources = [
            (SimpleNamespace(start=0, end=3), pages[0]),
            (SimpleNamespace(start=3, end=7), pages[1]),
        ]
        LocalCheckpointStore._assemble(pages[2], widths, sources, 1, 6)
        for layer, planes in enumerate(widths):
            offset = 0
            for plane, width in enumerate(planes):
                actual = (
                    pages[2]
                    .layer_tensor(layer)[offset : offset + 5 * width]
                    .view(5, width)
                )
                expected = (
                    (torch.arange(1, 6) + layer * 20 + plane * 10)
                    .to(torch.uint8)[:, None]
                    .expand_as(actual)
                )
                assert torch.equal(actual, expected)
                offset += 5 * width
        for page in pages:
            assert page.get_size() == layout.expected_bytes(page.valid_tokens, 3)
    finally:
        for page in pages:
            page.ref_count_down()
    assert allocator.total_allocated_size == 0


@pytest.mark.parametrize("factor", [1, 2])
def test_derived_connector_initializes_mixed_layout_without_staging(factor):
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )

    conn = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    conn._group_layouts = {}
    conn.use_gpu = False
    conn.use_mla = True
    conn.dsa_two_groups = True
    conn.dtype = torch.bfloat16
    conn.lmcache_chunk_size = 1024
    conn.indexer_c8_layout = IndexerC8Layout(c8_layers=(False, True))
    conn._reset_sparse_direct_layer_states = lambda: None
    conn._mirror_layout = lambda _: None
    caches = [
        (torch.empty(9, 128, 1, 128, dtype=torch.bfloat16),),
        (
            torch.empty(9 * factor, 128, 1, 128, dtype=torch.int8),
            torch.empty(9 * factor, 128, 1, 1, dtype=torch.float16),
        ),
    ]
    layout = conn._lazy_initialize_buffer(caches, kv_group=1, init_staging=False)
    assert layout.layer_token_bytes == (256, 130)
    assert layout.layer_slot_factors == (1, factor)
    assert layout.gpu_buffer_allocator is None
    assert layout.storage_dtype == torch.uint8
    assert (
        conn._lmc_plane_num_tokens(torch.empty(7 * 130, dtype=torch.uint8), 1, 1) == 7
    )
    assert (
        conn._lmc_plane_num_tokens(torch.empty(7 * 256, dtype=torch.uint8), 1, 0) == 7
    )


def test_page_pointer_fast_path_refuses_wrong_layer_count_before_offsets():
    from lmcache.v1.memory_management import (
        TensorMemoryAllocator,
        MemoryFormat,
        LayerPageSource,
    )
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )

    allocator = TensorMemoryAllocator(torch.empty(8192, dtype=torch.uint8))
    page = allocator.batched_allocate_layer_pages(
        [torch.Size([7 * 130])],
        [torch.uint8],
        1,
        1,
        MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=7,
        full_tokens=7,
    )[0]
    conn = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    try:
        assert (
            conn._layer_page_pointer_rows(
                [LayerPageSource((page,), 0), LayerPageSource((page,), 1)]
            )
            is None
        )
    finally:
        page.ref_count_down()


@pytest.mark.parametrize("widths", [(256, 130), (1152, 1152)])
def test_page_pointer_fast_path_resolves_once_per_page(widths):
    from lmcache.v1.memory_management import (
        TensorMemoryAllocator,
        MemoryFormat,
        LayerPageSource,
    )
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )

    allocator = TensorMemoryAllocator(torch.empty(65536, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        [torch.Size([8 * w]) for w in widths],
        [torch.uint8] * 2,
        2,
        2,
        MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=[7, 3],
        full_tokens=8,
    )
    conn = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    conn.num_layers = 2
    calls = []

    def resolve(page, **kwargs):
        calls.append((page, kwargs["required_bytes"]))
        return page.layer_data_ptr(0)

    conn._resolve_registered_cpu_source_device_ptr = resolve
    try:
        rows = conn._layer_page_pointer_rows(
            [LayerPageSource(tuple(pages), i) for i in range(2)]
        )
        assert rows == [[page.layer_data_ptr(i) for page in pages] for i in range(2)]
        assert calls == [(page, page.get_size()) for page in pages]
    finally:
        for page in pages:
            page.ref_count_down()
    assert allocator.total_allocated_size == 0
