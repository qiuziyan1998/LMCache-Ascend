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
