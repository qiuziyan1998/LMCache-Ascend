# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache_ascend.v1.cache_engine import (
    AscendLMCacheEngine,
    LOCAL_CPU_BACKEND_NAME,
)


def test_warm_cached_retrieve_skips_contains_and_uses_local_cpu() -> None:
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.storage_manager = MagicMock()
    engine.storage_manager.storage_backends = {LOCAL_CPU_BACKEND_NAME: MagicMock()}
    engine.retrieve_locations = None
    engine.num_layers = 2

    cached_keys: list[list] = [[MagicMock()], [MagicMock()]]
    cached_starts = [0]
    cached_ends = [256]
    ret_mask = torch.zeros(256, dtype=torch.bool)
    retrieve_kwargs = {
        "_retrieve_metadata_warm": True,
        "_use_cached_retrieve": True,
        "cached_retrieve_location": "RemoteBackend",
    }

    location, _, _, _ = engine._ensure_retrieve_chunk_metadata(
        tokens=[0] * 256,
        mask=None,
        request_configs=None,
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        ret_mask=ret_mask,
        retrieve_kwargs=retrieve_kwargs,
    )

    engine.storage_manager.contains.assert_not_called()
    assert location == LOCAL_CPU_BACKEND_NAME
    assert retrieve_kwargs["cached_retrieve_location"] == LOCAL_CPU_BACKEND_NAME


def test_shared_passive_cached_metadata_skips_storage_probe() -> None:
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.storage_manager = None
    engine.num_layers = 1
    engine._should_use_shared_layerwise_retrieve = lambda _kv_group: True
    engine._is_passive = lambda: False
    engine._is_indexer_passive = lambda: True
    engine._cached_layerwise_location_or_raise = MagicMock(
        side_effect=AssertionError("passive rank must not probe storage")
    )

    cached_keys: list[list] = [[MagicMock()]]
    cached_starts = [0]
    cached_ends = [1]
    ret_mask = torch.zeros(1, dtype=torch.bool)
    retrieve_kwargs = {"kv_group": 1}

    location, _, _, _ = engine._ensure_retrieve_chunk_metadata(
        tokens=[0],
        mask=None,
        request_configs=None,
        cached_keys=cached_keys,
        cached_starts=cached_starts,
        cached_ends=cached_ends,
        ret_mask=ret_mask,
        retrieve_kwargs=retrieve_kwargs,
    )

    engine._cached_layerwise_location_or_raise.assert_not_called()
    assert location is None
    assert ret_mask.tolist() == [True]
    assert retrieve_kwargs["_retrieve_metadata_warm"] is True
