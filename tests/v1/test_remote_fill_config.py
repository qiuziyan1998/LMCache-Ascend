# SPDX-License-Identifier: Apache-2.0
"""RemoteFill must not expose fixed protocol invariants as YAML knobs."""

# Third Party
import pytest

# First Party
import lmcache_ascend  # noqa: F401
from lmcache.v1.config import LMCacheEngineConfig


@pytest.fixture(autouse=True)
def _qualify_remote_fill_hardware_gate(monkeypatch) -> None:
    monkeypatch.setenv(
        "LMCACHE_REMOTE_FILL_H0_QUALIFICATION",
        "mooncake-sync-write-visible-v1",
    )


def _config(role: str) -> LMCacheEngineConfig:
    options = {
        "enable_remote_lmcache_store": True,
        "pd_role": role,
        "remote_url": "mooncakestore://metadata",
        "chunk_size": 1024,
    }
    if role == "receiver":
        options.update(local_cpu=True, max_local_cpu_size=1.0)
    config = LMCacheEngineConfig.from_defaults(**options)
    config.validate()
    return config


def test_remote_fill_sender_implies_fixed_direct_store_contract() -> None:
    config = _config("sender")

    assert config.store_async is True
    assert config.store_async_max_queue_size == 2
    assert config.use_layerwise is True
    assert config.dsa_two_groups is True
    assert config.enable_sparse_attention is True
    assert config.save_unfull_chunk is True
    assert config.pre_caching_hash_algorithm == "sha256_cbor"
    assert config.get_extra_config_value("save_only_first_rank") is True
    assert config.get_extra_config_value("use_ascend_direct") is True
    assert config.get_extra_config_value("save_chunk_meta") is False
    assert config.get_extra_config_value("mooncake_page_first_multi_buffer") is True
    assert config.get_extra_config_value("mooncake_layer_merged_page_objects") is True


def test_remote_fill_decoder_implies_strict_shared_publication() -> None:
    config = _config("receiver")

    assert config.enable_shared_cpu_cache is True
    assert config.shared_cpu_cache_strict is True
    assert config.use_layerwise is True
    assert config.get_extra_config_value("save_only_first_rank") is True


def test_unqualified_feature_flag_preserves_legacy_ascend_config(monkeypatch) -> None:
    monkeypatch.delenv("LMCACHE_REMOTE_FILL_H0_QUALIFICATION", raising=False)
    config = LMCacheEngineConfig.from_defaults(enable_remote_lmcache_store=True)

    config.validate()

    assert config.store_async is False
    assert config.use_layerwise is False
    assert config.enable_shared_cpu_cache is False
    assert config.get_extra_config_value("use_ascend_direct") is None
