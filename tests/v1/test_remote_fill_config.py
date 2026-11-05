# SPDX-License-Identifier: Apache-2.0
"""RemoteFill must not expose fixed protocol invariants as YAML knobs."""

# Third Party
import pytest

# First Party
import lmcache_ascend  # noqa: F401
from lmcache.v1.config import LMCacheEngineConfig


def _config(role: str, repair=None) -> LMCacheEngineConfig:
    options = {
        "enable_remote_lmcache_store": True,
        "pd_role": role,
        "remote_url": "mooncakestore://metadata",
        "chunk_size": 1024,
    }
    if repair is not None:
        options["remote_fill_prefix_hole_repair"] = repair
    if role == "receiver":
        options.update(local_cpu=True, max_local_cpu_size=1.0)
    config = LMCacheEngineConfig.from_defaults(**options)
    config.validate()
    return config


@pytest.mark.parametrize("role", ["sender", "receiver"])
@pytest.mark.parametrize("repair", [None, False, True])
def test_prefix_hole_repair_is_default_off_and_not_forced_by_remote_fill(role, repair):
    config = _config(role, repair)
    assert config.enable_remote_lmcache_store is True
    assert config.remote_fill_prefix_hole_repair is (repair is True)


@pytest.mark.parametrize("enabled", [False, True])
def test_prefix_hole_repair_yaml_and_environment(tmp_path, monkeypatch, enabled):
    path = tmp_path / "lmcache.yaml"
    path.write_text(f"remote_fill_prefix_hole_repair: {str(enabled).lower()}\n")
    config = LMCacheEngineConfig.from_file(str(path))
    assert config.remote_fill_prefix_hole_repair is enabled
    monkeypatch.setenv("LMCACHE_REMOTE_FILL_PREFIX_HOLE_REPAIR", str(enabled).lower())
    assert LMCacheEngineConfig.from_env().remote_fill_prefix_hole_repair is enabled


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


def test_disabled_feature_preserves_legacy_ascend_config() -> None:
    config = LMCacheEngineConfig.from_defaults(enable_remote_lmcache_store=False)

    config.validate()

    assert config.store_async is False
    assert config.use_layerwise is False
    assert config.enable_shared_cpu_cache is False
    assert config.get_extra_config_value("use_ascend_direct") is None


def test_direct_hbm_rejects_explicit_chunk_metadata_before_normalization() -> None:
    config = LMCacheEngineConfig.from_defaults(
        enable_remote_lmcache_store=True,
        enable_dsa_cold_compact_load=True,
        dsa_group1_load_mode="persistent_direct_hbm",
        pd_role="sender",
        remote_url="mooncakestore://metadata",
        extra_config={"save_chunk_meta": True},
    )

    with pytest.raises(ValueError, match="save_chunk_meta=false"):
        config.validate()


def test_direct_hbm_sender_does_not_require_decoder_cold_slab() -> None:
    config = LMCacheEngineConfig.from_defaults(
        enable_remote_lmcache_store=True,
        dsa_group1_load_mode="persistent_direct_hbm",
        pd_role="sender",
        remote_url="mooncakestore://metadata",
    )

    config.validate()

    assert config.enable_dsa_cold_compact_load is False
    assert config.enable_shared_cpu_cache is False
