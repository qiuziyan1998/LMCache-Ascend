# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the synthetic cold-bootstrap integration benchmark."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys


def _load_benchmark_module():
    benchmark_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "v1"
        / "kv_transfer"
        / "benchmark_sparse_cold_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_sparse_cold_bootstrap",
        benchmark_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cold_bootstrap_benchmark_preserves_page_first_topology_and_warm_reuse(
    monkeypatch,
):
    benchmark = _load_benchmark_module()
    monitor = SimpleNamespace(
        on_pin=lambda _obj: None,
        on_pin_many=lambda _objs: None,
        on_unpin=lambda _obj: None,
    )
    monkeypatch.setattr(
        benchmark.PinMonitor,
        "GetOrCreate",
        lambda _config=None: monitor,
    )
    args = SimpleNamespace(
        num_tokens=512,
        chunk_size=256,
        num_layers=3,
        latent_bytes_per_token=1152,
        indexer_bytes_per_token=256,
        remote_gbps=0.0,
        rpc_overhead_us=0.0,
        object_mode="production",
        kv_groups=[0, 1],
    )

    per_layer = benchmark.run_once(
        args,
        kv_group=0,
        prefix_mode="per-layer",
    )
    batched = benchmark.run_once(
        args,
        kv_group=0,
        prefix_mode="batched",
    )

    assert per_layer.local_probe_calls == 3
    assert batched.local_probe_calls == 1
    for result in (per_layer, batched):
        assert result.remote_calls == 1
        assert result.mooncake_page_objects == 2
        assert result.lmcache_memory_objects == 6
        assert result.pointer_objects == 6
        assert result.handle_objects == 6
        assert result.broadcast_envelopes == 3
        assert result.transferred_bytes == 512 * 1152 * 3
        assert result.warm_total_s > 0
        assert result.resolver_classification_s > 0
        assert result.resolver_validation_s == 0
        assert result.resolver_rollback_s == 0


def test_cold_bootstrap_token_database_config_is_schema_independent():
    benchmark = _load_benchmark_module()

    assert not hasattr(benchmark, "LMCacheEngineConfig")
    token_database, metadata = benchmark._make_token_database(
        chunk_size=128,
        num_layers=7,
    )

    config = token_database.config
    assert config.chunk_size == 128
    assert config.pre_caching_hash_algorithm == "builtin"
    assert config.save_unfull_chunk is True
    assert config.dsa_two_groups is True
    assert config.remote_url is None
    assert config.enable_pd is False
    assert config.get_extra_config_value("save_only_first_rank") is True
    assert config.get_extra_config_value("missing", "fallback") == "fallback"
    assert metadata.chunk_size == 128
    assert metadata.kv_shape[0] == 7


def test_synthetic_connector_satisfies_layerwise_connector_contract():
    benchmark = _load_benchmark_module()
    # First Party
    from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector

    connector = benchmark.SyntheticGPUConnector(
        benchmark.StageStats(),
        num_layers=3,
    )

    assert_layerwise_gpu_connector(connector)


def test_cold_bootstrap_pair_reuses_latent_chunk_hash_plan_for_indexer() -> None:
    benchmark = _load_benchmark_module()
    args = SimpleNamespace(
        num_tokens=512,
        chunk_size=256,
        num_layers=3,
        latent_bytes_per_token=1152,
        indexer_bytes_per_token=256,
        remote_gbps=0.0,
        rpc_overhead_us=0.0,
        object_mode="synthetic",
        kv_groups=[0, 1],
    )

    independent = benchmark._run_pair_once(
        args,
        prefix_mode="batched",
        chunk_plan_mode="independent",
    )
    reused = benchmark._run_pair_once(
        args,
        prefix_mode="batched",
        chunk_plan_mode="reuse",
    )

    assert independent[0].hash_plan_reuses == 0
    assert independent[1].hash_plan_reuses == 0
    assert reused[0].hash_plan_reuses == 0
    assert reused[1].hash_plan_reuses == 1
    assert reused[1].lmcache_memory_objects == independent[1].lmcache_memory_objects
    assert reused[1].mooncake_page_objects == independent[1].mooncake_page_objects
    assert reused[1].transferred_bytes == independent[1].transferred_bytes
