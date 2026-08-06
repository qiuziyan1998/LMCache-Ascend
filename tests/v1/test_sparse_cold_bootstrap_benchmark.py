# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the synthetic cold-bootstrap integration benchmark."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

# Third Party
import pytest


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
        assert result.resolver_remote_get_s > 0
        assert result.resolver_classification_s > 0
        assert result.resolver_validation_s == 0
        assert result.resolver_rollback_s == 0


def test_resolver_timing_hook_covers_all_production_stages() -> None:
    benchmark = _load_benchmark_module()
    stats = benchmark.StageStats()
    stages = (
        "windows",
        "remote_get",
        "results",
        "classification",
        "pinning",
        "scatter",
        "rollback",
    )

    assert benchmark._RESOLVER_EMITTED_STAGES == stages
    for index, stage in enumerate(stages, start=1):
        benchmark._record_resolver_stage(stats, stage, index / 1000)
        assert getattr(stats, f"resolver_{stage}_s") == pytest.approx(
            index / 1000
        )
    benchmark._record_resolver_stage(stats, "validation", 0.008)
    assert stats.resolver_validation_s == pytest.approx(0.008)
    assert benchmark._resolver_cpu_attributed_s(stats) == pytest.approx(
        sum(index / 1000 for index in (1, 3, 4, 5, 6, 8))
    )
    with pytest.raises(ValueError, match="Unknown resolver timing stage"):
        benchmark._record_resolver_stage(stats, "future_stage", 1.0)


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


def _ab_args(**overrides):
    values = {
        "num_tokens": 8,
        "chunk_size": 4,
        "num_layers": 2,
        "latent_bytes_per_token": 8,
        "indexer_bytes_per_token": 4,
        "remote_gbps": 0.0,
        "rpc_overhead_us": 0.0,
        "object_mode": "synthetic",
        "h2d_gbps": 22.0,
        "dense_consumer_layer_ms": 0.0,
        "normal_sync_ms": 0.0,
        "cold_compact_sync_ms": 0.0,
        "cold_compact_barrier_ms": 0.0,
        "ab_prefix_mode": "batched",
        "warmup": 0,
        "repeats": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ab_samples(benchmark):
    def sample(enabled, wall, sync):
        consumer_layers = {0: 0 if enabled else 2, 1: 2}
        return benchmark.ColdCompactProtocolSample(
            enabled=enabled,
            wall_s=wall,
            synchronization_barrier_s=sync,
            groups={
                group: benchmark.StageStats(
                    metadata_s=1.0,
                    remote_s=2.0,
                    remote_buffer_s=0.4,
                    remote_wrapper_s=0.1,
                    remote_transfer_s=1.0,
                    resolver_cpu_s=1.0,
                    pointer_s=0.25,
                    handle_s=0.25,
                    device_consumer_s=0.5 * layers,
                    device_consumer_layers=layers,
                )
                for group, layers in consumer_layers.items()
            },
        )

    return {
        "flag_unset": [sample(False, 20.0, 0.1)],
        "flag_true": [sample(True, 21.0, 2.1)],
    }


def test_cold_compact_ab_uses_production_order_without_double_counting() -> None:
    benchmark = _load_benchmark_module()
    report = benchmark.build_cold_compact_ab_report(_ab_args(), _ab_samples(benchmark))
    disabled = report.variants["flag_unset"]
    enabled = report.variants["flag_true"]

    assert disabled.consumer_modes == {0: "dense", 1: "dense"}
    assert enabled.consumer_modes == {0: "materialize_only", 1: "dense"}
    assert disabled.device_consumer_s == pytest.approx(2.0)
    assert enabled.device_consumer_s == pytest.approx(1.0)
    assert disabled.critical_path_s == pytest.approx(20.0)
    assert enabled.critical_path_s == pytest.approx(21.0)
    assert report.comparison["delta_s"] == pytest.approx(1.0)
    assert report.comparison["slowdown"] > 1
    for variant in (disabled, enabled):
        attributed = sum(
            (
                variant.lookup_metadata_s,
                variant.mooncake_s,
                variant.resolver_cpu_s,
                variant.pointer_handle_s,
                variant.bootstrap_other_s,
                variant.device_consumer_s,
                variant.synchronization_barrier_s,
            )
        )
        assert attributed == pytest.approx(variant.critical_path_s)


def test_cold_compact_ab_executes_distinct_production_protocols() -> None:
    benchmark = _load_benchmark_module()
    args = _ab_args()

    disabled = benchmark._run_cold_compact_protocol_once(args, enabled=False)
    enabled = benchmark._run_cold_compact_protocol_once(args, enabled=True)

    assert disabled.groups[0].device_consumer_layers == args.num_layers
    assert disabled.groups[1].device_consumer_layers == args.num_layers
    assert enabled.groups[0].device_consumer_layers == 0
    assert enabled.groups[1].device_consumer_layers == args.num_layers
    assert disabled.groups[0].device_bytes == 8 * 8 * 2
    assert disabled.groups[1].device_bytes == 8 * 4 * 2
    assert enabled.groups[1].device_bytes == 8 * 4 * 2
    assert sum(item.device_bytes for item in disabled.groups.values()) == (
        args.num_layers
        * (
            args.num_tokens * args.latent_bytes_per_token
            + args.num_tokens * args.indexer_bytes_per_token
        )
    )
    assert sum(item.device_bytes for item in enabled.groups.values()) == (
        args.num_layers * args.num_tokens * args.indexer_bytes_per_token
    )
    assert disabled.wall_s > 0 and enabled.wall_s > 0


def test_cold_compact_ab_exports_filtered_and_backward_compatible_json(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _load_benchmark_module()
    stats = benchmark.StageStats()
    results = [
        benchmark.BenchmarkResult(
            name="g0_batched_reuse",
            prefix_mode="batched",
            chunk_plan_mode="reuse",
            object_mode="synthetic",
            kv_group=0,
            num_layers=2,
            num_chunks=2,
            bytes_per_token=8,
            repeats=1,
            median=stats,
            samples=[stats],
        )
    ]
    args = _ab_args(
        kv_groups=[0, 1],
        prefix_modes=["batched"],
        chunk_plan_modes=["reuse"],
        object_mode="synthetic",
        remote_gbps=0.0,
        rpc_overhead_us=0.0,
        cold_compact_ab=True,
        output_json=tmp_path / "full.json",
        ab_output_json=tmp_path / "ab.json",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "run_pair_cases", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        benchmark,
        "run_cold_compact_ab_cases",
        lambda _args: _ab_samples(benchmark),
    )
    monkeypatch.setattr(benchmark, "print_results", lambda _results: None)
    monkeypatch.setattr(benchmark, "print_cold_compact_ab", lambda _report: None)

    benchmark.main()

    full = json.loads(args.output_json.read_text(encoding="utf-8"))
    filtered = json.loads(args.ab_output_json.read_text(encoding="utf-8"))
    assert set(full) == {"config", "results", "cold_compact_ab"}
    assert filtered["schema"] == 1
    assert filtered["model"] == "executed_production_ordered_protocol_v2"
    assert filtered["parameters"]["sample_count"] == 1
    assert set(filtered["variants"]) == {"flag_unset", "flag_true"}
    assert len(filtered["samples"]) == 1


def test_cold_compact_ab_rejects_incomplete_protocol_samples() -> None:
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="non-empty off/on samples"):
        benchmark.build_cold_compact_ab_report(_ab_args(), {"flag_unset": []})


def test_layer_merged_matrix_exercises_cold_compact_page_path() -> None:
    benchmark = _load_benchmark_module()
    report = benchmark.run_layer_merged_cold_compact_matrix(
        _ab_args(object_mode="production")
    )

    assert report["variants"]["merged_0_compact_1"]["physical_pages"] == 0
    assert report["variants"]["merged_1_compact_1"]["physical_pages"] > 0
