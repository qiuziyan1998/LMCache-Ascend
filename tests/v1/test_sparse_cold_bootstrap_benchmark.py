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


def test_cold_bootstrap_benchmark_preserves_page_first_topology_and_warm_reuse():
    benchmark = _load_benchmark_module()
    args = SimpleNamespace(
        num_tokens=512,
        chunk_size=256,
        num_layers=3,
        latent_bytes_per_token=1152,
        indexer_bytes_per_token=256,
        remote_gbps=0.0,
        rpc_overhead_us=0.0,
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
