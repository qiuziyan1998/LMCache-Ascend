# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import ast


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = (
    _REPO_ROOT
    / "benchmark"
    / "v1"
    / "kv_transfer"
    / "benchmark_sparse_k_transfer_tuning.py"
)
_MEM_KERNELS = _REPO_ROOT / "csrc" / "mem_kernels.cpp"
_PYBIND = _REPO_ROOT / "csrc" / "pybind.cpp"


def _source() -> str:
    return _BENCHMARK.read_text(encoding="utf-8")


def test_bandwidth_benchmark_is_valid_python() -> None:
    ast.parse(_source(), filename=str(_BENCHMARK))


def test_bandwidth_preset_exercises_large_existing_sparse_payloads() -> None:
    source = _source()
    assert '"bandwidth": Preset(' in source
    assert (
        "selected_counts=(256, 512, 1024, 2048, 4096, 8192, 16384)"
        in source
    )
    assert "_run_layers(harness, config, selected_counts)" in source
    assert "_ols_bandwidth_fit(" in source


def test_raw_reference_calls_acl_memcpy_without_synthetic_kernel() -> None:
    benchmark = _source()
    mem_kernels = _MEM_KERNELS.read_text(encoding="utf-8")
    pybind = _PYBIND.read_text(encoding="utf-8")

    assert "lmc_ops.benchmark_aclrt_memcpy_h2d(" in benchmark
    assert "aclrtMemcpyAsync(" in mem_kernels
    assert 'm.def("benchmark_aclrt_memcpy_h2d"' in pybind
    assert "benchmark_ascendc_mapped_h2d" not in benchmark
    assert "contiguous_h2d_bandwidth" not in mem_kernels
