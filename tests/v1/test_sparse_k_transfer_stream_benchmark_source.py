# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import ast


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = (
    _REPO_ROOT
    / "benchmark"
    / "v1"
    / "kv_transfer"
    / "benchmark_sparse_k_transfer_streams.py"
)


def _source() -> str:
    return _BENCHMARK.read_text(encoding="utf-8")


def test_stream_benchmark_is_valid_python() -> None:
    ast.parse(_source(), filename=str(_BENCHMARK))


def test_stream_benchmark_has_equal_payload_control_cases() -> None:
    source = _source()
    assert "SINGLE_SMALL = " in source
    assert "SINGLE_LARGE = " in source
    assert "SEQUENTIAL_PAIR = " in source
    assert "CONCURRENT_PAIR = " in source
    assert "SINGLE_LARGE: small_bytes * 2" in source
    assert "SEQUENTIAL_PAIR: small_bytes * 2" in source
    assert "CONCURRENT_PAIR: small_bytes * 2" in source


def test_concurrent_case_uses_independent_streams_and_destinations() -> None:
    source = _source()
    assert "destination=harness.dst_direct_fast" in source
    assert "destination=harness.dst_direct" in source
    assert "stream_a = torch.npu.Stream(" in source
    assert "stream_b = torch.npu.Stream(" in source
    assert "stream_a.wait_event(start_event)" in source
    assert "stream_b.wait_event(start_event)" in source
    assert "control_stream.wait_event(done_a)" in source
    assert "control_stream.wait_event(done_b)" in source


def test_stream_benchmark_copies_and_verifies_both_mla_planes() -> None:
    source = _source()
    assert "KV_FORMAT_MLA_LATENT" in source
    assert "for plane_index in range(2):" in source
    assert "torch.testing.assert_close(actual, expected, rtol=0, atol=0)" in source
