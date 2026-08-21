# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardware-adapter H0 orchestrator contract."""

# Standard
from pathlib import Path
import importlib.util
import sys

# Third Party
import pytest


def _module():
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "v1"
        / "kv_transfer"
        / "remote_fill_h0.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_h0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Adapter:
    closed = False
    partial_tokens = ()

    @staticmethod
    def _result(case, **fields):
        return {"status": "PASS", "evidence": [f"raw/{case}.json"], **fields}

    def production_components(self):
        return {
            name: True
            for name in (
                "global_te",
                "source_registration",
                "source_page_planner",
                "decoder_local_cpu_allocator",
                "native_transport",
            )
        }

    def full_pages(self):
        return self._result(
            "h0-a",
            byte_mismatches=0,
            canary_mismatches=0,
            vector_order_mismatches=0,
            group0_full_verified=True,
            group1_full_verified=True,
            combined_verified=True,
            noncontiguous_runs_verified=True,
            destination_identity_verified=True,
        )

    def partial_pages(self, valid_tokens):
        self.partial_tokens = valid_tokens
        return self._result(
            "h0-b",
            byte_mismatches=0,
            unused_byte_mutations=0,
            valid_token_mismatches=0,
            cold_compact_valid_tokens_verified=True,
        )

    def terminal_visibility(self, _guard_seconds):
        return self._result(
            "h0-c",
            late_source_reads=0,
            late_destination_mutations=0,
            source_reuse_verified=True,
        )

    def destination_reuse(self, _guard_seconds):
        return self._result(
            "h0-d",
            late_destination_mutations=0,
            destination_reuse_verified=True,
        )

    def registration_soak(self, iterations):
        return self._result(
            "h0-e",
            iterations=iterations,
            registration_leaks=0,
            owner_leaks=0,
            reservation_leaks=0,
            future_leaks=0,
            attempt_record_leaks=0,
            overlapping_registration_failures=0,
        )

    def production_bandwidth(self, _window_tokens):
        return self._result(
            "h0-f",
            groups={
                name: {
                    "bytes": 4096,
                    "p50_ms": 1,
                    "p95_ms": 2,
                    "p99_ms": 3,
                    "native_p50_ms": 1,
                    "setup_p50_ms": 0.5,
                    "effective_gbps": 4,
                    "cold_samples": 10,
                    "warm_samples": 10,
                }
                for name in ("group0", "group1", "combined")
            },
        )

    def dual_sink(self, _window_tokens):
        return self._result(
            "h0-g",
            persistent_ms=10,
            direct_ms=10,
            dual_ms=10,
            concurrency_efficiency=1,
            persistent_slowdown=1,
            direct_slowdown=1,
        )

    def close(self):
        self.closed = True


def test_h0_orchestrator_runs_exact_matrix_and_closes() -> None:
    module = _module()
    adapter = _Adapter()
    report = module.run_h0(
        adapter,
        {
            "schema": 1,
            "identity": "manifest",
            "inventory": {"cache": {"chunk_size": 256}},
        },
        chunk_size=256,
        window_tokens=4096,
        soak_iterations=100,
        guard_seconds=1,
    )
    assert set(report["cases"]) == {f"H0-{suffix}" for suffix in "ABCDEFG"}
    assert adapter.partial_tokens == (1, 64, 128, 255)
    assert adapter.closed is True


def test_h0_adapter_must_supply_real_raw_evidence() -> None:
    module = _module()

    class MissingEvidence(_Adapter):
        def full_pages(self):
            record = super().full_pages()
            record.pop("evidence")
            return record

    adapter = MissingEvidence()
    with pytest.raises(ValueError, match="H0-A has no raw evidence"):
        module.run_h0(
            adapter,
            {
                "schema": 1,
                "identity": "manifest",
                "inventory": {"cache": {"chunk_size": 256}},
            },
            chunk_size=256,
            window_tokens=4096,
            soak_iterations=100,
            guard_seconds=1,
        )
    assert adapter.closed is True


def test_h0_closes_when_component_discovery_fails() -> None:
    module = _module()

    class BrokenDiscovery(_Adapter):
        def production_components(self):
            raise RuntimeError("discovery failed")

    adapter = BrokenDiscovery()
    adapter.closed = False
    with pytest.raises(RuntimeError, match="discovery failed"):
        module.run_h0(
            adapter,
            {"schema": 1, "identity": "manifest"},
            chunk_size=256,
            window_tokens=4096,
            soak_iterations=100,
            guard_seconds=1,
        )
    assert adapter.closed is True
