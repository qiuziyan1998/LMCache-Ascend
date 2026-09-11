# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardware-adapter H0 orchestrator contract."""

# Standard
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import tempfile

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


def _manifest() -> dict:
    return {
        "schema": 1,
        "repositories": {
            name: {
                "path": "/repo",
                "branch": "branch",
                "sha": "1" * 40,
                "tree_sha": "2" * 40,
                "remote_origin": "https://example.invalid/repository.git",
                "submodules": [],
                "dirty": False,
            }
            for name in ("LMCache-NPU", "LMCache-Ascend", "vllm", "vllm-ascend")
        },
        "inventory": {
            "artifacts": {
                "mooncake_build": "moon",
                "mooncake_checksum": "a" * 64,
                "container_image_digest": "sha256:" + "b" * 64,
            },
            "software": dict.fromkeys(
                ("python", "pytorch", "torch_npu", "cann", "driver", "firmware"),
                "version",
            ),
            "model": dict.fromkeys(
                (
                    "serving_bundle_artifact_id",
                    "weights_identity",
                    "tokenizer_identity",
                    "quantization_identity",
                    "custom_code_identity",
                ),
                "identity",
            ),
            "cache": {
                "namespace": "namespace",
                "layout_tag": "layout",
                "page_abi": "abi",
                "token_hash_algorithm": "sha256",
                "python_hash_seed": 0,
                "chunk_size": 256,
                "remote_fill_window_size": 4096,
                "group_dimensions": [576, 128],
                "cache_layer_count": 79,
                "dtype": "bfloat16",
                "mtp": {"enabled": True, "num_speculative_tokens": 1},
            },
            "topology": {
                "tp": 8,
                "dp": 2,
                "ep": 8,
                "flashcomm": True,
                "graph_mode": "piecewise",
            },
            "network": {
                "nic": "nic",
                "link_state": "up",
                "routing": "rdma",
                "mtu": 4200,
                "cpu_socket": "socket0",
                "numa_policy": "node0",
                "memory_placement": "node0",
                "global_te_session": "decoder:1234",
                "mooncake_segment_placement": "prefiller-local",
            },
            "clock": {"source": "ptp", "max_observed_host_offset_us": 1},
        },
    }


class _Adapter:
    closed = False
    partial_tokens = ()

    def __init__(self, manifest_hash: str = "a" * 64) -> None:
        self.manifest_hash = manifest_hash

    def _result(self, case, **fields):
        adapter_path = Path(__file__).resolve()
        adapter_digest = sha256(adapter_path.read_bytes()).hexdigest()
        command = ["pytest", adapter_path.name]
        identity = {
            "producing_host": "test-host",
            "clock_domain": "test-host:boot",
            "trial_id": case,
            "manifest_sha256": self.manifest_hash,
            "adapter_module_sha256": adapter_digest,
            "command": command,
        }
        path = Path(tempfile.gettempdir()) / (
            f"remote-fill-h0-{case}-{self.manifest_hash}.jsonl"
        )
        path.write_text(
            json.dumps({"qualification_evidence_identity": identity}) + "\n",
            encoding="utf-8",
        )
        digest = sha256(path.read_bytes()).hexdigest()
        evidence = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            **{key: identity[key] for key in identity if key != "command"},
            "adapter_module_path": str(adapter_path),
            "adapter_module_sha256": adapter_digest,
            "adapter_module_size_bytes": adapter_path.stat().st_size,
            "command": command,
            "example_only": False,
        }
        return {"status": "PASS", "evidence": [evidence], **fields}

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

    @staticmethod
    def observed_layout():
        return {
            "chunk_size": 256,
            "window_tokens": 4096,
            "group_dimensions": [576, 128],
            "cache_layer_count": 79,
            "dtype": "bfloat16",
            "page_abi": "abi",
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
            persistent_verified=True,
            direct_verified=True,
            same_source_generation=True,
            same_manifest_digest=True,
            persistent_byte_count=4096,
            direct_byte_count=4096,
            source_byte_count=4096,
        )

    def close(self):
        self.closed = True


def test_h0_orchestrator_runs_exact_matrix_and_closes() -> None:
    module = _module()
    manifest = _manifest()
    manifest_hash = module._qualification_module().payload_sha256(manifest)
    adapter = _Adapter(manifest_hash)
    report = module.run_h0(
        adapter,
        manifest,
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
    manifest = _manifest()
    manifest_hash = module._qualification_module().payload_sha256(manifest)

    class MissingEvidence(_Adapter):
        def full_pages(self):
            record = super().full_pages()
            record.pop("evidence")
            return record

    adapter = MissingEvidence(manifest_hash)
    with pytest.raises(ValueError, match="H0-A has no raw evidence"):
        module.run_h0(
            adapter,
            manifest,
            chunk_size=256,
            window_tokens=4096,
            soak_iterations=100,
            guard_seconds=1,
        )
    assert adapter.closed is True


def test_h0_closes_when_component_discovery_fails() -> None:
    module = _module()
    manifest = _manifest()
    manifest_hash = module._qualification_module().payload_sha256(manifest)

    class BrokenDiscovery(_Adapter):
        def production_components(self):
            raise RuntimeError("discovery failed")

    adapter = BrokenDiscovery(manifest_hash)
    adapter.closed = False
    with pytest.raises(RuntimeError, match="discovery failed"):
        module.run_h0(
            adapter,
            manifest,
            chunk_size=256,
            window_tokens=4096,
            soak_iterations=100,
            guard_seconds=1,
        )
    assert adapter.closed is True
