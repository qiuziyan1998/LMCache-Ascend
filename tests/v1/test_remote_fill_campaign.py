# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed staged RemoteFill P1 campaign."""

# Standard
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import tempfile


def _module():
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "v1"
        / "kv_transfer"
        / "remote_fill_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(module) -> dict:
    qualification = module._qualification_module()
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
            for name in qualification.REPOSITORY_NAMES
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


def _evidence(manifest_hash: str) -> list[dict]:
    adapter_path = Path(__file__).resolve()
    adapter_digest = sha256(adapter_path.read_bytes()).hexdigest()
    command = ["pytest", adapter_path.name]
    identity = {
        "producing_host": "test-host",
        "clock_domain": "test-host:boot",
        "trial_id": "campaign-test",
        "manifest_sha256": manifest_hash,
        "adapter_module_sha256": adapter_digest,
        "command": command,
    }
    path = Path(tempfile.gettempdir()) / f"remote-fill-campaign-{manifest_hash}.jsonl"
    path.write_text(
        json.dumps({"qualification_evidence_identity": identity}) + "\n",
        encoding="utf-8",
    )
    digest = sha256(path.read_bytes()).hexdigest()
    return [
        {
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
    ]


def test_campaign_runs_fixed_matrix_with_alternated_abc_order() -> None:
    module = _module()
    manifest = _manifest(module)
    qualification = module._qualification_module()
    manifest_hash = qualification.payload_sha256(manifest)

    class Adapter:
        def __init__(self):
            self.prepared = []
            self.closed = False

        def prepare_mode(self, case, mode):
            self.prepared.append((case["case_id"], mode))

        def run_mode(self, case, mode):
            return {
                "mode": mode,
                "mode_contract": module.MODE_CONTRACTS[mode],
                "qualification_manifest_sha256": manifest_hash,
                "diagnostics_enabled": False,
                "warmups_separate": True,
                "cache_state": "cold",
                "cache_isolation_verified": True,
                "cold_trial_isolation": ("clear_all_tiers_before_each_measured_batch"),
                "prompt_tokens": case["prompt_tokens"],
                "concurrency": case["concurrency"],
                "decoder_load": case["decoder_load"],
                "measured_repetitions": 10,
                "independent_batches": 10,
                "workload_spec_sha256": "a" * 64,
                "raw_evidence": _evidence(manifest_hash),
            }

        def close(self):
            self.closed = True

    adapter = Adapter()
    report = module.run_campaign(
        adapter,
        manifest,
        max_supported_tokens=131072,
        tiers=(1, 2, 3, 4),
    )

    assert len(report["cases"]) == 18
    assert report["cases"][0]["execution_order"] == ("A", "B", "C")
    assert report["cases"][1]["execution_order"] == ("B", "C", "A")
    assert report["cases"][2]["execution_order"] == ("C", "A", "B")
    assert adapter.closed


def test_campaign_defaults_to_tier1_only() -> None:
    module = _module()

    cases = module.build_cases(131072, tiers=(1,))

    assert {case["tier"] for case in cases} == {1}
    assert len(cases) == 2


def test_campaign_rejects_wrong_mode_contract() -> None:
    module = _module()
    manifest = _manifest(module)
    manifest_hash = module._qualification_module().payload_sha256(manifest)

    class Adapter:
        def prepare_mode(self, case, mode):
            pass

        def run_mode(self, case, mode):
            return {
                "mode": mode,
                "mode_contract": "wrong_path",
                "qualification_manifest_sha256": manifest_hash,
                "diagnostics_enabled": False,
                "warmups_separate": True,
                "cache_state": "cold",
                "cache_isolation_verified": True,
                "cold_trial_isolation": ("clear_all_tiers_before_each_measured_batch"),
                "prompt_tokens": case["prompt_tokens"],
                "concurrency": case["concurrency"],
                "decoder_load": case["decoder_load"],
                "measured_repetitions": 10,
                "independent_batches": 10,
                "workload_spec_sha256": "a" * 64,
                "raw_evidence": _evidence(manifest_hash),
            }

        def close(self):
            pass

    try:
        module.run_campaign(
            Adapter(), manifest, max_supported_tokens=131072, tiers=(1,)
        )
    except ValueError as error:
        assert "invalid P1 result" in str(error)
    else:
        raise AssertionError("wrong mode contract was accepted")
