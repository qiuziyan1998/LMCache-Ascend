# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed staged RemoteFill P1 campaign."""

# Standard
from pathlib import Path
import importlib.util
import sys


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


def test_campaign_runs_fixed_matrix_with_alternated_abc_order() -> None:
    module = _module()
    manifest = {"schema": 1, "identity": "campaign"}
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
                "workload_spec_sha256": "a" * 64,
                "raw_evidence": [f"raw/{case['case_id']}-{mode}.jsonl"],
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
    manifest = {"schema": 1, "identity": "campaign"}
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
                "workload_spec_sha256": "a" * 64,
                "raw_evidence": ["raw.jsonl"],
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
