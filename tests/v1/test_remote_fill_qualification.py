# SPDX-License-Identifier: Apache-2.0
"""Tests for the RemoteFill qualification manifest contract."""

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
        / "remote_fill_qualification.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _c1_module():
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "v1"
        / "kv_transfer"
        / "remote_fill_c1.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_c1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inventory() -> dict:
    return {
        "artifacts": {
            "mooncake_build": "moon-v1",
            "mooncake_checksum": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
        },
        "software": {
            "python": "3.11.14",
            "pytorch": "2.7.1",
            "torch_npu": "2.7.1",
            "cann": "8.2",
            "driver": "25.0",
            "firmware": "7.7",
        },
        "model": {
            "serving_bundle_artifact_id": "bundle-v1",
            "weights_identity": "weights-v1",
            "tokenizer_identity": "tokenizer-v1",
            "quantization_identity": "w4a8-v1",
            "custom_code_identity": "model-code-v1",
        },
        "cache": {
            "namespace": "qualification-v1",
            "layout_tag": "layout-v3",
            "page_abi": "page-abi-v1",
            "token_hash_algorithm": "builtin",
            "python_hash_seed": 0,
            "chunk_size": 256,
            "remote_fill_window_size": 4096,
            "group_dimensions": [576, 128],
            "cache_layer_count": 79,
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
            "nic": "mlx5_0",
            "link_state": "up",
            "routing": "rdma",
            "mtu": 4200,
            "cpu_socket": "socket0",
            "numa_policy": "node0",
            "memory_placement": "node0-pinned",
            "global_te_session": "decoder:1234",
            "mooncake_segment_placement": "prefiller-local",
        },
        "clock": {"source": "ptp", "max_observed_host_offset_us": 10},
    }


def test_manifest_requires_complete_runtime_inventory() -> None:
    module = _module()
    manifest = {
        "schema": 1,
        "repositories": {
            name: {
                "path": "/repo",
                "branch": "branch",
                "sha": "1" * 40,
                "dirty": False,
            }
            for name in module.REPOSITORY_NAMES
        },
        "inventory": _inventory(),
    }
    module.validate_manifest(manifest)
    del manifest["inventory"]["clock"]["source"]
    with pytest.raises(ValueError, match="clock.source"):
        module.validate_manifest(manifest)


def test_dirty_repository_is_explicitly_nonqualifying() -> None:
    module = _module()
    manifest = {
        "schema": 1,
        "repositories": {
            name: {
                "path": "/repo",
                "branch": "branch",
                "sha": "1" * 40,
                "dirty": False,
            }
            for name in module.REPOSITORY_NAMES
        },
        "inventory": _inventory(),
    }
    manifest["repositories"]["LMCache-NPU"]["dirty"] = True
    with pytest.raises(ValueError, match="LMCache-NPU"):
        module.validate_manifest(manifest)
    module.validate_manifest(manifest, allow_dirty=True)


def _evidence_matrix(module) -> dict:
    return {
        item: {
            "status": "PASS_UNIT",
            "evidence": [f"tests::{item.lower()}"],
        }
        for item in module.VERIFICATION_ITEMS
    }


def test_evidence_matrix_requires_every_verification_item() -> None:
    module = _module()
    matrix = _evidence_matrix(module)
    module.validate_evidence_matrix(matrix)
    del matrix["V8"]
    with pytest.raises(ValueError, match="exactly V1 through V8"):
        module.validate_evidence_matrix(matrix)


def test_hardware_pending_is_not_a_c1_exit() -> None:
    module = _module()
    matrix = _evidence_matrix(module)
    for item in ("V1", "V2"):
        matrix[item]["status"] = "FIXED_AND_PASS"
    matrix["V6"] = {
        "status": "PENDING_HARDWARE",
        "evidence": ["TP8 campaign has not run"],
    }
    module.validate_evidence_matrix(matrix)
    with pytest.raises(ValueError, match="V6 remains pending"):
        module.validate_evidence_matrix(matrix, require_hardware=True)
    matrix["V6"]["status"] = "PASS_UNIT"
    with pytest.raises(ValueError, match="lacks passing hardware evidence"):
        module.validate_evidence_matrix(matrix, require_hardware=True)


def test_not_applicable_status_requires_proof() -> None:
    module = _module()
    matrix = _evidence_matrix(module)
    matrix["V4"] = {
        "status": "NOT_APPLICABLE_WITH_PROOF",
        "evidence": ["protocol operation removed in exact SHA"],
    }
    with pytest.raises(ValueError, match="explicit non-applicability proof"):
        module.validate_evidence_matrix(matrix)


def _h0_report(module) -> dict:
    cases = {
        name: {"status": "PASS", "evidence": [f"raw/{name}.json"]}
        for name in module.H0_CASES
    }
    cases["H0-A"].update(
        byte_mismatches=0,
        canary_mismatches=0,
        vector_order_mismatches=0,
        group0_full_verified=True,
        group1_full_verified=True,
        combined_verified=True,
        noncontiguous_runs_verified=True,
        destination_identity_verified=True,
    )
    cases["H0-B"].update(
        byte_mismatches=0,
        unused_byte_mutations=0,
        valid_token_mismatches=0,
        valid_tokens_tested=[1, 64, 128, 255],
        cold_compact_valid_tokens_verified=True,
    )
    cases["H0-C"].update(
        late_source_reads=0,
        late_destination_mutations=0,
        source_reuse_verified=True,
        guard_seconds=1,
    )
    cases["H0-D"].update(
        late_destination_mutations=0,
        destination_reuse_verified=True,
        guard_seconds=1,
    )
    cases["H0-E"].update(
        iterations=100,
        registration_leaks=0,
        owner_leaks=0,
        reservation_leaks=0,
        future_leaks=0,
        attempt_record_leaks=0,
        overlapping_registration_failures=0,
    )
    cases["H0-F"].update(window_tokens=4096)
    cases["H0-F"]["groups"] = {
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
    }
    cases["H0-G"].update(
        window_tokens=4096,
        persistent_ms=10,
        direct_ms=10,
        dual_ms=10,
        concurrency_efficiency=1,
        persistent_slowdown=1,
        direct_slowdown=1,
    )
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_h0_report",
        "activation": "mooncake-sync-write-visible-v1",
        "qualification_manifest_sha256": "a" * 64,
        "production_components": {
            name: True for name in module.H0_PRODUCTION_COMPONENTS
        },
        "cases": cases,
    }


def test_h0_report_rejects_any_safety_failure() -> None:
    module = _module()
    report = _h0_report(module)
    module.validate_h0_report(report)
    report["cases"]["H0-C"]["late_destination_mutations"] = 1
    with pytest.raises(ValueError, match="late_destination_mutations"):
        module.validate_h0_report(report)


def test_h0_report_requires_production_components_and_soak() -> None:
    module = _module()
    report = _h0_report(module)
    report["production_components"]["source_page_planner"] = False
    with pytest.raises(ValueError, match="production component"):
        module.validate_h0_report(report)
    report = _h0_report(module)
    with pytest.raises(ValueError, match="soak length"):
        module.validate_h0_report(report, minimum_soak_iterations=1000)


def test_h0_report_is_bound_to_exact_manifest() -> None:
    module = _module()
    report = _h0_report(module)
    manifest = {
        "schema": 1,
        "identity": "exact-run",
        "inventory": {"cache": {"chunk_size": 256}},
    }
    report["qualification_manifest_sha256"] = module.payload_sha256(manifest)
    module.validate_h0_report(report, manifest=manifest)
    with pytest.raises(ValueError, match="different manifest"):
        module.validate_h0_report(
            report,
            manifest={
                "schema": 1,
                "identity": "other",
                "inventory": {"cache": {"chunk_size": 256}},
            },
        )


def test_h0_report_rejects_nested_native_capabilities() -> None:
    module = _module()
    report = _h0_report(module)
    report["cases"]["H0-F"]["groups"]["group0"]["source_ptr"] = 123
    with pytest.raises(ValueError, match="private native capability"):
        module.validate_h0_report(report)


def _c1_report(module, manifest) -> dict:
    scenarios = {
        name: {"status": "PASS", "evidence": [f"raw/{name}.json"]}
        for name in module.C1_SCENARIOS
    }
    scenarios["tp8_passive_materialization"].update(
        group0_failure_collective_fallback=True,
        group1_failure_collective_fallback=True,
        computed_token_counts_agree=True,
    )
    scenarios["cold_miss"].update(
        terminal_outcome="LOCAL_FULL",
        required_store_end=128,
        persistent_common_end=128,
    )
    scenarios["joinable_partial_prefix"].update(
        persistent_boundary_used=True,
        paired_local_prefix_used=True,
        both_groups_common_frontier=True,
    )
    scenarios["decoder_local_first_hole"].update(
        abandoned_before_arm=True,
        no_suffix_publication=True,
        persistent_authoritative=True,
    )
    scenarios["full_persistent_prefix_hit"].update(
        transactions=0,
        native_calls=0,
        retained_source_bytes=0,
    )
    scenarios["final_partial_page"].update(
        valid_tokens_verified=True,
        both_groups_verified=True,
        persistence_verified=True,
        mtp_frontier_verified=True,
    )
    scenarios["mtp_first_resume"].update(
        required_store_end_verified=True,
        final_token_recompute_verified=True,
        speculative_row_ownership_verified=True,
        computed_token_count_verified=True,
        execution_route_verified=True,
        cache_selection_verified=True,
    )
    scenarios["dp2_routing"]["mappings"] = [
        "P0-D0",
        "P0-D1",
        "P1-D0",
        "P1-D1",
    ]
    scenarios["dp2_routing"]["nonselected_destination_allocated_pages"] = 0
    scenarios["repeated_prefix_and_eviction"].update(
        immediate_retained_hit_verified=True,
        evicted_persistent_fallback_verified=True,
        model_output_acceptable=True,
    )
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_c1_report",
        "qualification_manifest_sha256": module.payload_sha256(manifest),
        "production_components": {
            name: True for name in module.C1_PRODUCTION_COMPONENTS
        },
        "topology": {"tp": 8, "dp": 2, "num_speculative_tokens": 1},
        "scenarios": scenarios,
        "integrity": {name: 0 for name in module.C1_ZERO_INVARIANTS},
        "diagnostics": {"bounded": True, "diagnosed_requests": 9},
        "comparisons": {
            name: {"status": "PASS", "evidence": [f"raw/{name}.json"]}
            for name in module.C1_COMPARISONS
        },
        "dp_equivalence": {
            "measured": True,
            "cross_dp_reuse_qualified": True,
            "execution_identity_namespaced": False,
        },
    }


def test_c1_report_requires_full_topology_and_zero_integrity_failures() -> None:
    module = _module()
    manifest = {"schema": 1, "identity": "manifest"}
    report = _c1_report(module, manifest)
    module.validate_c1_report(report, manifest=manifest)
    report["integrity"]["one_group_only_uses"] = 1
    with pytest.raises(ValueError, match="one_group_only_uses"):
        module.validate_c1_report(report, manifest=manifest)
    report = _c1_report(module, manifest)
    report["scenarios"]["dp2_routing"]["mappings"].pop()
    with pytest.raises(ValueError, match="DP2 routing"):
        module.validate_c1_report(report, manifest=manifest)


def test_c1_runner_owns_fixed_scenario_order_and_cleanup() -> None:
    qualification = _module()
    runner = _c1_module()
    manifest = {"schema": 1, "identity": "manifest"}
    expected = _c1_report(qualification, manifest)
    calls: list[str] = []

    class Adapter:
        def production_components(self):
            return expected["production_components"]

        def topology(self):
            return expected["topology"]

        def run_scenario(self, name):
            calls.append(name)
            return expected["scenarios"][name]

        def integrity_counters(self):
            return expected["integrity"]

        def diagnostics(self):
            return expected["diagnostics"]

        def comparisons(self):
            return expected["comparisons"]

        def dp_equivalence(self):
            return expected["dp_equivalence"]

        def close(self):
            calls.append("close")

    report = runner.run_c1(Adapter(), manifest)

    assert calls == [*qualification.C1_SCENARIOS, "close"]
    assert report["kind"] == "direct_remote_lmcache_c1_report"
    qualification.validate_c1_report(report, manifest=manifest)


def test_p1_evaluation_applies_gates_without_enabling_o2() -> None:
    module = _module()
    manifest = {"schema": 1, "identity": "manifest"}
    report = {
        "schema": 1,
        "kind": "direct_remote_lmcache_p1_report",
        "qualification_manifest_sha256": module.payload_sha256(manifest),
        "modes": {
            name: {
                "diagnostics_enabled": False,
                "mode_contract": module.P1_MODE_CONTRACTS[name],
                "measured_repetitions": 30,
                "warmups_separate": True,
                "cache_state": "cold",
                "cold_trial_isolation": ("clear_all_tiers_before_each_measured_batch"),
                "qualification_manifest_sha256": module.payload_sha256(manifest),
                "workload_spec_sha256": "b" * 64,
            }
            for name in ("A", "B", "C")
        },
        "disabled_ttft_regression_percent": 0.5,
        "disabled_throughput_regression_percent": 0.5,
        "clean_prearm_fallback_ttft_regression_percent": 0.5,
        "median_ttft_gain_ms": 750,
        "median_ttft_gain_ci_low_ms": 600,
        "decoder_cache_ready_reduction_percent": 75,
        "decoder_throughput_regression_percent": 2,
        "decoder_p99_regression_percent": 4,
        "publication_efficiency_percent": 97,
        "retained_at_load_percent": 95,
        "p95_native_gap_ms": 12,
        "p50_native_duration_ms": 100,
        "post_prefill_tail_ms": 100,
        "full_window_native_overlap_percent": 90,
        "native_gap_material_to_ttft": True,
        "retention_assessed_as_usable": True,
        "clock_safe_traces": True,
        "client_ttft_authoritative": True,
        "active_decoder_profile": "moderate-production-replay",
        "raw_evidence": ["raw/p1.jsonl"],
        "workload_tiers": {
            "tier1": "PASS",
            "tier2": "PASS",
            "tier3": "PASS",
            "tier4": "PENDING_TARGETED_ABLATION",
        },
    }
    decision = module.evaluate_p1_report(report, manifest=manifest)
    assert decision["proceed_with_experimental_architecture"] is True
    assert decision["production_target_pass"] is True
    assert decision["o1_lookahead_eligible"] is True
    assert decision["o2_advance_eligible"] is False

    report["modes"]["C"]["mode_contract"] = "new_code_feature_disabled"
    with pytest.raises(ValueError, match="wrong execution path"):
        module.evaluate_p1_report(report, manifest=manifest)
    report["modes"]["C"]["mode_contract"] = module.P1_MODE_CONTRACTS["C"]
    report["publication_efficiency_percent"] = 90
    decision = module.evaluate_p1_report(report, manifest=manifest)
    assert decision["proceed_with_experimental_architecture"] is False
    assert decision["healthy_discarded_bytes_gate_pass"] is False
