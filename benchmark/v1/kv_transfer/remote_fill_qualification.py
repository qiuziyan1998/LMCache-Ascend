# SPDX-License-Identifier: Apache-2.0
"""Generate and validate Direct Remote LMCache qualification manifests."""

# Standard
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import importlib.metadata
import json
import math
import subprocess
import sys


SCHEMA_VERSION = 1
REPOSITORY_NAMES = ("LMCache-NPU", "LMCache-Ascend", "vllm", "vllm-ascend")
VERIFICATION_ITEMS = tuple(f"V{index}" for index in range(1, 9))
HARDWARE_VERIFICATION_ITEMS = frozenset({"V1", "V2", "V6"})
EVIDENCE_STATUSES = frozenset(
    {
        "PASS_UNIT",
        "PASS_INSPECTION",
        "PENDING_HARDWARE",
        "FIXED_AND_PASS",
        "NOT_APPLICABLE_WITH_PROOF",
    }
)
H0_CASES = tuple(f"H0-{suffix}" for suffix in "ABCDEFG")
H0_PRODUCTION_COMPONENTS = (
    "global_te",
    "source_registration",
    "source_page_planner",
    "decoder_local_cpu_allocator",
    "native_transport",
)
C1_SCENARIOS = (
    "cold_miss",
    "joinable_partial_prefix",
    "decoder_local_first_hole",
    "full_persistent_prefix_hit",
    "final_partial_page",
    "mtp_first_resume",
    "tp8_passive_materialization",
    "dp2_routing",
    "repeated_prefix_and_eviction",
)
C1_ZERO_INVARIANTS = (
    "stale_source_reads",
    "one_group_only_uses",
    "cross_rank_frontier_disagreements",
    "committed_key_overwrites",
    "false_persistent_completions",
    "destination_reuse_during_possible_dma",
    "invalid_partial_page_interpretations",
    "mtp_store_frontier_disagreements",
    "stale_epoch_generation_publications",
)
C1_COMPARISONS = (
    "source_to_decoder_local_group0",
    "source_to_decoder_local_group1",
    "direct_to_persistent",
    "selected_topk",
    "first_resume_hidden_logits",
    "eager_graph",
)
C1_PRODUCTION_COMPONENTS = (
    "vllm",
    "ascend_multi_connector",
    "lmcache",
    "mooncake",
    "tp8_shared_local_cpu",
    "dp2_routing",
    "mtp",
)
P1_MODE_CONTRACTS = {
    "A": "existing_production_path",
    "B": "new_code_feature_disabled",
    "C": "conservative_remote_fill",
}
P1_COLD_TRIAL_ISOLATION = {
    "clear_all_tiers_before_each_measured_batch",
    "unique_namespace_per_measured_batch",
}
REQUIRED_INVENTORY_PATHS = (
    "artifacts.mooncake_build",
    "artifacts.mooncake_checksum",
    "artifacts.container_image_digest",
    "software.python",
    "software.pytorch",
    "software.torch_npu",
    "software.cann",
    "software.driver",
    "software.firmware",
    "model.serving_bundle_artifact_id",
    "model.weights_identity",
    "model.tokenizer_identity",
    "model.quantization_identity",
    "model.custom_code_identity",
    "cache.namespace",
    "cache.layout_tag",
    "cache.page_abi",
    "cache.dtype",
    "cache.token_hash_algorithm",
    "cache.python_hash_seed",
    "cache.chunk_size",
    "cache.remote_fill_window_size",
    "cache.group_dimensions",
    "cache.cache_layer_count",
    "cache.mtp",
    "topology.tp",
    "topology.dp",
    "topology.ep",
    "topology.flashcomm",
    "topology.graph_mode",
    "network.nic",
    "network.link_state",
    "network.routing",
    "network.mtu",
    "network.cpu_socket",
    "network.numa_policy",
    "network.memory_placement",
    "network.global_te_session",
    "network.mooncake_segment_placement",
    "clock.source",
    "clock.max_observed_host_offset_us",
)


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_snapshot(root: Path) -> dict[str, Any]:
    """Return exact Git identity for one qualification repository."""

    resolved = root.resolve(strict=True)
    submodules = [
        line
        for line in _run_git(
            resolved, "submodule", "status", "--recursive"
        ).splitlines()
        if line
    ]
    return {
        "path": str(resolved),
        "branch": _run_git(resolved, "branch", "--show-current") or "<detached>",
        "sha": _run_git(resolved, "rev-parse", "HEAD"),
        "tree_sha": _run_git(resolved, "rev-parse", "HEAD^{tree}"),
        "remote_origin": _run_git(resolved, "config", "--get", "remote.origin.url"),
        "submodules": submodules,
        "dirty": bool(_run_git(resolved, "status", "--porcelain")),
    }


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an immutable artifact."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable digest used to bind reports to one manifest."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _value_at(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _usable(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value)) and value >= 0
    return True


def is_sha256(value: Any, *, prefixed: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if prefixed and not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:") if prefixed else value
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest.lower()
    )


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    converted = float(value)
    return math.isfinite(converted) and (
        minimum is None or converted >= minimum
    )


def _evidence_artifact_contains_identity(
    path: Path,
    record: Mapping[str, Any],
) -> bool:
    """Verify an evidence artifact contains its own immutable binding header."""

    decoder = json.JSONDecoder()
    expected = {
        field: record.get(field)
        for field in (
            "producing_host",
            "clock_domain",
            "trial_id",
            "manifest_sha256",
            "adapter_module_sha256",
        )
    }
    try:
        with path.open(encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if len(line) > 1024 * 1024:
                    continue
                start = line.find("{")
                if start < 0:
                    continue
                try:
                    payload, _ = decoder.raw_decode(line[start:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                identity = payload.get("qualification_evidence_identity", payload)
                if not isinstance(identity, Mapping):
                    continue
                if all(
                    identity.get(field) == value
                    for field, value in expected.items()
                ):
                    return identity.get("command") == record.get("command")
    except (OSError, UnicodeError):
        return False
    return False


def valid_evidence(value: Any) -> bool:
    """Return whether evidence uses verified, immutable artifact records."""

    if not isinstance(value, list) or not value:
        return False
    for record in value:
        if not isinstance(record, Mapping) or record.get("example_only") is not False:
            return False
        path_value = record.get("path")
        adapter_path_value = record.get("adapter_module_path")
        if not isinstance(path_value, str) or not isinstance(
            adapter_path_value, str
        ):
            return False
        path = Path(path_value)
        adapter_path = Path(adapter_path_value)
        if not path.is_absolute() or not adapter_path.is_absolute():
            return False
        try:
            stat = path.stat()
            adapter_stat = adapter_path.stat()
        except OSError:
            return False
        if (
            not path.is_file()
            or not adapter_path.is_file()
            or record.get("size_bytes") != stat.st_size
            or record.get("adapter_module_size_bytes") != adapter_stat.st_size
            or not is_sha256(record.get("sha256"))
            or not is_sha256(record.get("adapter_module_sha256"))
            or file_sha256(path) != record["sha256"]
            or file_sha256(adapter_path) != record["adapter_module_sha256"]
        ):
            return False
        for field in (
            "producing_host",
            "clock_domain",
            "trial_id",
            "manifest_sha256",
        ):
            if not isinstance(record.get(field), str) or not record[field].strip():
                return False
        if not is_sha256(record.get("manifest_sha256")):
            return False
        command = record.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            return False
        if not _evidence_artifact_contains_identity(path, record):
            return False
    return True


def evidence_matches_manifest(value: Any, manifest_sha256: str) -> bool:
    """Verify evidence records and bind each one to the exact manifest."""

    return valid_evidence(value) and all(
        record["manifest_sha256"] == manifest_sha256 for record in value
    )


def validate_manifest(
    manifest: Mapping[str, Any], *, allow_dirty: bool = False
) -> None:
    """Reject an incomplete, ambiguous, or dirty qualification manifest."""

    if manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("unsupported qualification manifest schema")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, Mapping) or set(repositories) != set(
        REPOSITORY_NAMES
    ):
        raise ValueError("manifest must contain exactly the four repositories")
    for name, snapshot in repositories.items():
        if not isinstance(snapshot, Mapping) or not all(
            snapshot.get(field)
            for field in ("path", "branch", "sha", "tree_sha", "remote_origin")
        ):
            raise ValueError(f"repository identity is incomplete: {name}")
        for field in ("sha", "tree_sha"):
            sha = snapshot[field]
            if (
                not isinstance(sha, str)
                or len(sha) not in (40, 64)
                or any(
                    character not in "0123456789abcdef"
                    for character in sha.lower()
                )
            ):
                raise ValueError(f"repository {field} is invalid: {name}")
        if not isinstance(snapshot.get("submodules"), list):
            raise ValueError(f"repository submodule state is invalid: {name}")
        if snapshot.get("dirty") and not allow_dirty:
            raise ValueError(f"qualification repository is dirty: {name}")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("manifest inventory must be a mapping")
    missing = [
        path
        for path in REQUIRED_INVENTORY_PATHS
        if not _usable(_value_at(inventory, path))
    ]
    if missing:
        raise ValueError("manifest inventory is incomplete: " + ", ".join(missing))
    if not is_sha256(_value_at(inventory, "artifacts.mooncake_checksum")):
        raise ValueError("artifacts.mooncake_checksum must be SHA-256")
    if not is_sha256(
        _value_at(inventory, "artifacts.container_image_digest"), prefixed=True
    ):
        raise ValueError("artifacts.container_image_digest must be SHA-256")
    group_dimensions = _value_at(inventory, "cache.group_dimensions")
    if (
        not isinstance(group_dimensions, list)
        or len(group_dimensions) != 2
        or any(not isinstance(value, int) or value <= 0 for value in group_dimensions)
    ):
        raise ValueError("cache.group_dimensions must contain two positive integers")
    for path in (
        "cache.chunk_size",
        "cache.remote_fill_window_size",
        "cache.cache_layer_count",
        "topology.tp",
        "topology.dp",
        "topology.ep",
        "network.mtu",
    ):
        value = _value_at(inventory, path)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} must be a positive integer")
    offset = _value_at(inventory, "clock.max_observed_host_offset_us")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not math.isfinite(float(offset))
        or offset < 0
    ):
        raise ValueError("clock.max_observed_host_offset_us must be nonnegative")
    python_hash_seed = _value_at(inventory, "cache.python_hash_seed")
    if (
        isinstance(python_hash_seed, bool)
        or not isinstance(python_hash_seed, int)
        or python_hash_seed < 0
    ):
        raise ValueError("cache.python_hash_seed must be nonnegative")
    if not isinstance(_value_at(inventory, "topology.flashcomm"), bool):
        raise ValueError("topology.flashcomm must be boolean")
    mtp = _value_at(inventory, "cache.mtp")
    if not isinstance(mtp, Mapping) or not isinstance(mtp.get("enabled"), bool):
        raise ValueError("cache.mtp must contain an enabled boolean")
    speculative = mtp.get("num_speculative_tokens")
    if (
        isinstance(speculative, bool)
        or not isinstance(speculative, int)
        or speculative < 0
    ):
        raise ValueError("cache.mtp.num_speculative_tokens must be nonnegative")


def validate_evidence_matrix(
    matrix: Mapping[str, Any], *, require_hardware: bool = False
) -> None:
    """Validate the complete V1--V8 claimed-fix evidence matrix.

    ``PENDING_HARDWARE`` is permitted while preparing H0, but never for a C1
    exit or a production-readiness record.
    """

    if set(matrix) != set(VERIFICATION_ITEMS):
        raise ValueError("evidence matrix must contain exactly V1 through V8")
    for item in VERIFICATION_ITEMS:
        record = matrix[item]
        if not isinstance(record, Mapping):
            raise ValueError(f"{item} evidence record must be a mapping")
        status = record.get("status")
        if status not in EVIDENCE_STATUSES:
            raise ValueError(f"{item} has an unsupported evidence status")
        if require_hardware and status == "PENDING_HARDWARE":
            raise ValueError(f"{item} remains pending hardware qualification")
        if (
            require_hardware
            and item in HARDWARE_VERIFICATION_ITEMS
            and status != "FIXED_AND_PASS"
        ):
            raise ValueError(f"{item} lacks passing hardware evidence")
        evidence = record.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(reference, str) or not reference.strip()
                for reference in evidence
            )
        ):
            raise ValueError(f"{item} must contain concrete evidence references")
        if status == "NOT_APPLICABLE_WITH_PROOF" and not record.get("proof"):
            raise ValueError(f"{item} requires explicit non-applicability proof")


def build_qualification_record(
    manifest: Mapping[str, Any],
    evidence_matrix: Mapping[str, Any],
    *,
    require_hardware: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Bind evidence to one exact manifest without changing either payload."""

    validate_manifest(manifest, allow_dirty=allow_dirty)
    validate_evidence_matrix(evidence_matrix, require_hardware=require_hardware)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "direct_remote_lmcache_qualification_record",
        "manifest": json.loads(json.dumps(manifest, sort_keys=True)),
        "claimed_fix_evidence": json.loads(json.dumps(evidence_matrix, sort_keys=True)),
    }


def _require_zero(record: Mapping[str, Any], *fields: str) -> None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, bool) or value != 0:
            raise ValueError(f"safety field must be zero: {field}")


def _reject_private_native_fields(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            if name.endswith(("_ptr", "_address", "_capability")):
                raise ValueError(f"private native capability in {path}.{name}")
            _reject_private_native_fields(nested, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_private_native_fields(nested, f"{path}[{index}]")


def validate_h0_report(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    minimum_soak_iterations: int = 100,
) -> None:
    """Reject an incomplete or unsafe H0 hardware report.

    The report is produced by the isolated deployment driver because only D
    can inspect its registered LocalCPU destination bytes. This validator
    deliberately accepts no raw pointer or capability fields.
    """

    if report.get("example_only") is True:
        raise ValueError("example H0 reports cannot qualify production")
    if report.get("schema") != SCHEMA_VERSION or report.get("kind") != (
        "direct_remote_lmcache_h0_report"
    ):
        raise ValueError("H0 report has an invalid schema or kind")
    if minimum_soak_iterations < 100:
        raise ValueError("H0 smoke qualification requires at least 100 transfers")
    _reject_private_native_fields(report)
    if report.get("activation") != "mooncake-sync-write-visible-v1":
        raise ValueError("H0 report has the wrong native activation contract")
    if not is_sha256(report.get("qualification_manifest_sha256")):
        raise ValueError("H0 report is not bound to a qualification manifest")
    if manifest is not None and report["qualification_manifest_sha256"] != (
        payload_sha256(manifest)
    ):
        raise ValueError("H0 report was produced for a different manifest")
    observed_layout = report.get("observed_layout")
    if not isinstance(observed_layout, Mapping):
        raise ValueError("H0 report lacks the observed production layout")
    if manifest is not None:
        expected_layout = {
            "chunk_size": _value_at(manifest, "inventory.cache.chunk_size"),
            "window_tokens": _value_at(
                manifest, "inventory.cache.remote_fill_window_size"
            ),
            "group_dimensions": _value_at(
                manifest, "inventory.cache.group_dimensions"
            ),
            "cache_layer_count": _value_at(
                manifest, "inventory.cache.cache_layer_count"
            ),
            "dtype": _value_at(manifest, "inventory.cache.dtype"),
            "page_abi": _value_at(manifest, "inventory.cache.page_abi"),
        }
        if dict(observed_layout) != expected_layout:
            raise ValueError("H0 observed layout differs from the manifest")
    components = report.get("production_components")
    if not isinstance(components, Mapping) or any(
        components.get(component) is not True for component in H0_PRODUCTION_COMPONENTS
    ):
        raise ValueError("H0 report did not use every required production component")
    cases = report.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(H0_CASES):
        raise ValueError("H0 report must contain exactly H0-A through H0-G")
    for case_name, case in cases.items():
        if not isinstance(case, Mapping) or case.get("status") != "PASS":
            raise ValueError(f"{case_name} did not pass")
        if not evidence_matches_manifest(
            case.get("evidence"), report["qualification_manifest_sha256"]
        ):
            raise ValueError(f"{case_name} has no raw evidence reference")
    full = cases["H0-A"]
    if not all(
        full.get(field) is True
        for field in (
            "group0_full_verified",
            "group1_full_verified",
            "combined_verified",
            "noncontiguous_runs_verified",
            "destination_identity_verified",
        )
    ):
        raise ValueError("H0-A did not prove every full-page/vector case")
    _require_zero(
        cases["H0-A"],
        "byte_mismatches",
        "canary_mismatches",
        "vector_order_mismatches",
    )
    _require_zero(
        cases["H0-B"],
        "byte_mismatches",
        "unused_byte_mutations",
        "valid_token_mismatches",
    )
    partial = cases["H0-B"]
    valid_tokens = partial.get("valid_tokens_tested")
    if (
        not isinstance(valid_tokens, list)
        or len(valid_tokens) != 4
        or any(not isinstance(value, int) or value <= 0 for value in valid_tokens)
        or partial.get("cold_compact_valid_tokens_verified") is not True
    ):
        raise ValueError("H0-B partial-page coverage is incomplete")
    if manifest is not None:
        chunk_size = _value_at(manifest, "inventory.cache.chunk_size")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise ValueError("H0 manifest lacks a valid cache chunk size")
        expected = sorted(
            {1, max(1, chunk_size // 4), max(1, chunk_size // 2), chunk_size - 1}
        )
        if valid_tokens != expected:
            raise ValueError("H0-B valid_tokens matrix differs from the manifest")
    _require_zero(cases["H0-C"], "late_source_reads", "late_destination_mutations")
    if (
        cases["H0-C"].get("source_reuse_verified") is not True
        or not isinstance(cases["H0-C"].get("guard_seconds"), (int, float))
        or isinstance(cases["H0-C"].get("guard_seconds"), bool)
        or cases["H0-C"]["guard_seconds"] <= 0
    ):
        raise ValueError("H0-C terminal visibility proof is incomplete")
    _require_zero(cases["H0-D"], "late_destination_mutations")
    if (
        cases["H0-D"].get("destination_reuse_verified") is not True
        or not isinstance(cases["H0-D"].get("guard_seconds"), (int, float))
        or isinstance(cases["H0-D"].get("guard_seconds"), bool)
        or cases["H0-D"]["guard_seconds"] <= 0
    ):
        raise ValueError("H0-D destination reuse proof is incomplete")
    soak = cases["H0-E"]
    iterations = soak.get("iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < minimum_soak_iterations
    ):
        raise ValueError("H0-E did not reach the required soak length")
    _require_zero(
        soak,
        "registration_leaks",
        "owner_leaks",
        "reservation_leaks",
        "future_leaks",
        "attempt_record_leaks",
        "overlapping_registration_failures",
    )
    bandwidth = cases["H0-F"].get("groups")
    bandwidth_window = cases["H0-F"].get("window_tokens")
    if (
        isinstance(bandwidth_window, bool)
        or not isinstance(bandwidth_window, int)
        or bandwidth_window < 4096
    ):
        raise ValueError("H0-F did not use production-sized windows")
    if not isinstance(bandwidth, Mapping) or set(bandwidth) != {
        "group0",
        "group1",
        "combined",
    }:
        raise ValueError("H0-F requires Group 0, Group 1, and combined bandwidth")
    for name, metrics in bandwidth.items():
        if not isinstance(metrics, Mapping) or any(
            not isinstance(metrics.get(field), (int, float))
            or isinstance(metrics.get(field), bool)
            or metrics[field] <= 0
            for field in (
                "bytes",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "native_p50_ms",
                "effective_gbps",
                "cold_samples",
                "warm_samples",
            )
        ):
            raise ValueError(f"H0-F has invalid {name} bandwidth metrics")
        setup = metrics.get("setup_p50_ms")
        if not isinstance(setup, (int, float)) or isinstance(setup, bool) or setup < 0:
            raise ValueError(f"H0-F has invalid {name} setup metrics")
    dual = cases["H0-G"]
    dual_window = dual.get("window_tokens")
    if (
        isinstance(dual_window, bool)
        or not isinstance(dual_window, int)
        or dual_window < 4096
    ):
        raise ValueError("H0-G did not use production-sized windows")
    for field in (
        "persistent_ms",
        "direct_ms",
        "dual_ms",
        "concurrency_efficiency",
        "persistent_slowdown",
        "direct_slowdown",
    ):
        value = dual.get(field)
        if not _finite_number(value, minimum=0.0):
            raise ValueError(f"H0-G is missing {field}")
    if not all(
        dual.get(field) is True
        for field in (
            "persistent_verified",
            "direct_verified",
            "same_source_generation",
            "same_manifest_digest",
        )
    ):
        raise ValueError("H0-G did not prove dual-sink data integrity")
    for field in (
        "persistent_byte_count",
        "direct_byte_count",
        "source_byte_count",
    ):
        if not _finite_number(dual.get(field), minimum=1.0):
            raise ValueError(f"H0-G is missing {field}")
    if not (
        dual["persistent_byte_count"]
        == dual["direct_byte_count"]
        == dual["source_byte_count"]
    ):
        raise ValueError("H0-G sink byte counts differ")


def validate_c1_report(
    report: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> None:
    """Validate the complete production-topology C1 correctness record."""

    if report.get("example_only") is True:
        raise ValueError("example C1 reports cannot qualify production")
    if report.get("schema") != SCHEMA_VERSION or report.get("kind") != (
        "direct_remote_lmcache_c1_report"
    ):
        raise ValueError("C1 report has an invalid schema or kind")
    if report.get("qualification_manifest_sha256") != payload_sha256(manifest):
        raise ValueError("C1 report was produced for a different manifest")
    components = report.get("production_components")
    if not isinstance(components, Mapping) or any(
        components.get(component) is not True for component in C1_PRODUCTION_COMPONENTS
    ):
        raise ValueError("C1 did not use every required production component")
    topology = report.get("topology")
    if not isinstance(topology, Mapping) or any(
        not isinstance(topology.get(field), int)
        or isinstance(topology.get(field), bool)
        or topology[field] < minimum
        for field, minimum in (("tp", 8), ("dp", 2), ("num_speculative_tokens", 1))
    ):
        raise ValueError("C1 requires TP8, DP2, and MTP first-resume coverage")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(C1_SCENARIOS):
        raise ValueError("C1 report must contain every required scenario")
    for name, scenario in scenarios.items():
        if not isinstance(scenario, Mapping) or scenario.get("status") != "PASS":
            raise ValueError(f"C1 scenario did not pass: {name}")
        if not evidence_matches_manifest(
            scenario.get("evidence"), report["qualification_manifest_sha256"]
        ):
            raise ValueError(f"C1 scenario has no evidence: {name}")
    cold = scenarios["cold_miss"]
    if cold.get("terminal_outcome") not in {"LOCAL_FULL", "PERSISTENT_ONLY"}:
        raise ValueError("C1 cold miss has an unsafe terminal outcome")
    for field in ("persistent_common_end", "required_store_end"):
        value = cold.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"C1 cold miss has an invalid {field}")
    if cold["persistent_common_end"] < cold["required_store_end"]:
        raise ValueError("C1 cold miss persistence is incomplete")
    partial = scenarios["joinable_partial_prefix"]
    if not all(
        partial.get(field) is True
        for field in (
            "persistent_boundary_used",
            "paired_local_prefix_used",
            "both_groups_common_frontier",
        )
    ):
        raise ValueError("C1 joinable partial-prefix proof is incomplete")
    first_hole = scenarios["decoder_local_first_hole"]
    if not all(
        first_hole.get(field) is True
        for field in (
            "abandoned_before_arm",
            "no_suffix_publication",
            "persistent_authoritative",
        )
    ):
        raise ValueError("C1 first-hole fallback proof is incomplete")
    full_hit = scenarios["full_persistent_prefix_hit"]
    _require_zero(full_hit, "transactions", "native_calls", "retained_source_bytes")
    final_partial = scenarios["final_partial_page"]
    if not all(
        final_partial.get(field) is True
        for field in (
            "valid_tokens_verified",
            "both_groups_verified",
            "persistence_verified",
            "mtp_frontier_verified",
        )
    ):
        raise ValueError("C1 final partial-page proof is incomplete")
    mtp = scenarios["mtp_first_resume"]
    if not all(
        mtp.get(field) is True
        for field in (
            "required_store_end_verified",
            "final_token_recompute_verified",
            "speculative_row_ownership_verified",
            "computed_token_count_verified",
            "execution_route_verified",
            "cache_selection_verified",
        )
    ):
        raise ValueError("C1 MTP first-resume proof is incomplete")
    tp8 = scenarios["tp8_passive_materialization"]
    if not all(
        tp8.get(field) is True
        for field in (
            "group0_failure_collective_fallback",
            "group1_failure_collective_fallback",
            "computed_token_counts_agree",
        )
    ):
        raise ValueError("C1 TP8 passive-rank failure coverage is incomplete")
    dp2 = scenarios["dp2_routing"]
    if set(dp2.get("mappings", ())) != {"P0-D0", "P0-D1", "P1-D0", "P1-D1"}:
        raise ValueError("C1 DP2 routing matrix is incomplete")
    _require_zero(dp2, "nonselected_destination_allocated_pages")
    repeated = scenarios["repeated_prefix_and_eviction"]
    if not all(
        repeated.get(field) is True
        for field in (
            "immediate_retained_hit_verified",
            "evicted_persistent_fallback_verified",
            "model_output_acceptable",
        )
    ):
        raise ValueError("C1 repeated-prefix/eviction proof is incomplete")
    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("C1 integrity counters are missing")
    _require_zero(integrity, *C1_ZERO_INVARIANTS)
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("bounded") is not True:
        raise ValueError("C1 diagnostics must be explicitly bounded")
    if not isinstance(diagnostics.get("diagnosed_requests"), int) or not (
        0 < diagnostics["diagnosed_requests"] <= 32
    ):
        raise ValueError("C1 diagnosed request count is invalid")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(C1_COMPARISONS):
        raise ValueError("C1 integrity comparison matrix is incomplete")
    for name, comparison in comparisons.items():
        if not isinstance(comparison, Mapping) or comparison.get("status") not in {
            "PASS",
            "NOT_APPLICABLE_WITH_PROOF",
        }:
            raise ValueError(f"C1 comparison did not pass: {name}")
        if not evidence_matches_manifest(
            comparison.get("evidence"), report["qualification_manifest_sha256"]
        ):
            raise ValueError(f"C1 comparison lacks evidence: {name}")
        if comparison["status"] == "NOT_APPLICABLE_WITH_PROOF" and not comparison.get(
            "proof"
        ):
            raise ValueError(f"C1 comparison lacks non-applicability proof: {name}")
    dp_equivalence = report.get("dp_equivalence")
    if not isinstance(dp_equivalence, Mapping) or (
        dp_equivalence.get("measured") is not True
        or not (
            dp_equivalence.get("cross_dp_reuse_qualified") is True
            or dp_equivalence.get("execution_identity_namespaced") is True
        )
    ):
        raise ValueError("C1 DP equivalence/reuse decision is incomplete")


def evaluate_p1_report(
    report: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a P1 aggregate and evaluate only the documented gates."""

    if report.get("example_only") is True:
        raise ValueError("example P1 reports cannot qualify production")
    if report.get("schema") != SCHEMA_VERSION or report.get("kind") != (
        "direct_remote_lmcache_p1_report"
    ):
        raise ValueError("P1 report has an invalid schema or kind")
    manifest_hash = payload_sha256(manifest)
    if report.get("qualification_manifest_sha256") != manifest_hash:
        raise ValueError("P1 report was produced for a different manifest")
    modes = report.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != {"A", "B", "C"}:
        raise ValueError("P1 requires exactly A, B, and C modes")
    if not isinstance(report.get("release_claim", False), bool):
        raise ValueError("P1 release_claim must be boolean")
    cache_states = set()
    workload_specs = set()
    for name, mode in modes.items():
        if not isinstance(mode, Mapping) or (
            mode.get("diagnostics_enabled") is not False
        ):
            raise ValueError(f"P1 mode {name} enabled content diagnostics")
        if mode.get("mode_contract") != P1_MODE_CONTRACTS[name]:
            raise ValueError(f"P1 mode {name} used the wrong execution path")
        if (
            mode.get("cache_state") == "cold"
            and mode.get("cold_trial_isolation") not in P1_COLD_TRIAL_ISOLATION
        ):
            raise ValueError(f"P1 mode {name} did not isolate cold trials")
        minimum_repetitions = 30 if report.get("release_claim") is True else 10
        if mode.get("measured_repetitions", 0) < minimum_repetitions:
            raise ValueError(
                f"P1 mode {name} has fewer than {minimum_repetitions} repetitions"
            )
        if mode.get("independent_batches", 0) < minimum_repetitions:
            raise ValueError(
                f"P1 mode {name} has fewer than {minimum_repetitions} "
                "independent repetition batches"
            )
        if mode.get("warmups_separate") is not True:
            raise ValueError(f"P1 mode {name} mixed warmup and measured trials")
        if mode.get("qualification_manifest_sha256") != manifest_hash:
            raise ValueError(f"P1 mode {name} used a different manifest")
        workload_specs.add(mode.get("workload_spec_sha256"))
        cache_states.add(mode.get("cache_state"))
    if cache_states != {"cold"}:
        raise ValueError("P1 A/B/C modes must use isolated cold cache states")
    if len(workload_specs) != 1 or not is_sha256(next(iter(workload_specs), None)):
        raise ValueError("P1 A/B/C modes used different workload specifications")
    if (
        report.get("clock_safe_traces") is not True
        or report.get("client_ttft_authoritative") is not True
    ):
        raise ValueError("P1 lacks clock-safe trace/client TTFT evidence")
    active_profile = report.get("active_decoder_profile")
    raw_evidence = report.get("raw_evidence")
    if (
        not isinstance(active_profile, str)
        or not active_profile.strip()
        or not evidence_matches_manifest(raw_evidence, manifest_hash)
    ):
        raise ValueError("P1 lacks active-load or raw evidence")
    workload_tiers = report.get("workload_tiers")
    if not isinstance(workload_tiers, Mapping) or any(
        workload_tiers.get(tier) != "PASS" for tier in ("tier1", "tier2", "tier3")
    ):
        raise ValueError("P1 workload campaign is incomplete")
    required_numbers = (
        "disabled_ttft_regression_percent",
        "disabled_throughput_regression_percent",
        "clean_prearm_fallback_ttft_regression_percent",
        "median_ttft_gain_ms",
        "median_ttft_gain_ci_low_ms",
        "decoder_cache_ready_reduction_percent",
        "decoder_throughput_regression_percent",
        "decoder_p99_regression_percent",
        "publication_efficiency_percent",
        "retained_at_load_percent",
        "local_full_sample_count",
        "local_full_rate_percent",
        "normal_path_fatal_restarts",
        "failed_measured_requests",
        "missing_traces",
        "unmatched_request_ids",
        "p95_native_gap_ms",
        "p50_native_duration_ms",
        "post_prefill_tail_ms",
        "full_window_native_overlap_percent",
    )
    values = {}
    for field in required_numbers:
        value = report.get(field)
        if not _finite_number(value):
            raise ValueError(f"P1 is missing numeric field: {field}")
        values[field] = float(value)
    if not isinstance(report.get("native_gap_material_to_ttft"), bool):
        raise ValueError("P1 native-gap materiality decision must be boolean")
    disabled_gate = (
        values["disabled_ttft_regression_percent"] <= 1
        and values["disabled_throughput_regression_percent"] <= 1
    )
    fallback_gate = values["clean_prearm_fallback_ttft_regression_percent"] <= 1
    publication_gate = values["publication_efficiency_percent"] >= 95
    retention_gate = (
        values["retained_at_load_percent"] >= 90
        and values["local_full_sample_count"] >= 30
        and values["local_full_rate_percent"] >= 90
    )
    clean_execution_gate = all(
        values[field] == 0
        for field in (
            "normal_path_fatal_restarts",
            "failed_measured_requests",
            "missing_traces",
            "unmatched_request_ids",
        )
    )
    proceed = (
        disabled_gate
        and fallback_gate
        and publication_gate
        and retention_gate
        and clean_execution_gate
        and values["decoder_cache_ready_reduction_percent"] >= 70
        and values["median_ttft_gain_ms"] >= 500
        and values["median_ttft_gain_ci_low_ms"] > 0
        and values["decoder_throughput_regression_percent"] <= 3
        and values["decoder_p99_regression_percent"] <= 5
    )
    lookahead = (
        values["p95_native_gap_ms"] >= max(5.0, 0.1 * values["p50_native_duration_ms"])
        and report.get("native_gap_material_to_ttft") is True
    )
    return {
        "proceed_with_experimental_architecture": proceed,
        "production_target_pass": (
            proceed
            and values["median_ttft_gain_ms"] >= 700
            and values["decoder_throughput_regression_percent"] <= 3
            and values["decoder_p99_regression_percent"] <= 5
            and values["full_window_native_overlap_percent"] >= 80
        ),
        "disabled_path_gate_pass": disabled_gate,
        "clean_fallback_gate_pass": fallback_gate,
        "healthy_discarded_bytes_gate_pass": publication_gate,
        "retention_gate_pass": retention_gate,
        "clean_execution_gate_pass": clean_execution_gate,
        "o1_lookahead_eligible": lookahead,
        "o2_advance_eligible": False,
        "o2_reason": "evaluate only after an O1 hardware result",
        "two_native_writes_eligible": False,
        "two_native_writes_reason": "requires a separate H0 concurrency experiment",
        "reported_retained_at_load_percent": values["retained_at_load_percent"],
        "reported_post_prefill_tail_ms": values["post_prefill_tail_ms"],
        "reported_full_window_native_overlap_percent": values[
            "full_window_native_overlap_percent"
        ],
    }


def build_manifest(
    inventory: Mapping[str, Any],
    repository_roots: Mapping[str, Path],
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Build a validated manifest without inventing unavailable runtime facts."""

    if set(repository_roots) != set(REPOSITORY_NAMES):
        raise ValueError("repository roots must name exactly the four repositories")
    normalized = json.loads(json.dumps(inventory, sort_keys=True, allow_nan=False))
    if not isinstance(normalized, dict):
        raise ValueError("inventory must be a mapping")
    software = normalized.setdefault("software", {})
    if not isinstance(software, dict):
        raise ValueError("inventory software must be a mapping")
    software.setdefault("python", sys.version.split()[0])
    for package, field in (("torch", "pytorch"), ("torch-npu", "torch_npu")):
        try:
            software.setdefault(field, importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {
            name: repository_snapshot(repository_roots[name])
            for name in REPOSITORY_NAMES
        },
        "inventory": normalized,
    }
    validate_manifest(manifest, allow_dirty=allow_dirty)
    return manifest


def default_repository_roots() -> dict[str, Path]:
    """Return repository roots for the documented four-repository workspace."""

    ascend = Path(__file__).resolve().parents[3]
    lmcache_root = ascend.parent
    vllm_root = lmcache_root.parent / "vllm-overall"
    return {
        "LMCache-NPU": lmcache_root / "LMCache-NPU",
        "LMCache-Ascend": ascend,
        "vllm": vllm_root / "vllm",
        "vllm-ascend": vllm_root / "vllm-ascend",
    }


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="operator runtime inventory JSON")
    parser.add_argument("output", type=Path, help="qualification manifest JSON")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record a development run; never use for release qualification",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="optional V1-V8 evidence JSON to bind to this exact manifest",
    )
    parser.add_argument(
        "--require-hardware",
        action="store_true",
        help="reject PENDING_HARDWARE evidence (C1/release records)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    manifest = build_manifest(
        inventory,
        default_repository_roots(),
        allow_dirty=args.allow_dirty,
    )
    output: Mapping[str, Any] = manifest
    if args.evidence is not None:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        output = build_qualification_record(
            manifest,
            evidence,
            require_hardware=args.require_hardware,
            allow_dirty=args.allow_dirty,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
