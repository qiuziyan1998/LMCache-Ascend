# SPDX-License-Identifier: Apache-2.0
"""Run the fixed staged RemoteFill P1 A/B/C campaign through a launcher adapter."""

# Standard
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol
import importlib.util
import json


MODE_CONTRACTS = {
    "A": "existing_production_path",
    "B": "new_code_feature_disabled",
    "C": "conservative_remote_fill",
}
COLD_TRIAL_ISOLATION = {
    "clear_all_tiers_before_each_measured_batch",
    "unique_namespace_per_measured_batch",
}


class P1Adapter(Protocol):
    """Deployment adapter that activates modes and resets cache state."""

    def prepare_mode(self, case: Mapping[str, Any], mode: str) -> None: ...

    def run_mode(self, case: Mapping[str, Any], mode: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _qualification_module():
    path = Path(__file__).with_name("remote_fill_qualification.py")
    spec = importlib.util.spec_from_file_location("remote_fill_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RemoteFill qualification contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cases(
    max_supported_tokens: int,
    *,
    tiers: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    """Return the documented staged matrix without a full Cartesian product."""

    if max_supported_tokens < 131072:
        raise ValueError("P1 maximum prompt length must be at least 128K")
    selected = set(tiers)
    if not selected or not selected <= {1, 2, 3, 4}:
        raise ValueError("P1 tiers must be a nonempty subset of 1 through 4")
    cases: list[dict[str, Any]] = []
    for length in (32768, 131072):
        cases.append(
            dict(
                case_id=f"tier1-{length}",
                tier=1,
                prompt_tokens=length,
                concurrency=1,
                decoder_load="idle",
            )
        )
    for length in dict.fromkeys((32768, 65536, 131072, max_supported_tokens)):
        cases.append(
            dict(
                case_id=f"tier2-{length}",
                tier=2,
                prompt_tokens=length,
                concurrency=1,
                decoder_load="idle",
            )
        )
    for load in ("idle", "moderate", "heavy"):
        for concurrency in (1, 8, 16, 32):
            cases.append(
                dict(
                    case_id=f"tier3-{load}-c{concurrency}",
                    tier=3,
                    prompt_tokens=131072,
                    concurrency=concurrency,
                    decoder_load=load,
                )
            )
    cases.append(
        dict(
            case_id="tier4-balanced-dp2",
            tier=4,
            prompt_tokens=131072,
            concurrency=8,
            decoder_load="moderate",
            topology="balanced_dp2",
        )
    )
    return tuple(case for case in cases if case["tier"] in selected)


def run_campaign(
    adapter: P1Adapter,
    manifest: Mapping[str, Any],
    *,
    max_supported_tokens: int,
    tiers: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Run cold-isolated A/B/C trials in alternated order and validate identity."""

    qualification = _qualification_module()
    records = []
    try:
        qualification.validate_manifest(manifest, allow_dirty=False)
        manifest_hash = qualification.payload_sha256(manifest)
        for index, case in enumerate(build_cases(max_supported_tokens, tiers=tiers)):
            order = tuple("ABC"[(index + offset) % 3] for offset in range(3))
            modes: dict[str, dict[str, Any]] = {}
            for mode in order:
                adapter.prepare_mode(case, mode)
                result = dict(adapter.run_mode(case, mode))
                if (
                    result.get("mode") != mode
                    or result.get("mode_contract") != MODE_CONTRACTS[mode]
                    or result.get("qualification_manifest_sha256") != manifest_hash
                    or result.get("diagnostics_enabled") is not False
                    or result.get("warmups_separate") is not True
                    or result.get("cache_state") != "cold"
                    or result.get("cache_isolation_verified") is not True
                    or result.get("cold_trial_isolation") not in COLD_TRIAL_ISOLATION
                    or result.get("prompt_tokens") != case["prompt_tokens"]
                    or result.get("concurrency") != case["concurrency"]
                    or result.get("decoder_load") != case["decoder_load"]
                    or result.get("measured_repetitions", 0) < 10
                    or result.get("independent_batches", 0) < 10
                    or not qualification.is_sha256(result.get("workload_spec_sha256"))
                    or not qualification.evidence_matches_manifest(
                        result.get("raw_evidence"), manifest_hash
                    )
                ):
                    raise ValueError(
                        f"invalid P1 result for {case['case_id']} mode {mode}"
                    )
                modes[mode] = result
            if len({value["workload_spec_sha256"] for value in modes.values()}) != 1:
                raise ValueError(f"A/B/C workload mismatch for {case['case_id']}")
            records.append({"case": case, "execution_order": order, "modes": modes})
        return {
            "schema": 1,
            "kind": "direct_remote_lmcache_p1_campaign",
            "qualification_manifest_sha256": manifest_hash,
            "tiers": sorted(set(tiers)),
            "cases": records,
        }
    finally:
        adapter.close()


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="deployment module:factory")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-supported-tokens", required=True, type=int)
    parser.add_argument(
        "--tier",
        action="append",
        type=int,
        choices=(1, 2, 3, 4),
        help="campaign tier to run; defaults to Tier 1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    module, separator, factory = args.adapter.partition(":")
    if not separator:
        raise ValueError("adapter must use module:factory syntax")
    adapter = getattr(import_module(module), factory)(config)
    report = run_campaign(
        adapter,
        manifest,
        max_supported_tokens=args.max_supported_tokens,
        tiers=tuple(args.tier or (1,)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
