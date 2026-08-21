# SPDX-License-Identifier: Apache-2.0
"""Run C1 through a deployment-owned production topology adapter."""

# Standard
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol
import importlib.util
import json


class C1Adapter(Protocol):
    """Adapter implemented beside the real TP8/DP2/MTP serving launcher."""

    def production_components(self) -> Mapping[str, bool]: ...

    def topology(self) -> Mapping[str, int]: ...

    def run_scenario(self, name: str) -> Mapping[str, Any]: ...

    def integrity_counters(self) -> Mapping[str, int]: ...

    def diagnostics(self) -> Mapping[str, Any]: ...

    def comparisons(self) -> Mapping[str, Mapping[str, Any]]: ...

    def dp_equivalence(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _qualification_module():
    path = Path(__file__).with_name("remote_fill_qualification.py")
    spec = importlib.util.spec_from_file_location("remote_fill_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RemoteFill qualification contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_c1(
    adapter: C1Adapter,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the exact C1 matrix and return a validated manifest-bound report."""

    qualification = _qualification_module()
    try:
        report = {
            "schema": qualification.SCHEMA_VERSION,
            "kind": "direct_remote_lmcache_c1_report",
            "qualification_manifest_sha256": qualification.payload_sha256(manifest),
            "production_components": dict(adapter.production_components()),
            "topology": dict(adapter.topology()),
            "scenarios": {},
        }
        for name in qualification.C1_SCENARIOS:
            report["scenarios"][name] = dict(adapter.run_scenario(name))
        report.update(
            integrity=dict(adapter.integrity_counters()),
            diagnostics=dict(adapter.diagnostics()),
            comparisons={
                name: dict(value) for name, value in adapter.comparisons().items()
            },
            dp_equivalence=dict(adapter.dp_equivalence()),
        )
        qualification.validate_c1_report(report, manifest=manifest)
        return report
    finally:
        adapter.close()


def _load_adapter(specification: str, config: Mapping[str, Any]) -> C1Adapter:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("adapter must use module:factory syntax")
    return getattr(import_module(module_name), factory_name)(config)


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="deployment module:factory")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    report = run_c1(_load_adapter(args.adapter, config), manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
