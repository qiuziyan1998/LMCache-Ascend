# SPDX-License-Identifier: Apache-2.0
"""Run H0-A--G through a deployment-owned production hardware adapter."""

# Standard
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol
import importlib.util
import json


class H0Adapter(Protocol):
    """Production-only adapter implemented beside the deployed P/D launcher.

    Every method must use production GlobalTE, registration, source planner,
    decoder LocalCPU allocator, NIC/NUMA placement, and native transport. Test
    destinations remain hidden and are inspected inside D before release.
    """

    def production_components(self) -> Mapping[str, bool]: ...

    def full_pages(self) -> Mapping[str, Any]: ...

    def partial_pages(self, valid_tokens: tuple[int, ...]) -> Mapping[str, Any]: ...

    def terminal_visibility(self, guard_seconds: float) -> Mapping[str, Any]: ...

    def destination_reuse(self, guard_seconds: float) -> Mapping[str, Any]: ...

    def registration_soak(self, iterations: int) -> Mapping[str, Any]: ...

    def production_bandwidth(self, window_tokens: int) -> Mapping[str, Any]: ...

    def dual_sink(self, window_tokens: int) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _qualification_module():
    path = Path(__file__).with_name("remote_fill_qualification.py")
    spec = importlib.util.spec_from_file_location("remote_fill_qualification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RemoteFill qualification contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_h0(
    adapter: H0Adapter,
    manifest: Mapping[str, Any],
    *,
    chunk_size: int,
    window_tokens: int,
    soak_iterations: int,
    guard_seconds: float,
) -> dict[str, Any]:
    """Run all H0 cases and return a manifest-bound validated report."""

    if chunk_size < 4 or window_tokens < 4096 or soak_iterations < 100:
        raise ValueError("H0 dimensions or soak length are invalid")
    if guard_seconds <= 0:
        raise ValueError("H0 terminal-visibility guard must be positive")
    valid_tokens = tuple(
        sorted({1, max(1, chunk_size // 4), max(1, chunk_size // 2), chunk_size - 1})
    )
    qualification = _qualification_module()
    try:
        report = {
            "schema": 1,
            "kind": "direct_remote_lmcache_h0_report",
            "activation": "mooncake-sync-write-visible-v1",
            "qualification_manifest_sha256": qualification.payload_sha256(manifest),
            "production_components": dict(adapter.production_components()),
            "cases": {},
        }
        cases = report["cases"]
        cases["H0-A"] = dict(adapter.full_pages())
        cases["H0-B"] = dict(adapter.partial_pages(valid_tokens))
        cases["H0-B"]["valid_tokens_tested"] = list(valid_tokens)
        cases["H0-C"] = dict(adapter.terminal_visibility(guard_seconds))
        cases["H0-C"]["guard_seconds"] = guard_seconds
        cases["H0-D"] = dict(adapter.destination_reuse(guard_seconds))
        cases["H0-D"]["guard_seconds"] = guard_seconds
        cases["H0-E"] = dict(adapter.registration_soak(soak_iterations))
        cases["H0-F"] = dict(adapter.production_bandwidth(window_tokens))
        cases["H0-F"]["window_tokens"] = window_tokens
        cases["H0-G"] = dict(adapter.dual_sink(window_tokens))
        cases["H0-G"]["window_tokens"] = window_tokens
        qualification.validate_h0_report(
            report,
            manifest=manifest,
            minimum_soak_iterations=soak_iterations,
        )
        return report
    finally:
        adapter.close()


def _load_adapter(specification: str, config: Mapping[str, Any]) -> H0Adapter:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("adapter must use module:factory syntax")
    factory = getattr(import_module(module_name), factory_name)
    return factory(config)


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="deployment module:factory")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--window-tokens", type=int, default=4096)
    parser.add_argument("--soak-iterations", type=int, default=100)
    parser.add_argument("--guard-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter_config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    adapter = _load_adapter(args.adapter, adapter_config)
    report = run_h0(
        adapter,
        manifest,
        chunk_size=args.chunk_size,
        window_tokens=args.window_tokens,
        soak_iterations=args.soak_iterations,
        guard_seconds=args.guard_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
