# SPDX-License-Identifier: Apache-2.0
"""Compare paired client-side RemoteFill A/B/C trial files."""

# Standard
from argparse import ArgumentParser
from pathlib import Path
import importlib.util
import json
import sys


def _workload_module():
    path = Path(__file__).with_name("remote_fill_workload.py")
    spec = importlib.util.spec_from_file_location("remote_fill_workload", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RemoteFill workload contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode-a", required=True, type=Path)
    parser.add_argument("--mode-b", required=True, type=Path)
    parser.add_argument("--mode-c", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    comparison = _workload_module().compare_modes(
        {
            "A": _read_jsonl(args.mode_a),
            "B": _read_jsonl(args.mode_b),
            "C": _read_jsonl(args.mode_c),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
