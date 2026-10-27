# SPDX-License-Identifier: Apache-2.0
"""Exercise full imported control-plane modules in isolated CPU processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/vllm_cpu_e2e.py"


def run_case(tmp_path, *args):
    report = tmp_path / "report.json"
    env = dict(os.environ, PYTHONHASHSEED="0")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--requests",
            "2",
            "--rounds",
            "2",
            "--decode-tokens",
            "4",
            "--output",
            str(report),
            *args,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=150,
        env=env,
    )
    (tmp_path / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result, report


@pytest.mark.parametrize("banks,seed", [(2, 1), (4, 0), (8, 2)])
def test_full_pd_control_path(tmp_path, banks, seed):
    result, report = run_case(tmp_path, "--banks", str(banks), "--seed", str(seed))
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    data = json.loads(report.read_text())
    assert data["status"] == "passed"
    assert data["tp"] == 4
    assert data["native_model_runner"]
    assert data["decode_tokens"] == 16
    assert data["mixed_batches"] > 0
    assert data["checked_rows"] > 10000
    assert data["remote_puts"] > 0 and data["remote_gets"] > 0


@pytest.mark.parametrize(
    "fault", ["skip-load-wait", "skip-save-wait", "corrupt-remote"]
)
def test_detects_missing_wait_and_transport_corruption(tmp_path, fault):
    result, report = run_case(tmp_path, "--fault", fault, "--rounds", "1")
    assert result.returncode != 0
    assert not report.exists()
    assert "KV mismatch" in result.stdout + result.stderr
