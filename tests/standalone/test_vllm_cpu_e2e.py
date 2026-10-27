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
            "--context-tokens",
            "635",
            "--batch-tokens",
            "256",
            "--cache-chunk-tokens",
            "128",
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
    assert data["transfer_validation"]["guarded_reads"] > 0
    assert data["transfer_validation"]["early_reads"] == 0


@pytest.mark.parametrize(
    "fault", ["skip-load-wait", "skip-save-wait", "early-ready-event", "corrupt-remote"]
)
def test_detects_missing_wait_and_transport_corruption(tmp_path, fault):
    result, report = run_case(tmp_path, "--fault", fault, "--rounds", "1")
    assert result.returncode != 0
    assert not report.exists()
    assert any(
        message in result.stdout + result.stderr
        for message in ("KV mismatch", "read before completion")
    )


def test_poison_is_visible_until_completion():
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from cpu_vllm_runtime import CPUHardware

    hardware = CPUHardware(ROOT)
    source = torch.full((32,), 7, dtype=torch.uint8)
    target = source.clone()
    hardware.queue.memory(source, source)
    hardware.queue.memory(target, target)
    hardware.dma([(target.data_ptr(), source.data_ptr(), 32)], False)
    event = hardware.Event()
    event.record()
    assert not event.query()
    hardware.queue.step()  # Device starts the DMA, but has not completed it.
    assert target.tolist() == [255] * 32
    assert not event.query()
    event.synchronize()
    assert event.query() and torch.equal(source, target)


def test_early_reader_fails_even_if_stale_bytes_are_identical():
    import ctypes
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from cpu_vllm_runtime import CPUHardware

    hardware = CPUHardware(ROOT)
    target = torch.full((32,), 7, dtype=torch.uint8)
    hardware.queue.memory(target, target)
    writer = hardware.submit_access(
        "load same bytes",
        lambda: ctypes.memset(target.data_ptr(), 7, 32),
        writes=[(target.data_ptr(), 32)],
    )
    consumer = hardware.Stream()
    with hardware.stream(consumer):
        read = hardware.submit_access(
            "compute early",
            lambda: None,
            reads=[(target.data_ptr(), 32)],
        )
    assert not writer.done and target.tolist() == [7] * 32
    with pytest.raises(AssertionError, match="read before completion"):
        read.action()


def test_sparse_transport_preserves_unordered_duplicate_selection():
    from types import SimpleNamespace
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from cpu_vllm_runtime import CPUHardware

    hardware = CPUHardware(ROOT)
    nope = torch.arange(16, dtype=torch.float16).reshape(8, 2)
    rope = torch.arange(100, 108, dtype=torch.float16).reshape(8, 1)
    chunks = [
        torch.cat(
            (nope[start : start + 4].flatten(), rope[start : start + 4].flatten())
        )
        for start in (0, 4)
    ]
    caches = [torch.full_like(nope, -9), torch.full_like(rope, -9)]
    for tensor in chunks + caches:
        hardware.queue.memory(tensor, tensor)
    selected, slots = torch.tensor([6, 0, 6, 3]), torch.tensor([5, 1, 2, 3])
    hardware.sparse_copy(
        SimpleNamespace(caches=caches),
        slots,
        selected,
        torch.tensor([chunk.data_ptr() for chunk in chunks]),
        4,
        8,
    )
    assert bool((caches[0] == -9).all())
    hardware.queue.drain()
    assert torch.equal(caches[0][slots], nope[selected])
    assert torch.equal(caches[1][slots], rope[selected])
    assert bool((caches[0][torch.tensor([0, 4, 6, 7])] == -9).all())
