# SPDX-License-Identifier: Apache-2.0
"""Tests for the RemoteFill client-side TTFT runner."""

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
        / "remote_fill_workload.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_workload", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    status = 200
    headers = {"X-Request-Id": "cmpl-trace"}

    def __init__(self) -> None:
        self._parts = [
            b'data: {"id":"cmpl-trace","choices":[{"text":"token"}]}\n\n',
            b'data: {"id":"cmpl-trace","choices":[],"usage":{"prompt_tokens":6}}\n\n',
            b"data: [DONE]\n\n",
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        del size
        return self._parts.pop(0) if self._parts else b""


def test_client_trial_records_proxy_trace_and_ttft() -> None:
    module = _module()
    item = module.WorkItem("case", "prompt", 0, False, 6)
    row = module.run_request(
        item,
        endpoint="http://proxy/v1/completions",
        mode="C",
        trial_id="trial",
        workload_id="workload",
        workload_spec_sha256="spec",
        qualification_manifest_sha256="manifest",
        cache_state="cold",
        model=None,
        max_tokens=1,
        timeout_seconds=1,
        api_key=None,
        opener=lambda *_args, **_kwargs: _Response(),
    )
    assert row["ok"] is True
    assert row["proxy_request_id"] == "cmpl-trace"
    assert row["response_bytes"] > 4
    assert row["ttft_ms"] >= 0
    assert row["first_generated_token_ms"] == row["ttft_ms"]
    assert row["ttfb_ms"] <= row["ttft_ms"]
    assert row["prompt_bytes"] == 6
    assert row["prompt_tokens"] == 6
    assert "prompt" not in row
    assert row["workload_spec_sha256"] == "spec"


def test_client_trial_rejects_prompt_token_mismatch() -> None:
    module = _module()
    item = module.WorkItem("case", "prompt", 0, False, 7)

    row = module.run_request(
        item,
        endpoint="http://proxy/v1/completions",
        mode="C",
        trial_id="trial",
        workload_id="workload",
        workload_spec_sha256="spec",
        qualification_manifest_sha256="manifest",
        cache_state="cold",
        model=None,
        max_tokens=1,
        timeout_seconds=1,
        api_key=None,
        opener=lambda *_args, **_kwargs: _Response(),
    )

    assert row["ok"] is False
    assert "prompt-token usage" in row["error"]


def test_work_items_keep_warmups_before_measured_trials() -> None:
    module = _module()
    items = module.build_work_items(
        [{"case_id": "case", "prompt": "prompt", "expected_prompt_tokens": 6}],
        repetitions=2,
        warmups=2,
        seed=0,
    )
    assert [item.warmup for item in items] == [True, True, False, False]
    assert [item.repetition for item in items] == [0, 1, 0, 1]


def test_summary_excludes_warmup_and_rejects_all_failures() -> None:
    module = _module()
    rows = [
        {"warmup": True, "ok": True, "ttft_ms": 1000, "mode": "A"},
        {
            "warmup": False,
            "ok": True,
            "ttft_ms": 10,
            "mode": "A",
            "trial_id": "trial",
            "workload_id": "workload",
            "workload_spec_sha256": "spec",
            "qualification_manifest_sha256": "manifest",
            "cache_state": "cold",
            "run_id": "trial",
            "batch_id": 0,
        },
        {
            "warmup": False,
            "ok": True,
            "ttft_ms": 20,
            "mode": "A",
            "trial_id": "trial",
            "workload_id": "workload",
            "workload_spec_sha256": "spec",
            "qualification_manifest_sha256": "manifest",
            "cache_state": "cold",
            "run_id": "trial",
            "batch_id": 1,
        },
    ]
    assert module.summarize(rows)["median_ttft_ms"] == 15
    for row in rows:
        row["ok"] = False
    with pytest.raises(ValueError, match="request failures"):
        module.summarize(rows)


def test_abc_comparison_is_paired_and_does_not_claim_complete_gate() -> None:
    module = _module()

    def rows(mode, ttft):
        return [
            {
                "case_id": "case",
                "prompt_sha256": "hash",
                "repetition": repetition,
                "cache_state": "cold",
                "warmup": False,
                "ok": True,
                "mode": mode,
                "ttft_ms": ttft,
                "workload_id": "workload",
                "workload_spec_sha256": "a" * 64,
                "qualification_manifest_sha256": "b" * 64,
                "prompt_tokens": 100,
                "run_id": f"trial-{mode}",
                "batch_id": repetition,
            }
            for repetition in range(10)
        ]

    comparison = module.compare_modes(
        {"A": rows("A", 1000), "B": rows("B", 1005), "C": rows("C", 400)}
    )
    assert comparison["disabled_ttft_gate_pass"] is True
    assert comparison["experimental_ttft_gate_pass"] is True
    assert comparison["proceed_gate_complete"] is False
    mismatched = rows("C", 400)
    mismatched[0]["prompt_sha256"] = "other"
    with pytest.raises(ValueError, match="pairwise comparable"):
        module.compare_modes(
            {"A": rows("A", 1000), "B": rows("B", 1005), "C": mismatched}
        )
    different_workload = rows("C", 400)
    different_workload[0]["workload_spec_sha256"] = "other"
    with pytest.raises(ValueError, match="inconsistent workload identity"):
        module.compare_modes(
            {
                "A": rows("A", 1000),
                "B": rows("B", 1005),
                "C": different_workload,
            }
        )
    different_manifest = rows("C", 400)
    for row in different_manifest:
        row["qualification_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different qualification manifests"):
        module.compare_modes(
            {
                "A": rows("A", 1000),
                "B": rows("B", 1005),
                "C": different_manifest,
            }
        )
