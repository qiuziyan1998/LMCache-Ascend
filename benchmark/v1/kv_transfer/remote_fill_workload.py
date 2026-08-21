# SPDX-License-Identifier: Apache-2.0
"""Measure authoritative client TTFT for RemoteFill A/B/C campaigns."""

# Standard
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.request import Request, urlopen
import json
import math
import os
import random
import socket
import statistics
import time


@dataclass(frozen=True)
class WorkItem:
    case_id: str
    prompt: str
    repetition: int
    warmup: bool


def _clock_domain() -> tuple[str, str]:
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = str(round(time.time() - time.monotonic()))
    return host, f"{host}:{boot}"


def load_prompts(path: Path) -> list[dict[str, str]]:
    """Load strict JSONL prompt records with stable case identities."""

    prompts: list[dict[str, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != {"case_id", "prompt"}:
            raise ValueError(f"invalid prompt record at line {line_number}")
        if not all(isinstance(record[key], str) and record[key] for key in record):
            raise ValueError(f"empty prompt record at line {line_number}")
        prompts.append(record)
    if not prompts or len({item["case_id"] for item in prompts}) != len(prompts):
        raise ValueError("prompt cases must be nonempty and uniquely identified")
    return prompts


def build_work_items(
    prompts: Iterable[dict[str, str]],
    *,
    repetitions: int,
    warmups: int,
    seed: int,
) -> list[WorkItem]:
    """Create separately labelled warmup and measured trials."""

    if repetitions <= 0 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    records = list(prompts)
    warmup_items = [
        WorkItem(record["case_id"], record["prompt"], index, True)
        for record in records
        for index in range(warmups)
    ]
    measured_items = [
        WorkItem(record["case_id"], record["prompt"], index, False)
        for record in records
        for index in range(repetitions)
    ]
    random.Random(seed).shuffle(warmup_items)
    random.Random(seed).shuffle(measured_items)
    return warmup_items + measured_items


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable identity of one JSON-compatible payload."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def run_request(
    item: WorkItem,
    *,
    endpoint: str,
    mode: str,
    trial_id: str,
    workload_id: str,
    workload_spec_sha256: str,
    qualification_manifest_sha256: str,
    cache_state: str,
    model: str | None,
    max_tokens: int,
    timeout_seconds: float,
    api_key: str | None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Run one streaming request and record first body-byte latency."""

    payload: dict[str, Any] = {
        "prompt": item.prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if model:
        payload["model"] = model
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Qualification-Trial": trial_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=body, headers=headers, method="POST")
    item_identity = {
        "case_id": item.case_id,
        "prompt_sha256": sha256(item.prompt.encode()).hexdigest(),
        "prompt_bytes": len(item.prompt.encode()),
        "repetition": item.repetition,
        "warmup": item.warmup,
    }
    host, clock_domain = _clock_domain()
    send_wall_ns = time.time_ns()
    send_monotonic = time.perf_counter()
    try:
        with opener(request, timeout=timeout_seconds) as response:
            first = response.read(1)
            first_monotonic = time.perf_counter()
            remaining = response.read()
            complete_monotonic = time.perf_counter()
            if not first:
                raise RuntimeError("response completed without a body byte")
            request_id = response.headers.get("X-Request-Id", "")
            status = int(getattr(response, "status", 200))
        return {
            "schema": 1,
            "kind": "direct_remote_lmcache_client_trial",
            "mode": mode,
            "trial_id": trial_id,
            "workload_id": workload_id,
            "workload_spec_sha256": workload_spec_sha256,
            "qualification_manifest_sha256": qualification_manifest_sha256,
            "cache_state": cache_state,
            **item_identity,
            "host": host,
            "clock_domain": clock_domain,
            "client_send_wall_time_ns": send_wall_ns,
            "client_send_monotonic_ms": round(send_monotonic * 1000, 3),
            "proxy_request_id": request_id,
            "http_status": status,
            "ttft_ms": round((first_monotonic - send_monotonic) * 1000, 3),
            "complete_ms": round((complete_monotonic - send_monotonic) * 1000, 3),
            "response_bytes": len(first) + len(remaining),
            "ok": 200 <= status < 300,
        }
    except Exception as error:
        return {
            "schema": 1,
            "kind": "direct_remote_lmcache_client_trial",
            "mode": mode,
            "trial_id": trial_id,
            "workload_id": workload_id,
            "workload_spec_sha256": workload_spec_sha256,
            "qualification_manifest_sha256": qualification_manifest_sha256,
            "cache_state": cache_state,
            **item_identity,
            "host": host,
            "clock_domain": clock_domain,
            "client_send_wall_time_ns": send_wall_ns,
            "client_send_monotonic_ms": round(send_monotonic * 1000, 3),
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _median_confidence_interval(
    values: list[float], *, samples: int = 2000
) -> tuple[float, float]:
    generator = random.Random(0)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return _nearest_rank(medians, 0.025), _nearest_rank(medians, 0.975)


def summarize(
    results: Iterable[dict[str, Any]], *, measured_wall_seconds: float | None = None
) -> dict[str, Any]:
    """Summarize measured successful trials without mixing warmups/errors."""

    rows = list(results)
    measured = [row for row in rows if not row["warmup"]]
    values = [float(row["ttft_ms"]) for row in measured if row.get("ok")]
    if not values:
        raise ValueError("no successful measured TTFT samples")
    identity_fields = (
        "mode",
        "trial_id",
        "workload_id",
        "workload_spec_sha256",
        "qualification_manifest_sha256",
        "cache_state",
    )
    if any(len({row.get(field) for row in measured}) != 1 for field in identity_fields):
        raise ValueError("measured trials contain mixed qualification identity")
    median = statistics.median(values)
    confidence_low, confidence_high = _median_confidence_interval(values)
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_client_summary",
        "mode": measured[0]["mode"],
        "trial_id": measured[0]["trial_id"],
        "workload_id": measured[0]["workload_id"],
        "workload_spec_sha256": measured[0]["workload_spec_sha256"],
        "qualification_manifest_sha256": measured[0]["qualification_manifest_sha256"],
        "cache_state": measured[0]["cache_state"],
        "warmups_separate": True,
        "measured": len(measured),
        "successful": len(values),
        "failed": len(measured) - len(values),
        "median_ttft_ms": median,
        "median_ttft_95_ci_ms": [confidence_low, confidence_high],
        "p95_ttft_ms": _nearest_rank(values, 0.95),
        "median_absolute_deviation_ms": statistics.median(
            abs(value - median) for value in values
        ),
        "min_ttft_ms": min(values),
        "max_ttft_ms": max(values),
        "request_throughput_per_second": (
            len(values) / measured_wall_seconds
            if measured_wall_seconds is not None and measured_wall_seconds > 0
            else None
        ),
    }


def compare_modes(
    mode_results: Mapping[str, Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare paired A/B/C client trials without mixing cache states."""

    if set(mode_results) != {"A", "B", "C"}:
        raise ValueError("comparison requires exactly A, B, and C modes")
    indexed: dict[str, dict[tuple[Any, ...], float]] = {}
    workload_specs: set[tuple[str, str]] = set()
    manifest_hashes: dict[str, list[str]] = {}
    for mode, rows_iterable in mode_results.items():
        rows = [row for row in rows_iterable if not row["warmup"]]
        if not rows or any(not row.get("ok") for row in rows):
            raise ValueError(f"mode {mode} contains missing or failed measured trials")
        if any(row.get("mode") != mode for row in rows):
            raise ValueError(f"mode {mode} contains mislabelled trials")
        identities_for_mode = {
            (str(row.get("workload_id", "")), str(row.get("workload_spec_sha256", "")))
            for row in rows
        }
        if len(identities_for_mode) != 1 or not all(next(iter(identities_for_mode))):
            raise ValueError(f"mode {mode} has inconsistent workload identity")
        if not _is_sha256(next(iter(identities_for_mode))[1]):
            raise ValueError(f"mode {mode} has an invalid workload digest")
        workload_specs.update(identities_for_mode)
        manifest_hashes[mode] = sorted(
            {str(row.get("qualification_manifest_sha256", "")) for row in rows}
        )
        if len(manifest_hashes[mode]) != 1 or not manifest_hashes[mode][0]:
            raise ValueError(f"mode {mode} has inconsistent manifest identity")
        if not _is_sha256(manifest_hashes[mode][0]):
            raise ValueError(f"mode {mode} has an invalid manifest digest")
        values: dict[tuple[Any, ...], float] = {}
        for row in rows:
            identity = (
                row["case_id"],
                row["prompt_sha256"],
                row["repetition"],
                row["cache_state"],
            )
            if identity in values:
                raise ValueError(f"mode {mode} contains duplicate trial identity")
            values[identity] = float(row["ttft_ms"])
        indexed[mode] = values
    if len(workload_specs) != 1:
        raise ValueError("A/B/C runs use different workload specifications")
    if len({values[0] for values in manifest_hashes.values()}) != 1:
        raise ValueError("A/B/C runs use different qualification manifests")
    identities = set(indexed["A"])
    if any(set(indexed[mode]) != identities for mode in ("B", "C")):
        raise ValueError("A/B/C trials are not pairwise comparable")
    a_values = [indexed["A"][identity] for identity in sorted(identities)]
    b_values = [indexed["B"][identity] for identity in sorted(identities)]
    c_values = [indexed["C"][identity] for identity in sorted(identities)]
    gains = [
        baseline - direct for baseline, direct in zip(a_values, c_values, strict=True)
    ]
    median_gain = statistics.median(gains)
    gain_low, gain_high = _median_confidence_interval(gains)
    median_a = statistics.median(a_values)
    median_b = statistics.median(b_values)
    disabled_regression = (median_b - median_a) / median_a * 100 if median_a else 0.0
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_abc_comparison",
        "paired_samples": len(identities),
        "workload_id": next(iter(workload_specs))[0],
        "workload_spec_sha256": next(iter(workload_specs))[1],
        "qualification_manifest_sha256": {
            mode: values[0] for mode, values in manifest_hashes.items()
        },
        "cache_states": sorted({identity[3] for identity in identities}),
        "median_ttft_ms": {
            "A": median_a,
            "B": median_b,
            "C": statistics.median(c_values),
        },
        "disabled_regression_percent": disabled_regression,
        "median_remote_fill_gain_ms": median_gain,
        "median_remote_fill_gain_95_ci_ms": [gain_low, gain_high],
        "disabled_ttft_gate_pass": disabled_regression <= 1.0,
        "experimental_ttft_gate_pass": median_gain >= 500 and gain_low > 0,
        "release_sample_count_pass": len(identities) >= 30,
        "proceed_gate_complete": False,
        "proceed_gate_incomplete_reason": (
            "decoder critical-path reduction and active-load interference "
            "must be joined from the stage trace"
        ),
    }


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--mode", required=True, choices=("A", "B", "C"))
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--qualification-manifest", required=True, type=Path)
    parser.add_argument("--cache-state", required=True, choices=("cold", "warm"))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_tokens <= 0 or args.concurrency <= 0 or args.timeout_seconds <= 0:
        raise ValueError("max-tokens, concurrency, and timeout must be positive")
    prompts = load_prompts(args.prompts)
    manifest = json.loads(args.qualification_manifest.read_text(encoding="utf-8"))
    qualification_manifest_sha256 = payload_sha256(manifest)
    workload_spec_sha256 = payload_sha256(
        {
            "workload_id": args.workload_id,
            "prompts": [
                {
                    "case_id": record["case_id"],
                    "prompt_sha256": sha256(record["prompt"].encode()).hexdigest(),
                }
                for record in prompts
            ],
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "cache_state": args.cache_state,
        }
    )
    items = build_work_items(
        prompts,
        repetitions=args.repetitions,
        warmups=args.warmups,
        seed=args.seed,
    )
    api_key = os.environ.get(args.api_key_env)
    warmup_items = [item for item in items if item.warmup]
    measured_items = [item for item in items if not item.warmup]

    def execute(executor: ThreadPoolExecutor, selected: list[WorkItem]):
        return list(
            executor.map(
                lambda item: run_request(
                    item,
                    endpoint=args.endpoint,
                    mode=args.mode,
                    trial_id=args.trial_id,
                    workload_id=args.workload_id,
                    workload_spec_sha256=workload_spec_sha256,
                    qualification_manifest_sha256=qualification_manifest_sha256,
                    cache_state=args.cache_state,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                    api_key=api_key,
                ),
                selected,
            )
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = execute(executor, warmup_items)
        measured_started = time.perf_counter()
        results.extend(execute(executor, measured_items))
        measured_wall_seconds = time.perf_counter() - measured_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    summary = summarize(results, measured_wall_seconds=measured_wall_seconds)
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
