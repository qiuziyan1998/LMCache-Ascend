# SPDX-License-Identifier: Apache-2.0
"""Measure authoritative client TTFT for RemoteFill A/B/C campaigns."""

# Standard
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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
    expected_prompt_tokens: int | None = None
    batch_id: int = 0


def _clock_domain() -> tuple[str, str]:
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = str(round(time.time() - time.monotonic()))
    return host, f"{host}:{boot}"


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Load strict JSONL prompt records with stable case identities."""

    prompts: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != {
            "case_id",
            "prompt",
            "expected_prompt_tokens",
        }:
            raise ValueError(f"invalid prompt record at line {line_number}")
        if not all(
            isinstance(record[key], str) and record[key]
            for key in ("case_id", "prompt")
        ) or not (
            isinstance(record["expected_prompt_tokens"], int)
            and record["expected_prompt_tokens"] > 0
        ):
            raise ValueError(f"empty prompt record at line {line_number}")
        prompts.append(record)
    if not prompts or len({item["case_id"] for item in prompts}) != len(prompts):
        raise ValueError("prompt cases must be nonempty and uniquely identified")
    return prompts


def build_work_items(
    prompts: Iterable[dict[str, Any]],
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
        WorkItem(
            record["case_id"],
            record["prompt"],
            index,
            True,
            int(record["expected_prompt_tokens"]),
        )
        for record in records
        for index in range(warmups)
    ]
    measured_items = [
        WorkItem(
            record["case_id"],
            record["prompt"],
            index,
            False,
            int(record["expected_prompt_tokens"]),
        )
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


def _generated_content(payload: Mapping[str, Any]) -> str:
    """Return generated text from one completion or chat-completion event."""

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content
    return ""


def _read_streaming_response(
    response: Any,
) -> tuple[float, float, int, str, int]:
    """Read SSE incrementally and locate the first generated model token."""

    first_byte_at: float | None = None
    first_token_at: float | None = None
    response_bytes = 0
    request_id = ""
    prompt_tokens: int | None = None
    pending = b""
    read_chunk = getattr(response, "read1", None)
    if not callable(read_chunk):
        read_chunk = response.read
    while True:
        chunk = read_chunk(4096)
        now = time.perf_counter()
        if not chunk:
            break
        if first_byte_at is None:
            first_byte_at = now
        response_bytes += len(chunk)
        pending += chunk
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            line = raw_line.strip()
            if not line or line.startswith(b":"):
                continue
            if not line.startswith(b"data:"):
                continue
            encoded = line[5:].strip()
            if encoded == b"[DONE]":
                continue
            try:
                event = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("stream contained malformed JSON data") from error
            if not isinstance(event, dict):
                raise RuntimeError("stream event must be a JSON object")
            if event.get("error"):
                raise RuntimeError(f"stream returned an error: {event['error']}")
            if not request_id and isinstance(event.get("id"), str):
                request_id = event["id"]
            usage = event.get("usage")
            if isinstance(usage, Mapping) and isinstance(
                usage.get("prompt_tokens"), int
            ):
                prompt_tokens = int(usage["prompt_tokens"])
            if first_token_at is None and _generated_content(event):
                first_token_at = now
    if first_byte_at is None:
        raise RuntimeError("response completed without a body byte")
    if first_token_at is None:
        raise RuntimeError("response completed without a generated token")
    if prompt_tokens is None or prompt_tokens <= 0:
        raise RuntimeError("response did not report positive prompt-token usage")
    return first_byte_at, first_token_at, response_bytes, request_id, prompt_tokens


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
    """Run one streaming request and measure the first generated token."""

    payload: dict[str, Any] = {
        "prompt": item.prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
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
        "run_id": trial_id,
        "batch_id": item.batch_id,
        "repetition_id": item.repetition,
    }
    host, clock_domain = _clock_domain()
    send_wall_ns = time.time_ns()
    send_monotonic = time.perf_counter()
    try:
        with opener(request, timeout=timeout_seconds) as response:
            (
                first_byte_monotonic,
                first_token_monotonic,
                response_bytes,
                streamed_request_id,
                prompt_tokens,
            ) = _read_streaming_response(response)
            complete_monotonic = time.perf_counter()
            request_id = response.headers.get("X-Request-Id", "") or streamed_request_id
            if not request_id:
                raise RuntimeError("response did not provide a request identity")
            if (
                item.expected_prompt_tokens is not None
                and prompt_tokens != item.expected_prompt_tokens
            ):
                raise RuntimeError(
                    "server prompt-token usage does not match the workload: "
                    f"expected={item.expected_prompt_tokens}, observed={prompt_tokens}"
                )
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
            "ttfb_ms": round((first_byte_monotonic - send_monotonic) * 1000, 3),
            "first_generated_token_ms": round(
                (first_token_monotonic - send_monotonic) * 1000, 3
            ),
            # Compatibility alias; every gate now means first generated token.
            "ttft_ms": round((first_token_monotonic - send_monotonic) * 1000, 3),
            "complete_ms": round((complete_monotonic - send_monotonic) * 1000, 3),
            "response_bytes": response_bytes,
            "prompt_tokens": prompt_tokens,
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


def _clustered_medians(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    """Collapse correlated concurrent requests into repetition-batch samples."""

    clusters: dict[tuple[Any, Any], list[float]] = {}
    for row in rows:
        cluster = (row.get("run_id"), row.get("batch_id"))
        if cluster[0] in (None, "") or not isinstance(cluster[1], int):
            raise ValueError("measured trials require run_id and integer batch_id")
        clusters.setdefault(cluster, []).append(float(row[field]))
    return [statistics.median(clusters[key]) for key in sorted(clusters)]


def summarize(
    results: Iterable[dict[str, Any]], *, measured_wall_seconds: float | None = None
) -> dict[str, Any]:
    """Summarize measured successful trials without mixing warmups/errors."""

    rows = list(results)
    measured = [row for row in rows if not row["warmup"]]
    if any(not row.get("ok") for row in measured):
        raise ValueError("measured qualification trials contain request failures")
    successful = [row for row in measured if row.get("ok")]
    values = [float(row["ttft_ms"]) for row in successful]
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
    cluster_values = _clustered_medians(successful, "ttft_ms")
    confidence_low, confidence_high = _median_confidence_interval(cluster_values)
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
        "successful": len(successful),
        "failed": len(measured) - len(values),
        "independent_batches": len(cluster_values),
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
                row.get("prompt_tokens"),
                row.get("batch_id"),
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
    ordered_identities = sorted(identities)
    gains = [
        indexed["A"][identity] - indexed["C"][identity]
        for identity in ordered_identities
    ]
    clustered_gains: dict[Any, list[float]] = {}
    for identity, gain in zip(ordered_identities, gains, strict=True):
        clustered_gains.setdefault(identity[5], []).append(gain)
    independent_gains = [
        statistics.median(clustered_gains[batch])
        for batch in sorted(clustered_gains)
    ]
    median_gain = statistics.median(gains)
    gain_low, gain_high = _median_confidence_interval(independent_gains)
    median_a = statistics.median(a_values)
    median_b = statistics.median(b_values)
    disabled_regression = (median_b - median_a) / median_a * 100 if median_a else 0.0
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_abc_comparison",
        "paired_samples": len(identities),
        "independent_batches": len(independent_gains),
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
        "release_sample_count_pass": len(independent_gains) >= 30,
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
    # Local import keeps the standalone workload helpers importable in tests.
    import remote_fill_qualification as qualification

    qualification.validate_manifest(manifest, allow_dirty=False)
    if args.cache_state == "cold" and args.warmups:
        raise ValueError(
            "cold qualification requires --warmups=0 unless a deployment adapter "
            "provides reset evidence between warmup and measured requests"
        )
    qualification_manifest_sha256 = payload_sha256(manifest)
    workload_spec_sha256 = payload_sha256(
        {
            "workload_id": args.workload_id,
            "prompts": [
                {
                    "case_id": record["case_id"],
                    "prompt_sha256": sha256(record["prompt"].encode()).hexdigest(),
                    "expected_prompt_tokens": record["expected_prompt_tokens"],
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
        batched = [
            replace(item, batch_id=index // args.concurrency)
            for index, item in enumerate(selected)
        ]
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
                batched,
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
