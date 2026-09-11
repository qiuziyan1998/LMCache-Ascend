# SPDX-License-Identifier: Apache-2.0
"""Build a clock-safe per-request RemoteFill stage trace from cold-perf logs."""

# Standard
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Iterable
import json
import math
import statistics


_MARKER = "[LMCACHE_COLD_PERF]"


def _decode_payload(line: str) -> dict[str, Any] | None:
    marker = line.find(_MARKER)
    if marker < 0:
        return None
    start = line.find("{", marker + len(_MARKER))
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(line[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _matches_trace(payload: dict[str, Any], trace: str) -> bool:
    identities = [
        payload.get("req_id"),
        payload.get("proxy_request_id"),
        payload.get("transfer_id"),
        payload.get("remote_fill_transfer_id"),
    ]
    identities.extend(payload.get("request_ids") or ())
    return any(trace == str(identity) for identity in identities if identity)


def load_events(role: str, path: Path, trace: str) -> list[dict[str, Any]]:
    """Load one role's matching events and enforce clock identity fields."""

    events = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            payload = _decode_payload(line)
            if payload is None or not _matches_trace(payload, trace):
                continue
            clock_valid = all(
                payload.get(field) for field in ("host", "clock_domain")
            ) and (
                isinstance(payload.get("wall_time_ns"), int)
                and isinstance(payload.get("monotonic_ms"), (int, float))
            )
            if not clock_valid:
                raise ValueError(
                    f"{path} contains a matching event without clock identity"
                )
            events.append({**payload, "role": role, "source_log": str(path)})
    return events


def _first(events: Iterable[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matching = [event for event in events if event.get("event") == name]
    if not matching:
        return None
    domains = {event["clock_domain"] for event in matching}
    if len(domains) != 1:
        raise ValueError(f"{name} crossed clock domains")
    return min(matching, key=lambda event: float(event["monotonic_ms"]))


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return a disjoint union of valid monotonic intervals."""

    ordered = sorted((start, end) for start, end in intervals if end >= start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _local_delta(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> float | None:
    if start is None or end is None:
        return None
    if start["clock_domain"] != end["clock_domain"]:
        raise ValueError("refusing to subtract timestamps from different clock domains")
    return round(float(end["monotonic_ms"]) - float(start["monotonic_ms"]), 3)


def _sum_field(events: Iterable[dict[str, Any]], field: str) -> float:
    return round(sum(float(event.get(field, 0.0)) for event in events), 3)


def _event_values(
    events: Iterable[dict[str, Any]], event_name: str, field: str
) -> list[float]:
    return [
        float(event[field])
        for event in events
        if event.get("event") == event_name
        and isinstance(event.get(field), (int, float))
        and not isinstance(event.get(field), bool)
    ]


def _prefiller_model_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [
        event
        for event in events
        if event.get("role") == "prefiller"
        and event.get("event") == "prefiller_model_chunk_complete"
        and event.get("tp_rank") == 0
    ]
    if not chunks:
        return {
            "chunk_count": 0,
            "model_forward_cpu_ms": None,
            "model_prefill_span_ms": None,
        }
    domains = {event["clock_domain"] for event in chunks}
    if len(domains) != 1:
        raise ValueError("prefiller model chunks crossed clock domains")
    starts = [float(event["model_started_monotonic_ms"]) for event in chunks]
    ends = [float(event["model_ended_monotonic_ms"]) for event in chunks]
    return {
        "chunk_count": len(chunks),
        "clock_domain": next(iter(domains)),
        "model_started_monotonic_ms": min(starts),
        "model_ended_monotonic_ms": max(ends),
        "model_forward_cpu_ms": round(
            sum(float(event["model_forward_cpu_ms"]) for event in chunks), 3
        ),
        "model_prefill_span_ms": round(max(ends) - min(starts), 3),
    }


def build_trace(
    events: Iterable[dict[str, Any]],
    *,
    trace: str,
    client_trials: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Produce only local-clock durations plus authoritative client TTFT."""

    rows = list(events)
    if not rows:
        raise ValueError("no cold-performance events matched the trace")
    proxy = [row for row in rows if row["role"] == "proxy"]
    prefiller_model = _prefiller_model_summary(rows)
    windows = sorted(
        (row for row in rows if row.get("event") == "remote_fill_window_complete"),
        key=lambda row: float(row["monotonic_ms"]),
    )
    idle_gaps = []
    native_overlaps = []
    for previous, current in zip(windows, windows[1:], strict=False):
        if previous["clock_domain"] != current["clock_domain"]:
            raise ValueError("RemoteFill windows crossed clock domains")
        if previous.get("native_ended_monotonic_ms") and current.get(
            "native_started_monotonic_ms"
        ):
            delta = float(current["native_started_monotonic_ms"]) - float(
                previous["native_ended_monotonic_ms"]
            )
            idle_gaps.append(round(max(0.0, delta), 3))
            native_overlaps.append(round(max(0.0, -delta), 3))
    commits = [row for row in rows if row.get("event") == "remote_fill_local_commit"]
    actual_loads = [
        row for row in rows if row.get("event") == "remote_fill_actual_load"
    ]
    client = [
        row for row in client_trials if trace == str(row.get("proxy_request_id", ""))
    ]
    client_ttft = [
        float(row["first_generated_token_ms"])
        for row in client
        if row.get("ok")
        and isinstance(row.get("first_generated_token_ms"), (int, float))
    ]
    request_received = _first(proxy, "proxy_request_received")
    prefill_dispatch = _first(proxy, "proxy_prefiller_dispatch")
    prefill_response = _first(proxy, "proxy_prefill_response_received")
    decoder_send = _first(proxy, "proxy_decoder_send_start")
    decoder_first_byte = _first(proxy, "proxy_decoder_first_byte_received")
    transfer_ids = sorted(
        {
            str(value)
            for row in rows
            for value in (
                row.get("transfer_id"),
                row.get("remote_fill_transfer_id"),
            )
            if value
        }
    )
    if len(transfer_ids) > 1:
        raise ValueError(
            "one qualification trace matched multiple RemoteFill transfer attempts"
        )
    source_ready_values = [
        float(row["source_fences_ready_monotonic_ms"])
        for row in windows
        if row.get("direct_satisfied")
        and isinstance(row.get("source_fences_ready_monotonic_ms"), (int, float))
        and float(row["source_fences_ready_monotonic_ms"]) > 0
    ]
    production_frontier = max(source_ready_values) if source_ready_values else None
    model_start = prefiller_model.get("model_started_monotonic_ms")
    model_domain = prefiller_model.get("clock_domain")
    native_intervals = [
        (
            float(row["native_started_monotonic_ms"]),
            float(row["native_ended_monotonic_ms"]),
        )
        for row in windows
        if row.get("direct_satisfied")
        and row.get("native_started_monotonic_ms")
        and row.get("native_ended_monotonic_ms")
    ]
    if (
        native_intervals
        and model_domain is not None
        and any(
            row["clock_domain"] != model_domain
            for row in windows
            if row.get("native_started_monotonic_ms")
        )
    ):
        raise ValueError("prefiller model and native windows crossed clock domains")
    full_intervals = [
        (
            float(row["native_started_monotonic_ms"]),
            float(row["native_ended_monotonic_ms"]),
        )
        for row in windows
        if row.get("full_window") is True
        and row.get("direct_satisfied")
        and row.get("native_started_monotonic_ms")
        and row.get("native_ended_monotonic_ms")
    ]
    native_intervals = _merge_intervals(native_intervals)
    full_intervals = _merge_intervals(full_intervals)
    full_native_ms = sum(end - start for start, end in full_intervals)
    full_overlap_ms = (
        sum(
            max(
                0.0,
                min(end, float(production_frontier))
                - max(start, float(model_start)),
            )
            for start, end in full_intervals
        )
        if model_start is not None and production_frontier is not None
        else 0.0
    )
    return {
        "schema": 1,
        "kind": "direct_remote_lmcache_clock_safe_trace",
        "trace": trace,
        "hosts": sorted({str(row["host"]) for row in rows}),
        "clock_domains": sorted({str(row["clock_domain"]) for row in rows}),
        "transfer_ids": transfer_ids,
        "event_count": len(rows),
        "authoritative_client_ttft_ms": client_ttft,
        "proxy_local_phases_ms": {
            "routing_before_prefill": _local_delta(request_received, prefill_dispatch),
            "prefiller_round_trip": _local_delta(prefill_dispatch, prefill_response),
            "handoff_before_decoder_send": _local_delta(prefill_response, decoder_send),
            "decoder_send_to_first_byte": _local_delta(
                decoder_send, decoder_first_byte
            ),
        },
        "remote_fill_window_totals_ms": {
            field: _sum_field(windows, field)
            for field in (
                "reserve_ms",
                "arm_ms",
                "source_event_wait_ms",
                "source_registration_ms",
                "native_slot_wait_ms",
                "native_ms",
                "report_ms",
            )
        },
        "prefiller_model": prefiller_model,
        "kv_production_frontier_monotonic_ms": production_frontier,
        "kv_production_frontier_basis": "producer_fences_ready",
        "post_prefill_native_tail_ms": (
            round(
                max(
                    0.0,
                    max(end for _, end in native_intervals) - production_frontier,
                ),
                3,
            )
            if native_intervals and production_frontier is not None
            else None
        ),
        "full_window_native_overlap_percent": (
            round(full_overlap_ms / full_native_ms * 100, 3)
            if full_native_ms > 0
            else None
        ),
        "native_idle_gap_ms": idle_gaps,
        "native_overlap_ms": native_overlaps,
        "native_gap_p95_ms": (
            sorted(idle_gaps)[max(0, math.ceil(0.95 * len(idle_gaps)) - 1)]
            if idle_gaps
            else None
        ),
        "finish_control_ms": [
            float(row.get("finish_control_ms", 0.0))
            for row in rows
            if row.get("event") == "remote_fill_producer_terminal"
        ],
        "decoder_commit_ms": [float(row.get("elapsed_ms", 0.0)) for row in commits],
        "decoder_commit_lock_wait_ms": [
            float(row.get("lock_wait_ms", 0.0)) for row in commits
        ],
        "decoder_commit_lock_hold_ms": [
            float(row.get("lock_hold_ms", 0.0)) for row in commits
        ],
        "actual_load_outcomes": [
            {
                "outcome": row.get("outcome"),
                "required_store_end": row.get("required_store_end"),
                "common_local_end": row.get("common_local_end"),
                "actual_load_end": row.get("actual_load_end"),
                "remote_suffix": row.get("remote_suffix"),
            }
            for row in actual_loads
        ],
        "decoder_local_stage_samples_ms": {
            "scheduler_lookup": _event_values(
                (row for row in rows if row["role"] == "decoder"),
                "scheduler_lookup",
                "elapsed_ms",
            ),
            "cache_retrieve": _event_values(
                (row for row in rows if row["role"] == "decoder"),
                "cold_compact_retrieve_complete",
                "elapsed_ms",
            ),
            "worker_load": _event_values(
                (row for row in rows if row["role"] == "decoder"),
                "worker_load_complete",
                "elapsed_ms",
            ),
            "input_prepare": _event_values(
                (row for row in rows if row["role"] == "decoder"),
                "decoder_input_prepare_complete",
                "elapsed_ms",
            ),
            "first_forward": _event_values(
                (
                    row
                    for row in rows
                    if row["role"] == "decoder" and row.get("tp_rank", 0) == 0
                ),
                "decoder_forward_return",
                "cpu_elapsed_ms",
            ),
            "sample": _event_values(
                (row for row in rows if row["role"] == "decoder"),
                "decoder_sample_complete",
                "elapsed_ms",
            ),
        },
        "client_ttft_median_ms": (
            statistics.median(client_ttft) if client_ttft else None
        ),
        "double_counting_note": (
            "window and commit timings explain prefiller round-trip time and are not "
            "additional TTFT components"
        ),
    }


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="role=path; use proxy, prefiller, and decoder roles",
    )
    parser.add_argument("--client", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_entries: list[tuple[str, Path]] = []
    for entry in args.log:
        role, separator, path = entry.partition("=")
        if not separator or role not in {"proxy", "prefiller", "decoder"}:
            raise ValueError("--log must use proxy=, prefiller=, or decoder=")
        log_entries.append((role, Path(path)))
    events = [
        event
        for role, path in log_entries
        for event in load_events(role, path, args.trace)
    ]
    transfer_ids = {
        str(value)
        for event in events
        for value in (
            event.get("transfer_id"),
            event.get("remote_fill_transfer_id"),
        )
        if value
    }
    if len(transfer_ids) > 1:
        raise ValueError(
            "trace matched multiple transfer attempts; select one exact "
            "request identity"
        )
    events.extend(
        event
        for transfer_id in transfer_ids
        for role, path in log_entries
        for event in load_events(role, path, transfer_id)
    )
    unique = {
        (
            event["role"],
            event["source_log"],
            event.get("event"),
            event.get("pid"),
            event.get("monotonic_ms"),
        ): event
        for event in events
    }
    trace = build_trace(
        unique.values(), trace=args.trace, client_trials=_read_jsonl(args.client)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(trace, sort_keys=True))


if __name__ == "__main__":
    main()
