# SPDX-License-Identifier: Apache-2.0
"""Summarize LMCache cold-start transfer and layerwise materialization logs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Any


PERF_MARKER = "[LMCACHE_COLD_PERF] "
READY_RE = re.compile(
    r"\[DSA_COLD_COMPACT\] request=(\S+) generation=(\d+) "
    r"status=(\S+).*?elapsed_ms=([0-9.]+)"
)
LEASE_RE = re.compile(r"LEASE_EXPIRED|lease_expired_before_data_transfer_completed")
FAIL_RE = re.compile(r"status=failed|Traceback|LMCache ERROR")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.3f}"


def fmt_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:,.2f}{unit}"
        amount /= 1024.0
    return f"{amount:,.2f}TiB"


def iter_perf_events(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    cursor = 0
    while True:
        marker = text.find(PERF_MARKER, cursor)
        if marker < 0:
            break
        start = marker + len(PERF_MARKER)
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start
            continue
        if isinstance(payload, dict):
            events.append(payload)
        cursor = start + consumed
    return events


def event_group(event: dict[str, Any]) -> int | None:
    groups = event.get("kv_groups")
    if isinstance(groups, list) and len(groups) == 1:
        return int(groups[0])
    group = event.get("kv_group")
    return int(group) if group is not None else None


def describe(values: list[float]) -> str:
    if not values:
        return "count=0"
    return (
        f"count={len(values)} total={sum(values):,.3f}ms "
        f"avg={mean(values):,.3f}ms p50={percentile(values, 0.50):,.3f}ms "
        f"p95={percentile(values, 0.95):,.3f}ms "
        f"max={max(values):,.3f}ms"
    )


def layer_span(
    events: list[dict[str, Any]],
    event_name: str,
    *,
    rank: int | None = None,
) -> tuple[float | None, list[tuple[int, int, float]]]:
    selected: dict[int, float] = {}
    for event in events:
        if event.get("event") != event_name:
            continue
        if event.get("phase") != "dsa_cold_compact_latent":
            continue
        if event_group(event) != 0:
            continue
        if rank is not None and int(event.get("rank", -1)) != rank:
            continue
        if event.get("status", "ok") != "ok":
            continue
        layer = event.get("layer")
        timestamp = event.get("monotonic_ms")
        if layer is not None and timestamp is not None:
            selected[int(layer)] = float(timestamp)
    ordered = sorted(selected.items())
    if len(ordered) < 2:
        return None, []
    gaps = [
        (left_layer, right_layer, right_time - left_time)
        for (left_layer, left_time), (right_layer, right_time) in zip(
            ordered, ordered[1:]
        )
    ]
    return ordered[-1][1] - ordered[0][1], gaps


def print_slowest(
    title: str,
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    if not rows:
        return
    print(f"  {title}:")
    for row in sorted(
        rows,
        key=lambda item: float(item.get("elapsed_ms", 0.0)),
        reverse=True,
    )[:limit]:
        print(
            "    "
            f"rank={row.get('rank', '-')} layer={row.get('layer', '-')} "
            f"elapsed={fmt_ms(float(row.get('elapsed_ms', 0.0)))}ms "
            f"objects={row.get('objects', '-')}"
        )


def summarize_request(
    req_id: str,
    events: list[dict[str, Any]],
    ready: dict[str, Any] | None,
    pages: list[dict[str, Any]],
    *,
    slow_ms: float,
    top: int,
) -> None:
    print()
    print(f"=== request {req_id} ===")
    if ready:
        print(
            "  result: "
            f"generation={ready['generation']} status={ready['status']} "
            f"cold_elapsed={fmt_ms(ready['elapsed_ms'])}ms"
        )
    else:
        print("  result: no DSA_COLD_COMPACT completion line")

    capacities = [
        event for event in events if event.get("event") == "capacity_preflight"
    ]
    for event in capacities:
        print(
            "  capacity: "
            f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms "
            f"scan_skipped={event.get('scan_skipped', False)} "
            f"fits={event.get('fits')} "
            f"required={fmt_bytes(event.get('required_bytes'))} "
            f"available={fmt_bytes(event.get('available_bytes'))}"
        )

    resolvers = [
        event for event in events if event.get("event") == "remote_resolver"
    ]
    for event in sorted(resolvers, key=lambda item: event_group(item) or -1):
        print(
            "  remote: "
            f"group={event_group(event)} phase={event.get('phase')} "
            f"status={event.get('status')} "
            f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms "
            f"get={fmt_ms(float(event.get('remote_get_ms', 0.0)))}ms "
            f"classify={fmt_ms(float(event.get('classification_ms', 0.0)))}ms "
            f"pin={fmt_ms(float(event.get('pinning_ms', 0.0)))}ms"
        )

    for event in pages:
        name = event.get("event")
        group = event_group(event)
        if name == "mooncake_page_lookup":
            print(
                "  page lookup: "
                f"group={group} found={event.get('found_pages')}/"
                f"{event.get('complete_pages')} "
                f"legacy_keys={event.get('legacy_keys')} "
                f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms"
            )
        elif name == "mooncake_page_get":
            print(
                "  page get: "
                f"group={group} completed={event.get('completed_pages')}/"
                f"{event.get('submitted_pages')} bytes={fmt_bytes(event.get('bytes'))} "
                f"alloc={fmt_ms(float(event.get('allocation_ms', 0.0)))}ms "
                f"submit={fmt_ms(float(event.get('submission_ms', 0.0)))}ms "
                f"transfer={fmt_ms(float(event.get('transfer_ms', 0.0)))}ms "
                f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms"
            )
        elif name == "mooncake_legacy_get":
            print(
                "  legacy get: "
                f"group={group} keys={event.get('keys')} "
                f"bytes_read={fmt_bytes(event.get('bytes_read'))} "
                f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms"
            )

    pointer_batches = [
        event
        for event in events
        if event.get("event") == "sparse_pointer_batch_prepare"
    ]
    for event in pointer_batches:
        print(
            "  rank0 pointer batch: "
            f"objects={event.get('objects')} "
            f"elapsed={fmt_ms(float(event.get('elapsed_ms', 0.0)))}ms "
            f"resolve={fmt_ms(float(event.get('ptr_resolve_ms', 0.0)))}ms "
            f"device_build={fmt_ms(float(event.get('ptr_tensor_build_ms', 0.0)))}ms "
            f"concat={fmt_ms(float(event.get('ptr_tensor_concat_ms', 0.0)))}ms"
        )

    finalizes = [
        event
        for event in events
        if event.get("event") == "passive_pointer_batch_finalize"
    ]
    finalize_times = [float(event.get("elapsed_ms", 0.0)) for event in finalizes]
    if finalize_times:
        print(f"  passive pointer finalize: {describe(finalize_times)}")
        print_slowest("slowest passive finalizes", finalizes, limit=top)

    passive = [
        event
        for event in events
        if event.get("event") == "passive_layer_prepare"
        and event.get("phase") == "dsa_cold_compact_latent"
    ]
    passive_times = [float(event.get("elapsed_ms", 0.0)) for event in passive]
    if passive_times:
        slow = [
            event
            for event in passive
            if float(event.get("elapsed_ms", 0.0)) >= slow_ms
        ]
        print(
            "  passive layer prepare: "
            f"{describe(passive_times)} slow>={slow_ms:g}ms={len(slow)}"
        )
        print_slowest("slowest passive layers", passive, limit=top)

    span, gaps = layer_span(events, "shared_handle_broadcast", rank=0)
    if span is not None:
        gap_values = [gap for _, _, gap in gaps]
        print(
            "  rank0 layer publication: "
            f"span={fmt_ms(span)}ms gaps={describe(gap_values)}"
        )
        for left, right, gap in sorted(
            gaps, key=lambda item: item[2], reverse=True
        )[:top]:
            print(f"    gap layer={left}->{right} {fmt_ms(gap)}ms")

    error_events = [
        event for event in events if event.get("status") == "error"
    ]
    if error_events:
        counts = Counter(str(event.get("event")) for event in error_events)
        print(f"  structured errors: {dict(counts)}")

    ready_ms = float(ready["elapsed_ms"]) if ready else None
    remote_ms = sum(float(event.get("elapsed_ms", 0.0)) for event in resolvers)
    capacity_ms = sum(
        float(event.get("elapsed_ms", 0.0)) for event in capacities
    )
    pointer_ms = sum(
        float(event.get("elapsed_ms", 0.0)) for event in pointer_batches
    )
    passive_finalize_max = max(finalize_times, default=0.0)
    if ready_ms is not None:
        measured = {
            "remote_resolvers_sum": remote_ms,
            "rank0_layer_publication_span": span or 0.0,
            "capacity": capacity_ms,
            "rank0_pointer_batch": pointer_ms,
            "max_passive_pointer_finalize": passive_finalize_max,
        }
        dominant = max(measured, key=measured.get)
        print(
            "  bottleneck hint: "
            f"largest measured component={dominant} "
            f"({fmt_ms(measured[dominant])}ms); components overlap, "
            "so do not add them as an exact critical path."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="vLLM/LMCache D-node log")
    parser.add_argument(
        "--request",
        help="Only summarize one request ID",
    )
    parser.add_argument(
        "--slow-ms",
        type=float,
        default=1000.0,
        help="Threshold for a slow passive layer (default: 1000)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of slowest rows/gaps to show (default: 10)",
    )
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    perf_events = iter_perf_events(text)

    ready_by_req: dict[str, dict[str, Any]] = {}
    for match in READY_RE.finditer(text):
        ready_by_req[match.group(1)] = {
            "generation": int(match.group(2)),
            "status": match.group(3),
            "elapsed_ms": float(match.group(4)),
        }

    events_by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages_by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_pages: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    page_names = {
        "mooncake_page_lookup",
        "mooncake_page_get",
        "mooncake_legacy_get",
    }
    for event in perf_events:
        req_id = event.get("req_id")
        if req_id:
            req = str(req_id)
            events_by_req[req].append(event)
            if event.get("event") == "remote_resolver":
                group = event_group(event)
                pid = int(event.get("pid", -1))
                if group is not None:
                    pages_by_req[req].extend(
                        pending_pages.pop((pid, group), [])
                    )
            continue
        if event.get("event") in page_names:
            group = event_group(event)
            pid = int(event.get("pid", -1))
            if group is not None:
                pending_pages[(pid, group)].append(event)

    request_ids = set(events_by_req) | set(ready_by_req)
    if args.request:
        request_ids = {args.request}
    ordered_requests = sorted(
        request_ids,
        key=lambda req: min(
            (
                float(event.get("monotonic_ms", float("inf")))
                for event in events_by_req.get(req, [])
            ),
            default=float("inf"),
        ),
    )

    print(f"log={args.log}")
    print(
        f"perf_events={len(perf_events)} requests={len(ordered_requests)} "
        f"lease_errors={len(LEASE_RE.findall(text))} "
        f"failure_markers={len(FAIL_RE.findall(text))}"
    )
    for req_id in ordered_requests:
        summarize_request(
            req_id,
            events_by_req.get(req_id, []),
            ready_by_req.get(req_id),
            pages_by_req.get(req_id, []),
            slow_ms=args.slow_ms,
            top=args.top,
        )


if __name__ == "__main__":
    main()

