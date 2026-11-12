#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize KVFENCE observations without copying server logs or KV payloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


def dependencies(record: dict[str, Any], name: str = "dependencies") -> list[dict]:
    """Flatten the direct dependencies and one recorded upstream level."""
    found = {}
    for row in record.get(name, []):
        for item in (row, *row.get("after", [])):
            found[item["id"]] = item
    return list(found.values())


def summarize(paths: Iterable[Path], request_id: str | None, top: int) -> dict:
    """Return aggregate sync times and the largest observed waits."""
    begin = {}
    stats = defaultdict(Counter)
    largest = []
    violations = []
    checkpoints = Counter()
    checkpoint_violations = []
    errors = []
    malformed = 0
    decoder = json.JSONDecoder()
    step_requests = {}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if "[KVFENCE] " not in line:
                    continue
                try:
                    record, _ = decoder.raw_decode(line.split("[KVFENCE] ", 1)[1])
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                event = record.get("event")
                step_key = (
                    str(path),
                    record.get("host"),
                    record.get("pid"),
                    record.get("step"),
                )
                if event == "step_requests":
                    step_requests[step_key] = record.get("requests", [])
                    continue
                if event not in {"checkpoint", "sync_begin", "sync_end", "sync_error"}:
                    continue
                deps = dependencies(
                    record, "before_sync" if "before_sync" in record else "dependencies"
                )
                if request_id and not (
                    request_id in (record.get("req") or "")
                    or any(request_id in (row.get("req") or "") for row in deps)
                    or any(request_id in req for req in step_requests.get(step_key, []))
                ):
                    continue
                key = (
                    str(path),
                    record.get("host"),
                    record.get("pid"),
                    record.get("sync"),
                )
                if event == "sync_begin":
                    begin[key] = dict(record, file=str(path), line=number)
                    continue
                loads = [row for row in deps if row.get("kind") == "load"]
                category = (
                    "load_pending"
                    if any(row.get("ready") is False for row in loads)
                    else "load_query_unknown"
                    if any(row.get("ready") is None for row in loads)
                    else "loads_ready"
                    if loads
                    else "no_load_record"
                )
                if event == "checkpoint":
                    checkpoints[category] += 1
                    if record.get("ordering_violation"):
                        checkpoint_violations.append(
                            dict(
                                file=str(path),
                                line=number,
                                pid=record.get("pid"),
                                step=record.get("step"),
                                layer=record.get("layer"),
                                events=record["ordering_violation"][:5],
                            )
                        )
                    continue
                begin.pop(key, None)
                if event == "sync_error":
                    errors.append(
                        dict(
                            file=str(path),
                            line=number,
                            stage=record.get("stage"),
                            error=record.get("error"),
                        )
                    )
                    continue
                stage = record.get("stage", "unknown")
                bucket = stats[stage]
                elapsed = record.get("sync_ms", 0)
                bucket["calls"] += 1
                bucket["sync_ms"] += elapsed
                bucket["max_ms"] = max(bucket["max_ms"], elapsed)
                bucket[category] += 1
                detail = dict(
                    file=str(path),
                    line=number,
                    host=record.get("host"),
                    pid=record.get("pid"),
                    stage=stage,
                    step=record.get("step"),
                    group=record.get("group"),
                    layer=record.get("layer"),
                    bank=record.get("bank"),
                    sync_ms=elapsed,
                    category=category,
                    loads=len(loads),
                    pending_loads=sum(row.get("ready") is False for row in loads),
                    input_marker_pending=any(
                        row.get("kind") == "sfa_inputs" and row.get("ready") is False
                        for row in deps
                    ),
                    requests=sorted({row["req"] for row in deps if row.get("req")})[:3],
                    request_count=len({row["req"] for row in deps if row.get("req")}),
                    pending_after=record.get("pending_after", [])[:5],
                )
                if not detail["requests"]:
                    batch_requests = step_requests.get(step_key, [])
                    detail["requests"] = batch_requests[:3]
                    detail["request_count"] = len(batch_requests)
                if detail["pending_after"]:
                    violations.append(detail)
                largest.append(detail)
                largest.sort(key=lambda row: row["sync_ms"], reverse=True)
                del largest[top:]
    unfinished = next(iter(begin.values()), None)
    if unfinished is not None:
        unfinished = {
            key: unfinished.get(key)
            for key in (
                "file",
                "line",
                "host",
                "pid",
                "step",
                "stage",
                "group",
                "layer",
                "bank",
                "req",
            )
        }
    return dict(
        stages={name: dict(count) for name, count in sorted(stats.items())},
        checkpoints=dict(checkpoints),
        checkpoint_ordering_violations=len(checkpoint_violations),
        first_checkpoint_violations=checkpoint_violations[:top],
        unfinished_syncs=len(begin),
        first_unfinished=unfinished,
        pending_after_count=len(violations),
        first_pending_after=violations[:top],
        sync_errors=errors[:top],
        largest_syncs=largest,
        malformed_records=malformed,
        interpretation="Pending means the event was not complete at the sample; "
        "sync_ms includes compute and host submission, not only DMA. Queries are non-atomic.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--request-id")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    report = summarize(args.logs, args.request_id, args.top)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
