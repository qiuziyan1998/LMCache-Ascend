# SPDX-License-Identifier: Apache-2.0
"""Opt-in one-byte-per-token KV observations, without content hashes.

Only the first raw byte of each token/component is exported. Each caller
waits for its device readback and writes the observation before returning.
There are no diagnostic queues or background readers. This intentionally
blocks serving when DEBUG is enabled. Sampling cannot prove equality of
the unsampled bytes.
"""

from __future__ import annotations

# Standard
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any
import json
import os
import re
import socket
import traceback

# Third Party
import torch

# First Party
from lmcache_ascend.v1.kv_trace import (
    _emit,
    diagnostic as diagnostic,
    enabled as enabled,
    packet_planes,
    segments,
)

FORMAT = "token-byte0-v1"
BLOCK_TOKENS = 4096
REFERENCE_SEGMENT_TOKENS = BLOCK_TOKENS


def request_key(req: str) -> str:
    """Return the shared P/D UUID, or a sanitized non-UUID request name."""
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", req)
    return match[0].lower() if match else re.sub(r"[^a-zA-Z0-9_.-]", "_", req)


def token_bytes(value: torch.Tensor) -> torch.Tensor:
    """Select byte zero per token; never reduce or copy the complete payload."""
    if not value.is_contiguous():
        raise ValueError("feature_source_must_be_contiguous")
    rows = value.shape[0]
    if not rows:
        return torch.empty(0, dtype=torch.uint8, device=value.device)
    raw = value.detach().view(torch.uint8).reshape(rows, -1)
    return raw[:, 0].contiguous()


class FeatureFile:
    """Append observations, reusing unchanged logical 4096-token extents.

    The binary file contains sampled bytes only. JSONL observations refer to
    immutable byte ranges, so subsequent changes never overwrite a baseline.
    The recorder serializes access with its reentrant lock.
    """

    def __init__(self, directory: Path, base: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.base = directory, base
        self.payload = (directory / "features.bin").open("xb")
        self.index = (directory / "index.jsonl").open("x", encoding="utf-8")
        self.latest: dict[tuple, tuple[bytes, int]] = {}
        self.sequence = 0
        self.write_event(dict(stage="FILE_BEGIN", format=FORMAT, **base))

    def write_event(self, record: dict[str, Any]) -> None:
        """Write a flushed metadata/completion record without tensor content."""
        self.index.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.index.flush()

    def write(self, record: dict[str, Any], value: torch.Tensor) -> None:
        """Save one CPU feature vector, retaining immutable extent references."""
        record = dict(record)
        width = int(record.get("byte_width", 1))
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        start, end = int(record["start"]), int(record["end"])
        if len(raw) != (end - start) * width:
            raise ValueError("feature_length_disagrees_with_logical_range")
        pieces = []
        for begin, stop in segments(start, end, BLOCK_TOKENS):
            data = raw[(begin - start) * width : (stop - start) * width]
            key = (
                record["group"],
                record["layer"],
                record["component"],
                record.get("kind", "token"),
                begin,
                stop,
                width,
            )
            previous = self.latest.get(key)
            if previous is not None and previous[0] == data:
                offset = previous[1]
            else:
                offset = self.payload.tell()
                self.payload.write(data)
                self.latest[key] = data, offset
            pieces.append([begin, stop, offset])
        self.payload.flush()
        self.sequence += 1
        self.write_event(dict(record, observation=self.sequence, pieces=pieces))

    def close(self) -> None:
        """Close a completed request and release its prefix reuse cache."""
        self.payload.close()
        self.index.close()
        self.latest.clear()


class KVTrace:
    """Collect small feature vectors using the existing DEBUG hook interface."""

    def __init__(
        self,
        worker: int,
        role: str,
        emit: Any = _emit,
        segment_tokens: int = BLOCK_TOKENS,
        root: Any = None,
    ) -> None:
        self.base = dict(
            schema=2,
            algorithm=FORMAT,
            host=socket.gethostname(),
            pid=os.getpid(),
            rank=worker,
            role=role,
        )
        self.emit, self.role = emit, role
        self.segment_tokens = segment_tokens
        self.root = Path(root) if root is not None else Path.cwd() / "kv-feature-dump"
        self.lock = RLock()
        self.first_failure: dict[str, Any] | None = None
        self.files: dict[str, FeatureFile] = {}
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.failed = False
        self.quarantine: list[Any] = []

    def error(self, req: str, stage: str, reason: str, **fields: Any) -> None:
        """Stop further device probes after the first failure; report it once."""
        with self.lock:
            if self.failed:
                return
            self.failed = True
            self.first_failure = dict(req=req, stage=stage, error=reason, **fields)
            self.emit(dict(self.base, **self.first_failure))

    def accepts(self, req: str) -> bool:
        """Only an actual sampling failure stops observation."""
        return not self.failed

    def claim(self, req: str, stage: str, group: int, layer: int) -> bool:
        """Limit D observations to the first real decode transfer per layer."""
        with self.lock:
            if not self.accepts(req):
                return False
            key = req, stage, group, layer
            if key in self.seen:
                return False
            self.seen[key] = None
            return True

    def record(self, req: str, stage: str, **fields: Any) -> dict[str, Any]:
        """Build a request-scoped metadata record."""
        return dict(req=req, stage=stage, **fields)

    def capture(self, record: dict[str, Any], producer: Any) -> None:
        """Wait for this observation and flush it to disk before returning."""
        with self.lock:
            if not self.accepts(record["req"]):
                return
            host = value = event = None
            completed = False
            try:
                value = producer()
                if value.device.type == "cpu":
                    host = value.clone()
                else:
                    host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                    host.copy_(value, non_blocking=True)
                    event = torch.npu.Event()
                    event.record(torch.npu.current_stream(value.device))
                if event is not None:
                    # The caller uses the actual load/compute stream. This
                    # fence covers the sample and its D2H, not just submission.
                    event.synchronize()
                completed = True
                self._file(record["req"]).write(record, host)
            except Exception as error:
                # Keep storage alive if device completion could not be proved.
                if (
                    not completed
                    and host is not None
                    and value is not None
                    and value.device.type != "cpu"
                ):
                    self.quarantine.append((host, value, event))
                self.error(
                    record["req"],
                    record["stage"],
                    type(error).__name__,
                    detail=str(error)[:500],
                    stack=traceback.format_exc()[-1500:],
                )

    def tensor(
        self,
        req: str,
        stage: str,
        group: int,
        layer: int,
        frontier: int,
        start: int,
        end: int,
        component: int,
        value: torch.Tensor,
        **fields: Any,
    ) -> None:
        """Save byte zero per token, or full small integer mapping metadata."""
        mapping = stage.startswith("MAP_")
        record = self.record(
            req,
            stage,
            group=group,
            layer=layer,
            frontier=frontier,
            start=start,
            end=end,
            component=component,
            **fields,
        )
        if mapping:
            record.update(kind="mapping", byte_width=value.element_size())
        self.capture(
            record, lambda: value.contiguous() if mapping else token_bytes(value)
        )

    def paged(
        self,
        req: str,
        stage: str,
        group: int,
        layer: int,
        frontier: int,
        planes: Any,
        table: torch.Tensor,
        start: int,
        end: int,
        *,
        block_size: int,
        factor: int = 1,
        block_map: Any = None,
        segment_tokens: Any = None,
        **fields: Any,
    ) -> None:
        """Gather one byte, rather than a whole KV row, at each logical slot.

        Device indices are clamped before every gather. Original P mappings
        are archived separately for offline bounds/expected-map checks.
        """
        if not self.accepts(req) or end <= start:
            return
        if table.ndim != 1 or (end - 1) // block_size >= table.numel():
            self.error(req, stage, "feature_table_range", group=group, layer=layer)
            return
        for component, plane in enumerate(planes):
            if not plane.is_contiguous():
                self.error(req, stage, "feature_plane_not_contiguous")
                return
            row_bytes = plane[0, 0].numel() * plane.element_size()
            capacity = plane.shape[0] * plane.shape[1]
            record = self.record(
                req,
                stage,
                group=group,
                layer=layer,
                frontier=frontier,
                start=start,
                end=end,
                component=component,
                components=len(planes),
                **fields,
            )

            def sample(plane=plane, row_bytes=row_bytes, capacity=capacity):
                positions = torch.arange(
                    start, end, dtype=torch.long, device=table.device
                )
                blocks = table.index_select(0, positions // block_size).long()
                if block_map is not None:
                    mapping = block_map.to(table.device)
                    blocks = mapping.index_select(
                        0, blocks.clamp(0, mapping.numel() - 1)
                    )
                slots = blocks * (block_size * factor) + positions % block_size
                safe = slots.clamp(0, capacity - 1)
                raw = plane.detach().view(torch.uint8).reshape(-1)
                return raw.index_select(0, safe * row_bytes)

            self.capture(record, sample)

    def cpu_packet(
        self,
        req: str,
        stage: str,
        group: int,
        layer: int,
        frontier: int,
        start: int,
        end: int,
        tensor: torch.Tensor,
        capacity: int,
        byte_widths: Any,
        *,
        interleaved: bool = False,
        batches: Any = None,
        **fields: Any,
    ) -> None:
        """Sample completed CPU packets; no independent full-payload D mirror."""
        if not self.accepts(req):
            return
        planes = packet_planes(tensor, capacity, byte_widths, interleaved)
        for component, plane in enumerate(planes):
            value = plane[: end - start, 0].clone()
            batches.setdefault((stage, component), []).append((start, end, value))

    def flush_packets(
        self,
        req: str,
        group: int,
        layer: int,
        frontier: int,
        batches: Any,
        **fields: Any,
    ) -> None:
        """Merge adjacent CPU feature ranges before deduplicated file writes."""
        for (stage, component), rows in batches.items():
            runs: list[list[Any]] = []
            for start, end, value in rows:
                if runs and runs[-1][1] == start:
                    runs[-1][1] = end
                    runs[-1][2].append(value)
                else:
                    runs.append([start, end, [value]])
            for start, end, values in runs:
                self.capture(
                    self.record(
                        req,
                        stage,
                        group=group,
                        layer=layer,
                        frontier=frontier,
                        start=start,
                        end=end,
                        component=component,
                        **fields,
                    ),
                    lambda values=values: torch.cat(values),
                )

    def cpu_work(
        self,
        req: str,
        callback: Any,
        owners: Any,
        events: Any,
        *,
        stage: str = "P_CPU_STORE",
        **fields: Any,
    ) -> None:
        """Read CPU sources now, after their already-recorded producer events.

        Passive TP views cannot extend the owner's shared allocation lifetime.
        Complete the read inside the calling operation, before its lease ends.
        """
        with self.lock:
            if not self.accepts(req):
                return
            owners = tuple({id(obj): obj for obj in owners}.values())
            events = tuple(events)
            held = []
            waiting = completed = False
            try:
                for obj in owners:
                    obj.ref_count_up()
                    held.append(obj)
                waiting = True
                for event in events:
                    event.synchronize()
                completed = True
                callback()
            except Exception as error:
                self.error(
                    req,
                    stage,
                    type(error).__name__,
                    detail=str(error)[:500],
                    **fields,
                )
            finally:
                if waiting and not completed:
                    self.quarantine.append((callback, held, events))
                else:
                    self._release(held, self.record(req, stage))

    def finish(self, req: str, **fields: Any) -> None:
        """Write completion and close the file synchronously."""
        with self.lock:
            failure = self.first_failure
            end = self.record(req, "END", failed=failure is not None, **fields)
            if failure is not None:
                end["cause"] = failure
            if req in self.files:
                file = self.files.pop(req)
                try:
                    file.write_event(end)
                finally:
                    file.close()
            self.seen = OrderedDict((k, v) for k, v in self.seen.items() if k[0] != req)
            self.emit(dict(self.base, **end))

    def drain(self, timeout: float = 60.0) -> bool:
        """No queued work remains; retain the existing shutdown safety check."""
        with self.lock:
            return not self.quarantine

    def _file(self, req: str) -> FeatureFile:
        if req not in self.files:
            identity = "{role}-{host}-r{rank}-p{pid}".format(**self.base)
            directory = self.root / request_key(req) / identity
            self.files[req] = FeatureFile(directory, dict(self.base, req=req))
            self.emit(
                dict(self.base, req=req, stage="FEATURE_FILE", directory=str(directory))
            )
        return self.files[req]

    def _release(self, owners: Any, record: dict[str, Any]) -> None:
        for owner in owners:
            try:
                owner.ref_count_down()
            except Exception as error:
                self.error(
                    record["req"],
                    "feature_release",
                    type(error).__name__,
                    detail=str(error)[:500],
                )
