# SPDX-License-Identifier: Apache-2.0
"""Opt-in one-byte-per-token KV observations, without content hashes.

Only the first raw byte of each token/component is exported. File writes,
prefix reuse and CPU source reads run on one background worker. Observations
use the existing bank stream and completion fences; serving never waits for
the file writer. Sampling cannot prove equality of the unsampled bytes.
"""

from __future__ import annotations

# Standard
from collections import OrderedDict
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Event, Thread, get_ident
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
MAX_PENDING = 512
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
    All methods are called by the recorder's single background worker.
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
        self.queue: Queue = Queue()
        self.slots = BoundedSemaphore(MAX_PENDING)
        self.files: dict[str, FeatureFile] = {}
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.failed = False
        self.quarantine: list[Any] = []
        self.thread = Thread(target=self._run, daemon=True, name="kv-feature-writer")
        self.thread.start()

    def error(self, req: str, stage: str, reason: str, **fields: Any) -> None:
        """Stop further device probes after the first failure; report it once."""
        if self.failed:
            return
        self.failed = True
        self.emit(dict(self.base, req=req, stage=stage, error=reason, **fields))

    def claim(self, req: str, stage: str, group: int, layer: int) -> bool:
        """Limit D observations to the first real decode transfer per layer."""
        if self.failed:
            return False
        key = req, stage, group, layer
        if key in self.seen:
            return False
        self.seen[key] = None
        return True

    def record(self, req: str, stage: str, **fields: Any) -> dict[str, Any]:
        """Build a request-scoped metadata record for background storage."""
        return dict(req=req, stage=stage, **fields)

    def capture(self, record: dict[str, Any], producer: Any) -> None:
        """Reserve capacity before enqueueing NPU sampling or async readback."""
        if self.failed:
            return
        inline = get_ident() == self.thread.ident
        if not inline and not self.slots.acquire(blocking=False):
            self.error(record["req"], record["stage"], "feature_queue_full")
            return
        host = value = event = None
        try:
            value = producer()
            if value.device.type == "cpu":
                host = value.clone()
            else:
                host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                host.copy_(value, non_blocking=True)
                event = torch.npu.Event()
                event.record(torch.npu.current_stream(value.device))
            if inline:
                if event is not None:
                    event.synchronize()  # Background worker only.
                self._file(record["req"]).write(record, host)
            else:
                self.queue.put(("sample", record, host, value, event))
        except Exception as error:
            # A failed event record does not prove an async host write ended.
            if host is not None and value is not None and value.device.type != "cpu":
                self.quarantine.append((host, value, event))
            if not inline:
                self.slots.release()
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
        if self.failed or end <= start:
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
        if self.failed:
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
        """Lease source objects until real producer completion and CPU sampling."""
        if self.failed:
            return
        inline = get_ident() == self.thread.ident
        if not inline and not self.slots.acquire(blocking=False):
            self.error(req, stage, "feature_queue_full")
            return
        held = []
        released_slot = inline
        try:
            for obj in {id(obj): obj for obj in owners}.values():
                obj.ref_count_up()
                held.append(obj)
            if inline:
                completed = False
                try:
                    for event in events:
                        event.synchronize()
                    completed = True
                    callback()
                finally:
                    if completed:
                        self._release(held, self.record(req, stage))
                    else:
                        self.quarantine.append((callback, held, tuple(events)))
                    held = []
            else:
                self.queue.put(
                    (
                        "work",
                        self.record(req, stage, **fields),
                        callback,
                        held,
                        tuple(events),
                    )
                )
        except Exception as error:
            self._release(held, self.record(req, stage))
            if not released_slot:
                self.slots.release()
            self.error(req, stage, type(error).__name__, detail=str(error)[:500])

    def finish(self, req: str, **fields: Any) -> None:
        """Queue a completion marker after earlier observations, without waiting."""
        self.queue.put(("end", self.record(req, "END", **fields), None, None, None))

    def drain(self, timeout: float = 60.0) -> bool:
        """Shutdown-only wait; unsafe pending device writes prevent slab release."""
        done = Event()
        self.queue.put(("drain", {}, done, None, None))
        return done.wait(timeout) and not self.quarantine

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

    def _run(self) -> None:
        while True:
            kind, record, host, device, event = self.queue.get()
            completed = event is None or kind not in ("sample", "work")
            try:
                if kind == "drain":
                    host.set()
                elif kind == "end":
                    if record["req"] in self.files:
                        file = self.files.pop(record["req"])
                        file.write_event(dict(record, failed=self.failed))
                        file.close()
                    self.seen = OrderedDict(
                        (k, v) for k, v in self.seen.items() if k[0] != record["req"]
                    )
                    self.emit(dict(self.base, **record, failed=self.failed))
                elif kind == "work":
                    for producer in event:
                        # Background only, borrowing existing real DMA fences.
                        producer.synchronize()
                    completed = True
                    if not self.failed:
                        host()
                else:
                    if event is not None:
                        event.synchronize()  # Background feature readback only.
                    completed = True
                    if not self.failed:
                        self._file(record["req"]).write(record, host)
            except Exception as error:
                if not completed:
                    self.quarantine.append((host, device, event))
                self.error(
                    record.get("req", "unknown"),
                    record.get("stage", kind),
                    type(error).__name__,
                    detail=str(error)[:500],
                    stack=traceback.format_exc()[-1500:],
                )
            finally:
                if kind == "work" and completed:
                    self._release(device, record)
                if kind in ("sample", "work"):
                    self.slots.release()
                self.queue.task_done()
                del host, device, event
