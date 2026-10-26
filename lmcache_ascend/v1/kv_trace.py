# SPDX-License-Identifier: Apache-2.0
"""Full-content segmented KV diagnostics; logs contain hashes, never tensors.

The existing LMCACHE_PREFILL_STORE_DEBUG switch owns this diagnostic. Device
payloads are reduced on their consuming stream; only two int32 checksums per
segment leave the device. ``sum16x2-v1`` contains a plain sum and a position-
weighted sum of unsigned 16-bit words. CPU and device use the same algorithm.
"""

from __future__ import annotations

# Standard
from collections import OrderedDict
from functools import wraps
from queue import Full, Queue
from threading import Event, Lock, Thread, get_ident, main_thread
from typing import Any, Iterator, Sequence
import json
import logging
import os
import socket
import sys
import time
import traceback

# Third Party
import torch

ALGORITHM = "sum16x2-v1"
SEGMENT_TOKENS = 1024
REFERENCE_SEGMENT_TOKENS = 128
MAX_PENDING = 8192
MAX_REQUESTS = 1024
MAX_HASH_BYTES = 4 * 1024 * 1024
HASHES_PER_LINE = 32
STALL_SECONDS = 60.0
_LOGGER = logging.getLogger("lmcache.kv_trace")
_EMIT_LOCK = Lock()


def enabled() -> bool:
    """Use the existing diagnostic switch, read before workers start."""
    return os.environ.get("LMCACHE_PREFILL_STORE_DEBUG", "0") == "1"


def diagnostic(function: Any) -> Any:
    """Turn probe failures into explicit coverage errors, never serving asserts."""

    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            trace = getattr(self, "kv_trace", None)
            requests = list(getattr(self, "kv_trace_requests", {}))
            if function.__name__ == "trace_cpu_kv" and len(args) > 1:
                requests = [args[1]]
            elif (
                function.__name__
                in (
                    "enqueue_store_kv_trace",
                    "enqueue_load_kv_trace",
                    "trace_prepared_kv",
                    "trace_direct_kv",
                    "finish_kv_trace",
                )
                and args
            ):
                requests = [args[0]]
            elif function.__name__ == "configure_kv_trace" and args:
                requests = [request.req_id for request in args[0]]
            for request in requests:
                record = dict(
                    req=request,
                    stage=function.__name__,
                    error=type(error).__name__,
                    detail=str(error)[:160],
                )
                if trace is not None:
                    trace.emit(dict(trace.base, **record))
                else:
                    _emit(record)
            return None

    return wrapped


def content_checksum(value: torch.Tensor) -> torch.Tensor:
    """Reduce every payload byte to two int32 words on value's device.

    Integer overflow is intentional modulo 2**32. The first sum includes the
    valid byte count; the second weights words by their one-based positions.
    No scalar extraction or device-to-host payload copy occurs here.
    """
    return content_checksums(
        value.detach().contiguous().reshape(-1).view(torch.uint8).reshape(1, -1)
    )[0]


def content_checksums(values: torch.Tensor) -> torch.Tensor:
    """Compute two integer sums per segment; return [segments, 2] int32.

    All bytes participate. There are no seed loops or avalanche rounds. The
    weighted sum detects ordinary reordering that a plain sum would miss.
    This is an inexpensive diagnostic checksum, not a cryptographic hash.
    """
    raw = values.detach().contiguous().view(torch.uint8).reshape(values.shape[0], -1)
    valid_bytes = raw.shape[1]
    padding = valid_bytes % 2
    if padding:
        raw = torch.cat(
            (
                raw,
                torch.zeros(
                    (raw.shape[0], padding), dtype=torch.uint8, device=raw.device
                ),
            ),
            dim=1,
        )
    elif raw.storage_offset() % 2:
        raw = raw.clone()
    words = raw.view(torch.int16).to(torch.int32) & 0xFFFF
    positions = torch.arange(
        1, words.shape[1] + 1, dtype=torch.int32, device=raw.device
    )
    plain = words.sum(dim=1, dtype=torch.int32) ^ valid_bytes
    weighted = (words * positions).sum(dim=1, dtype=torch.int32)
    return torch.stack((plain, weighted), dim=1)


def prefix_checksums(value: torch.Tensor) -> torch.Tensor:
    """Hash every token prefix of one short boundary segment on its device.

    Handles arbitrary D truncation, including an odd final byte, without
    comparing different ranges or copying the KV payload to the host.
    """
    raw = value.detach().contiguous().view(torch.uint8).reshape(value.shape[0], -1)
    row_bytes = raw.shape[1]
    flat = raw.reshape(-1)
    padding = flat.numel() % 2
    if padding:
        flat = torch.cat(
            (flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device))
        )
    elif flat.storage_offset() % 2:
        flat = flat.clone()
    words = flat.view(torch.int16).to(torch.int32) & 0xFFFF
    positions = torch.arange(
        1, words.numel() + 1, dtype=torch.int32, device=words.device
    )
    lengths = (
        torch.arange(1, raw.shape[0] + 1, dtype=torch.int32, device=words.device)
        * row_bytes
    )
    complete, remainder = lengths // 2, lengths % 2
    last_complete = (complete - 1).clamp_min(0).long()
    plain = words.cumsum(dim=0, dtype=torch.int32).index_select(0, last_complete)
    weighted = (
        (words * positions)
        .cumsum(dim=0, dtype=torch.int32)
        .index_select(0, last_complete)
    )
    plain = torch.where(complete > 0, plain, 0)
    weighted = torch.where(complete > 0, weighted, 0)
    if row_bytes % 2:
        last = (
            words.index_select(0, complete.clamp_max(words.numel() - 1).long()) & 0xFF
        )
        last = torch.where(remainder > 0, last, 0)
        plain += last
        weighted += last * (complete + 1)
    return torch.stack((plain ^ lengths, weighted), dim=1)


def _emit(record: dict[str, Any]) -> None:
    with _EMIT_LOCK:
        if not _LOGGER.handlers:
            _LOGGER.setLevel(logging.INFO)
            _LOGGER.propagate = False
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            _LOGGER.addHandler(handler)
    _LOGGER.info("[KVTRACE] %s", json.dumps(record, separators=(",", ":")))


def segments(
    start: int, end: int, size: int = SEGMENT_TOKENS
) -> Iterator[tuple[int, int]]:
    """Split absolute logical positions at canonical storage-chunk boundaries."""
    while start < end:
        stop = min(end, (start // size + 1) * size)
        yield start, stop
        start = stop


def packet_planes(
    tensor: torch.Tensor,
    capacity: int,
    byte_widths: Sequence[int],
    interleaved: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Decode canonical plane-stacked bytes, including C8 key and FP16 scale."""
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() != capacity * sum(byte_widths):
        raise ValueError("packet_size_disagrees_with_validated_layout")
    if interleaved:
        return tuple(
            raw.reshape(capacity, sum(byte_widths)).split(tuple(byte_widths), dim=1)
        )
    result, offset = [], 0
    for width in byte_widths:
        count = capacity * int(width)
        result.append(raw[offset : offset + count].reshape(capacity, int(width)))
        offset += count
    return tuple(result)


class KVTrace:
    """Emit full per-layer coverage and asynchronously collect device digests.

    The queue retains tiny checksums and completion events only. Full KV stays
    in its existing allocation; there are no tensor archives or dump folders.
    Every omitted observation is an explicit error, never a successful match.
    """

    def __init__(
        self,
        worker: int,
        role: str,
        emit: Any = _emit,
        segment_tokens: int = SEGMENT_TOKENS,
    ) -> None:
        self.worker, self.role, self.emit = worker, role, emit
        if segment_tokens <= 0:
            raise ValueError("segment_tokens must be positive")
        self.segment_tokens = segment_tokens
        self.base = dict(
            schema=1,
            algorithm=ALGORITHM,
            host=socket.gethostname(),
            pid=os.getpid(),
            rank=worker,
            role=role,
        )
        self.queue: Queue = Queue(MAX_PENDING)
        self.lock = Lock()
        self.pending_job: tuple[float, dict, bool] | None = None
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.cpu_baselines: dict[tuple, torch.Tensor] = {}
        self.thread = Thread(target=self._run, daemon=True, name="kv-trace")
        self.thread.start()
        self.watchdog = Thread(target=self._watch, daemon=True, name="kv-trace-watch")
        self.watchdog.start()

    def check_stall(self, now: float | None = None) -> None:
        """Log bounded Python stack summaries once per stalled hash job.

        Host inspection only: no signal handlers, device calls or new fences.
        A slow observation is informational, never a hash mismatch or assert.
        """
        now = time.monotonic() if now is None else now
        with self.lock:
            pending = self.pending_job
            if pending is None or pending[2] or now - pending[0] < STALL_SECONDS:
                return
            since, record, _ = pending
            self.pending_job = (since, record, True)
        frames = sys._current_frames()

        def stack(ident: Any) -> list[str]:
            frame = frames.get(ident)
            if frame is None:
                return []
            return [
                f"{os.path.basename(row.filename)}:{row.lineno}:{row.name}"
                for row in traceback.extract_stack(frame, limit=8)
            ]

        self.emit(
            dict(
                self.base,
                req=record["req"],
                stage="STALL",
                blocked_stage=record["stage"],
                group=record.get("group"),
                layer=record.get("layer"),
                elapsed_s=round(now - since, 1),
                queued=self.queue.qsize(),
                main_stack=stack(main_thread().ident),
                hash_stack=stack(self.thread.ident),
            )
        )

    def _watch(self) -> None:
        tick = Event()
        while not tick.wait(5):
            try:
                self.check_stall()
            except Exception:
                # A best-effort host watchdog cannot fail a serving request.
                continue

    def error(self, req: str, stage: str, reason: str, **fields: Any) -> None:
        """Record an explicit coverage failure without failing the request."""
        self.emit(dict(self.base, req=req, stage=stage, error=reason, **fields))

    def claim(self, req: str, stage: str, group: int, layer: int) -> bool:
        """Claim a first-decode observation once per request/group/layer."""
        key = req, stage, group, layer
        with self.lock:
            if key in self.seen:
                return False
            self.seen[key] = None
            if len(self.seen) > MAX_REQUESTS * 256:
                self.seen.popitem(last=False)
        return True

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
        """Checksum all bytes of one logical segment without exporting its KV."""
        record = dict(
            req=req,
            stage=stage,
            group=group,
            layer=layer,
            frontier=frontier,
            start=start,
            end=end,
            component=component,
            shape=list(value.shape),
            dtype=str(value.dtype),
            **fields,
        )
        try:
            digest = content_checksum(value)
            if value.device.type == "cpu":
                self._result(record, digest)
                return
            # Only the two checksum words leave the device.
            host = torch.empty(digest.shape, dtype=torch.int32, pin_memory=True)
            host.copy_(digest, non_blocking=True)
            event = torch.npu.Event()
            event.record(torch.npu.current_stream(digest.device))
            if get_ident() == self.thread.ident:
                event.synchronize()
                self._result(record, host)
            else:
                self.queue.put_nowait((record, host, digest, event))
        except Full:
            self.error(req, stage, "checksum_queue_full", group=group, layer=layer)
        except Exception as error:
            self.error(req, stage, type(error).__name__, group=group, layer=layer)

    def paged(
        self,
        req: str,
        stage: str,
        group: int,
        layer: int,
        frontier: int,
        planes: Sequence[torch.Tensor],
        table: torch.Tensor,
        start: int,
        end: int,
        *,
        block_size: int,
        factor: int = 1,
        block_map: Any = None,
        segment_tokens: int | None = None,
        **fields: Any,
    ) -> None:
        """Read logical positions using the model's actual physical block table."""
        try:
            self.emit(
                dict(
                    self.base,
                    req=req,
                    stage="PLAN",
                    phase=stage,
                    group=group,
                    layer=layer,
                    frontier=frontier,
                    start=start,
                    end=end,
                    components=len(planes),
                    **fields,
                )
            )
            if table.ndim != 1:
                raise ValueError("request_block_table_must_be_one_dimensional")
            rows = [[] for _ in planes]
            bounds = []
            segment_size = segment_tokens or self.segment_tokens
            pending = list(segments(start, end, segment_size))
            width = max(
                int(plane[0, 0].numel() * plane.element_size()) for plane in planes
            )
            max_segments = max(1, MAX_HASH_BYTES // (segment_size * width))
            cursor = 0
            while cursor < len(pending):
                span = pending[cursor][1] - pending[cursor][0]
                stop_cursor = cursor + 1
                while (
                    stop_cursor < len(pending)
                    and stop_cursor - cursor < max_segments
                    and pending[stop_cursor][1] - pending[stop_cursor][0] == span
                ):
                    stop_cursor += 1
                selected_bounds = pending[cursor:stop_cursor]
                begin, stop = selected_bounds[0][0], selected_bounds[-1][1]
                positions = torch.arange(
                    begin, stop, device=table.device, dtype=torch.long
                )
                blocks = table.index_select(0, positions // block_size).long()
                if block_map is not None:
                    blocks = block_map.to(table.device).index_select(0, blocks)
                # Mixed C8 can consume half a physical bundle per logical block.
                slots = blocks * (block_size * factor) + positions % block_size
                for component, plane in enumerate(planes):
                    flat = plane.reshape(-1, *plane.shape[2:])
                    value = flat.index_select(0, slots).contiguous()
                    raw = value.view(torch.uint8).reshape(len(selected_bounds), -1)
                    rows[component].extend(content_checksums(raw).unbind(0))
                bounds.extend(selected_bounds)
                cursor = stop_cursor
            for component, checksums in enumerate(rows):
                if not checksums:
                    continue
                digest = torch.stack(checksums)
                record = dict(
                    req=req,
                    stage=stage,
                    group=group,
                    layer=layer,
                    frontier=frontier,
                    component=component,
                    bounds=bounds,
                    **fields,
                )
                if digest.device.type == "cpu":
                    self._result(record, digest)
                else:
                    host = torch.empty(digest.shape, dtype=torch.int32, pin_memory=True)
                    host.copy_(digest, non_blocking=True)
                    event = torch.npu.Event()
                    event.record(torch.npu.current_stream(digest.device))
                    self.queue.put_nowait((record, host, digest, event))
        except Exception as error:
            self.error(req, stage, type(error).__name__, group=group, layer=layer)

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
        byte_widths: Sequence[int],
        *,
        mirror_device: Any = None,
        interleaved: bool = False,
        batches: Any = None,
        source_identity: int | None = None,
        **fields: Any,
    ) -> None:
        """Hash CPU payload planes; optionally verify a bounded independent H2D.

        The mirror has one storage chunk at a time, never the full prompt. Its
        only consumer is the checksum; it cannot change the model's KV cache.
        """
        try:
            planes = packet_planes(tensor, capacity, byte_widths, interleaved)
            if batches is not None and stage == "D_CPU":
                pending = list(segments(start, end, self.segment_tokens))
                cursor = 0
                while cursor < len(pending):
                    span = pending[cursor][1] - pending[cursor][0]
                    stop_cursor = cursor + 1
                    while (
                        stop_cursor < len(pending)
                        and pending[stop_cursor][1] - pending[stop_cursor][0] == span
                    ):
                        stop_cursor += 1
                    bounds = pending[cursor:stop_cursor]
                    for component, plane in enumerate(planes):
                        values = (
                            plane[bounds[0][0] - start : bounds[-1][1] - start]
                            .contiguous()
                            .reshape(len(bounds), -1)
                        )
                        digests = content_checksums(values)
                        batches.setdefault((stage, component), []).extend(
                            (begin, stop, digest)
                            for (begin, stop), digest in zip(
                                bounds, digests.unbind(0), strict=True
                            )
                        )
                        if mirror_device is not None:
                            mirror = values.to(device=mirror_device, non_blocking=True)
                            mirror_digests = content_checksums(mirror)
                            batches.setdefault(("D_MIRROR", component), []).extend(
                                (begin, stop, digest)
                                for (begin, stop), digest in zip(
                                    bounds, mirror_digests.unbind(0), strict=True
                                )
                            )
                    cursor = stop_cursor
                return
            for begin, stop in segments(start, end, self.segment_tokens):
                for component, plane in enumerate(planes):
                    value = plane[begin - start : stop - start]
                    if batches is not None:
                        key = (
                            req,
                            group,
                            layer,
                            begin,
                            stop,
                            component,
                            source_identity,
                        )
                        reusable = (
                            source_identity is not None
                            and stop - begin == self.segment_tokens
                            and begin % self.segment_tokens == 0
                        )
                        # Full MemoryObjs are leased and immutable for the
                        # request. Reuse only their CPU baseline; the actual
                        # NPU bank is checksummed afresh at every model step.
                        cached = (
                            self.cpu_baselines.get(key)
                            if stage == "P_CPU_LOAD" and reusable
                            else None
                        )
                        digest = content_checksum(value) if cached is None else cached
                        if reusable and stage in ("P_CPU_LOAD", "P_CPU_STORE"):
                            self.cpu_baselines[key] = digest
                        batches.setdefault((stage, component), []).append(
                            (begin, stop, digest)
                        )
                        if mirror_device is not None:
                            mirror = value.to(device=mirror_device, non_blocking=True)
                            batches.setdefault(("D_MIRROR", component), []).append(
                                (begin, stop, content_checksum(mirror))
                            )
                        continue
                    self.tensor(
                        req,
                        stage,
                        group,
                        layer,
                        frontier,
                        begin,
                        stop,
                        component,
                        value,
                        **fields,
                    )
                    if mirror_device is not None:
                        mirror = value.to(device=mirror_device, non_blocking=True)
                        self.tensor(
                            req,
                            "D_MIRROR",
                            group,
                            layer,
                            frontier,
                            begin,
                            stop,
                            component,
                            mirror,
                            **fields,
                        )
        except Exception as error:
            self.error(req, stage, type(error).__name__, group=group, layer=layer)

    def boundary_prefixes(
        self,
        req: str,
        group: int,
        layer: int,
        frontier: int,
        planes: Any,
        table: torch.Tensor,
        start: int,
        end: int,
        block_size: int,
    ) -> None:
        """Record all possible final-block D boundaries with one short gather."""
        if end <= start:
            return
        try:
            positions = torch.arange(start, end, dtype=torch.long, device=table.device)
            blocks = table.index_select(0, positions // block_size).long()
            slots = blocks * block_size + positions % block_size
            batches = {}
            for component, plane in enumerate(planes):
                values = plane.reshape(-1, *plane.shape[2:]).index_select(0, slots)
                digests = prefix_checksums(values)
                batches[("P_D_BOUNDARY", component)] = [
                    (start, start + i + 1, digest)
                    for i, digest in enumerate(digests.unbind(0))
                ]
            self.flush_packets(req, group, layer, frontier, batches, final=True)
        except Exception as error:
            self.error(
                req, "P_D_BOUNDARY", type(error).__name__, group=group, layer=layer
            )

    def flush_packets(
        self,
        req: str,
        group: int,
        layer: int,
        frontier: int,
        batches: Any,
        **fields: Any,
    ) -> None:
        """Write one compact digest vector per layer/component, not per token."""
        for (stage, component), entries in batches.items():
            if not entries:
                continue
            digest = torch.stack([entry[2] for entry in entries])
            record = dict(
                req=req,
                stage=stage,
                group=group,
                layer=layer,
                frontier=frontier,
                component=component,
                bounds=[entry[:2] for entry in entries],
                **fields,
            )
            if digest.device.type == "cpu":
                self._result(record, digest)
            else:
                host = torch.empty(digest.shape, dtype=torch.int32, pin_memory=True)
                host.copy_(digest, non_blocking=True)
                event = torch.npu.Event()
                event.record(torch.npu.current_stream(digest.device))
                try:
                    self.queue.put_nowait((record, host, digest, event))
                except Full:
                    self.error(
                        req, stage, "checksum_queue_full", group=group, layer=layer
                    )

    def finish(self, req: str, **fields: Any) -> None:
        """Queue a completion marker after all earlier checksum readbacks."""
        try:
            self.queue.put_nowait(
                (dict(req=req, stage="END", **fields), None, None, None)
            )
        except Full:
            self.error(req, "END", "checksum_queue_full")

    def drain(self, timeout: float = 60.0) -> bool:
        """Shutdown-only barrier before the owner unmaps CPU source storage."""
        completed = Event()
        deadline = time.monotonic() + timeout
        try:
            self.queue.put(
                (completed.set, dict(req="shutdown", stage="DRAIN"), [], ()),
                timeout=max(0.0, timeout),
            )
        except Full:
            return False
        return completed.wait(max(0.0, deadline - time.monotonic()))

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
        """Retain existing CPU buffers until background hashing has completed."""
        held = []
        try:
            for obj in {id(obj): obj for obj in owners}.values():
                obj.ref_count_up()
                held.append(obj)
            self.queue.put_nowait(
                (callback, dict(req=req, stage=stage, **fields), held, tuple(events))
            )
        except Exception as error:
            for obj in held:
                obj.ref_count_down()
            self.error(req, stage, type(error).__name__, detail=str(error)[:160])

    def _result(self, record: dict[str, Any], value: torch.Tensor) -> None:
        if get_ident() == self.thread.ident:
            with self.lock:
                if self.pending_job is not None:
                    self.pending_job = (time.monotonic(), self.pending_job[1], False)
        words = value.tolist()
        if "bounds" in record:
            record = dict(record)
            bounds = record.pop("bounds")
            hashes = [
                [start, end, "".join(f"{int(x) & 0xFFFFFFFF:08x}" for x in row)]
                for (start, end), row in zip(bounds, words, strict=True)
            ]
            # Keep each worker line below a few KiB; long JSON writes from
            # several ranks can otherwise interleave in the service log pipe.
            for offset in range(0, len(hashes), HASHES_PER_LINE):
                self.emit(
                    dict(
                        self.base,
                        **record,
                        hashes=hashes[offset : offset + HASHES_PER_LINE],
                    )
                )
            return
        digest = "".join(f"{int(word) & 0xFFFFFFFF:08x}" for word in words)
        if "expected_hash" in record:
            record = dict(record, match=digest == record["expected_hash"])
        self.emit(dict(self.base, **record, hash=digest))

    def _run(self) -> None:
        while True:
            record, host, device, event = self.queue.get()
            if callable(record) or event is not None:
                with self.lock:
                    self.pending_job = (
                        time.monotonic(),
                        host if callable(record) else record,
                        False,
                    )
            try:
                if callable(record):
                    for producer in event:
                        producer.synchronize()
                    record()
                elif event is None:
                    self.emit(dict(self.base, **record))
                    if record.get("stage") == "END":
                        for key in tuple(self.cpu_baselines):
                            if key[0] == record["req"]:
                                self.cpu_baselines.pop(key, None)
                else:
                    event.synchronize()  # Background checksum readback only.
                    self._result(record, host)
            except Exception as error:
                if callable(record):
                    self.error(
                        host["req"],
                        host["stage"],
                        type(error).__name__,
                        detail=str(error)[:160],
                    )
                else:
                    self.error(record["req"], record["stage"], type(error).__name__)
            finally:
                with self.lock:
                    self.pending_job = None
                if callable(record):
                    for obj in device:
                        try:
                            obj.ref_count_down()
                        except Exception as error:
                            self.error(host["req"], "release", type(error).__name__)
                self.queue.task_done()
                del record, host, device, event
