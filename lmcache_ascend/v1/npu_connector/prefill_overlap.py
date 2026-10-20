# SPDX-License-Identifier: Apache-2.0
"""Opt-in, asynchronous device-event accounting for P-node bank rotation.

No timing event is used as an inference dependency. The submission thread
only records events; queries, elapsed_time, JSON and logging run off-thread.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from queue import Empty, Full, LifoQueue, Queue
from threading import Event, Thread
from typing import Any, Callable, Optional
import atexit
import ctypes
from ctypes.util import find_library
import json
import os
import socket


LOG_MARKER = "[PREFILL_OVERLAP]"
MAX_PENDING_STEPS = 32
MAX_POOLED_EVENTS = 4096
POLL_SECONDS = 0.02
IDLE_POLL_SECONDS = 0.5


class AclEventTimer:
    """Read completed ACL event times without torch-npu's implicit queue drain.

    Event.elapsed_time() calls emptyAllNPUStream() and synchronizes events in
    torch-npu. The collector has already queried completion, so call only the
    ACL elapsed-time API. CDLL releases the GIL while executing the native API.
    Library loading is lazy and happens on the collector, never in a callback.
    """

    def __init__(self, library_loader: Optional[Callable[[], Any]] = None) -> None:
        self.library_loader = library_loader
        self.library: Any = None
        self.function: Any = None

    def __call__(self, start: Any, end: Any) -> float:
        if self.function is None:
            self.library = (
                self.library_loader()
                if self.library_loader is not None
                else ctypes.CDLL(find_library("ascendcl") or "libascendcl.so")
            )
            self.function = self.library.aclrtEventElapsedTime
            self.function.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.function.restype = ctypes.c_int
        elapsed = ctypes.c_float()
        result = self.function(
            ctypes.byref(elapsed),
            ctypes.c_void_p(int(start.npu_event)),
            ctypes.c_void_p(int(end.npu_event)),
        )
        if result:
            raise RuntimeError(f"aclrtEventElapsedTime failed: ACL error {result}")
        return float(elapsed.value)


def single_request_span(
    requests: list[Any],
    attn_metadata: Any,
) -> tuple[tuple[str, int, int], ...]:
    """Use CPU metadata to reject multi-request and dummy model batches.

    Connector metadata may omit requests that perform no cache operations.
    Check the actual attention batch as well, without device tensor readback.
    """
    if len(requests) != 1 or not isinstance(attn_metadata, Mapping):
        return ()
    request = requests[0]
    span = getattr(request, "prefill_overlap_span", None)
    if span is None or span[0] >= span[1]:
        return ()
    metadata = next(iter(attn_metadata.values()), None)
    ids = getattr(metadata, "req_ids", None)
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or set(ids) != {request.req_id}:
            return ()
    else:
        # CPU shape only: query_start_loc has one more entry than requests.
        offsets = getattr(metadata, "query_start_loc_cpu", None)
        if offsets is None or len(offsets) != 2:
            return ()
    return ((request.req_id, int(span[0]), int(span[1])),)


@dataclass
class TimingInterval:
    """Own two dedicated timing events until their measurements are consumed."""

    kind: str
    group: int
    bank: Optional[int]
    layer: int
    model_layer: int
    start: Any
    end: Any


@dataclass
class StepTimings:
    step: int
    requests: tuple[tuple[str, int, int], ...]
    intervals: list[TimingInterval] = field(default_factory=list)
    tails: dict[int, Any] = field(default_factory=dict)
    status: str = "complete"


def summarize_step(
    batch: StepTimings,
    elapsed_time: Callable[[Any, Any], float],
) -> dict[str, Any]:
    """Read a completed batch and average per-layer coverage, excluding L0.

    A model-layer start is its LAST bank consumer gate's END event, so both
    latent and indexer waits precede that start. Indexer waits carry the actual
    model-layer number, not the sparse group ordinal.
    For each consecutive pair, coverage = 1 - wait(next) / (start(next)-start).
    The final statistic is the arithmetic mean, NOT a ratio of summed times.
    Negative measurements are preserved for diagnosis instead of clamped.
    """
    starts: dict[int, Any] = {}
    latent_layers: set[int] = set()
    waits: dict[int, float] = {}
    banks: dict[tuple[int, Optional[int]], dict[str, Any]] = {}
    for item in batch.intervals:
        duration = elapsed_time(item.start, item.end)
        if item.kind == "wait":
            waits[item.model_layer] = waits.get(item.model_layer, 0.0) + duration
            starts[item.model_layer] = item.end
            if item.group == 0:
                latent_layers.add(item.model_layer)
        else:
            entry = banks.setdefault(
                (item.group, item.bank),
                dict(group=item.group, bank=item.bank, load_ms=0.0, save_ms=0.0),
            )
            entry[item.kind + "_ms"] += duration

    ratios, wait_ms, interval_ms = [], 0.0, 0.0
    layers = sorted(latent_layers)
    for previous, current in zip(layers, layers[1:], strict=False):
        if current != previous + 1:
            continue
        period = elapsed_time(starts[previous], starts[current])
        if period <= 0:
            continue
        waited = waits.get(current, 0.0)
        ratios.append(1.0 - waited / period)
        wait_ms += waited
        interval_ms += period
    load_ms = sum(row["load_ms"] for row in banks.values())
    save_ms = sum(row["save_ms"] for row in banks.values())
    return dict(
        schema_version=1,
        status=batch.status,
        scope="single_request_step",
        start_definition="last_bank_ready",
        step=batch.step,
        requests=[
            dict(req_id=req, token_start=start, token_end=end)
            for req, start, end in batch.requests
        ],
        coverage=(sum(ratios) / len(ratios) if ratios else None),
        layer_samples=len(ratios),
        wait_ms=wait_ms,
        interval_ms=interval_ms,
        first_layer_wait_ms=waits.get(layers[0], 0.0) if layers else 0.0,
        all_wait_ms=sum(waits.values()),
        load_ms=load_ms,
        save_ms=save_ms,
        dma_ms=load_ms + save_ms,
        out_of_range_layers=sum(not 0.0 <= ratio <= 1.0 for ratio in ratios),
        banks=list(banks.values()),
    )


class PrefillOverlapRecorder:
    """Keep bounded event batches; collect only completed work in a daemon.

    ``event_factory`` must return device events with timing enabled. ``emit``
    and ``initialize_thread`` run in the collector thread. Events are private
    to statistics, so recycling them cannot change bank or publication fences.
    """

    def __init__(
        self,
        event_factory: Callable[[], Any],
        emit: Callable[[str], None],
        *,
        worker: dict[str, Any],
        elapsed_time: Callable[[Any, Any], float],
        initialize_thread: Optional[Callable[[], None]] = None,
        start_worker: bool = True,
    ) -> None:
        self.event_factory = event_factory
        self.emit = emit
        self.worker = dict(host=socket.gethostname(), pid=os.getpid(), **worker)
        self.elapsed_time = elapsed_time
        self.initialize_thread = initialize_thread
        self.active: Optional[StepTimings] = None
        self.sequence = 0
        self.pending: Queue[StepTimings] = Queue(MAX_PENDING_STEPS)
        self.pool: LifoQueue[Any] = LifoQueue(MAX_POOLED_EVENTS)
        self.head: Optional[StepTimings] = None
        self.dropped_steps = 0
        self.failed = False
        self.stop = Event()
        self.thread: Optional[Thread] = None
        if start_worker:
            self.thread = Thread(target=self._run, name="prefill-overlap", daemon=True)
            self.thread.start()
            atexit.register(self.close)

    def begin_step(self, requests: tuple[tuple[str, int, int], ...]) -> None:
        """Begin a single-request step; empty/dummy/multi-request steps are free."""
        if self.active is not None:
            self.finish_step("incomplete")
        self.sequence += 1
        if len(requests) != 1 or self.failed:
            return
        if self.pending.full():
            self.dropped_steps += 1
            return
        self.active = StepTimings(self.sequence, requests)

    def start_interval(
        self,
        kind: str,
        stream: Any,
        group: int,
        bank: Optional[int],
        layer: int,
        model_layer: Optional[int] = None,
    ) -> Optional[TimingInterval]:
        """Record a start, after existing producer waits on the SAME stream."""
        if self.active is None or self.failed:
            return None
        ticket = TimingInterval(
            kind,
            group,
            bank,
            layer,
            layer if model_layer is None else model_layer,
            self._event(),
            self._event(),
        )
        ticket.start.record(stream)
        return ticket

    def end_interval(self, ticket: Optional[TimingInterval], stream: Any) -> None:
        """Record an end; never query or synchronize on the submission thread."""
        if ticket is None or self.active is None:
            return
        ticket.end.record(stream)
        self.active.intervals.append(ticket)
        self.active.tails[id(stream)] = ticket.end

    def finish_step(self, status: str = "complete") -> None:
        """Publish an immutable event batch without waiting for device work."""
        batch, self.active = self.active, None
        if batch is None:
            return
        batch.status = status
        try:
            self.pending.put_nowait(batch)
        except Full:
            self.dropped_steps += 1

    def poll(self) -> bool:
        """Collect one ready batch. Called by the daemon, or CPU-only tests."""
        if self.head is None:
            try:
                self.head = self.pending.get_nowait()
            except Empty:
                return False
        batch = self.head
        # Last event on each observed stream fences its earlier timing pairs.
        if not all(event.query() for event in batch.tails.values()):
            return False
        result = summarize_step(batch, self.elapsed_time)
        result.update(self.worker, dropped_steps=self.dropped_steps)
        self.emit(LOG_MARKER + " " + json.dumps(result, separators=(",", ":")))
        self.head = None
        for item in batch.intervals:
            for event in (item.start, item.end):
                try:
                    self.pool.put_nowait(event)
                except Full:
                    pass
        return True

    def close(self) -> None:
        """Give completed summaries a bounded shutdown drain; no device fence."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _event(self) -> Any:
        try:
            return self.pool.get_nowait()
        except Empty:
            return self.event_factory()

    def _run(self) -> None:
        try:
            if self.initialize_thread is not None:
                self.initialize_thread()
            while True:
                if self.head is None:
                    try:
                        self.head = self.pending.get(timeout=IDLE_POLL_SECONDS)
                    except Empty:
                        if self.stop.is_set():
                            break
                        continue
                if self.poll():
                    continue
                if self.stop.is_set():
                    break
                self.stop.wait(POLL_SECONDS)
        except Exception as error:
            self.failed = True
            self.emit(
                LOG_MARKER
                + " "
                + json.dumps(
                    dict(
                        schema_version=1,
                        status="collector_error",
                        error=str(error),
                        **self.worker,
                    )
                )
            )
