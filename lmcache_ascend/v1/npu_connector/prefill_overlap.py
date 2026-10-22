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
import time


LOG_MARKER = "[PREFILL_OVERLAP]"
COVERAGE_DEFINITION = "same_bank_reuse_v1"
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


def request_spans(
    requests: list[Any],
    attn_metadata: Any,
) -> tuple[tuple[str, int, int], ...]:
    """Identify all prefill requests in a model batch using CPU metadata.

    Connector metadata may omit requests that perform no cache operations.
    Check the actual attention batch as well, without device tensor readback.
    """
    if not requests or not isinstance(attn_metadata, Mapping):
        return ()
    spans = []
    for request in requests:
        span = getattr(request, "prefill_overlap_span", None)
        if span is None or span[0] >= span[1]:
            return ()
        spans.append((request.req_id, int(span[0]), int(span[1])))
    expected_ids = {row[0] for row in spans}
    if len(expected_ids) != len(spans):
        return ()
    metadata = next(iter(attn_metadata.values()), None)
    ids = getattr(metadata, "req_ids", None)
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or set(ids) != expected_ids:
            return ()
    else:
        # CPU shape only: query_start_loc has one more entry than requests.
        offsets = getattr(metadata, "query_start_loc_cpu", None)
        if offsets is None or len(offsets) != len(spans) + 1:
            return ()
    return tuple(spans)


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
    rotation_slot: Optional[int] = None


@dataclass
class StepTimings:
    step: int
    requests: tuple[tuple[str, int, int], ...]
    intervals: list[TimingInterval] = field(default_factory=list)
    tails: dict[int, Any] = field(default_factory=dict)
    status: str = "complete"
    # Submission time only anchors offline plots; durations use device events.
    start_time_ns: int = field(default_factory=time.time_ns)


def summarize_step(
    batch: StepTimings,
    elapsed_time: Callable[[Any, Any], float],
) -> dict[str, Any]:
    """Compare consecutive consumers of the SAME physical bank in a KV group.

    A model-layer start is its LAST bank consumer gate's END event, so both
    latent and indexer waits precede that start. Indexer waits carry the actual
    model-layer number, not the sparse group ordinal.
    The denominator spans the previous and current model-layer ready events
    for that bank, not adjacent model layers. Each gate contributes its own
    bank wait. Average valid bank ratios within each model layer, then average
    model layers so layers with an indexer do not receive extra weight.
    Negative measurements are preserved for diagnosis instead of clamped.

    With mixed request phases, one consumer gate joins multiple banks. Its
    rotation_slot identifies the group-layer ordinal modulo bank count: the
    same per-request physical-bank assignment returns after a full rotation.
    These batch gates are reported separately from physical-bank DMA totals.
    """
    starts: dict[int, Any] = {}
    waits: dict[int, float] = {}
    consumers: dict[tuple[int, Optional[int], Optional[int], int], float] = {}
    banks: dict[tuple[int, Optional[int]], dict[str, Any]] = {}
    batch_slots: dict[tuple[int, int], dict[str, Any]] = {}
    for item in batch.intervals:
        duration = elapsed_time(item.start, item.end)
        if item.kind == "wait":
            waits[item.model_layer] = waits.get(item.model_layer, 0.0) + duration
            starts[item.model_layer] = item.end
            key = (item.group, item.bank, item.rotation_slot, item.model_layer)
            consumers[key] = consumers.get(key, 0.0) + duration
        else:
            entry = banks.setdefault(
                (item.group, item.bank),
                dict(group=item.group, bank=item.bank, load_ms=0.0, save_ms=0.0),
            )
            entry[item.kind + "_ms"] += duration

    layer_ratios: dict[int, list[float]] = {}
    previous_layers: dict[tuple[int, str, int], int] = {}
    bank_ratios: dict[tuple[int, str, int], list[float]] = {}
    wait_ms, interval_ms, initial_bank_wait_ms = 0.0, 0.0, 0.0
    unknown_bank_waits, invalid_intervals = 0, 0
    for (group, bank, rotation_slot, current), waited in consumers.items():
        if bank is None and rotation_slot is None:
            unknown_bank_waits += 1
            continue
        if bank is None:
            identity = (group, "batch_rotation", rotation_slot)
            entry = batch_slots.setdefault(
                (group, rotation_slot),
                dict(group=group, rotation_slot=rotation_slot),
            )
        else:
            identity = (group, "physical", bank)
            entry = banks.setdefault(
                (group, bank),
                dict(group=group, bank=bank, load_ms=0.0, save_ms=0.0),
            )
        previous = previous_layers.get(identity)
        previous_layers[identity] = current
        if previous is None:
            initial_bank_wait_ms += waited
            entry["initial_wait_ms"] = waited
            continue
        period = elapsed_time(starts[previous], starts[current])
        if period <= 0:
            invalid_intervals += 1
            continue
        ratio = 1.0 - waited / period
        layer_ratios.setdefault(current, []).append(ratio)
        bank_ratios.setdefault(identity, []).append(ratio)
        entry["wait_ms"] = entry.get("wait_ms", 0.0) + waited
        entry["interval_ms"] = entry.get("interval_ms", 0.0) + period
        wait_ms += waited
        interval_ms += period
    ratios = [sum(values) / len(values) for values in layer_ratios.values()]
    entries = [
        ((group, "physical", bank), entry) for (group, bank), entry in banks.items()
    ] + [
        ((group, "batch_rotation", slot), entry)
        for (group, slot), entry in batch_slots.items()
    ]
    for identity, entry in entries:
        values = bank_ratios.get(identity, [])
        entry["reuse_samples"] = len(values)
        entry["coverage"] = sum(values) / len(values) if values else None
    load_ms = sum(row["load_ms"] for row in banks.values())
    save_ms = sum(row["save_ms"] for row in banks.values())
    return dict(
        schema_version=2,
        coverage_definition=COVERAGE_DEFINITION,
        coverage_aggregation="mean_layer_mean_bank_ratios",
        status=batch.status,
        scope="batch_step" if len(batch.requests) > 1 else "single_request_step",
        start_definition="last_bank_ready",
        step=batch.step,
        batch_start_time_ns=batch.start_time_ns,
        time_basis="host_batch_submission",
        requests=[
            dict(req_id=req, token_start=start, token_end=end)
            for req, start, end in batch.requests
        ],
        coverage=(sum(ratios) / len(ratios) if ratios else None),
        layer_samples=len(ratios),
        bank_reuse_samples=sum(len(values) for values in bank_ratios.values()),
        wait_ms=wait_ms,
        interval_ms=interval_ms,
        first_layer_wait_ms=waits[min(waits)] if waits else 0.0,
        initial_bank_wait_ms=initial_bank_wait_ms,
        all_wait_ms=sum(waits.values()),
        unknown_bank_waits=unknown_bank_waits,
        invalid_reuse_intervals=invalid_intervals,
        load_ms=load_ms,
        save_ms=save_ms,
        dma_ms=load_ms + save_ms,
        out_of_range_layers=sum(
            any(not 0.0 <= ratio <= 1.0 for ratio in values)
            for values in layer_ratios.values()
        ),
        banks=list(banks.values()),
        batch_rotation_slots=list(batch_slots.values()),
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
        """Begin one batch measurement; empty/dummy steps are free."""
        if self.active is not None:
            self.finish_step("incomplete")
        self.sequence += 1
        if not requests or self.failed:
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
        rotation_slot: Optional[int] = None,
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
            rotation_slot,
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
                        schema_version=2,
                        coverage_definition=COVERAGE_DEFINITION,
                        status="collector_error",
                        error=str(error),
                        **self.worker,
                    )
                )
            )
