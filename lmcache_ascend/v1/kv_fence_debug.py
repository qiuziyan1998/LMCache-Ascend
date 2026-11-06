# SPDX-License-Identifier: Apache-2.0
"""Opt-in, payload-free isolation of P/D device dependency boundaries.

LMCACHE_PREFILL_FENCE_DEBUG=check|p_load|p_sfa|p_vup|p_use|p_reuse|p_submit|p_save|d_load|all
Comma-separated stages are accepted, except p_use/p_reuse/p_submit alternatives.
``p_sfa`` and ``p_vup`` each require an isolated run, optionally with ``check``.
``all`` keeps p_use and excludes p_sfa/p_vup. ``check`` never waits on a device.
``p_submit`` drains the compute/bank host submission queues, not NPU execution.
LMCACHE_PREFILL_FENCE_GROUPS=0,1 selects KV groups (default: both).
LMCACHE_PREFILL_FENCE_LAYERS=start:end selects group-local layer ordinals,
with an exclusive end (default: all). No tensors, files or worker threads are
created. Incompatible heavy diagnostics are rejected, not silently overridden.
Append ``,trace`` to log every diagnostic synchronization and its registered
event dependencies. ``trace`` alone samples pre-SFA dependencies without waiting.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Callable, Mapping, Sequence


_STAGES = frozenset(
    ("p_load", "p_sfa", "p_vup", "p_use", "p_reuse", "p_submit", "p_save", "d_load")
)
_ENV = "LMCACHE_PREFILL_FENCE_DEBUG"


class _Event:
    def __init__(self, fence: Any, fields: dict[str, Any]) -> None:
        self.fence = fence
        self.fields = fields
        # One level only: retaining a recursive history would grow forever.
        self.after: tuple[tuple[Any, dict[str, Any]], ...] = ()


class FenceTrace:
    """Observe registered waits; this is not a runtime critical-path profiler."""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.emit = emit
        self.sequence = 0
        self.sync_sequence = 0
        self.events: dict[int, _Event] = {}
        self.streams: dict[tuple[Any, ...], Any] = {}
        self.waits: dict[tuple[Any, ...], dict[int, _Event]] = {}
        self.tails: dict[tuple[Any, ...], _Event] = {}
        self.request_events: dict[str, dict[int, _Event]] = {}

    def begin_step(self) -> None:
        """Keep only live stream dependencies across chunk boundaries."""
        retained = list(self.tails.values())
        retained.extend(event for row in self.waits.values() for event in row.values())
        retained.extend(
            event for row in self.request_events.values() for event in row.values()
        )
        self.events = {id(event.fence): event for event in retained}

    def forget_request(self, req: str) -> None:
        """Drop old event labels after the caller finishes request cleanup."""
        self.request_events.pop(req, None)

    def _stream_key(self, stream: Any) -> tuple[Any, ...]:
        # Do not read npu_stream: that accessor can drain Torch-NPU's task queue.
        # current_stream() can return a fresh Python wrapper for the same stream.
        stream_id = getattr(stream, "stream_id", None)
        key = (
            (
                "npu",
                str(getattr(stream, "device", getattr(stream, "device_index", None))),
                stream_id,
            )
            if stream_id is not None
            else ("object", id(stream))
        )
        self.streams[key] = stream
        return key

    def _event(self, fence: Any, **fields: Any) -> _Event:
        key = id(fence)
        event = self.events.get(key)
        if event is None:
            self.sequence += 1
            event = _Event(fence, dict(id=f"e{self.sequence}", **fields))
            self.events[key] = event
        return event

    def record(self, fence: Any, stream: Any, **fields: Any) -> None:
        """Label an existing event recorded after work on its source stream."""
        stream_key = self._stream_key(stream)
        event = self._event(fence, **fields)
        event.fields["source_stream"] = stream_key
        req = fields.get("req")
        if req is not None:
            self.request_events.setdefault(str(req), {})[id(fence)] = event
        predecessors = dict(self.waits.pop(stream_key, {}))
        tail = self.tails.get(stream_key)
        if tail is not None:
            predecessors[id(tail.fence)] = tail
        event.after = tuple(
            (parent.fence, parent.fields)
            for parent in predecessors.values()
            if parent.fence is not fence
        )
        # p_use follows the input marker. Preserve its direct load dependencies
        # without retaining a recursively growing graph of earlier layers.
        event.after += tuple(
            dependency
            for parent in predecessors.values()
            if parent.fields.get("kind") == "sfa_inputs"
            for dependency in parent.after
            if dependency[1].get("kind") in ("load", "save")
        )
        self.tails[stream_key] = event

    def join(self, stream: Any, fence: Any, **fields: Any) -> None:
        """Remember a wait that the caller has already submitted to this stream."""
        stream_key = self._stream_key(stream)
        event = self._event(fence, **fields)
        self.waits.setdefault(stream_key, {})[id(fence)] = event

    @staticmethod
    def _state(fence: Any, fields: dict[str, Any]) -> dict[str, Any]:
        try:
            ready = bool(fence.query())
        except Exception as exc:
            return dict(fields, ready=None, query_error=str(exc)[:240])
        return dict(fields, ready=ready)

    def _snapshot(self, events: list[_Event]) -> list[dict[str, Any]]:
        return [
            dict(
                self._state(event.fence, event.fields),
                after=[self._state(fence, fields) for fence, fields in event.after],
            )
            for event in events
        ]

    def _dependencies(self, fence: Any, target: str) -> list[_Event]:
        if target == "event":
            return [self._event(fence, kind="unregistered")]
        key = self._stream_key(fence)
        events = dict(self.waits.get(key, {}))
        tail = self.tails.get(key)
        if tail is not None:
            events[id(tail.fence)] = tail
        return list(events.values())

    def _prune_ready_waits(self, stream: Any, snapshot: list[dict[str, Any]]) -> None:
        key = self._stream_key(stream)
        completed = {row["id"] for row in snapshot if row["ready"] is True}
        row = self.waits.get(key, {})
        for identity, event in list(row.items()):
            if event.fields["id"] in completed:
                row.pop(identity)

    def checkpoint(self, stream: Any, **fields: Any) -> None:
        """Observe the same pre-SFA point in a run with no diagnostic sync."""
        events = self._dependencies(stream, "stream")
        snapshot = self._snapshot(events)
        self.emit(
            dict(
                event="checkpoint",
                target="stream",
                dependencies=snapshot,
                ordering_violation=sorted(
                    {
                        dependency["id"]
                        for row in snapshot
                        if row["ready"] is True
                        for dependency in row["after"]
                        if dependency["ready"] is False
                    }
                ),
                device_synchronize=False,
                **fields,
            )
        )
        self._prune_ready_waits(stream, snapshot)

    def synchronize(self, fence: Any, *, target: str, **fields: Any) -> None:
        """Log entry/return for exactly one caller-requested synchronization.

        Queries are non-atomic samples. False may mean host submission or device
        execution is pending. Duration is host wall time, not an event's GPU time.
        """
        self.sync_sequence += 1
        sync_id = self.sync_sequence
        events = self._dependencies(fence, target)
        entry = self._snapshot(events)
        emit_start = time.perf_counter()
        self.emit(
            dict(
                event="sync_begin",
                sync=sync_id,
                target=target,
                dependencies=entry,
                **fields,
            )
        )
        begin_log_ms = (time.perf_counter() - emit_start) * 1000
        # Logging itself can allow DMA to finish. Keep this second observation
        # so the log does not attribute that time to the blocking sync call.
        before = self._snapshot(events)
        started = time.perf_counter()
        try:
            fence.synchronize()
        except BaseException as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self.emit(
                dict(
                    event="sync_error",
                    sync=sync_id,
                    target=target,
                    sync_ms=round(elapsed, 3),
                    begin_log_ms=round(begin_log_ms, 3),
                    before_sync=before,
                    error=str(exc)[:400],
                    **fields,
                )
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000
        after = self._snapshot(events)
        pending_after = sorted(
            {
                item["id"]
                for row in after
                for item in (row, *row["after"])
                if item["ready"] is False
            }
        )
        self.emit(
            dict(
                event="sync_end",
                sync=sync_id,
                target=target,
                sync_ms=round(elapsed, 3),
                begin_log_ms=round(begin_log_ms, 3),
                before_sync=before,
                dependencies=after,
                pending_after=pending_after,
                **fields,
            )
        )
        if target == "stream":
            self._prune_ready_waits(fence, after)


class KVFenceDebug:
    """Check host handoffs and optionally synchronize one existing boundary."""

    @classmethod
    def from_environment(cls, role: str, rank: int) -> KVFenceDebug | None:
        """Return None when disabled; reject ambiguous/invalid experiments."""
        raw = os.environ.get(_ENV, "").strip().lower()
        if raw in ("", "0", "off"):
            return None
        return cls(role, rank, os.environ)

    def __init__(
        self,
        role: str,
        rank: int,
        env: Mapping[str, str],
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        raw = env.get(_ENV, "check").strip().lower()
        stages = set(raw.split(","))
        trace_enabled = "trace" in stages
        stages.discard("trace")
        if stages == {"all"}:
            stages = set(_STAGES) - {"p_sfa", "p_vup", "p_reuse", "p_submit"}
        if (not stages and not trace_enabled) or stages - (_STAGES | {"check"}):
            raise ValueError(f"Invalid {_ENV}: {raw!r}")
        if len(stages & {"p_use", "p_reuse", "p_submit"}) > 1:
            raise ValueError("Select only one of p_use and p_reuse and p_submit")
        if "p_sfa" in stages and stages - {"p_sfa", "check"}:
            raise ValueError(
                "Use p_sfa alone or with check to isolate the SFA input boundary"
            )
        if "p_vup" in stages and stages - {"p_vup", "check"}:
            raise ValueError(
                "Use p_vup alone or with check to isolate the V-up output boundary"
            )
        for name in ("LMCACHE_PREFILL_STORE_DEBUG", "LMCACHE_PREFILL_AUDIT"):
            if env.get(name, "0") == "1":
                raise ValueError(f"Set {name}=0 when using {_ENV}")
        self.stages = frozenset(stages - {"check"})
        self.check_protocol = "check" in stages
        groups = env.get("LMCACHE_PREFILL_FENCE_GROUPS", "0,1")
        try:
            self.groups = frozenset(int(value) for value in groups.split(","))
            bounds = env.get("LMCACHE_PREFILL_FENCE_LAYERS", "0:").split(":")
            if len(bounds) != 2:
                raise ValueError("expected start:end")
            self.first = int(bounds[0]) if bounds[0] else 0
            self.end = int(bounds[1]) if bounds[1] else None
            if (
                not self.groups
                or self.groups - {0, 1}
                or self.first < 0
                or (self.end is not None and self.end <= self.first)
            ):
                raise ValueError("invalid range")
        except ValueError as exc:
            raise ValueError("Invalid LMCACHE_PREFILL_FENCE_GROUPS/LAYERS") from exc
        if "p_sfa" in stages and 0 not in self.groups:
            raise ValueError("p_sfa requires latent KV group 0")
        if "p_vup" in stages and 0 not in self.groups:
            raise ValueError("p_vup requires latent KV group 0")
        self.role = role
        self.base = dict(
            host=socket.gethostname(), pid=os.getpid(), rank=rank, role=role
        )
        self.emit = emit or self._print
        self.trace = FenceTrace(self._emit_trace) if trace_enabled else None
        self.seen: set[tuple[str, int]] = set()
        # At most one retained event per group/bank, never a per-request history.
        self._reuse_fences: dict[tuple[int, int | None], Any] = {}
        self.expected: dict[tuple[int, int, int], set[str]] = {}
        self.submitted: dict[tuple[int, int, int], set[str]] = {}
        self.latest: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.consumed: set[tuple[int, int, int]] = set()
        self.step = 0
        self.begun = False
        self.emit(
            dict(
                self.base,
                event="enabled",
                stages=sorted(self.stages),
                groups=sorted(self.groups),
                layers=[self.first, self.end],
                payload_sampling=False,
                protocol_check=self.check_protocol,
                dependency_trace=trace_enabled,
                trace_scope=(
                    "registered bank waits plus SFA input marker"
                    if trace_enabled
                    else None
                ),
            )
        )

    @staticmethod
    def _print(record: dict[str, Any]) -> None:
        print("[KVFENCE] " + json.dumps(record, separators=(",", ":")), flush=True)

    def _emit_trace(self, record: dict[str, Any]) -> None:
        self.emit(dict(self.base, step=self.step, **record))

    def trace_record(self, kind: str, fence: Any, stream: Any, **fields: Any) -> None:
        """Label an existing event without creating or waiting for an event."""
        if self.trace is not None:
            self.trace.record(fence, stream, kind=kind, step=self.step, **fields)

    def trace_join(self, stream: Any, fence: Any, **fields: Any) -> None:
        """Describe an already-submitted stream wait, including both KV groups."""
        if self.trace is not None:
            self.trace.join(stream, fence, **fields)

    def observe_existing(
        self, fence: Any, *, stage: str, target: str = "event", **fields: Any
    ) -> None:
        """Trace a required production sync without adding or filtering it."""
        if self.trace is None:
            fence.synchronize()
        else:
            self.trace.synchronize(fence, target=target, stage=stage, **fields)

    def fail(self, code: str, group: int, layer: int, bank: int, **fields: Any) -> None:
        """Emit one actionable protocol error and stop before using the bank."""
        record = dict(
            self.base,
            event="error",
            code=code,
            step=self.step,
            group=group,
            layer=layer,
            bank=bank,
            **fields,
        )
        self.emit(record)
        raise RuntimeError("KVFENCE " + json.dumps(record, separators=(",", ":")))

    def begin_step(
        self,
        requests: Sequence[Any],
        layer_counts: Sequence[int],
        bank_count: int,
        chunk_size: int,
    ) -> None:
        """Declare required loads from scheduler metadata before any submission.

        A true zero-load request is allowed to wait on an earlier save only.
        A declared hit must submit every request's load before the shared bank
        is consumed. Counts/indices use the same aligned mask as the adapter.
        """
        if self.role != "P" or not (self.check_protocol or self.trace is not None):
            return
        if self.trace is not None:
            self.trace.begin_step()
        self.step += 1
        if self.trace is not None:
            self._emit_trace(
                dict(
                    event="step_requests",
                    requests=[request.req_id for request in requests],
                    bank_count=bank_count,
                )
            )
        if not self.check_protocol:
            return
        if bank_count <= 0 or chunk_size <= 0:
            raise ValueError("KVFENCE requires positive bank/chunk sizes")
        self.begun = True
        self.expected.clear()
        self.submitted.clear()
        self.latest.clear()
        self.consumed.clear()
        for request in requests:
            spec = request.load_spec
            if (
                spec is None
                or not spec.can_load
                or request.is_sparse_decode
                or getattr(spec, "dsa_cold_compact_load", False)
            ):
                continue
            count = min(int(spec.lmcache_cached_tokens), len(request.token_ids))
            masked = min(int(spec.vllm_cached_tokens), count) // chunk_size * chunk_size
            if count <= masked:
                continue
            for group, layers in enumerate(layer_counts):
                for layer in range(layers):
                    key = (group, layer, layer % bank_count)
                    self.expected.setdefault(key, set()).add(request.req_id)

    def loaded(
        self,
        req: str,
        group: int,
        layer: int,
        bank: int,
        event: Any,
    ) -> None:
        """Remember actual submission order, including all requests on one FIFO."""
        if not self.check_protocol:
            return
        key = (group, layer, bank)
        if not self.begun:
            self.fail("missing_step_declaration", group, layer, bank, req=req)
        if key in self.consumed:
            self.fail("load_after_consume", group, layer, bank, req=req)
        self.submitted.setdefault(key, set()).add(req)
        self.latest.setdefault(key, {})[req] = event

    def before_read(
        self, group: int, layer: int, bank: int, events: Mapping[str, Any]
    ) -> None:
        """Reject absent/stale/wrong handoffs without inspecting NPU contents."""
        if not self.check_protocol:
            return
        key = (group, layer, bank)
        if not self.begun:
            self.fail("missing_step_declaration", group, layer, bank)
        if key in self.consumed:
            return  # The adapter can join a layer twice on the same compute stream.
        expected = self.expected.get(key, set())
        submitted = self.submitted.get(key, set())
        if expected != submitted:
            self.fail(
                "request_load_mismatch",
                group,
                layer,
                bank,
                missing=sorted(expected - submitted),
                unexpected=sorted(submitted - expected),
            )
        latest = self.latest.get(key, {})
        wrong = sorted(
            req
            for req in set(latest) | set(events)
            if latest.get(req) is not events.get(req)
        )
        if wrong:
            self.fail(
                "wrong_load_event",
                group,
                layer,
                bank,
                requests=wrong,
                expected_events=len(latest),
                actual_events=len(events),
            )
        self.consumed.add(key)
        self.latest.pop(key, None)
        self.submitted.pop(key, None)

    def before_sfa(
        self,
        group: int,
        layer: int,
        bank: int,
        current_stream: Callable[[], Any],
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Finish current-stream input work immediately before latent SFA.

        The callback is resolved only for selected P layers. No final-use
        event is recorded or replaced, and no payload is inspected or copied.
        """
        if (
            self.role != "P"
            or group != 0
            or group not in self.groups
            or layer < self.first
            or (self.end is not None and layer >= self.end)
        ):
            return
        stream = None
        if self.trace is not None and event_factory is not None:
            stream = current_stream()
            ready = event_factory()
            ready.record(stream)
            self.trace_record(
                "sfa_inputs", ready, stream, group=group, layer=layer, bank=bank
            )
        if "p_sfa" in self.stages:
            self.wait(
                "p_sfa",
                group,
                layer,
                stream if stream is not None else current_stream(),
                bank=bank,
                target="stream",
            )
        elif self.trace is not None and not self.stages:
            self.trace.checkpoint(
                stream if stream is not None else current_stream(),
                stage="p_sfa",
                group=group,
                layer=layer,
                bank=bank,
            )

    def after_vup(
        self,
        group: int,
        layer: int,
        bank: int,
        current_stream: Callable[[], Any],
    ) -> None:
        """Finish P latent V-up output before submitting output projection.

        Called after V-up submission. Resolve the stream only for selected
        layers, preserving the existing SFA final-use event and bank handoffs.
        """
        if (
            "p_vup" not in self.stages
            or self.role != "P"
            or group != 0
            or group not in self.groups
            or layer < self.first
            or (self.end is not None and layer >= self.end)
        ):
            return
        self.wait("p_vup", group, layer, current_stream(), bank=bank, target="stream")

    def before_reuse(
        self,
        group: int,
        bank: int,
        use: tuple[int, int, Any] | None,
    ) -> None:
        """Fence the last bank reader before save or overwrite, on every rank.

        First loads can reuse a bank from the preceding chunk, so use the
        stored event even across transfer generations. With no earlier reader
        there is nothing to wait for. Layer filtering selects the source layer
        N, including when the pending operation is load N+bank_count.
        """
        if "p_reuse" not in self.stages or use is None:
            return
        _generation, layer, event = use
        self.wait("p_reuse", group, layer, event, bank=bank)

    def after_use_wait(
        self,
        group: int,
        layer: int,
        bank: int,
        current_stream: Callable[[], Any],
        dma_stream: Any,
    ) -> None:
        """Drain host submissions after enqueueing DMA's last-reader wait.

        In torch-npu 12d689a0, Stream.npu_stream calls NPUStream::stream(),
        which runs MakeSureQueueEmpty() before returning the native handle.
        With per-stream queues, both compute and DMA must be drained. With
        the shared queue, the second drain is normally already empty. This
        does not call aclrtSynchronizeEvent/Stream/Device or poll completion.
        It is a targeted runtime experiment, not the capture-time upstream
        emptyAllNPUStream() fix: unrelated streams/devices are not drained.
        """
        if (
            "p_submit" not in self.stages
            or self.role != "P"
            or group not in self.groups
            or layer < self.first
            or (self.end is not None and layer >= self.end)
        ):
            return
        try:
            # Resolve only after filtering, so other modes do not touch NPU APIs.
            for stream in (current_stream(), dma_stream):
                handle = stream.npu_stream
                if not isinstance(handle, int) or handle == 0:
                    raise RuntimeError("Task-queue drain returned no NPU stream handle")
        except Exception as exc:
            self.fail(
                "submission_drain_failed",
                group,
                layer,
                bank,
                stage="p_submit",
                detail=str(exc)[:400],
            )
        marker = ("p_submit", group)
        if marker not in self.seen:
            self.seen.add(marker)
            self.emit(
                dict(
                    self.base,
                    event="active",
                    stage="p_submit",
                    group=group,
                    layer=layer,
                    bank=bank,
                    queues=["compute", "bank_dma"],
                    device_synchronize=False,
                )
            )

    def wait(
        self,
        stage: str,
        group: int,
        layer: int,
        fence: Any,
        *,
        req: str | None = None,
        bank: int | None = None,
        target: str = "event",
    ) -> None:
        """Wait only at the selected boundary, never by taking a global device fence.

        ``fence`` is an existing event or the exact transfer stream. Callers
        place ordinary fences immediately after that operation. p_reuse waits
        on the original reader event at the later DMA submission boundary.
        """
        if (
            stage not in self.stages
            or not stage.startswith(self.role.lower())
            or group not in self.groups
            or layer < self.first
            or (self.end is not None and layer >= self.end)
        ):
            return
        reuse_key = (group, bank)
        if stage == "p_reuse" and self._reuse_fences.get(reuse_key) is fence:
            return
        marker = (stage, group)
        if marker not in self.seen:
            self.seen.add(marker)
            self.emit(
                dict(
                    self.base,
                    event="active",
                    stage=stage,
                    group=group,
                    layer=layer,
                    bank=bank,
                    req=req,
                )
            )
        try:
            if self.trace is None:
                fence.synchronize()
            else:
                self.trace.synchronize(
                    fence,
                    target=target,
                    stage=stage,
                    group=group,
                    layer=layer,
                    bank=bank,
                    req=req,
                )
        except Exception as exc:
            self.fail(
                "device_fence_failed",
                group,
                layer,
                -1 if bank is None else bank,
                req=req,
                stage=stage,
                detail=str(exc)[:400],
            )
        if stage == "p_reuse":
            # Multiple requests, then the next load, share the reader event.
            # Synchronize it once; a new event on this bank replaces this entry.
            self._reuse_fences[reuse_key] = fence
