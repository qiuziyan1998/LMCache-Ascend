# SPDX-License-Identifier: Apache-2.0
"""Worker checkpoint capture using M-owned pages and explicit participant leases."""

# Standard
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field
from threading import RLock
from typing import Annotated, Any
from weakref import proxy
import time

# Third Party
import msgspec
import torch

# First Party
from lmcache.integration.vllm.preemption_checkpoint import (
    CaptureSpec,
    CheckpointResult,
    SealSpec,
)
from lmcache.integration.vllm.storage_process import cancel_unadmitted_operation
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.rpc.storage_protocol import (
    SourceGeometry,
    pack_checkpoint_tokens_u32,
    unpack_uints,
)
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle, SharedHandleBatch
from lmcache_ascend.v1.preemption_checkpoint import (
    CaptureJob,
    CheckpointWorker,
    clear_failure_tracebacks,
)


@dataclass
class _Reservation:
    future: Any
    job: CaptureJob
    reply: dict | None = None
    holds: list[LocalLeaseHandle] = field(default_factory=list)
    imported: bool = False
    fenced: bool = False
    sealed: bool = False
    releasing: bool = False
    released: bool = False
    acknowledging: bool = False
    retired: set[str] = field(default_factory=set)
    cancel_future: Any = None
    cleanup_future: Any = None
    resume_on: Any = None
    cancel_deadline: float = 0.0


class _CheckpointControls:
    def __init__(self, worker: Any) -> None:
        self.worker = proxy(worker)

    def forget(self, req_id: str, generation: int | None = None) -> None:
        """Forget M's advisory offer on the existing request/generation event."""
        worker = self.worker
        if worker.writer:
            worker.submit_control(
                "checkpoint_forget",
                {"generation": generation},
                req_id,
                generation or 0,
                f"forget-{generation}",
            )

    def restore_event(self, identity: tuple, attempt: int, event: str) -> None:
        """Forward bounded lifecycle notifications without waiting in W."""
        worker = self.worker
        if (
            worker.writer
            and event != "begin"
            and getattr(worker.engine.config, "decode_checkpoint_writeback", False)
        ):
            worker.submit_control(
                "checkpoint_restore_event",
                {"generation": identity[1], "attempt": attempt, "event": event},
                identity[0],
                identity[1],
                f"restore-{attempt}",
            )

    def normalize(self, *args: Any, **kwargs: Any) -> None:
        """Restore normalization belongs to the coarse M prepare operation."""
        raise RuntimeError(
            "Process checkpoint restore must use its prepared source ticket"
        )


class ProcessCheckpointWorker(CheckpointWorker):
    """Reuse native capture and accepted-token controls without a worker CPU owner."""

    def __init__(self, engine: Any) -> None:
        super().__init__(engine, local_store=_CheckpointControls(self))
        self.context = engine.storage_process
        if self.context is None:
            raise ValueError(
                "Process checkpoint capture requires early storage bootstrap"
            )
        self.client = self.context.client
        self.writer = engine.metadata.is_first_rank()
        self.reservations: dict[tuple[str, int], _Reservation] = {}
        self.pending_retirements: dict[str, Any] = {}
        self.pending_controls: dict[tuple, Any] = {}
        self._sources: dict[tuple[str, int], Any] = {}
        self._lock = RLock()
        receiver = proxy(self)
        self.context.drain_guards.append(lambda: receiver.sources_drained())

    def sources_drained(self) -> bool:
        """Keep mappings registered until every capture and retirement is terminal."""
        with self._lock:
            return (
                not self.reservations
                and not self.pending_retirements
                and not self.pending_controls
                and not self.restore_owners
            )

    def capture(
        self,
        spec: CaptureSpec,
        caches: dict[int, list],
        block_size: int,
        prefix_state: Any = None,
    ) -> None:
        """Only TP0 reserves/captures; every rank still owns its restore holds."""
        if not self.writer:
            return
        key = (spec.req_id, spec.generation)
        if key in self.jobs:
            # EngineCore may repeat a control until its result arrives. The
            # existing captured generation no longer needs its old prefix lease.
            super().capture(spec, caches, block_size)
            return
        with self._lock:
            reconciling = key in self.reservations
            occupied = len(set(self.jobs) | self.reservations.keys())
        if reconciling:
            self.results.append(
                CheckpointResult(
                    *key, "failed", reason="checkpoint generation is still reconciling"
                )
            )
            return
        if occupied >= self.max_jobs:
            self.results.append(
                CheckpointResult(*key, "failed", reason="checkpoint staging is busy")
            )
            return
        source = getattr(prefix_state, "prepared_sparse_sources", {}).get(0)
        try:
            if source is not None:
                self._sources[key] = (
                    self.engine.compact_source_importer.source_descriptor(source)
                )
            # M validates and retains the source descriptor during reservation.
            # W has no owner cache and must not run local-mode prefix acquisition.
            super().capture(spec, caches, block_size)
        except Exception as error:
            # Descriptor validation happens before the base capture owns a job.
            if key not in self.reservations:
                self.results.append(CheckpointResult(*key, "failed", reason=str(error)))
            else:
                raise
        finally:
            self._sources.pop(key, None)

    def seal(self, seal: SealSpec) -> None:
        """Relay accepted/computed tokens once through the existing seal executor."""
        if self.writer:
            super().seal(seal)

    def cancel(self, req_id: str, generation: int | None = None) -> None:
        """Queue cleanup without waiting for service replies, including during seal."""
        super().cancel(req_id, generation)
        with self._lock:
            states = tuple(self.reservations.values())
        for state in states:
            spec = state.job.spec
            if spec.req_id == req_id and (
                generation is None or spec.generation == generation
            ):
                state.job.cancelled = True
                self._release(state.job)

    def release_restore(
        self, req_id: str, generation: int, load_generation: int
    ) -> None:
        """Retire normalization ownership at the existing all-TP receive ACK."""
        self.engine.compact_source_importer.release_restore(req_id, load_generation)
        self.restore_event(req_id, generation, load_generation, "ack")
        for owner in self.restore_owners.pop((req_id, generation, load_generation), ()):
            if isinstance(owner, LocalLeaseHandle):
                owner.release("fenced")
            else:
                owner.ref_count_down()

    def submit_control(
        self,
        method: str,
        payload: dict,
        req_id: str,
        generation: int,
        attempt: str,
        *,
        retry_deadline: float | None = None,
    ) -> Any:
        """Queue one CPU-only metadata operation and retain its terminal ACK."""
        future = self.client.request(
            method,
            payload,
            request_id=req_id,
            request_attempt=attempt,
            load_generation=generation,
            purpose="checkpoint_control",
        )
        with self._lock:
            self.pending_controls[future.identity.key] = future
        receiver = proxy(self)
        deadline = retry_deadline
        if method == "checkpoint_restore_event" and deadline is None:
            deadline = time.monotonic() + self.timeout

        def completed(done: Any) -> None:
            try:
                result = done.result()
                if result.status == "BUSY" and method == "checkpoint_restore_event":
                    receiver.executor.submit(
                        receiver._retry_restore_control, done, payload, deadline
                    )
                    return
                if result.status not in (
                    "COMPLETED",
                    "FAILED_SAFE",
                    "CANCELLED_NOT_STARTED",
                ):
                    return
                acknowledged = receiver.client.acknowledge_terminal(done.identity)
                with receiver._lock:
                    receiver.pending_controls[done.identity.key] = acknowledged

                def retired(ack: Any) -> None:
                    if ack.result().status == "ACKNOWLEDGED":
                        with receiver._lock:
                            receiver.pending_controls.pop(done.identity.key, None)

                acknowledged.add_done_callback(retired)
            except BaseException:
                return  # The pending map retains uncertainty, including the mapping.

        future.add_done_callback(completed)
        return future

    def _retry_restore_control(
        self, future: Any, payload: dict, deadline: float, cancelled: bool = False
    ) -> None:
        """Reconcile refused notifications off-thread using existing cancel proof."""
        try:
            if time.monotonic() >= deadline:
                return
            if not cancelled:
                result = cancel_unadmitted_operation(
                    self.client, future, max(0.001, deadline - time.monotonic())
                )
                if result.status not in ("BUSY", "CANCELLED_NOT_STARTED"):
                    return  # Keep ambiguous operation evidence and drain guard.
                cancelled = result.status == "CANCELLED_NOT_STARTED"
            if cancelled:
                identity = future.identity
                self.submit_control(
                    "checkpoint_restore_event",
                    payload,
                    identity.request_id,
                    identity.load_generation,
                    identity.request_attempt,
                    retry_deadline=deadline,
                )
                with self._lock:
                    self.pending_controls.pop(identity.key, None)
                return
        except BufferError:
            pass  # Cancellation/ACK credit can recover just like submission credit.
        except BaseException as error:
            clear_failure_tracebacks(error)
            return  # No timeout or failed IPC establishes notification completion.
        # Retirement queued on this executor can be what frees capacity.
        # Yield after one refusal instead of occupying its sole worker.
        time.sleep(0.005)
        self.executor.submit(
            self._retry_restore_control, future, payload, deadline, cancelled
        )

    def poll(self) -> tuple[CheckpointResult, ...]:
        """Preserve normal control results and retry a short refused CPU cancel."""
        with self._lock:
            states = tuple(self.reservations.values())
        for state in states:
            job = state.job
            if state.released:
                self._ack_released(state.future.identity)
            elif job.cancelled and job.future is None and not job.quarantined:
                self._release(job)
                if state.released:
                    self.jobs.pop((job.spec.req_id, job.spec.generation), None)
            if job.quarantined:
                raise RuntimeError(
                    "Checkpoint has unresolved service or native ownership"
                )
        return super().poll()

    def close(self) -> None:
        """Drain source views and refuse teardown while service ownership is unknown."""
        if self.restore_owners:
            raise RuntimeError(
                "Checkpoint restores still await all-worker acknowledgement"
            )
        for req_id, generation in tuple(self.jobs):
            self.cancel(req_id, generation)
        with self._lock:
            states = tuple(self.reservations.values())
        for state in states:
            state.job.cancelled = True
            self._release(state.job)
        deadline = time.monotonic() + self.timeout
        while True:
            # Poll may enqueue the final forget for a completed cancelled seal.
            # Include that control in the drain before stopping this executor.
            self.poll()
            if self.sources_drained():
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Process checkpoint ownership has not drained")
            time.sleep(0.005)
        self.executor.shutdown(wait=True)

    def _reserve_fragments(self, job: CaptureJob, caches: dict[int, list]) -> int:
        spec = job.spec
        source = self._sources.get((spec.req_id, spec.generation))
        attempt = (
            source["identity"].request_attempt
            if source
            else f"checkpoint-{spec.generation}"
        )
        payload = dict(
            purpose="checkpoint",
            generation=spec.generation,
            base=spec.base,
            end=spec.end,
            resident_start=spec.resident_start,
            prefix_end=spec.prefix_end,
            request_configs=dict(spec.request_configs or {}),
        )
        if source:
            payload.update(
                source_operation=source, source_lease_id=source["manifest"].lease_id
            )
        future = self.client.request(
            "reserve_write",
            payload,
            request_id=spec.req_id,
            request_attempt=attempt,
            load_generation=spec.generation,
            purpose="checkpoint_capture",
        )
        state = _Reservation(future, job)
        with self._lock:
            self.reservations[spec.req_id, spec.generation] = state
        result = future.result(self.timeout)
        if result.status == "COMPLETED" and result.value.get("status") == "SKIPPED":
            self._ack_released(future.identity, no_leases=True)
            raise MemoryError("Checkpoint capture refused while cache is frozen")
        if result.status != "COMPLETED" or result.error:
            if result.status == "FAILED_UNKNOWN":
                job.quarantined = True
            raise MemoryError(
                result.error or f"CPU checkpoint reservation: {result.status}"
            )
        state.reply = result.value
        self._import_fragments(job, state)
        return int(state.reply["captured_end"])

    def _import_fragments(self, job: CaptureJob, state: _Reservation) -> None:
        reply, mapping, layout = (
            state.reply,
            self.context.mapping,
            self.engine.storage_layout,
        )
        receiver = proxy(self)
        identity, digest = state.future.identity, state.future.payload_digest
        # A trusted reservation reply already exported participant leases, even
        # when its slab/frontier or a later group's geometry is rejected. Track
        # every unique lease before validating data that precedes local import.
        acquired = {hold.lease_id for hold in state.holds}
        for group in reply["groups"]:
            lease_id = group["lease_id"]
            if lease_id not in acquired:
                state.holds.append(
                    LocalLeaseHandle(
                        lease_id=lease_id,
                        mapping=mapping,
                        retire=lambda lease, proof: receiver._retire(
                            identity, digest, lease, proof
                        ),
                    )
                )
                acquired.add(lease_id)
        if (
            reply["slab_name"] != mapping.shm_name
            or reply["slab_size"] != mapping.size
            or reply["slab_generation"] != mapping.generation
            or not job.spec.prefix_end < reply["captured_end"] <= job.spec.end
        ):
            raise ValueError(
                "Checkpoint reservation has stale slab or capture identity"
            )
        allocator = self.engine.shared_cpu_cache_passive_allocator
        seen = set()
        for group in reply["groups"]:
            group_id = group["group_id"]
            if group_id not in (0, 1) or group_id in seen:
                raise ValueError(
                    "Checkpoint reservation has duplicate or invalid groups"
                )
            seen.add(group_id)
        if 1 not in seen or (
            job.spec.resident_start < reply["captured_end"] and 0 not in seen
        ):
            raise ValueError("Checkpoint reservation omits a required KV group")
        for group in reply["groups"]:
            group_id = group["group_id"]
            starts, ends = tuple(group["starts"]), tuple(group["ends"])
            count, layers = len(starts), len(layout.groups[group_id].layer_names)
            offsets, extents = (
                unpack_uints(group["page_offsets"], count),
                unpack_uints(group["page_extents"], count),
            )
            indices = unpack_uints(group["geometry_ids"], count, 32)
            geometries = msgspec.convert(
                group["geometries"], type=tuple[SourceGeometry, ...]
            )
            cursor = (
                max(job.spec.base, job.spec.resident_start)
                if group_id == 0
                else job.spec.base
            )
            prepared_geometry = {}
            for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                if start != cursor or not start < end <= reply["captured_end"]:
                    raise ValueError(
                        "Checkpoint reservation coverage has a hole or overlap"
                    )
                geometry_id = indices[index]
                if geometry_id >= len(geometries):
                    raise ValueError("Checkpoint geometry index is out of bounds")
                geometry = geometries[geometry_id]
                if (
                    geometry.token_capacity != end - start
                    or offsets[index] + extents[index] > mapping.size
                ):
                    raise ValueError("Checkpoint page exceeds its destination bounds")
                key = (geometry_id, extents[index])
                if key not in prepared_geometry:
                    geometry.validate(layers, extents[index])
                    position, layer_starts = 0, []
                    for size in geometry.layer_sizes:
                        layer_starts.append(position)
                        position += size
                    if (
                        geometry.layer_starts != tuple(layer_starts)
                        or len(set(geometry.layer_dtypes)) != 1
                    ):
                        raise ValueError("Checkpoint physical layer geometry disagrees")
                    prepared_geometry[key] = (
                        [torch.Size(shape) for shape in geometry.layer_shapes],
                        getattr(torch, geometry.layer_dtypes[0].removeprefix("torch.")),
                    )
                shapes, dtype = prepared_geometry[key]
                batch = SharedHandleBatch(
                    mapping.shm_name,
                    0,
                    layers,
                    1,
                    [geometry.layer_sizes[0]],
                    [0],
                    [],
                    [offsets[index]],
                    [extents[index]],
                )
                page = allocator.create_page_view(
                    batch,
                    chunk_index=0,
                    shape=shapes[0],
                    dtype=dtype,
                    fmt=MemoryFormat(layout.groups[group_id].memory_format),
                    cached_positions=range(start, end),
                    layer_shapes=shapes,
                )
                job.fragments.setdefault(group_id, []).append((start, end, page))
                cursor = end
            if cursor != reply["captured_end"]:
                raise ValueError("Checkpoint reservation omits its captured tail")

    def _begin_capture(self, job: CaptureJob) -> None:
        state = self.reservations[job.spec.req_id, job.spec.generation]
        for hold in state.holds:
            hold.mark_imported()
        state.imported = True

    def _capture_completed(self, job: CaptureJob) -> None:
        state = self.reservations[job.spec.req_id, job.spec.generation]
        # The reservation retains its producer holds until seal or cancellation.
        # Both controls already accept this final-access proof, so reporting local
        # capture completion needs no separate service acknowledgement.
        state.fenced = True

    def _publish_local(self, job: CaptureJob, seal: SealSpec) -> int:
        job.publish_started = time.monotonic()
        if job.cancelled:
            return 0
        state = self.reservations[job.spec.req_id, job.spec.generation]
        payload: dict[str, Any] = {
            "reservation_id": state.reply["reservation_id"],
            "phase": "seal",
            "proof": "fenced",
        }
        descriptor = getattr(self.context, "descriptor", {})
        if descriptor.get("checkpoint_tokens_u32_le_v1") is True:
            try:
                frame = pack_checkpoint_tokens_u32(seal.tokens)
            except msgspec.ValidationError:
                # Wider supported integers retain their exact legacy encoding.
                # Strict conversion still rejects bools, strings and negatives.
                payload["tokens"] = msgspec.convert(
                    seal.tokens,
                    type=tuple[Annotated[int, msgspec.Meta(ge=0)], ...],
                    strict=True,
                )
            else:
                payload.update(token_count=len(frame) // 4, tokens_u32_le=frame)
        else:
            payload["tokens"] = seal.tokens
        self._call(
            job,
            "commit_write",
            payload,
        )
        state.sealed = True
        job.publish_finished = time.monotonic()
        return len(seal.tokens)

    def _call(self, job: CaptureJob, method: str, payload: dict) -> Any:
        state = self.reservations[job.spec.req_id, job.spec.generation]
        future = self.submit_control(
            method,
            payload,
            job.spec.req_id,
            job.spec.generation,
            state.future.identity.request_attempt,
        )
        result = future.result(self.timeout)
        if result.status == "FAILED_UNKNOWN":
            job.quarantined = True
            raise NativeExternalPageTransferUnknownError(
                f"checkpoint {method}: {result.error or 'service completion unknown'}",
                future,
            )
        if result.status != "COMPLETED" or result.error:
            raise RuntimeError(
                result.error or f"Checkpoint control returned {result.status}"
            )
        return result.value

    def _retire(self, identity: Any, digest: bytes, lease: str, proof: str) -> None:
        future = self.client.retire(identity, digest, lease, proof)
        with self._lock:
            self.pending_retirements[lease] = future
        receiver = proxy(self)

        def retired(done: Any) -> None:
            try:
                if done.result().status == "RETIRED":
                    with receiver._lock:
                        receiver.pending_retirements.pop(lease, None)
                        state = receiver.reservations.get(
                            (identity.request_id, identity.load_generation)
                        )
                        if state is not None:
                            state.retired.add(lease)
                    receiver._ack_released(identity)
            except BaseException:
                return

        future.add_done_callback(retired)

    def _ack_released(self, identity: Any, *, no_leases: bool = False) -> None:
        with self._lock:
            state = self.reservations.get(
                (identity.request_id, identity.load_generation)
            )
            if state is None or state.acknowledging:
                return
            if not no_leases and (
                not state.released
                or any(hold.lease_id not in state.retired for hold in state.holds)
            ):
                return
            state.acknowledging = True
        acknowledged = self.client.acknowledge_terminal(identity)
        receiver = proxy(self)

        def done(future: Any) -> None:
            try:
                with receiver._lock:
                    if future.result().status == "ACKNOWLEDGED":
                        receiver.reservations.pop(
                            (identity.request_id, identity.load_generation), None
                        )
                    else:
                        state.acknowledging = False
            except BaseException:
                return

        acknowledged.add_done_callback(done)

    def _release(self, job: CaptureJob) -> None:
        key = (job.spec.req_id, job.spec.generation)
        with self._lock:
            state = self.reservations.get(key)
        if state is None:
            CheckpointWorker._release(job)
            return
        if job.quarantined:
            return
        if state.released:
            self._ack_released(state.future.identity)
            return
        with self._lock:
            if state.releasing or (
                state.resume_on is not None and not state.resume_on.done()
            ):
                return
            state.releasing = True
            # The base capture calls release only after its final-access fence,
            # including the recovery fence following partially prepared work.
            if state.imported:
                state.fenced = True
            try:
                state.cleanup_future = self.executor.submit(
                    self._release_reserved, state
                )
            except RuntimeError:
                state.releasing = False
                job.quarantined = True

    def _resume_release(self, state: _Reservation, future: Any) -> None:
        """Continue after the original reply, even with no foreground decode poll."""
        if state.resume_on is future:
            return
        state.resume_on = future
        key = (state.job.spec.req_id, state.job.spec.generation)
        operation = state.future.identity.key
        receiver = proxy(self)

        def completed(_: Any) -> None:
            # Queue behind the current cleanup/seal on the existing executor.
            # Capturing only the key avoids Future -> reservation ownership cycles.
            try:
                receiver.executor.submit(receiver._continue_release, key, operation)
            except RuntimeError:
                with receiver._lock:
                    retained = receiver.reservations.get(key)
                    if (
                        retained is not None
                        and retained.future.identity.key == operation
                    ):
                        retained.job.quarantined = True

        future.add_done_callback(completed)

    def _continue_release(self, key: tuple[str, int], operation: tuple) -> None:
        with self._lock:
            state = self.reservations.get(key)
            if state is None or state.future.identity.key != operation:
                return
            state.resume_on = None
        self._release(state.job)

    def _release_reserved(self, state: _Reservation) -> None:
        """Run bounded CPU control waits on the existing checkpoint executor."""
        try:
            self._finish_release(state)
        except BaseException as error:
            state.job.quarantined = True
            clear_failure_tracebacks(error)
        finally:
            with self._lock:
                state.releasing = False

    def _finish_release(self, state: _Reservation) -> None:
        job = state.job
        # A cancellation queued during sealing follows it in this same executor.
        published = job.future
        if published is not None and job.cancelled:
            try:
                published.result()
            except NativeExternalPageTransferUnknownError:
                job.quarantined = True
                return
            except BaseException as error:
                clear_failure_tracebacks(error)
        if state.reply is None:
            try:
                result = state.future.result() if state.future.done() else None
            except BaseException:
                job.quarantined = True
                return
            if result is not None and result.status == "BUSY":
                deadline = time.monotonic() + self.timeout
                while result.status == "BUSY":
                    remaining = max(0.0, deadline - time.monotonic())
                    result = cancel_unadmitted_operation(
                        self.client, state.future, remaining
                    )
                    if result.status == "BUSY":
                        if not remaining:
                            raise TimeoutError(
                                "CPU reservation cancellation deadline expired"
                            )
                        time.sleep(min(0.005, remaining))
            if result is not None and result.status == "COMPLETED":
                state.reply = result.value
                if result.value.get("status") == "SKIPPED":
                    self._ack_released(state.future.identity, no_leases=True)
                    return
                try:
                    self._import_fragments(job, state)
                except Exception as error:
                    # Rejected exported geometry still owns every lease adopted
                    # before validation; cancellation needs no device operands.
                    clear_failure_tracebacks(error)
            elif result is not None and result.status in (
                "FAILED_SAFE",
                "CANCELLED_NOT_STARTED",
            ):
                self._ack_released(state.future.identity, no_leases=True)
                return
            elif result is not None and result.status == "FAILED_UNKNOWN":
                job.quarantined = True
                return
            else:
                self.client.cancel(state.future.identity)
                self.client.replay(state.future.identity)
                self._resume_release(state, state.future)
                return
        if not state.sealed:
            if not self._cancel_reserved(state):
                return
        if job.cancelled:
            # Also covers a seal whose response timed out before M completed it.
            # Forget only after cancellation proves publication is no longer live.
            self.local.forget(job.spec.req_id, job.spec.generation)
        CheckpointWorker._release(job)
        for hold in state.holds:
            hold.release("fenced" if state.imported else "never_imported")
        state.released = True
        self._ack_released(state.future.identity)

    def _cancel_reserved(self, state: _Reservation) -> bool:
        job = state.job
        if not state.cancel_deadline:
            state.cancel_deadline = time.monotonic() + self.timeout
        while True:
            remaining = max(0.0, state.cancel_deadline - time.monotonic())
            if state.cancel_future is None:
                try:
                    state.cancel_future = self.submit_control(
                        "cancel_write",
                        {
                            "reservation_id": state.reply["reservation_id"],
                            "proof": "fenced" if state.imported else "never_imported",
                        },
                        job.spec.req_id,
                        job.spec.generation,
                        state.future.identity.request_attempt,
                    )
                except BufferError:
                    if not remaining:
                        raise
                    time.sleep(min(0.005, remaining))
                    continue
            future = state.cancel_future
            if not future.done():
                self._resume_release(state, future)
                return False
            result = future.result()
            if result.status == "BUSY":
                result = cancel_unadmitted_operation(self.client, future, remaining)
                if result.status == "CANCELLED_NOT_STARTED":
                    with self._lock:
                        self.pending_controls.pop(future.identity.key, None)
                elif result.status == "BUSY":
                    # A full service registry can refuse the lifecycle cancel as
                    # well as the work submission. Retain and reconcile this same
                    # original operation until its non-admission is proven.
                    if time.monotonic() >= state.cancel_deadline:
                        raise TimeoutError(
                            "CPU checkpoint cancellation admission deadline expired"
                        )
                    time.sleep(min(0.005, remaining))
                    continue
            if result.status == "FAILED_UNKNOWN":
                job.quarantined = True
                return False
            retry = result.status == "CANCELLED_NOT_STARTED" or (
                result.status == "COMPLETED"
                and isinstance(result.value, dict)
                and result.value.get("status") == "PENDING"
            )
            if retry:
                if time.monotonic() >= state.cancel_deadline:
                    raise TimeoutError("CPU checkpoint cancellation deadline expired")
                state.cancel_future = None
                time.sleep(min(0.005, remaining))
                continue
            if (
                result.status != "COMPLETED"
                or result.error
                or result.value.get("status") != "CANCELLED"
            ):
                raise RuntimeError("CPU checkpoint cancellation is unresolved")
            return True
