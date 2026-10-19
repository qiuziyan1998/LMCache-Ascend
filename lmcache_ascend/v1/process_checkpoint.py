# SPDX-License-Identifier: Apache-2.0
"""Worker checkpoint capture using M-owned pages and explicit participant leases."""

# Standard
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
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
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.rpc.storage_protocol import SourceGeometry, unpack_uints
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle, SharedHandleBatch
from lmcache_ascend.v1.preemption_checkpoint import CaptureJob, CheckpointWorker


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
    reconcile_queued: bool = False
    released: bool = False
    acknowledging: bool = False
    retired: set[str] = field(default_factory=set)


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
            super().capture(spec, caches, block_size, prefix_state)
            return
        with self._lock:
            reconciling = key in self.reservations
        if reconciling:
            self.results.append(
                CheckpointResult(
                    *key, "failed", reason="checkpoint generation is still reconciling"
                )
            )
            return
        source = getattr(prefix_state, "prepared_sparse_sources", {}).get(0)
        try:
            if source is not None:
                self._sources[key] = (
                    self.engine.compact_source_importer.source_descriptor(source)
                )
            super().capture(spec, caches, block_size, prefix_state)
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

    def release_restore(
        self, req_id: str, generation: int, load_generation: int
    ) -> None:
        """Retire normalization ownership at the existing all-TP receive ACK."""
        self.engine.compact_source_importer.release_restore(req_id, load_generation)
        for owner in self.restore_owners.pop((req_id, generation, load_generation), ()):
            if isinstance(owner, LocalLeaseHandle):
                owner.release("fenced")
            else:
                owner.ref_count_down()

    def submit_control(
        self, method: str, payload: dict, req_id: str, generation: int, attempt: str
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

        def completed(done: Any) -> None:
            try:
                result = done.result()
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
        super().close()
        deadline = time.monotonic() + self.timeout
        while not self.sources_drained() and time.monotonic() < deadline:
            self.poll()
            time.sleep(0.005)
        if not self.sources_drained():
            raise RuntimeError("Process checkpoint ownership has not drained")

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
            widths = tuple(
                tuple(value) if isinstance(value, list) else value
                for value in group["plane_widths"]
            )
            for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                if start != cursor or not start < end <= reply["captured_end"]:
                    raise ValueError(
                        "Checkpoint reservation coverage has a hole or overlap"
                    )
                geometry = geometries[indices[index]]
                geometry.validate(layers, extents[index])
                if (
                    geometry.token_capacity != end - start
                    or offsets[index] + extents[index] > mapping.size
                    or geometry.layer_starts
                    != tuple(
                        sum(geometry.layer_sizes[:layer]) for layer in range(layers)
                    )
                    or len(set(geometry.layer_dtypes)) != 1
                ):
                    raise ValueError(
                        "Checkpoint reservation physical geometry disagrees "
                        "with its page"
                    )
                shapes = [torch.Size(shape) for shape in geometry.layer_shapes]
                dtype = getattr(torch, geometry.layer_dtypes[0].removeprefix("torch."))
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
                job.fragments.setdefault(group_id, []).append(
                    (start, end, page, widths)
                )
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
        state.fenced = True
        self._call(
            job,
            "commit_write",
            {
                "reservation_id": state.reply["reservation_id"],
                "phase": "captured",
                "proof": "fenced",
            },
        )

    def _publish_local(self, job: CaptureJob, seal: SealSpec) -> int:
        job.publish_started = time.monotonic()
        if job.cancelled:
            return 0
        state = self.reservations[job.spec.req_id, job.spec.generation]
        self._call(
            job,
            "commit_write",
            {
                "reservation_id": state.reply["reservation_id"],
                "phase": "seal",
                "proof": "fenced",
                "tokens": seal.tokens,
            },
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
            self.jobs[key] = job
            return
        if state.released:
            self._ack_released(state.future.identity)
            return
        if state.reply is None:
            try:
                result = state.future.result() if state.future.done() else None
            except BaseException:
                job.quarantined = True
                self.jobs[key] = job
                return
            if result is not None and result.status == "COMPLETED":
                state.reply = result.value
                if result.value.get("status") == "SKIPPED":
                    self._ack_released(state.future.identity, no_leases=True)
                    return
                self._import_fragments(job, state)
            elif result is not None and result.status in (
                "FAILED_SAFE",
                "CANCELLED_NOT_STARTED",
            ):
                self._ack_released(state.future.identity, no_leases=True)
                return
            elif result is not None and result.status == "FAILED_UNKNOWN":
                job.quarantined = True
                self.jobs[key] = job
                return
            else:
                self.client.cancel(state.future.identity)
                self.client.replay(state.future.identity)
                if not state.reconcile_queued:
                    state.reconcile_queued = True

                    def resolved(_: Any) -> None:
                        # Cancellation never starts a new operation or imports
                        # device operands. The model poll retries after a late
                        # response, including after executor shutdown.
                        job.cancelled = True

                    state.future.add_done_callback(resolved)
                return
        if not state.sealed:
            try:
                result = self._call(
                    job,
                    "cancel_write",
                    {
                        "reservation_id": state.reply["reservation_id"],
                        "proof": "fenced" if state.imported else "never_imported",
                    },
                )
            except Exception:
                job.cancelled = True
                self.jobs[key] = job
                return
            if result.get("status") == "PENDING":
                job.cancelled = True
                self.jobs[key] = job
                return
        state.releasing = True
        CheckpointWorker._release(job)
        for hold in state.holds:
            hold.release("fenced" if state.imported else "never_imported")
        state.released = True
        self._ack_released(state.future.identity)
