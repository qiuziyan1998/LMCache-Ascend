# SPDX-License-Identifier: Apache-2.0
"""Bounded checkpoint capture and background persistence, owned by one worker.

Only the existing replicated MLA writer creates storage entries. Captured
fragments are private until accepted token IDs arrive from EngineCore.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace
from typing import Any
import time

from lmcache.integration.vllm.preemption_checkpoint import (
    CaptureSpec,
    CheckpointResult,
    SealSpec,
)
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
import torch


def clear_failure_tracebacks(error: BaseException) -> None:
    """Drop terminal I/O frames without relying on cyclic garbage collection."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(
            e for e in (current.__cause__, current.__context__) if e is not None
        )
        pending.extend(getattr(current, "exceptions", ()))
        current.__traceback__ = None


@dataclass
class CaptureJob:
    spec: CaptureSpec
    # group -> (start, end, page, per-plane widths)
    fragments: dict[int, list[tuple[int, int, Any, tuple[int, ...]]]] = field(
        default_factory=dict
    )
    plans: list[Any] = field(default_factory=list)
    future: Future | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.monotonic)
    quarantined: bool = False
    persist_started: float = 0.0
    persist_finished: float = 0.0


class CaptureBufferLease:
    """Logical fragment over a reusable, allocator-accounted registered slab."""

    def __init__(
        self, page: Any, tokens: int, widths: tuple[int, ...], release: Any
    ) -> None:
        self.page, self.tokens, self.widths, self.release = (
            page,
            tokens,
            widths,
            release,
        )
        self.raw_data, self.metadata = page.raw_data, page.metadata

    def get_dtype(self) -> Any:
        return self.page.get_dtype()

    def layer_data_ptr(self, layer: int) -> int:
        return self.page.layer_data_ptr(layer)

    def layer_tensor(self, layer: int) -> Any:
        return self.page.layer_tensor(layer)[: self.tokens * sum(self.widths)]

    def ref_count_down(self) -> None:
        if self.release is None:
            raise RuntimeError("Checkpoint buffer lease released twice")
        release, self.release = self.release, None
        release(self.page)


def fragment_vectors(
    fragments: list, start: int, end: int, layers: int
) -> tuple[list[int], list[int]]:
    """Describe a logical page as layer/plane/token-ordered owned CPU spans."""
    selected = [
        (max(start, a), min(end, b), a, b, page, widths)
        for a, b, page, widths in fragments
        if a < end and b > start
    ]
    selected.sort(key=lambda item: item[0])
    cursor = start
    for a, b, *_ in selected:
        if a != cursor:
            raise ValueError("Checkpoint fragment coverage has a hole or overlap")
        cursor = b
    if cursor != end or not selected:
        raise ValueError("Checkpoint fragment coverage is incomplete")
    widths = selected[0][-1]
    if any(item[-1] != widths for item in selected):
        raise ValueError("Checkpoint CPU fragments have incompatible layouts")
    pointers, sizes = [], []
    for layer in range(layers):
        preceding_width = 0
        for width in widths:
            for a, b, origin, limit, page, _ in selected:
                item_size = page.get_dtype().itemsize
                pointers.append(
                    page.layer_data_ptr(layer)
                    + ((limit - origin) * preceding_width + (a - origin) * width)
                    * item_size
                )
                sizes.append((b - a) * width * item_size)
            preceding_width += width
    return pointers, sizes


class CheckpointWorker:
    """Capture synchronously at HBM reuse; persist CPU-owned spans in background."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.jobs: dict[tuple[str, int], CaptureJob] = {}
        self.results: list[CheckpointResult] = []
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-store"
        )
        self.max_jobs = max(1, int(engine.config.store_async_max_queue_size or 2))
        self.timeout = float(engine.config.blocking_timeout_secs)
        chunk = int(engine.config.chunk_size)
        self.capacity_tokens = (
            max(
                chunk,
                int(
                    engine.config.get_extra_config_value(
                        "decode_window_save_window_size", chunk
                    )
                    or chunk
                ),
            )
            + chunk
        )
        self.buffer_lock = Lock()
        self.free_buffers: dict[int, list] = {0: [], 1: []}
        self.buffer_counts = {0: 0, 1: 0}
        self.buffer_widths: dict[int, tuple[int, ...]] = {}

    def _allocate_fragment(
        self, group: int, tokens: int, caches: Any = None
    ) -> tuple[Any, tuple[int, ...]]:
        if not 0 < tokens <= self.capacity_tokens:
            raise MemoryError("Checkpoint fragment exceeds bounded staging capacity")
        with self.buffer_lock:
            if self.free_buffers[group]:
                page = self.free_buffers[group].pop()
                widths = self.buffer_widths[group]
            else:
                # Group 0 can need a separate old partial-page prefix; Group 1
                # is entirely resident. Pool growth never evicts the CPU cache.
                limit = self.max_jobs * (2 if group == 0 else 1)
                if self.buffer_counts[group] >= limit:
                    raise MemoryError("Checkpoint capture buffers are busy")
                page, widths = self.engine.allocate_checkpoint_fragment(
                    group, self.capacity_tokens, caches
                )
                self.buffer_counts[group] += 1
                self.buffer_widths[group] = widths

        def release(page: Any) -> None:
            with self.buffer_lock:
                self.free_buffers[group].append(page)

        return CaptureBufferLease(page, tokens, widths, release), widths

    def capture(
        self, spec: CaptureSpec, caches: dict[int, list], block_size: int
    ) -> None:
        """Take a private snapshot before the model runner can reuse source HBM."""
        key = (spec.req_id, spec.generation)
        if key in self.jobs:
            job = self.jobs[key]
            self.results.append(
                CheckpointResult(
                    *key,
                    "failed" if job.cancelled else "captured",
                    spec.end,
                    "duplicate generation",
                )
            )
            return
        job = CaptureJob(spec)
        events = []
        cpu_plans = []
        device_work_started = False
        try:
            if self.engine.is_frozen():
                raise ValueError("checkpoint capture refused while cache is frozen")
            if len(self.jobs) >= self.max_jobs:
                raise ValueError("checkpoint staging is busy")
            if set(caches) != {0, 1}:
                raise ValueError("checkpoint requires both KV groups")
            for group in (0, 1):
                start = spec.resident_start if group == 0 else spec.base
                if not spec.base <= start < spec.end:
                    raise ValueError("checkpoint has no resident suffix")
                page, widths = self._allocate_fragment(
                    group, spec.end - start, caches[group]
                )
                job.fragments[group] = [(start, spec.end, page, widths)]
                blocks = spec.blocks[group]
                if (spec.end + block_size - 1) // block_size > len(blocks):
                    raise ValueError("checkpoint exceeds original block table")
                slots = torch.tensor(
                    [
                        blocks[i // block_size] * block_size + i % block_size
                        for i in range(start, spec.end)
                    ],
                    dtype=torch.long,
                    pin_memory=True,
                )
                if any(blocks[i // block_size] <= 0 for i in range(start, spec.end)):
                    raise ValueError("checkpoint source includes released/null blocks")
                cpu_plans.append((group, start, page, slots))
            # Admit both groups before uploading any metadata. A second-group
            # OOM must not cause a device wait for an unusable first-group plan.
            for group, start, page, slots in cpu_plans:
                device_work_started = True
                plan = self.engine.gpu_connector.prepare_group_capture(
                    [
                        [
                            SimpleNamespace(
                                tensor=page.layer_tensor(layer), metadata=page.metadata
                            )
                        ]
                        for layer in range(self.engine.num_layers)
                    ],
                    [start],
                    [spec.end],
                    slot_mapping=slots,
                    slot_mapping_base=start,
                    kv_group=group,
                    kvcaches=caches[group],
                )
                job.plans.append(plan)
            prepared_at = time.monotonic()
            for plan in job.plans:
                events.append(self.engine.gpu_connector.enqueue_group_capture(plan))
            enqueued_at = time.monotonic()
            # Both groups use one store stream. The last event covers the whole
            # capture. No layer/group payload is read by the host before this.
            events[-1].synchronize()
            captured_at = time.monotonic()
            job.plans.clear()
            self.jobs[key] = job
            self.results.append(
                CheckpointResult(
                    *key,
                    "captured",
                    spec.end,
                    timings_ms={
                        "prepare_ms": (prepared_at - job.started) * 1000,
                        "enqueue_ms": (enqueued_at - prepared_at) * 1000,
                        "capture_wait_ms": (captured_at - enqueued_at) * 1000,
                    },
                )
            )
        except Exception as error:
            try:
                # Also fences a native command that threw before recording its
                # event, and uploads issued by a partially prepared plan.
                if device_work_started:
                    self.engine.gpu_connector.finish_checkpoint_capture()
            except Exception:
                job.quarantined = True
                self.jobs[key] = job
                raise RuntimeError(
                    "Checkpoint capture fence failed; source owners retained"
                ) from error
            self._release(job)
            self.results.append(CheckpointResult(*key, "failed", reason=str(error)))
            clear_failure_tracebacks(error)

    def seal(self, seal: SealSpec) -> None:
        """Persist only accepted/computed positions of a captured generation."""
        key = (seal.req_id, seal.generation)
        job = self.jobs.get(key)
        if job is None or job.cancelled or job.future is not None:
            return
        if (
            not max(job.spec.base, job.spec.resident_start)
            < len(seal.tokens)
            <= job.spec.end
        ):
            self.cancel(*key)
            self.results.append(
                CheckpointResult(*key, "failed", reason="invalid seal frontier")
            )
            return
        job.future = self.executor.submit(self._persist, job, seal)

    def configure_capacity(self, window_size: int, chunk_size: int) -> None:
        """Use the adapter's resolved window setting, including its env override."""
        if self.jobs or any(self.buffer_counts.values()):
            raise RuntimeError("Cannot resize checkpoint buffers after capture starts")
        self.capacity_tokens = max(window_size, chunk_size) + chunk_size

    def _persist(self, job: CaptureJob, seal: SealSpec) -> int:
        job.persist_started = time.monotonic()
        spec = job.spec
        storage = self.engine.storage_manager
        database = self.engine.token_database
        try:
            if self.engine.is_frozen():
                raise ValueError("checkpoint persistence refused while cache is frozen")
            # Recover the nonresident portion of the first extended chunk only
            # after releasing HBM. Use the old exact partial key, not the new key.
            if spec.resident_start > spec.base:
                for start, end, key in database.process_tokens(
                    tokens=list(seal.tokens[: spec.resident_start]),
                    request_configs=spec.request_configs,
                    kv_group=0,
                ):
                    if start < spec.base:
                        continue
                    get_prefix = getattr(self.engine, "get_checkpoint_prefix", None)
                    cached = (
                        get_prefix(key, 0, end - start)
                        if get_prefix is not None
                        else None
                    )
                    page, widths = cached or self._allocate_fragment(0, end - start)
                    fragment = (start, end, page, widths)
                    job.fragments[0].append(fragment)
                    ptrs, sizes = fragment_vectors(
                        [fragment], start, end, self.engine.num_layers
                    )
                    if cached is None:
                        storage.batched_get_external_pages(
                            [key], [ptrs], [sizes], (page.raw_data,), spec.req_id
                        )
            keys, pointers, sizes = [], [], []
            for group in (0, 1):
                covered_end = spec.base
                for start, end, key in database.process_tokens(
                    tokens=list(seal.tokens),
                    request_configs=spec.request_configs,
                    kv_group=group,
                ):
                    if start < spec.base:
                        continue
                    if start != covered_end:
                        raise ValueError("Checkpoint storage keys have a coverage gap")
                    ptrs, lengths = fragment_vectors(
                        job.fragments[group], start, end, self.engine.num_layers
                    )
                    keys.append(key)
                    pointers.append(ptrs)
                    sizes.append(lengths)
                    covered_end = end
                if covered_end != len(seal.tokens):
                    raise ValueError("Checkpoint storage keys omit the accepted partial tail")
            owners = tuple(
                page.raw_data
                for fragments in job.fragments.values()
                for _, _, page, _ in fragments
            )
            storage.batched_put_external_pages(
                keys, pointers, sizes, owners, None, spec.req_id
            ).result()
            job.persist_finished = time.monotonic()
            return len(seal.tokens)
        except NativeExternalPageTransferUnknownError:
            # Unknown DMA completion cannot retire registered sources safely.
            job.quarantined = True
            raise

    def poll(self) -> tuple[CheckpointResult, ...]:
        """Advance control-only requests and retire completed buffer owners."""
        if not self.jobs and not self.results:
            return ()
        for key, job in tuple(self.jobs.items()):
            if job.quarantined:
                raise RuntimeError("Checkpoint has an unresolved native transfer")
            if job.future is not None and job.future.done():
                try:
                    end = job.future.result()
                    if not job.cancelled:
                        self.results.append(
                            CheckpointResult(
                                *key,
                                "ready",
                                end,
                                timings_ms={
                                    "persist_ms": (
                                        job.persist_finished - job.persist_started
                                    )
                                    * 1000,
                                    "publish_delay_ms": (
                                        time.monotonic() - job.persist_finished
                                    )
                                    * 1000,
                                },
                            )
                        )
                except NativeExternalPageTransferUnknownError:
                    raise RuntimeError(
                        "Checkpoint DMA completion is unknown; owners quarantined"
                    )
                except Exception as error:
                    if not job.cancelled:
                        self.results.append(
                            CheckpointResult(*key, "failed", reason=str(error))
                        )
                    clear_failure_tracebacks(error)
                job.future = None
                self._release(job)
                del self.jobs[key]
            elif time.monotonic() - job.started > self.timeout and not job.cancelled:
                job.cancelled = True
                self.results.append(
                    CheckpointResult(*key, "failed", reason="checkpoint deadline")
                )
                if job.future is None:
                    self._release(job)
                    del self.jobs[key]
        results, self.results = tuple(self.results), []
        return results

    def cancel(self, req_id: str, generation: int | None = None) -> None:
        """Stop publication, retaining any buffers still owned by native I/O."""
        for key, job in tuple(self.jobs.items()):
            if key[0] != req_id or (generation is not None and key[1] != generation):
                continue
            job.cancelled = True
            if job.quarantined:
                # Keep the quarantine visible to close()/poll(). Removing this
                # entry would allow engine teardown with unresolved native I/O.
                continue
            if job.future is None:
                self._release(job)
                del self.jobs[key]

    def close(self) -> None:
        """Drain background readers before retiring registered buffers."""
        for req_id, generation in tuple(self.jobs):
            self.cancel(req_id, generation)
        self.executor.shutdown(wait=True)
        self.poll()
        with self.buffer_lock:
            for pages in self.free_buffers.values():
                for page in pages:
                    page.ref_count_down()
                pages.clear()

    @staticmethod
    def _release(job: CaptureJob) -> None:
        if job.quarantined:
            return
        for fragments in job.fragments.values():
            for _, _, page, _ in fragments:
                page.ref_count_down()
        job.fragments.clear()
        job.plans.clear()
