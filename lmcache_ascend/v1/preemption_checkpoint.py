# SPDX-License-Identifier: Apache-2.0
"""Partial checkpoint capture and local publication, owned by one worker.

Only the existing replicated MLA writer creates storage entries. Captured
fragments are private until accepted token IDs arrive from EngineCore.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from lmcache_ascend.v1.local_checkpoint import (
    CheckpointPage,
    LocalCheckpoint,
    LocalCheckpointStore,
)
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
    prefix_sources: tuple[CheckpointPage, ...] = ()
    captured_end: int = 0
    future: Future | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.monotonic)
    quarantined: bool = False
    publish_started: float = 0.0
    publish_finished: float = 0.0


class CheckpointWorker:
    """Capture synchronously at HBM reuse; publish accepted CPU-owned spans locally."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.jobs: dict[tuple[str, int], CaptureJob] = {}
        self.results: list[CheckpointResult] = []
        self.restore_owners: dict[tuple[str, int, int], list[Any]] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-store"
        )
        self.max_jobs = max(1, int(engine.config.store_async_max_queue_size or 2))
        self.timeout = float(engine.config.blocking_timeout_secs)
        self.chunk_size = int(engine.config.chunk_size)
        self.local = LocalCheckpointStore(engine)

    def begin_restore(self, req_id: str, generation: int, load_generation: int) -> None:
        """Keep release controls active until the scheduler acknowledges every TP."""
        self.restore_owners.setdefault((req_id, generation, load_generation), [])

    def hold_restore(
        self, req_id: str, generation: int, load_generation: int, owners: list[Any]
    ) -> None:
        """Transfer normalized-source ownership before starting any consumer."""
        self.restore_owners[req_id, generation, load_generation].extend(owners)

    def release_restore(
        self, req_id: str, generation: int, load_generation: int
    ) -> None:
        """Retire one attempt only after its all-worker receive acknowledgement."""
        for page in self.restore_owners.pop((req_id, generation, load_generation), ()):
            page.ref_count_down()

    def _allocate_fragment(
        self, group: int, tokens: int, caches: Any = None
    ) -> tuple[Any, tuple[int, ...]]:
        if not 0 < tokens <= self.chunk_size:
            raise MemoryError("Checkpoint fragment exceeds one chunk")
        return self.engine.allocate_checkpoint_fragment(group, tokens, caches)

    def capture(
        self,
        spec: CaptureSpec,
        caches: dict[int, list],
        block_size: int,
        prefix_state: Any = None,
    ) -> None:
        """Take a private snapshot before the model runner can reuse source HBM."""
        key = (spec.req_id, spec.generation)
        if key in self.jobs:
            job = self.jobs[key]
            self.results.append(
                CheckpointResult(
                    *key,
                    "failed" if job.cancelled else "captured",
                    job.captured_end,
                    "duplicate generation",
                )
            )
            return
        job = CaptureJob(spec)
        events = []
        device_work_started = False
        try:
            if self.engine.is_frozen():
                raise ValueError("checkpoint capture refused while cache is frozen")
            if len(self.jobs) >= self.max_jobs:
                raise ValueError("checkpoint staging is busy")
            if set(caches) != {0, 1}:
                raise ValueError("checkpoint requires both KV groups")
            if not (
                0 <= spec.base <= spec.prefix_end <= spec.end
                and spec.base <= spec.resident_start <= spec.end
            ):
                raise ValueError("Invalid checkpoint frontiers")
            if prefix_state is not None and prefix_state.cached_keys:
                job.prefix_sources = tuple(
                    CheckpointPage(a, min(b, spec.resident_start), key)
                    for a, b, key in zip(
                        prefix_state.cached_starts,
                        prefix_state.cached_ends,
                        prefix_state.cached_keys[0],
                        strict=True,
                    )
                    if b > spec.prefix_end and a < spec.resident_start
                )
            for group in (0, 1):
                start = spec.resident_start if group == 0 else spec.base
                blocks = spec.blocks[group]
                if (spec.end + block_size - 1) // block_size > len(blocks) or any(
                    blocks[i // block_size] <= 0 for i in range(start, spec.end)
                ):
                    raise ValueError("Checkpoint source includes released/null blocks")
            cursor, reclaimed = spec.base, False
            while cursor < spec.end:
                desired_end = min(
                    spec.end, (cursor // self.chunk_size + 1) * self.chunk_size
                )
                end = desired_end
                while end > cursor:
                    candidate = {}
                    try:
                        for group in (0, 1):
                            start = (
                                max(cursor, spec.resident_start)
                                if group == 0
                                else cursor
                            )
                            if start >= end:
                                continue
                            try:
                                page, widths = self._allocate_fragment(
                                    group, end - start, caches[group]
                                )
                            except MemoryError:
                                if reclaimed:
                                    raise
                                reclaimed = True
                                missing = {
                                    g: caches[g]
                                    for g in (group, 1)
                                    if g not in candidate
                                }
                                if not self.engine.reclaim_checkpoint_capacity(
                                    end - cursor, missing
                                ):
                                    raise
                                page, widths = self._allocate_fragment(
                                    group, end - start, caches[group]
                                )
                            candidate[group] = (start, end, page, widths)
                    except MemoryError:
                        for _, _, page, _ in candidate.values():
                            page.ref_count_down()
                        end = cursor + (end - cursor) // 2
                        continue
                    except BaseException:
                        for _, _, page, _ in candidate.values():
                            page.ref_count_down()
                        raise
                    for group, fragment in candidate.items():
                        job.fragments.setdefault(group, []).append(fragment)
                    cursor = end
                    break
                if end != desired_end:
                    break  # Keep one smaller tail; bound failed-allocation work.
            if cursor <= spec.prefix_end:
                raise MemoryError("Checkpoint CPU staging allocation refused")
            job.captured_end = cursor
            # Admit the entire selected paired prefix before device preparation.
            for group, fragments in job.fragments.items():
                starts = [a for a, _, _, _ in fragments]
                ends = [b for _, b, _, _ in fragments]
                blocks = spec.blocks[group]
                slots = torch.tensor(
                    [
                        blocks[i // block_size] * block_size + i % block_size
                        for a, b in zip(starts, ends, strict=True)
                        for i in range(a, b)
                    ],
                    dtype=torch.long,
                    pin_memory=True,
                )
                device_work_started = True
                plan = self.engine.gpu_connector.prepare_group_capture(
                    [
                        [
                            SimpleNamespace(
                                tensor=page.layer_tensor(layer), metadata=page.metadata
                            )
                            for _, _, page, _ in fragments
                        ]
                        for layer in range(self.engine.num_layers)
                    ],
                    starts,
                    ends,
                    slot_mapping=slots,
                    slot_mapping_base=starts[0],
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
                    job.captured_end,
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
            not max(job.spec.base, job.spec.prefix_end)
            < len(seal.tokens)
            <= job.captured_end
        ):
            self.cancel(*key)
            self.results.append(
                CheckpointResult(*key, "failed", reason="invalid seal frontier")
            )
            return
        job.future = self.executor.submit(self._publish_local, job, seal)

    def configure_capacity(self, window_size: int, chunk_size: int) -> None:
        """Keep the existing setup API; allocation follows actual chunk ranges."""
        if self.jobs:
            raise RuntimeError("Cannot resize checkpoint chunks during capture")
        self.chunk_size = chunk_size

    def _publish_local(self, job: CaptureJob, seal: SealSpec) -> int:
        """Seal into the ordinary local cache; never persist generated KV."""
        job.publish_started = time.monotonic()
        if self.engine.is_frozen():
            raise ValueError("checkpoint publication refused while cache is frozen")
        spec, end = job.spec, len(seal.tokens)
        groups: list[list[CheckpointPage]] = [list(job.prefix_sources), []]
        keys, pages = [], []
        for group, fragments in job.fragments.items():
            for start, stop, page, _ in fragments:
                if start >= end:
                    continue
                key = self.engine.checkpoint_page_key(spec, group, start, stop)
                groups[group].append(CheckpointPage(start, min(stop, end), key))
                keys.append(key)
                pages.append(page)
        if job.cancelled:
            return 0
        # Later intervals enter LRU first, preserving useful leading coverage.
        self.engine.checkpoint_backend().batched_submit_layer_pages(
            keys[::-1], pages[::-1]
        )
        self.local.publish(
            spec.req_id,
            spec.generation,
            LocalCheckpoint(
                seal.tokens,
                spec.prefix_end,
                tuple(tuple(sorted(g, key=lambda p: p.start)) for g in groups),
            ),
        )
        job.publish_finished = time.monotonic()
        return end

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
                                    "local_publish_ms": (
                                        job.publish_finished - job.publish_started
                                    )
                                    * 1000,
                                    "publish_delay_ms": (
                                        time.monotonic() - job.publish_finished
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
                if job.cancelled:
                    self.local.forget(*key)
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
        self.local.forget(req_id, generation)
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
        if self.restore_owners:
            raise RuntimeError(
                "Checkpoint restores still await all-worker acknowledgement"
            )
        for req_id, generation in tuple(self.jobs):
            self.cancel(req_id, generation)
        self.executor.shutdown(wait=True)
        self.poll()

    @staticmethod
    def _release(job: CaptureJob) -> None:
        if job.quarantined:
            return
        for fragments in job.fragments.values():
            for _, _, page, _ in fragments:
                page.ref_count_down()
        job.fragments.clear()
        job.plans.clear()
