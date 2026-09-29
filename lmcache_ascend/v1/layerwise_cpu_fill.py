# SPDX-License-Identifier: Apache-2.0
"""Keep allocator pages alive while RemoteFill reads completed CPU staging.

A tensor reference alone does not prevent LMCache's allocator from recycling
its storage. This lease owns MemoryObj references, not another payload copy.
"""

from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, wait
from threading import Lock
from time import monotonic
from typing import Any


class LayerwiseCPUFillLease:
    """Retain completed CPU pages until a known native terminal outcome.

    The caller must supply the real D2H completion events separately. Creating
    a lease proves ownership only, not completion of the producer's writes.
    """

    def __init__(self, pages: tuple[Any, ...]) -> None:
        self._lock = Lock()
        self._pages: list[Any] = []
        try:
            for page in {id(page): page for page in pages}.values():
                if not page.is_valid() or page.raw_data.device.type != "cpu":
                    raise ValueError("Layerwise RemoteFill needs valid CPU pages")
                page.ref_count_up()
                self._pages.append(page)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Release once, only after all native readers are known to be done."""
        with self._lock:
            pages, self._pages = self._pages, []
        for page in pages:
            page.ref_count_down()


class LayerwisePutQueue:
    """Bound asynchronous CPU-page puts without a barrier every model forward.

    Single producer (the model thread). Backends own the DMA source references.
    Byte/count limits apply to submitted group batches, not model layers.
    A single oversized batch is allowed only in an otherwise empty queue.
    Failures remain sticky: an early LocalCPU hit is never proof of persistence.
    """

    def __init__(self, max_bytes: int, max_batches: int, timeout: float) -> None:
        if max_bytes <= 0 or max_batches <= 0 or timeout < 0:
            raise ValueError(
                "Layerwise put limits must be positive, timeout nonnegative"
            )
        self.max_bytes = max_bytes
        self.max_batches = max_batches
        self.timeout = timeout
        self.pending: deque[tuple[int, tuple[Future, ...]]] = deque()
        self.pending_bytes = 0
        self.error: BaseException | None = None
        self._request_futures: dict[str, set[Future]] = {}
        self._key_futures: dict[Any, set[Future]] = {}

    def fail(self, error: BaseException) -> None:
        """Latch synchronous submission failures as well as future failures."""
        if self.error is None:
            self.error = error

    def poll(self) -> None:
        """Reap completed puts without waiting; propagate any observed error."""
        if self.error is not None:
            raise self.error
        # Reap out-of-order completions too: one slow DMA must not keep fully
        # completed later batches charged against admission capacity.
        try:
            remaining = deque()
            remaining_bytes = 0
            completed = set()
            for size, futures in self.pending:
                done = True
                for future in futures:
                    if future.done():
                        future.result()
                        completed.add(future)
                    else:
                        done = False
                if not done:
                    remaining.append((size, futures))
                    remaining_bytes += size
            self.pending = remaining
            self.pending_bytes = remaining_bytes
            if not completed:
                return
            # Only outstanding transfers need dependency bookkeeping. No page
            # payloads or permanent per-request key history are retained here.
            for dependencies in (self._request_futures, self._key_futures):
                for key, futures in list(dependencies.items()):
                    futures.difference_update(completed)
                    if not futures:
                        del dependencies[key]
        except BaseException as error:
            self.error = error
            raise

    def reserve(self, size: int) -> None:
        """Apply backpressure only at a batch boundary when limits are reached."""
        self.poll()
        deadline = monotonic() + self.timeout
        while self.pending and (
            len(self.pending) >= self.max_batches
            or self.pending_bytes + size > self.max_bytes
        ):
            self._wait_first(deadline)

    def add(
        self,
        size: int,
        futures: list[Future],
        *,
        req_id: str = "",
        keys: Iterable[Any] = (),
    ) -> None:
        """Track a submitted put; its caller must first reserve this capacity."""
        if futures:
            self.pending.append((size, tuple(futures)))
            self.pending_bytes += size
            if req_id:
                self._request_futures.setdefault(req_id, set()).update(futures)
            for key in keys:
                self._key_futures.setdefault(key, set()).update(futures)

    def track_keys(self, req_id: str, keys: Iterable[Any]) -> None:
        """Inherit unfinished puts when reusing locally published CPU pages.

        Called at chunk enumeration, not per layer. Reuses already generated
        cache keys, so it neither rehashes the prompt nor waits for a transfer.
        """
        if not req_id or not self._key_futures:
            return
        for key in keys:
            futures = self._key_futures.get(key)
            if futures:
                self._request_futures.setdefault(req_id, set()).update(futures)

    def drain_requests(self, req_ids: Iterable[str]) -> None:
        """Fence these requests and reused-prefix puts, not unrelated requests."""
        self.poll()
        futures = set()
        for req_id in req_ids:
            futures.update(self._request_futures.get(req_id, ()))
        if futures:
            _, pending = wait(futures, timeout=self.timeout)
            if pending:
                self.error = TimeoutError("Layerwise CPU remote puts did not complete")
                raise self.error
        self.poll()

    def drain(self) -> None:
        """Fence all pending puts before allocator teardown."""
        self.poll()
        deadline = monotonic() + self.timeout
        while self.pending:
            self._wait_first(deadline)
        self.poll()

    def _wait_first(self, deadline: float) -> None:
        _, pending = wait(self.pending[0][1], timeout=max(0.0, deadline - monotonic()))
        if pending:
            self.error = TimeoutError("Layerwise CPU remote puts did not complete")
            raise self.error
        self.poll()
