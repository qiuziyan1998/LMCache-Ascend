# SPDX-License-Identifier: Apache-2.0
"""Bounded waiting-checkpoint ownership and background Mooncake write-back."""

from dataclasses import dataclass, field, replace
from threading import Event, Thread
from typing import Any
import logging
import time

from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache_ascend.v1.local_checkpoint import (
    CheckpointPage,
    LocalCheckpoint,
    LocalCheckpointStore,
    clear_failure_tracebacks,
)

logger = logging.getLogger(__name__)


@dataclass
class Retention:
    manifest: LocalCheckpoint
    groups: list
    remote: tuple = ()
    attempts: dict = field(default_factory=dict)
    cancelled: bool = False
    retry_at: float = 0.0
    pages: dict = field(init=False)

    def __post_init__(self) -> None:
        self.pages = {
            (g, s): p for g, entries in enumerate(self.groups) for s, p in entries
        }


class CheckpointWritebackStore(LocalCheckpointStore):
    """Protect actual CPU owners; never run network work in local publication."""

    def __init__(self, engine: Any, retire_executor: Any) -> None:
        super().__init__(engine)
        self.executor = retire_executor
        self.limit = int(engine.config.decode_checkpoint_retention_max_bytes)
        self.reserve = int(engine.config.decode_checkpoint_free_reserve_bytes)
        self.records: dict[tuple[str, int], Retention] = {}
        self.charges: dict[int, tuple[Any, int]] = {}
        self.used = 0
        self.wake = Event()
        self.thread: Thread | None = None
        self.closed = False
        self.fatal: BaseException | None = None
        self.quarantine: list = []

    def validate_capacity(self) -> None:
        """Validate the real heap at preflight, after StorageManager exists."""
        _, total = self.engine.checkpoint_backend().get_allocator_capacity_bytes()
        if not 0 < self.limit < total or not 0 < self.reserve < total:
            raise ValueError(
                "Checkpoint retention and reserve must be positive "
                "and smaller than the CPU heap"
            )

    def publish(self, req_id: str, generation: int, manifest: LocalCheckpoint) -> None:
        """Protect a complete paired offer before making it lookup-visible."""
        manifest = replace(
            manifest,
            identity=(req_id, generation),
            retention_id=object(),
            request_configs=(
                dict(manifest.request_configs)
                if manifest.request_configs is not None
                else None
            ),
        )
        end, groups = super().acquire(manifest, len(manifest.tokens))
        accepted = False
        try:
            with self.lock:
                if self.closed:
                    return
                if manifest.identity in self.records:
                    return
                self.manifests[manifest.identity] = manifest
                owners = {id(p): p for g in groups for _, p in g}
                added = sum(
                    p.get_physical_size()
                    for i, p in owners.items()
                    if i not in self.charges
                )
                if end == len(manifest.tokens) and self.used + added <= self.limit:
                    self._charge_locked(owners)
                    self.records[manifest.identity] = Retention(manifest, groups)
                    accepted = True
                    if self.thread is None:
                        self.thread = Thread(
                            target=self._run, name="checkpoint-spill", daemon=True
                        )
                        try:
                            self.thread.start()
                        except Exception as error:
                            self.thread = None
                            self.records.pop(manifest.identity)
                            self._uncharge_locked(owners)
                            accepted = False
                            logger.warning(
                                "Checkpoint spill thread unavailable: %s", error
                            )
            if not accepted:
                logger.info(
                    "Checkpoint retention refused: request=%s generation=%s bytes=%s",
                    req_id,
                    generation,
                    added,
                )
        finally:
            if not accepted:
                self.release(groups)
            self.wake.set()

    def forget(self, req_id: str, generation: int | None = None) -> None:
        """Tombstone synchronously; retire idle owners without waiting for I/O."""
        retired = []
        with self.lock:
            for key in tuple(self.manifests):
                if key[0] == req_id and (generation is None or key[1] == generation):
                    self.manifests.pop(key, None)
                    record = self.records.pop(key, None)
                    if record is not None:
                        record.cancelled = True
                        retired.append(record.groups)
                        record.groups = [[], []]
                        record.pages.clear()
        for groups in retired:
            self.executor.submit(self._retire, groups)
        self.wake.set()

    def restore_event(self, identity: tuple, attempt: int, event: str) -> None:
        """Join admitted and all-worker acknowledged events for one attempt."""
        retire = False
        with self.lock:
            record = self.records.get(identity)
            if record is None:
                return
            if event == "begin":
                events = record.attempts.setdefault(attempt, set())
            else:
                events = record.attempts.get(attempt)
                if events is None:
                    return  # A stale/duplicate acknowledgement owns nothing.
            events.add(event)
            retire = {"admitted", "ack"} <= events and "failed" not in events
            if {"failed", "ack"} <= events:
                del record.attempts[attempt]
        if retire:
            self.forget(*identity)
        self.wake.set()

    def get_page(
        self, manifest: LocalCheckpoint, group: int, source: CheckpointPage
    ) -> Any:
        """Resolve retained owners even after an explicit cache-alias removal."""
        with self.lock:
            record = self.records.get(manifest.identity)
            if (
                record is not None
                and record.manifest.retention_id is manifest.retention_id
            ):
                page = record.pages.get((group, source))
                if page is not None:
                    page.ref_count_up()
                    return page
        return super().get_page(manifest, group, source)

    def available(self, req_id: str, generation: int, end: int) -> int:
        manifest = self.manifest(req_id, generation)
        if manifest is None:
            return 0
        common, groups = self._acquire(
            manifest, min(end, len(manifest.tokens)), probe=True
        )
        self.release(groups)
        return common

    def acquire(self, manifest: LocalCheckpoint, end: int) -> tuple[int, list]:
        return self._acquire(manifest, end, probe=False)

    def close(self) -> None:
        """Stop submissions and drain native owners before allocator shutdown."""
        with self.lock:
            self.closed = True
            identities = tuple(self.manifests)
        for identity in identities:
            self.forget(*identity)
        self.wake.set()
        if self.thread is not None:
            self.thread.join()
        if self.fatal is not None:
            raise RuntimeError(
                "Checkpoint write-back has quarantined native owners"
            ) from self.fatal

    def _charge_locked(self, owners: dict[int, Any]) -> None:
        for i, page in owners.items():
            previous = self.charges.get(i)
            if previous is None:
                self.used += page.get_physical_size()
            self.charges[i] = (page, previous[1] + 1 if previous else 1)

    def _retire(self, groups: list) -> int:
        owners = {id(p): p for g in groups for _, p in g}
        self.release(groups)
        with self.lock:
            return self._uncharge_locked(owners)

    def _uncharge_locked(self, owners: dict) -> int:
        released = 0
        for i, page in owners.items():
            _, count = self.charges[i]
            if count == 1:
                self.used -= page.get_physical_size()
                released += page.get_physical_size()
                del self.charges[i]
            else:
                self.charges[i] = (page, count - 1)
        return released

    def _run(self) -> None:
        try:
            self._monitor()
        except BaseException as error:
            self.fatal = error
            clear_failure_tracebacks(error)
            logger.error(
                "Checkpoint pressure monitor failed; restart required: %s", error
            )

    def _monitor(self) -> None:
        credit, previous_free, draining, credit_until = 0, None, False, 0.0
        while True:
            with self.lock:
                if self.closed or self.fatal is not None:
                    return
                active = any(r.groups[0] or r.groups[1] for r in self.records.values())
            self.wake.wait(0.25 if active else None)
            self.wake.clear()
            with self.lock:
                if self.closed:
                    return
            if self.engine.is_frozen():
                continue
            free, _ = self.engine.checkpoint_backend().get_allocator_capacity_bytes()
            credit = (
                0
                if free >= self.reserve or time.monotonic() >= credit_until
                else max(
                    0,
                    credit
                    - max(0, (free if previous_free is None else previous_free) - free),
                )
            )
            previous_free = free
            with self.lock:
                draining = self.used >= self.limit * (0.6 if draining else 0.8)
                if not draining and free + credit >= self.reserve:
                    continue
                record = next(
                    (
                        r
                        for r in self.records.values()
                        if not r.remote
                        and not r.cancelled
                        and not r.attempts
                        and r.retry_at <= time.monotonic()
                    ),
                    None,
                )
                if record is None:
                    continue
                groups = [[(s, p) for s, p in g] for g in record.groups]
                for g in groups:
                    for _, page in g:
                        page.ref_count_up()
                self._charge_locked({id(p): p for g in groups for _, p in g})
            try:
                remote = self._spill(record, groups)
                retired = [[], []]
                with self.lock:
                    if remote and not record.cancelled:
                        record.remote = remote
                        retired, record.groups = record.groups, retired
                        record.pages.clear()
                self._retire(retired)
            except NativeExternalPageTransferUnknownError as error:
                # Tensor Python lifetimes alone do not protect allocator storage.
                self.quarantine.append(groups)
                self.fatal = error
                clear_failure_tracebacks(error)
                logger.error(
                    "Checkpoint native completion unknown; retaining source owners"
                )
                return
            except Exception as error:
                record.retry_at = time.monotonic() + 1.0
                logger.warning("Checkpoint spill deferred: %s", error)
                clear_failure_tracebacks(error)
            finally:
                if self.fatal is None:
                    released = self._retire(groups)
                    if released:
                        credit += released
                        credit_until = time.monotonic() + 1.0
            record = None  # Do not retain a finished request's history while idle.

    def _spill(self, record: Retention, groups: list) -> tuple:
        """One canonical page per native PUT bounds workspace and lock occupancy."""
        manifest = record.manifest
        remote = [[], []]
        chunk = int(self.engine.config.chunk_size)
        base = manifest.prefix_end // chunk * chunk
        reserve = self._allocation_retry()
        tokens = list(manifest.tokens)  # TokenDatabase accepts lists, not tuples.
        chunk_plan = {}

        for group in (0, 1):
            covered = base
            select_sources = self._source_selector(groups[group])
            for start, end, key in self.engine.checkpoint_token_results(
                tokens, manifest.request_configs, group, chunk_plan
            ):
                if start < base:
                    continue
                if (
                    record.cancelled
                    or record.attempts
                    or self.closed
                    or self.engine.is_frozen()
                ):
                    return ()
                if start != covered or not start < end <= len(manifest.tokens):
                    raise ValueError("Checkpoint persistence has a range gap")
                page = self.build_page(
                    manifest,
                    select_sources(start, end),
                    group,
                    start,
                    end,
                    key,
                    manifest.request_configs,
                    reserve,
                )
                try:
                    if (
                        record.cancelled
                        or record.attempts
                        or self.closed
                        or self.engine.is_frozen()
                    ):
                        return ()  # Boundary preparation may have waited for I/O.
                    self.engine.persist_checkpoint_page(
                        key, page, manifest.identity[0]
                    ).result()
                except NativeExternalPageTransferUnknownError:
                    self.quarantine.append(page)
                    raise
                finally:
                    if not any(page is obj for obj in self.quarantine):
                        page.ref_count_down()
                remote[group].append(CheckpointPage(start, end, key))
                covered = end
            if covered != len(manifest.tokens):
                raise ValueError("Checkpoint persistence omits the accepted tail")
        return tuple(tuple(g) for g in remote)

    def _acquire(
        self, manifest: LocalCheckpoint, end: int, *, probe: bool
    ) -> tuple[int, list]:
        with self.lock:
            record = self.records.get(manifest.identity)
            if (
                record is not None
                and record.manifest.retention_id is not manifest.retention_id
            ):
                record = None
            remote = record.remote if record else ()
            if record and record.groups[0] and record.groups[1]:
                # A failed normalization can have replaced/removed cache aliases.
                # Retained raw sources remain authoritative for this generation.
                manifest = record.manifest
        if self.engine.is_frozen():
            return super().acquire(manifest, end)
        if not remote:
            common, held = super().acquire(manifest, end)
            if common == end:
                return common, held
            with self.lock:
                # Spill confirmation can release protection between the source
                # snapshot and acquisition. Recheck that one-way transition.
                remote = (
                    record.remote
                    if record is not None
                    and self.records.get(manifest.identity) is record
                    else ()
                )
            if not remote:
                return common, held
            self.release(held)
        groups, plans, missing = [[], []], [[], []], []
        try:
            # Decide by canonical window. Never splice overlapping raw and remote
            # pages without retaining each page's original token/plane origin.
            for group in (0, 1):
                sources, index = manifest.groups[group], 0
                for descriptor in remote[group]:
                    left, right = (
                        max(manifest.prefix_end, descriptor.start),
                        min(end, descriptor.end),
                    )
                    if left >= right:
                        continue
                    held, cursor = [], left
                    plans[group].append((descriptor, held, right))
                    while index < len(sources):
                        source = sources[index]
                        if source.end <= cursor:
                            index += 1
                            continue
                        if source.start > cursor or cursor >= right:
                            break
                        page = self.get_page(manifest, group, source)
                        if page is None:
                            break
                        held.append((source, page))
                        cursor = min(right, source.end)
                    if cursor != right:
                        self.release([held])
                        missing.append(descriptor.key)
            exists = (
                dict(
                    zip(
                        missing,
                        self.engine.probe_checkpoint_pages(missing),
                        strict=True,
                    )
                )
                if missing
                else {}
            )
            ends = []
            for group in (0, 1):
                cursor = manifest.prefix_end
                for descriptor, held, right in plans[group]:
                    if held:
                        groups[group].extend(held)
                        held.clear()
                    else:
                        if not exists.get(descriptor.key, False):
                            break
                        if not probe:
                            try:
                                page = self.engine.load_checkpoint_page(
                                    descriptor.key,
                                    group,
                                    descriptor.end - descriptor.start,
                                    manifest.identity[0],
                                )
                            except MemoryError:
                                # No device work; retain the shorter paired proof.
                                break
                            if page is None:
                                break
                            groups[group].append((descriptor, page))
                    cursor = right
                ends.append(cursor)
            common = min(ends)
            for held in groups:
                while held and held[-1][0].start >= common:
                    held.pop()[1].ref_count_down()
            return common, groups
        except BaseException:
            self.release(groups)
            raise
        finally:
            for plan in plans:
                for _, held, _ in plan:
                    self.release([held])
