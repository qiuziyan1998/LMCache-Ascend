# SPDX-License-Identifier: Apache-2.0
"""Evictable local checkpoint offers; only an acquired restore owns pages."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from lmcache.integration.vllm.preemption_checkpoint import CheckpointRestoreMiss


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


@dataclass(frozen=True)
class CheckpointPage:
    start: int
    end: int
    key: Any


@dataclass(frozen=True)
class LocalCheckpoint:
    tokens: tuple[int, ...]
    prefix_end: int
    groups: tuple[tuple[CheckpointPage, ...], tuple[CheckpointPage, ...]]
    request_configs: Any = None
    identity: tuple[str, int] | None = None
    retention_id: object | None = None


class LocalCheckpointStore:
    """Keep manifests without pinning their allocator-owned LocalCPU pages."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.lock = Lock()
        self.manifests: dict[tuple[str, int], LocalCheckpoint] = {}

    def publish(self, req_id: str, generation: int, manifest: LocalCheckpoint) -> None:
        """Publish an advisory frontier after both groups finish capture/sealing."""
        with self.lock:
            self.manifests[req_id, generation] = manifest

    def forget(self, req_id: str, generation: int | None = None) -> None:
        """Discard offers, leaving ordinary LRU responsible for unowned pages."""
        with self.lock:
            for key in tuple(self.manifests):
                if key[0] == req_id and (generation is None or key[1] == generation):
                    del self.manifests[key]

    def manifest(self, req_id: str, generation: int) -> LocalCheckpoint | None:
        with self.lock:
            return self.manifests.get((req_id, generation))

    def acquire(
        self, manifest: LocalCheckpoint, end: int
    ) -> tuple[int, list[list[tuple[CheckpointPage, Any]]]]:
        """Retain the paired contiguous prefix; caller must release both lists.

        This is a short ownership lease, never a reservation while waiting for
        HBM. Missing/evicted pages shorten coverage instead of causing a spin.
        """
        groups: list[list[tuple[CheckpointPage, Any]]] = [[], []]
        ends = []
        try:
            for group, sources in enumerate(manifest.groups):
                cursor = manifest.prefix_end
                for source in sources:
                    if source.end <= cursor:
                        continue
                    if source.start > cursor or cursor >= end:
                        break
                    page = self.get_page(manifest, group, source)
                    if page is None:
                        break
                    groups[group].append((source, page))
                    cursor = min(source.end, end)
                ends.append(cursor)
            common = min(ends)
            for held in groups:
                while held and held[-1][0].start >= common:
                    held.pop()[1].ref_count_down()
            return common, groups
        except BaseException:
            self.release(groups)
            raise

    def get_page(
        self, manifest: LocalCheckpoint, group: int, source: CheckpointPage
    ) -> Any:
        """Acquire and validate an actual cache owner, or report a local miss."""
        pages, count = self.engine.checkpoint_backend().batched_get_layer_page_prefix(
            [source.key]
        )
        if not count:
            return None
        page = pages[0]
        try:
            if not page.is_valid() or page.valid_tokens < source.end - source.start:
                raise ValueError("Invalid local checkpoint page")
            self.engine.validate_checkpoint_page(group, page)
            return page
        except BaseException:
            page.ref_count_down()
            raise

    def build_page(
        self,
        manifest: LocalCheckpoint,
        sources: list,
        group: int,
        start: int,
        end: int,
        key: Any,
        configs: Any,
        reserve: Any,
    ) -> Any:
        """Return one owned canonical page without changing aliases or manifests."""
        selected = [(s, p) for s, p in sources if s.start < end and s.end > start]
        if (
            len(selected) == 1
            and selected[0][0].start == start
            and selected[0][0].end == end
            and selected[0][1].valid_tokens == end - start
        ):
            page = selected[0][1]
            page.ref_count_up()
            return page
        cached = self.engine.get_checkpoint_prefix(key, group, end - start)
        if cached is not None:
            return cached[0]
        prefix = page = None
        try:
            base = (
                manifest.prefix_end
                // int(self.engine.config.chunk_size)
                * int(self.engine.config.chunk_size)
            )
            if start == base and selected and selected[0][0].start > base:
                prefix = reserve(
                    lambda: self.engine.load_checkpoint_prefix(
                        manifest.tokens[: manifest.prefix_end], group, configs
                    ),
                    group,
                    manifest.prefix_end - base,
                )
                selected.insert(
                    0,
                    (
                        CheckpointPage(
                            base, min(manifest.prefix_end, selected[0][0].start), None
                        ),
                        prefix,
                    ),
                )
            page, widths = reserve(
                lambda: self.engine.allocate_checkpoint_fragment(group, end - start),
                group,
                end - start,
            )
            self._assemble(page, widths, selected, start, end)
            return page
        except BaseException:
            if page is not None:
                page.ref_count_down()
            raise
        finally:
            if prefix is not None:
                prefix.ref_count_down()

    @staticmethod
    def release(groups: list[list[tuple[CheckpointPage, Any]]]) -> None:
        for held in groups:
            for _, page in held:
                page.ref_count_down()
            held.clear()

    def available(self, req_id: str, generation: int, end: int) -> int:
        """Probe without retaining pages across scheduling/HBM admission."""
        manifest = self.manifest(req_id, generation)
        if manifest is None:
            return 0
        common, groups = self.acquire(manifest, min(end, len(manifest.tokens)))
        self.release(groups)
        return common

    def normalize(
        self, req_id: str, generation: int, tokens: list[int], request_configs: Any
    ) -> tuple[int, list[Any]]:
        """Acquire and normalize the local tail for the ordinary shared loader.

        Only boundary/cropped pages need CPU assembly. Full captured pages are
        admitted by reference. No generated KV is sent to persistent storage.
        """
        manifest = self.manifest(req_id, generation)
        if manifest is None:
            raise CheckpointRestoreMiss(0, "Local checkpoint offer is unavailable")
        if tuple(tokens) != manifest.tokens[: len(tokens)]:
            raise ValueError("Local checkpoint generation/history is unavailable")
        common, groups = self.acquire(manifest, len(tokens))
        owners: list[Any] = []
        reserve = self._allocation_retry()
        covered = manifest.prefix_end

        try:
            if common != len(tokens):
                raise CheckpointRestoreMiss(
                    common, "Local checkpoint was evicted before restore"
                )
            chunk = int(self.engine.config.chunk_size)
            base = manifest.prefix_end // chunk * chunk
            chunk_plan = {}
            for group in (0, 1):
                select_sources = self._source_selector(groups[group])
                keys, pages, normalized = [], [], []
                covered = base
                for start, end, key in self.engine.checkpoint_token_results(
                    tokens, request_configs, group, chunk_plan
                ):
                    if start < base:
                        continue
                    if start != covered or not start < end <= len(tokens):
                        raise ValueError(
                            "Normalized checkpoint coverage has a gap or overlap"
                        )
                    page = self.build_page(
                        manifest,
                        select_sources(start, end),
                        group,
                        start,
                        end,
                        key,
                        request_configs,
                        reserve,
                    )
                    owners.append(page)
                    keys.append(key)
                    pages.append(page)
                    normalized.append(CheckpointPage(start, end, keys[-1]))
                    covered = end
                if covered != len(tokens):
                    raise ValueError(
                        "Normalized checkpoint coverage omits the partial tail"
                    )
                self.engine.checkpoint_backend().batched_submit_layer_pages(keys, pages)
                # Move checkpoint-owned cache references instead of creating
                # two cache aliases that would permanently make refs > 1.
                updated = list(manifest.groups)
                updated[group] = tuple(normalized)
                with self.lock:
                    previous = manifest
                    manifest = replace(manifest, groups=tuple(updated))
                    if self.manifests.get((req_id, generation)) is previous:
                        self.manifests[req_id, generation] = manifest
                for source, _ in groups[group]:
                    if self.engine.is_checkpoint_page_key(source.key):
                        self.engine.checkpoint_backend().remove(source.key)
                # The offer remains advisory. Retain the actual canonical
                # winners before device work, even when admission kept a
                # compatible existing page instead of our duplicate.
                installed, count = (
                    self.engine.checkpoint_backend().batched_get_layer_page_prefix(keys)
                )
                owners.extend(installed)
                if count != len(keys):
                    raise CheckpointRestoreMiss(
                        normalized[count].start,
                        "Normalized checkpoint was evicted during admission",
                    )
                for page in installed:
                    self.engine.validate_checkpoint_page(group, page)
            return base, owners
        except BaseException as error:
            for page in owners:
                page.ref_count_down()
            if isinstance(error, MemoryError):
                # No device restore has started. Retry only before the failed
                # boundary; the scheduler re-proves both groups on that retry.
                raise CheckpointRestoreMiss(
                    min(covered, common, len(tokens) - 1),
                    "Local checkpoint boundary workspace is unavailable",
                ) from error
            raise
        finally:
            self.release(groups)

    @staticmethod
    def _source_selector(sources: list) -> Callable[[int, int], list]:
        """Select overlaps for increasing windows from retained ordered sources."""
        index = 0

        def select(start: int, end: int) -> list:
            nonlocal index
            while index < len(sources) and sources[index][0].end <= start:
                index += 1
            stop = index
            while stop < len(sources) and sources[stop][0].start < end:
                stop += 1
            return sources[index:stop]

        return select

    def _allocation_retry(self) -> Callable[[Callable[[], Any], int, int], Any]:
        """Give one operation a private, single-reclaim allocation budget."""
        reclaimed = False

        def reserve(factory: Callable[[], Any], group: int, count: int) -> Any:
            nonlocal reclaimed
            try:
                return factory()
            except MemoryError:
                if reclaimed:
                    raise
                reclaimed = True
                if not self.engine.reclaim_checkpoint_capacity(count, {group: None}):
                    raise
                return factory()

        return reserve

    @staticmethod
    def _assemble(
        destination: Any, widths: tuple[int, ...], sources: list, start: int, end: int
    ) -> None:
        """Copy CPU spans in layer/plane/token order with explicit coverage."""
        cursor = start
        for source, _ in sources:
            left, right = max(start, source.start), min(end, source.end)
            if left != cursor:
                raise ValueError("Local checkpoint boundary has a hole or overlap")
            cursor = right
        if cursor != end:
            raise ValueError("Local checkpoint boundary is incomplete")
        for layer in range(destination.num_layers):
            target = destination.layer_tensor(layer).reshape(-1)
            preceding = 0
            for width in (
                widths[layer] if widths and isinstance(widths[0], tuple) else widths
            ):
                for source, page in sources:
                    left, right = max(start, source.start), min(end, source.end)
                    src = page.layer_tensor(layer).reshape(-1)
                    a = page.valid_tokens * preceding + (left - source.start) * width
                    b = (end - start) * preceding + (left - start) * width
                    target[b : b + (right - left) * width].copy_(
                        src[a : a + (right - left) * width]
                    )
                preceding += width
