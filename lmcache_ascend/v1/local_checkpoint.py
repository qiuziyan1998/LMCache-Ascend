# SPDX-License-Identifier: Apache-2.0
"""Evictable local checkpoint offers; only an acquired restore owns pages."""

from dataclasses import dataclass, replace
from threading import Lock
from typing import Any


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
        backend = self.engine.checkpoint_backend()
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
                    pages, count = backend.batched_get_layer_page_prefix([source.key])
                    if not count:
                        break
                    page = pages[0]
                    groups[group].append((source, page))
                    if (
                        not page.is_valid()
                        or page.valid_tokens < source.end - source.start
                    ):
                        raise ValueError("Invalid local checkpoint page")
                    self.engine.validate_checkpoint_page(group, page)
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
        if manifest is None or tuple(tokens) != manifest.tokens[: len(tokens)]:
            raise ValueError("Local checkpoint generation/history is unavailable")
        common, groups = self.acquire(manifest, len(tokens))
        owners: list[Any] = []
        reclaimed = False

        def reserve(factory: Any, group: int, count: int) -> Any:
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

        try:
            if common != len(tokens):
                raise ValueError("Local checkpoint was evicted before restore")
            chunk = int(self.engine.config.chunk_size)
            base = manifest.prefix_end // chunk * chunk
            for group in (0, 1):
                sources = list(groups[group])
                keys, pages, normalized = [], [], []
                covered = base
                for start, end, key in self.engine.token_database.process_tokens(
                    tokens=tokens, request_configs=request_configs, kv_group=group
                ):
                    if start < base:
                        continue
                    if start != covered or not start < end <= len(tokens):
                        raise ValueError(
                            "Normalized checkpoint coverage has a gap or overlap"
                        )
                    selected = [
                        (source, page)
                        for source, page in sources
                        if source.start < end and source.end > start
                    ]
                    if (
                        len(selected) == 1
                        and selected[0][0].start == start
                        and selected[0][0].end == end
                        and selected[0][1].valid_tokens == end - start
                    ):
                        page = selected[0][1]
                        page.ref_count_up()
                    else:
                        cached = self.engine.get_checkpoint_prefix(
                            key, group, end - start
                        )
                        if cached is not None:
                            page, _ = cached
                        else:
                            if start == base and sources[0][0].start > base:
                                # Fetch the exact original partial page only
                                # when assembling a boundary that is not cached.
                                prefix = reserve(
                                    lambda: self.engine.load_checkpoint_prefix(
                                        manifest.tokens[: manifest.prefix_end],
                                        group,
                                        request_configs,
                                    ),
                                    group,
                                    manifest.prefix_end - base,
                                )
                                owners.append(prefix)
                                prefix_stop = min(
                                    manifest.prefix_end, sources[0][0].start
                                )
                                selected.insert(
                                    0,
                                    (CheckpointPage(base, prefix_stop, None), prefix),
                                )
                            page, widths = reserve(
                                lambda: self.engine.allocate_checkpoint_fragment(
                                    group, end - start
                                ),
                                group,
                                end - start,
                            )
                            owners.append(page)
                            self._assemble(page, widths, selected, start, end)
                            page = owners.pop()
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
                # Admission can retain an already-compatible canonical object.
                # Own the actual cache sources, not only our discarded duplicate.
                installed, count = (
                    self.engine.checkpoint_backend().batched_get_layer_page_prefix(keys)
                )
                owners.extend(installed)
                if count != len(keys):
                    raise ValueError(
                        "Normalized checkpoint was evicted during admission"
                    )
                for page in installed:
                    self.engine.validate_checkpoint_page(group, page)
                # Move checkpoint-owned cache references instead of creating
                # two cache aliases that would permanently make refs > 1.
                updated = list(manifest.groups)
                updated[group] = tuple(normalized)
                with self.lock:
                    manifest = replace(manifest, groups=tuple(updated))
                    if (req_id, generation) in self.manifests:
                        self.manifests[req_id, generation] = manifest
                for source, _ in groups[group]:
                    if self.engine.is_checkpoint_page_key(source.key):
                        self.engine.checkpoint_backend().remove(source.key)
            return base, owners
        except BaseException:
            for page in owners:
                page.ref_count_down()
            raise
        finally:
            self.release(groups)

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
            for width in widths:
                for source, page in sources:
                    left, right = max(start, source.start), min(end, source.end)
                    src = page.layer_tensor(layer).reshape(-1)
                    a = page.valid_tokens * preceding + (left - source.start) * width
                    b = (end - start) * preceding + (left - start) * width
                    target[b : b + (right - left) * width].copy_(
                        src[a : a + (right - left) * width]
                    )
                preceding += width
