# SPDX-License-Identifier: Apache-2.0
"""Shared direct-store source batches and pure remote-fill window planning.

Batches borrow the existing tensor owners and causal events. Selection keeps
those exact owner/event tuples; merging deduplicates owners by identity as before.
Callers retain each batch until its persistent/native completion contract allows
release. These functions own no mutable request state, locks, threads or engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, TYPE_CHECKING

from lmcache.v1.remote_fill import ControlPage
from lmcache.v1.remote_fill.native import DirectPushPageSource, DirectPushSourcePlan

if TYPE_CHECKING:
    from lmcache.utils import CacheEngineKey
    import torch


@dataclass(slots=True)
class DirectPageBatch:
    req_id: str
    keys: List[CacheEngineKey]
    ptrs: List[List[int]]
    sizes: List[List[int]]
    owners: tuple[torch.Tensor, ...]
    ready_event: Any
    group_ends: dict[int, int]
    ranges: tuple[tuple[int, int], ...] = ()
    ready_events: tuple[Any, ...] = ()


def select_remote_fill_batch_pages(
    batch: DirectPageBatch,
    pages: tuple[ControlPage, ...],
) -> DirectPageBatch:
    """Select one bounded control window from an existing source batch."""

    sources = {
        (key.to_string(), int(key.kv_group)): (
            key,
            page_ptrs,
            page_sizes,
            page_range,
        )
        for key, page_ptrs, page_sizes, page_range in zip(
            batch.keys,
            batch.ptrs,
            batch.sizes,
            batch.ranges,
            strict=True,
        )
    }
    if len(sources) != len(batch.keys):
        raise ValueError("remote-fill source page identity is duplicated")
    selected = [
        sources[(page.canonical_key, page.kv_group)] for page in pages
    ]
    group_ends: dict[int, int] = {}
    for page, (_, _, _, (_, end)) in zip(pages, selected, strict=True):
        group_ends[page.kv_group] = max(
            group_ends.get(page.kv_group, 0), end
        )
    return DirectPageBatch(
        req_id=batch.req_id,
        keys=[item[0] for item in selected],
        ptrs=[item[1] for item in selected],
        sizes=[item[2] for item in selected],
        owners=batch.owners,
        ready_event=batch.ready_event,
        group_ends=group_ends,
        ranges=tuple(item[3] for item in selected),
        ready_events=batch.ready_events,
    )


def build_remote_fill_source_plan(batch: DirectPageBatch) -> DirectPushSourcePlan:
    """Build native source vectors while retaining the exact owners and fences."""
    pages = tuple(
        DirectPushPageSource(
            canonical_key=key.to_string(),
            kv_group=int(key.kv_group),
            source_ptrs=tuple(page_ptrs),
            source_lengths=tuple(page_sizes),
        )
        for key, page_ptrs, page_sizes in zip(
            batch.keys, batch.ptrs, batch.sizes, strict=True
        )
    )
    if not batch.ready_events:
        raise ValueError("remote fill requires complete producer fences")
    return DirectPushSourcePlan(
        pages=pages,
        owners=batch.owners,
        producer_events=batch.ready_events,
    )


def merge_deferred_remote_fill_batches(
    batches: list[DirectPageBatch],
    ready_events: tuple[Any, ...],
) -> DirectPageBatch:
    """Join window-owned source plans under the final causal fence."""

    if not batches or not ready_events:
        raise ValueError("deferred remote fill requires sources and fences")
    req_id = batches[0].req_id
    if any(batch.req_id != req_id for batch in batches):
        raise ValueError("deferred remote-fill batches span requests")
    owners: dict[int, torch.Tensor] = {}
    group_ends: dict[int, int] = {}
    for batch in batches:
        owners.update((id(owner), owner) for owner in batch.owners)
        for group, end in batch.group_ends.items():
            group_ends[group] = max(group_ends.get(group, 0), end)
    return DirectPageBatch(
        req_id=req_id,
        keys=[key for batch in batches for key in batch.keys],
        ptrs=[ptrs for batch in batches for ptrs in batch.ptrs],
        sizes=[sizes for batch in batches for sizes in batch.sizes],
        owners=tuple(owners.values()),
        ready_event=ready_events[-1],
        group_ends=group_ends,
        ranges=tuple(page for batch in batches for page in batch.ranges),
        ready_events=ready_events,
    )

