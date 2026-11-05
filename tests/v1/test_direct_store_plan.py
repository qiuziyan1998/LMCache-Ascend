# SPDX-License-Identifier: Apache-2.0
"""Source ownership contracts for pure direct-store window planning."""
from types import SimpleNamespace

import pytest

from lmcache_ascend.v1.direct_store_plan import (
    DirectPageBatch,
    build_remote_fill_source_plan,
    merge_deferred_remote_fill_batches,
    select_remote_fill_batch_pages,
)


def _batch(req_id="request", group=0):
    key = SimpleNamespace(kv_group=group, to_string=lambda: f"key-{group}")
    event = object()
    return DirectPageBatch(
        req_id=req_id,
        keys=[key],
        ptrs=[[1024]],
        sizes=[[64]],
        owners=(object(),),
        ready_event=event,
        group_ends={group: 16},
        ranges=((0, 16),),
        ready_events=(event,),
    )


def test_window_selection_keeps_source_owner_and_event_identity():
    batch = _batch()
    page = SimpleNamespace(canonical_key="key-0", kv_group=0)
    selected = select_remote_fill_batch_pages(batch, (page,))
    assert selected is not batch
    assert selected.owners is batch.owners
    assert selected.ready_events is batch.ready_events
    assert selected.ready_event is batch.ready_event
    assert selected.keys[0] is batch.keys[0]
    assert selected.ptrs[0] is batch.ptrs[0]
    assert selected.sizes[0] is batch.sizes[0]
    assert selected.ranges[0] is batch.ranges[0]
    assert selected.group_ends == {0: 16}


def test_duplicate_source_identity_is_rejected():
    batch = _batch()
    batch.keys *= 2
    batch.ptrs *= 2
    batch.sizes *= 2
    batch.ranges *= 2
    with pytest.raises(ValueError, match="identity is duplicated"):
        select_remote_fill_batch_pages(batch, ())


def test_source_plan_keeps_owners_and_causal_events():
    batch = _batch()
    plan = build_remote_fill_source_plan(batch)
    assert plan.owners is batch.owners
    assert plan.producer_events is batch.ready_events
    assert plan.pages[0].source_ptrs == (1024,)
    assert plan.pages[0].source_lengths == (64,)


def test_source_plan_requires_complete_fences():
    batch = _batch()
    batch.ready_events = ()
    with pytest.raises(ValueError, match="complete producer fences"):
        build_remote_fill_source_plan(batch)


def test_merge_preserves_row_identity_and_deduplicates_owners():
    first, second = _batch(), _batch(group=1)
    second.owners = first.owners
    events = (object(), object())
    merged = merge_deferred_remote_fill_batches([first, second], events)
    assert merged.owners == first.owners
    assert merged.owners[0] is first.owners[0]
    assert merged.ptrs[0] is first.ptrs[0]
    assert merged.ptrs[1] is second.ptrs[0]
    assert merged.ready_events is events
    assert merged.ready_event is events[-1]
    assert merged.group_ends == {0: 16, 1: 16}


@pytest.mark.parametrize("missing", ["batches", "events"])
def test_merge_requires_sources_and_fences(missing):
    with pytest.raises(ValueError, match="requires sources and fences"):
        merge_deferred_remote_fill_batches(
            [] if missing == "batches" else [_batch()],
            () if missing == "events" else (object(),),
        )


def test_merge_rejects_cross_request_owners():
    with pytest.raises(ValueError, match="span requests"):
        merge_deferred_remote_fill_batches([_batch("a"), _batch("b")], (object(),))
