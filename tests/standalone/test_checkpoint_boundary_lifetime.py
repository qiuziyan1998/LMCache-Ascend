# SPDX-License-Identifier: Apache-2.0
"""Assembly temporaries must not outlive copying or replace decode owners."""

from test_preemption_checkpoint import Page, fake_engine, fill, publish, start_capture
from test_preemption_checkpoint import api as checkpoint_api
import pytest
import torch

api = checkpoint_api


@pytest.mark.parametrize("shared", [False, True])
def test_release_old_boundary_before_next_group_but_keep_new_decode_pages(
    api, monkeypatch, shared
):
    engine = fake_engine()
    store, _, _ = start_capture(api, monkeypatch, engine, prefix=6, end=14)
    store.poll()
    publish(api, store, 13)
    original = Page(2, 2, (2, 1))
    fill(original, 4, 0)
    original.refs = 2 if shared else 0  # Cache plus another request, or temporary.
    baseline_refs = original.refs

    def prefix(*args):
        original.ref_count_up()
        return original

    allocate = engine.allocate_checkpoint_fragment
    checked = []

    def allocate_after_boundary(group, count, caches=None):
        if group == 1:
            assert original.refs == baseline_refs
            checked.append(True)
            if not shared:
                # Simulate immediate allocator reuse of the retired CPU span.
                original.raw_data.fill_(255)
        return allocate(group, count, caches)

    engine.load_checkpoint_prefix = prefix
    engine.allocate_checkpoint_fragment = allocate_after_boundary
    _, owners = store.local.normalize("r", 1, list(range(13)), None)
    try:
        assert checked and original not in owners
        plans = list(engine.token_database.process_tokens(tokens=list(range(13))))
        pages = [engine.backend.pages[key] for a, _, key in plans if a >= 4]
        pointers = torch.tensor(
            [[p.layer_data_ptr(i) for p in pages] for i in range(2)]
        )
        assert not torch.any(pointers == original.layer_data_ptr(0))
        for start, end, key in plans:
            if start < 4:
                continue
            page = engine.backend.pages[key]
            assert page in owners and page.refs > 1
            assert not engine.backend.evict(key)
            expected = Page(2, end - start, (2, 1))
            fill(expected, start, 0)
            assert torch.equal(page.raw_data, expected.raw_data)
    finally:
        for page in owners:
            page.ref_count_down()
        store.close()
    assert original.refs == baseline_refs


def test_assembly_failure_releases_temporary_and_destination_once(api, monkeypatch):
    engine = fake_engine()
    store, _, _ = start_capture(api, monkeypatch, engine, prefix=6, end=14)
    store.poll()
    publish(api, store, 13)
    original = Page(2, 2, (2, 1))
    engine.load_checkpoint_prefix = lambda *args: original
    store.local._assemble = lambda *args: (_ for _ in ()).throw(ValueError("bad spans"))
    with pytest.raises(ValueError, match="bad spans"):
        store.local.normalize("r", 1, list(range(13)), None)
    assert original.refs == 0
    assert all(page.refs <= 1 for page in engine.allocated)
    store.close()


@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_boundary_failure_uses_real_retry_and_protects_all_sources(
    api, monkeypatch, retry_succeeds
):
    engine = fake_engine()
    store, _, _ = start_capture(api, monkeypatch, engine, prefix=6, end=14)
    store.poll()
    publish(api, store, 13)
    allocate = engine.allocate_checkpoint_fragment
    attempts, reclaims = [], []

    def allocate_boundary(group, count, caches=None):
        if group == 0 and count == 4:
            attempts.append((group, count))
            if len(attempts) == 1 or not retry_succeeds:
                raise MemoryError("no suitable span")
        return allocate(group, count, caches)

    def reclaim(count, groups):
        reclaims.append((count, groups))
        assert all(not engine.backend.evict(key) for key in engine.backend.pages)
        return True  # Capacity alone must never become restore success.

    engine.allocate_checkpoint_fragment = allocate_boundary
    engine.reclaim_checkpoint_capacity = reclaim
    if retry_succeeds:
        _, owners = store.local.normalize("r", 1, list(range(13)), None)
        for page in owners:
            page.ref_count_down()
    else:
        with pytest.raises(api[0].CheckpointRestoreMiss, match="workspace"):
            store.local.normalize("r", 1, list(range(13)), None)
    assert attempts == [(0, 4), (0, 4)]
    assert reclaims == [(4, {0: None})]
    assert all(page.refs <= 1 for page in engine.allocated)
    store.close()


def test_later_group_failure_and_shorter_retry_do_not_release_new_latent_pages(
    api, monkeypatch
):
    engine = fake_engine()
    store, _, _ = start_capture(api, monkeypatch, engine, prefix=6, end=14)
    store.poll()
    publish(api, store, 13)
    original = Page(2, 2, (2, 1))
    fill(original, 4, 0)
    engine.load_checkpoint_prefix = lambda *args: original
    allocate = engine.allocate_checkpoint_fragment

    def fail_index(group, count, caches=None):
        if group == 1:
            assert original.refs == 0
            raise MemoryError("index workspace refused")
        return allocate(group, count, caches)

    engine.allocate_checkpoint_fragment = fail_index
    with pytest.raises(api[0].CheckpointRestoreMiss) as error:
        store.local.normalize("r", 1, list(range(13)), None)
    assert error.value.available_end == 12 and original.refs == 0
    engine.allocate_checkpoint_fragment = allocate
    _, owners = store.local.normalize("r", 1, list(range(12)), None)
    try:
        for group in (0, 1):
            plans = engine.token_database.process_tokens(
                tokens=list(range(12)), kv_group=group
            )
            for start, end, key in plans:
                if start < 4:
                    continue
                page = engine.backend.pages[key]
                assert page.refs > 1 and not engine.backend.evict(key)
                expected = Page(2, end - start, (2, 1) if group == 0 else (1,))
                fill(expected, start, group)
                assert torch.equal(page.raw_data, expected.raw_data)
    finally:
        for page in owners:
            page.ref_count_down()
        store.close()
    assert original.refs == 0 and all(page.refs <= 1 for page in engine.allocated)
