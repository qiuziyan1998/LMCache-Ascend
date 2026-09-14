# SPDX-License-Identifier: Apache-2.0
"""Independent combined-pressure and lifetime audit of incremental capture."""

from dataclasses import replace

from test_incremental_checkpoint import active_state, assert_bytes, setup_saved
from test_incremental_checkpoint import api as checkpoint_api
from test_incremental_checkpoint import key_types as key_types_fixture
from test_preemption_checkpoint import finish
import pytest

api = checkpoint_api
key_types = key_types_fixture


@pytest.mark.parametrize("latent_hole", [None, 8, 12])
@pytest.mark.parametrize("index_holes", [(), (4,), (8,), (8, 12)])
@pytest.mark.parametrize("fragment_limit", [1, 4])
def test_combined_holes_and_pressure_preserve_paired_data(
    api, monkeypatch, key_types, latent_hole, index_holes, fragment_limit
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    state = active_state(engine, 13)
    for group, holes in [(0, (latent_hole,)), (1, index_holes)]:
        for start, _, key in engine.token_database.process_tokens(
            tokens=list(range(13)), kv_group=group
        ):
            if start in holes:
                engine.backend.remove(key)
    allocate = engine.allocate_checkpoint_fragment

    def bounded_allocate(group, count, caches=None):
        if count > fragment_limit:
            raise MemoryError("fragmented")
        return allocate(group, count, caches)

    engine.allocate_checkpoint_fragment = bounded_allocate
    store.cancel("r", 1)
    spec = replace(
        spec,
        generation=2,
        end=19,
        resident_start=13,
        blocks=((0, 0, 0, 4, 5), (1, 2, 3, 4, 5)),
    )
    store.capture(spec, {0: [1], 1: [2]}, 4, state)
    result = store.poll()[0]
    if fragment_limit == 1 and 4 in index_holes:
        # One shortened [4,5) fragment cannot bridge the original prompt end 6.
        # The bounded allocator must refuse rather than advertise generated KV.
        assert result.status == "failed"
        store.close()
        assert all(page.refs <= 1 for page in engine.allocated)
        return
    assert result.status == "captured", result
    upper = 19 if latent_hole is None else latent_hole
    assert 6 < result.end <= upper
    if not index_holes:
        assert result.end >= min(13, upper)
    job = store.jobs["r", 2]
    assert all(a >= 13 for a, _, _, _ in job.fragments.get(0, ()))
    store.seal(api[0].SealSpec("r", 2, tuple(range(result.end))))
    assert finish(store)[0].end == result.end
    engine.allocate_checkpoint_fragment = allocate
    _, owners = store.local.normalize("r", 2, list(range(result.end)), None)
    assert_bytes(engine, result.end)
    for page in owners:
        page.ref_count_down()
    store.close()
    assert all(page.refs <= 1 for page in engine.allocated)


def test_deadline_releases_borrowed_pages_without_pinning_waiting_offers(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)
    store.capture(
        replace(spec, generation=2, resident_start=13),
        {0: [1], 1: [2]},
        4,
        active_state(engine, 13),
    )
    assert store.poll()[0].status == "captured"
    job = store.jobs["r", 2]
    borrowed = tuple(job.source_owners)
    assert borrowed and all(page.refs > 1 for page in borrowed)
    job.started -= store.timeout + 1
    assert store.poll()[0].status == "failed"
    assert not store.jobs and all(page.refs == 1 for page in borrowed)
    store.close()


def test_unknown_capture_completion_retains_reused_sources_until_restart(
    api, monkeypatch, key_types
):
    engine, store, spec = setup_saved(api, monkeypatch, key_types)

    def fail(*args):
        raise RuntimeError("injected native failure")

    engine.gpu_connector.enqueue_group_capture = fail
    engine.gpu_connector.finish_checkpoint_capture = fail
    with pytest.raises(RuntimeError, match="source owners retained"):
        store.capture(
            replace(spec, generation=2, resident_start=13),
            {0: [1], 1: [2]},
            4,
            active_state(engine, 13),
        )
    job = store.jobs["r", 2]
    assert job.quarantined and job.source_owners
    store.cancel("r", 2)
    assert all(page.refs > 1 for page in job.source_owners)
    with pytest.raises(RuntimeError, match="unresolved native transfer"):
        store.poll()
    # Test teardown only: the fake native operation never accessed device memory.
    job.quarantined = False
    store.close()
    assert all(page.refs <= 1 for page in engine.allocated)
