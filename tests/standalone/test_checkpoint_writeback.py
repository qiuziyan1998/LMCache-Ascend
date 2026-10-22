# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint control/assembly; CPU allocations and remote I/O are fakes."""

from concurrent.futures import Future
import ast
import gc
import importlib.util
from pathlib import Path
import sys
from threading import Event
import time

import pytest
import torch

from test_preemption_checkpoint import (
    api as checkpoint_api,
    fake_engine,
    start_capture,
    publish,
    Page,
    fill,
)

from test_local_checkpoint_restore import implementation

api = checkpoint_api


def wait(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "background operation did not complete"
        time.sleep(0.002)


@pytest.fixture
def deployment(api, monkeypatch):
    monkeypatch.setitem(sys.modules, "lmcache_ascend.v1.preemption_checkpoint", api[1])
    name = "lmcache_ascend.v1.checkpoint_writeback"
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/checkpoint_writeback.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        Page, "get_physical_size", lambda p: p.raw_data.numel(), raising=False
    )
    engine = fake_engine((2, 3))
    engine.config.decode_checkpoint_writeback = True
    engine.config.decode_checkpoint_retention_max_bytes = 10000
    engine.config.decode_checkpoint_free_reserve_bytes = 1000
    engine.free = 90000
    engine.backend.get_allocator_capacity_bytes = lambda: (engine.free, 100000)
    engine.remote, engine.puts, engine.gets = {}, [], []

    def put(key, page, req_id):
        engine.puts.append(key)
        engine.remote[key] = (page.raw_data.clone(), page.valid_tokens)
        result = Future()
        result.set_result(None)
        return result

    def get(key, group, tokens, req_id):
        engine.gets.append(key)
        if key not in engine.remote:
            return None
        page, _ = engine.allocate_checkpoint_fragment(group, tokens)
        page.raw_data.copy_(engine.remote[key][0])
        return page

    engine.persist_checkpoint_page = put
    engine.probe_checkpoint_pages = lambda keys: [k in engine.remote for k in keys]
    engine.load_checkpoint_page = get
    stores = []

    def create(accepted=11, **kwargs):
        store, capture, _ = start_capture(api, monkeypatch, engine, **kwargs)
        stores.append(store)
        assert store.poll()[0].status == "captured"
        publish(api, store, accepted)
        return store

    yield engine, create, module
    for store in stores:
        if store.local.fatal is None:
            store.close()


def spill(store, engine):
    engine.free = 0
    store.local.wake.set()
    wait(lambda: bool(store.local.records["r", 1].remote))
    wait(lambda: store.local.used == 0)


def assert_rows(groups):
    for group, held in enumerate(groups):
        for source, page in held:
            expected = Page(
                page.num_layers, page.valid_tokens, (2, 1) if group == 0 else (1,)
            )
            fill(expected, source.start, group)
            assert torch.equal(page.raw_data, expected.raw_data)


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize(
    "enabled,budget", [(False, 10000), (True, 10000), (True, 100000)]
)
def test_writeback_capacity_validation_uses_initialized_heap_and_tp_envelope(
    deployment, api, enabled, budget, world_size
):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    engine, _, _ = deployment
    envelopes = []
    mapping = NS(preflight_device_ptr=lambda: None, close=lambda: None)
    base = implementation(
        "../LMCache-NPU/lmcache/v1/cache_engine.py",
        "LMCacheEngine",
        {
            "_shared_local_cpu_backend",
            "_post_init_shared_cpu_cache",
            "_shared_cpu_cache_startup_envelope",
        },
        object,
        time=time,
        startup_phase=lambda *a, **kw: nullcontext(),
        SharedSlabMapping=NS(from_rank0_allocator=lambda **kw: mapping),
        logger=NS(info=lambda *a, **kw: None, exception=lambda *a, **kw: None),
    )
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"checkpoint_backend", "_post_init_shared_cpu_cache"},
        base,
    )
    obj = cls()
    obj.config = engine.config
    obj.config.decode_checkpoint_writeback = enabled
    obj.config.decode_checkpoint_retention_max_bytes = budget
    obj.metadata = NS(
        is_first_rank=lambda: True, world_size=world_size, worker_id=0, first_rank=0
    )
    obj.enable_shared_cpu_cache = obj.shared_cpu_cache_strict = True
    obj.shared_cpu_cache_mapping = None
    obj.shared_cpu_cache_name = obj.shared_cpu_cache_slab_size = None
    obj.shared_cpu_cache_generation = 0
    obj.broadcast_object_fn = lambda envelope, rank: envelopes.append(envelope)
    # Actual Ascend __init__ creates CheckpointWorker before base post_init
    # constructs StorageManager. The constructor must not query that heap.
    obj.storage_manager = None
    obj.checkpoint_worker = api[1].CheckpointWorker(obj)
    try:
        assert not envelopes
        engine.backend.use_hot = True
        engine.backend.memory_allocator = NS(
            buffer=torch.empty(100000, dtype=torch.uint8), shm_name="test"
        )
        obj.storage_manager = NS(local_cpu_backend=engine.backend)
        if enabled and budget >= 100000:
            with pytest.raises(ValueError, match="smaller than the CPU heap"):
                obj._post_init_shared_cpu_cache()
            assert [e["status"] for e in envelopes] == (
                ["error"] if world_size > 1 else []
            )
            if world_size == 1:
                return
            # The actual passive preflight consumes that same failure envelope.
            passive = cls()
            passive.enable_shared_cpu_cache = True
            passive.metadata = NS(
                is_first_rank=lambda: False, world_size=4, worker_id=1, first_rank=0
            )
            passive.checkpoint_worker = None
            passive.broadcast_object_fn = lambda *a: envelopes[0]
            with pytest.raises(ValueError, match="passive preflight failed"):
                passive._post_init_shared_cpu_cache()
        else:
            obj._post_init_shared_cpu_cache()
            assert [e["status"] for e in envelopes] == (
                ["ok"] if world_size > 1 else []
            )
    finally:
        obj.checkpoint_worker.close()


def test_protected_ready_is_local_and_budget_counts_physical_owners(deployment):
    engine, create, _ = deployment
    store = create()
    record = store.local.records["r", 1]
    owners = {id(p): p for g in record.groups for _, p in g}
    assert store.local.used == sum(p.get_physical_size() for p in owners.values())
    assert not engine.puts and not engine.gets
    for group in record.manifest.groups:
        for source in group:
            assert not engine.backend.evict(source.key)
    assert store.local.available("r", 1, 11) == 11


def test_budget_refusal_preserves_unprotected_baseline(deployment):
    engine, create, _ = deployment
    engine.config.decode_checkpoint_retention_max_bytes = 1
    store = create()
    assert not store.local.records and store.local.used == 0
    assert store.local.available("r", 1, 11) == 11
    assert engine.backend.evict(("local", "r", 1, 0, 4, 8))
    assert store.local.available("r", 1, 11) == 4


def test_remote_spill_crops_accepted_planes_and_restores_after_eviction(deployment):
    engine, create, _ = deployment
    store = create()
    spill(store, engine)
    assert len(engine.puts) == 6  # Three pages in each unequal-layer group.
    for key, (raw, tokens) in engine.remote.items():
        expected = Page((2, 3)[key.group], tokens, (2, 1) if key.group == 0 else (1,))
        fill(expected, key.start, key.group)
        assert torch.equal(raw, expected.raw_data)
    for key in tuple(engine.backend.pages):
        assert engine.backend.evict(key)
    assert store.local.available("r", 1, 11) == 11
    assert not engine.gets  # Metadata lookup is not a payload load.
    base, owners = store.local.normalize("r", 1, list(range(11)), None)
    assert base == 0 and len(engine.gets) == 6
    manifest = store.local.manifest("r", 1)
    end, groups = store.local.acquire(manifest, 11)
    assert end == 11
    assert_rows(groups)
    store.local.release(groups)
    for page in owners:
        page.ref_count_down()


@pytest.mark.parametrize("group", [0, 1])
def test_missing_remote_page_shortens_paired_coverage(deployment, group):
    engine, create, _ = deployment
    store = create()
    spill(store, engine)
    for key in tuple(engine.backend.pages):
        engine.backend.evict(key)
    lost = next(k for k in engine.remote if k.group == group and k.start == 4)
    del engine.remote[lost]
    assert store.local.available("r", 1, 11) == 4
    with pytest.raises(
        sys.modules[
            "lmcache.integration.vllm.preemption_checkpoint"
        ].CheckpointRestoreMiss
    ):
        store.local.normalize("r", 1, list(range(11)), None)


def test_alternating_sources_read_only_missing_pages(deployment):
    engine, create, _ = deployment
    store = create()
    spill(store, engine)
    for key in tuple(engine.backend.pages):
        if key[3] == 0 and key[4] == 4 or key[3] == 1 and key[4] != 4:
            engine.backend.evict(key)
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    assert {(k.group, k.start) for k in engine.gets} == {(0, 4), (1, 0), (1, 8)}
    for page in owners:
        page.ref_count_down()


def test_lookup_racing_spill_confirmation_keeps_remote_coverage(
    deployment, monkeypatch
):
    engine, create, _ = deployment
    store = create()
    get_page = store.local.get_page
    first = True

    def complete_spill_before_acquiring_page(*args):
        nonlocal first
        if first:
            first = False
            # Lookup selected protected local sources. The background spill can
            # finish and make them evictable before it takes its own references.
            spill(store, engine)
            for key in tuple(engine.backend.pages):
                assert engine.backend.evict(key)
        return get_page(*args)

    monkeypatch.setattr(store.local, "get_page", complete_spill_before_acquiring_page)
    assert store.local.available("r", 1, 11) == 11
    assert not engine.gets  # A lookup must only probe remote presence.


@pytest.mark.parametrize("during_prepare", [False, True])
def test_frozen_cache_defers_checkpoint_spill(deployment, monkeypatch, during_prepare):
    engine, create, _ = deployment
    store = create()
    record = store.local.records["r", 1]
    frozen = not during_prepare
    engine.is_frozen = lambda: frozen
    build_page = store.local.build_page

    def freeze_after_prepare(*args):
        nonlocal frozen
        page = build_page(*args)
        frozen = True
        return page

    monkeypatch.setattr(store.local, "build_page", freeze_after_prepare)
    assert store.local._spill(record, record.groups) == ()
    assert not engine.puts
    assert store.local.used > 0


@pytest.mark.parametrize("probe", [False, True])
def test_frozen_restore_uses_only_local_sources(deployment, probe):
    engine, create, _ = deployment
    store = create()
    spill(store, engine)
    for key in tuple(engine.backend.pages):
        assert engine.backend.evict(key)
    engine.is_frozen = lambda: True

    def forbidden_remote_lookup(keys):
        pytest.fail("freeze mode must not access remote sources")

    engine.probe_checkpoint_pages = forbidden_remote_lookup
    manifest = store.local.manifest("r", 1)
    end, groups = store.local._acquire(manifest, 11, probe=probe)
    store.local.release(groups)
    assert end == manifest.prefix_end and not engine.gets


def test_blocked_put_does_not_block_local_publication_or_cancel_cleanup(deployment):
    engine, create, module = deployment
    store = create()
    started, unblock = Event(), Future()
    original = engine.persist_checkpoint_page

    def blocked(key, page, req):
        original(key, page, req)
        started.set()
        return unblock

    engine.persist_checkpoint_page = blocked
    engine.free = 0
    store.local.wake.set()
    assert started.wait(5)
    try:
        # The real local publication executor remains usable during PUT.
        assert store.executor.submit(lambda: 7).result(timeout=1) == 7
        record = store.local.records["r", 1]
        pages = [p for g in record.groups for _, p in g]
        store.cancel("r", 1)
        assert store.executor.submit(lambda: 7).result(timeout=1) == 7
        assert store.local.manifest("r", 1) is None
        assert store.local.used > 0 and all(p.refs > 0 for p in pages)
    finally:
        unblock.set_result(None)
    wait(lambda: store.local.used == 0)


def test_resume_uses_local_pages_while_put_is_unfinished(deployment):
    engine, create, _ = deployment
    store = create()
    entered, gate = Event(), Future()
    engine.persist_checkpoint_page = lambda *args: (entered.set(), gate)[1]
    engine.free = 0
    store.local.wake.set()
    assert entered.wait(5)
    try:
        store.begin_restore("r", 1, 7)
        _, owners = store.local.normalize("r", 1, list(range(11)), None)
        store.hold_restore("r", 1, 7, owners)
        assert not engine.gets
        store.restore_event("r", 1, 7, "admitted")
        assert ("r", 1) in store.local.records
        store.release_restore("r", 1, 7)
        assert ("r", 1) not in store.local.records
    finally:
        gate.set_result(None)
    wait(lambda: store.local.used == 0)


@pytest.mark.parametrize(
    "events", [("ack", "admitted"), ("admitted", "ack"), ("failed", "ack")]
)
def test_restore_notifications_join_and_failed_attempt_keeps_retention(
    deployment, events
):
    engine, create, _ = deployment
    store = create()
    store.begin_restore("r", 1, 7)
    for event in events:
        if event == "ack":
            store.release_restore("r", 1, 7)
        else:
            store.restore_event("r", 1, 7, event)
    if "failed" in events:
        assert store.local.records["r", 1].groups and store.local.used > 0
        assert not store.local.records["r", 1].attempts
    else:
        assert not store.local.records
        wait(lambda: store.local.used == 0)


@pytest.mark.parametrize("failed_group", [0, 1])
def test_put_failure_keeps_local_protection(deployment, failed_group):
    engine, create, _ = deployment
    store = create()
    called = Event()
    original = engine.persist_checkpoint_page

    def fail(*args):
        if args[0].group != failed_group:
            return original(*args)
        called.set()
        f = Future()
        f.set_exception(RuntimeError("remote full"))
        return f

    engine.persist_checkpoint_page = fail
    engine.free = 0
    store.local.wake.set()
    assert called.wait(5)
    wait(lambda: store.local.records["r", 1].retry_at > 0)
    assert not store.local.records["r", 1].remote
    assert not engine.backend.evict(("local", "r", 1, 0, 4, 8))


def test_cancel_during_boundary_preparation_does_not_start_put(deployment):
    engine, create, _ = deployment
    store = create()
    original = engine.load_checkpoint_prefix

    def cancel(*args):
        store.cancel("r", 1)
        return original(*args)

    engine.load_checkpoint_prefix = cancel
    engine.free = 0
    store.local.wake.set()
    wait(lambda: not store.local.records)
    wait(lambda: store.local.used == 0)
    assert not engine.puts


@pytest.mark.parametrize("stopped", ["cancelled", "restoring", "closed", "frozen"])
def test_stopped_spill_skips_history_copy_and_hash_planning(deployment, stopped):
    from dataclasses import replace

    engine, create, _ = deployment
    store = create()
    record = store.local.records["r", 1]
    calls = []

    class History(tuple):
        def __iter__(self):
            calls.append("copy-history")
            return super().__iter__()

    record.manifest = replace(record.manifest, tokens=History(record.manifest.tokens))
    original = engine.checkpoint_token_results

    def plan(*args, **kwargs):
        calls.append("plan-hashes")
        return original(*args, **kwargs)

    engine.checkpoint_token_results = plan
    if stopped == "cancelled":
        record.cancelled = True
    elif stopped == "restoring":
        record.attempts[1] = {"begin"}
    elif stopped == "closed":
        store.local.closed = True
    else:
        engine.is_frozen = lambda: True
    assert store.local._spill(record, record.groups) == ()
    assert not calls
    assert not engine.puts


def test_spill_cancelled_after_selection_retires_without_preparing(deployment):
    engine, create, _ = deployment
    store = create()
    selected, resume = Event(), Event()
    original = store.local._spill
    calls = []
    plan = engine.checkpoint_token_results

    def prepare(*args, **kwargs):
        calls.append("plan-hashes")
        return plan(*args, **kwargs)

    def gated(record, groups):
        selected.set()
        assert resume.wait(5)
        return original(record, groups)

    engine.checkpoint_token_results = prepare
    store.local._spill = gated
    engine.free = 0
    store.local.wake.set()
    try:
        assert selected.wait(5)
        store.cancel("r", 1)
    finally:
        resume.set()
    wait(lambda: store.local.used == 0)
    assert not calls and not engine.puts and not store.local.records
    assert store.local.fatal is None
    assert all(page.refs == 1 for page in engine.backend.pages.values())


def test_gc_disabled_finish_releases_retention(deployment):
    engine, create, _ = deployment
    enabled = gc.isenabled()
    gc.disable()
    try:
        store = create()
        store.local.forget("r")
        wait(lambda: store.local.used == 0)
        assert not store.local.records
        for page in engine.backend.pages.values():
            assert page.refs == 1
    finally:
        if enabled:
            gc.enable()


def test_retained_page_survives_explicit_alias_removal(deployment):
    engine, create, _ = deployment
    store = create()
    for key in tuple(engine.backend.pages):
        engine.backend.remove(key)
    assert store.local.available("r", 1, 11) == 11
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    assert not engine.gets
    for page in owners:
        page.ref_count_down()


def test_failed_normalization_keeps_original_retained_ranges_usable(deployment):
    engine, create, _ = deployment
    store = create()
    allocate = engine.allocate_checkpoint_fragment

    def fail(group, count, caches=None):
        if group == 1 and count == 3:
            raise MemoryError("boundary workspace full")
        return allocate(group, count, caches)

    engine.allocate_checkpoint_fragment = fail
    with pytest.raises(
        sys.modules[
            "lmcache.integration.vllm.preemption_checkpoint"
        ].CheckpointRestoreMiss
    ):
        store.local.normalize("r", 1, list(range(11)), None)
    for key in tuple(engine.backend.pages):
        engine.backend.evict(key)
    assert store.local.available("r", 1, 11) == 11


def test_shared_generations_are_charged_once(deployment):
    engine, create, _ = deployment
    store = create()
    before = store.local.used
    manifest = store.local.manifest("r", 1)
    store.local.publish("r", 2, manifest)
    assert store.local.used == before
    store.local.forget("r", 1)
    store.executor.submit(lambda: None).result(timeout=1)
    assert store.local.used == before
    assert store.local.available("r", 2, 11) == 11
    store.local.forget("r", 2)
    wait(lambda: store.local.used == 0)


def test_publication_reuses_its_unique_owner_set(deployment, monkeypatch):
    _, create, module = deployment
    store = create()
    before = store.local.used
    acquire = module.LocalCheckpointStore.acquire
    traversals = []

    class CountedGroups(list):
        def __iter__(self):
            traversals.append(1)
            return super().__iter__()

    def counted_acquire(self, *args):
        end, groups = acquire(self, *args)
        return end, CountedGroups(groups)

    monkeypatch.setattr(module.LocalCheckpointStore, "acquire", counted_acquire)
    store.local.publish("r", 2, store.local.manifest("r", 1))
    # One owner-set traversal and one range index; charging reuses the set.
    assert len(traversals) == 2
    assert store.local.used == before  # Shared physical owners are charged once.


@pytest.mark.parametrize("failures", [{1}, {1, 3}])
def test_spill_and_normalize_have_independent_single_reclaim_budgets(
    deployment, failures
):
    engine, create, _ = deployment
    store = create()
    record = store.local.records["r", 1]
    allocate = engine.allocate_checkpoint_fragment
    calls, reclaims = [], []

    def pressured_allocate(*args):
        calls.append(args)
        if len(calls) in failures:
            raise MemoryError("injected pressure")
        return allocate(*args)

    engine.allocate_checkpoint_fragment = pressured_allocate
    engine.reclaim_checkpoint_capacity = lambda *args: (reclaims.append(args), True)[1]
    engine.get_checkpoint_prefix = lambda *args, **kwargs: None
    miss = sys.modules[
        "lmcache.integration.vllm.preemption_checkpoint"
    ].CheckpointRestoreMiss
    for restore in (False, True):
        calls.clear()
        reclaims.clear()

        def run(restore):
            if restore:
                _, owners = store.local.normalize("r", 1, list(range(11)), None)
                for page in owners:
                    page.ref_count_down()
            else:
                assert store.local._spill(record, record.groups)

        if len(failures) > 1:
            with pytest.raises(miss if restore else MemoryError):
                run(restore)
        else:
            run(restore)
        assert len(reclaims) == 1


def test_repreemption_holds_nonresident_latent_until_new_retention(deployment, api):
    from types import SimpleNamespace as NS

    engine, create, _ = deployment
    store = create()
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    sources = list(
        engine.token_database.process_tokens(tokens=list(range(11)), kv_group=0)
    )
    state = NS(
        cached_starts=[a for a, _, _ in sources],
        cached_ends=[b for _, b, _ in sources],
        cached_keys=[[k for _, _, k in sources]],
    )
    store.local.forget("r", 1)
    store.executor.submit(lambda: None).result(timeout=1)
    try:
        store.capture(
            api[0].CaptureSpec("r", 2, 0, 15, 11, ((1, 2, 3, 4),) * 2, prefix_end=3),
            {0: [1], 1: [2]},
            4,
            state,
        )
        assert store.poll()[0].status == "captured"
    finally:
        for page in owners:
            page.ref_count_down()  # The preempted request releases its old lease.
    assert all(not engine.backend.evict(key) for _, _, key in sources)
    store.seal(api[0].SealSpec("r", 2, tuple(range(14))))
    wait(lambda: store.jobs["r", 2].future.done())
    assert store.poll()[0].status == "ready"
    engine.free = 0
    store.local.wake.set()
    wait(lambda: bool(store.local.records["r", 2].remote))
    wait(lambda: store.local.used == 0)
    for key in tuple(engine.backend.pages):
        assert engine.backend.evict(key)
    _, owners = store.local.normalize("r", 2, list(range(14)), None)
    try:
        end, groups = store.local.acquire(store.local.manifest("r", 2), 14)
        try:
            assert end == 14
            assert_rows(groups)
        finally:
            store.local.release(groups)
    finally:
        for page in owners:
            page.ref_count_down()


def test_waiting_metadata_detaches_request_config_mapping(deployment):
    from dataclasses import replace

    _, create, _ = deployment
    store = create()
    configs = {"lmcache.tag.tenant": "original"}
    manifest = replace(store.local.manifest("r", 1), request_configs=configs)
    store.local.publish("r", 2, manifest)
    configs["lmcache.tag.tenant"] = "changed"
    assert (
        store.local.manifest("r", 2).request_configs["lmcache.tag.tenant"] == "original"
    )


def test_restore_allocation_failure_is_a_safe_miss(deployment):
    engine, create, _ = deployment
    store = create()
    spill(store, engine)
    for key in tuple(engine.backend.pages):
        engine.backend.evict(key)

    def fail(*args):
        raise MemoryError("full")

    engine.load_checkpoint_page = fail
    with pytest.raises(
        sys.modules[
            "lmcache.integration.vllm.preemption_checkpoint"
        ].CheckpointRestoreMiss
    ):
        store.local.normalize("r", 1, list(range(11)), None)


def test_spill_unknown_completion_quarantines_source_and_workspace(deployment):
    engine, create, _ = deployment
    store = create()
    unknown = sys.modules[
        "lmcache.v1.remote_fill.native"
    ].NativeExternalPageTransferUnknownError

    def fail(*args):
        result = Future()
        result.set_exception(unknown("DMA still active"))
        return result

    engine.persist_checkpoint_page = fail
    engine.free = 0
    store.local.wake.set()
    wait(lambda: store.local.fatal is not None)
    store.local.forget("r")
    store.executor.submit(lambda: None).result(timeout=1)
    assert store.local.used > 0 and store.local.quarantine
    with pytest.raises(RuntimeError, match="quarantined"):
        store.close()
    # Only this CPU fake test establishes terminal completion after the check.
    for held in store.local.quarantine:
        if isinstance(held, list):
            store.local._retire(held)
        else:
            held.ref_count_down()
    store.local.quarantine.clear()
    store.local.fatal = None
    store.close()


def test_old_spill_cannot_publish_to_reused_identity(deployment):
    engine, create, _ = deployment
    store = create()
    entered, gate = Event(), Future()
    engine.persist_checkpoint_page = lambda *args: (entered.set(), gate)[1]
    old = store.local.manifest("r", 1)
    engine.free = 0
    store.local.wake.set()
    assert entered.wait(5)
    try:
        store.local.forget("r", 1)
        engine.free = 90000
        store.local.publish("r", 1, old)
        new = store.local.records["r", 1]
    finally:
        gate.set_result(None)
    wait(lambda: all(count == 1 for _, count in store.local.charges.values()))
    assert store.local.records["r", 1] is new and not new.remote


def test_normalization_cannot_replace_a_new_generation_record(deployment):
    engine, create, _ = deployment
    store = create()
    old = store.local.manifest("r", 1)
    replacement = []

    def replace_during_put():
        engine.backend.before_put = lambda: None
        store.local.forget("r", 1)
        store.local.publish("r", 1, old)
        replacement.append(store.local.manifest("r", 1))

    engine.backend.before_put = replace_during_put
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    try:
        assert store.local.manifest("r", 1) is replacement[0]
    finally:
        for page in owners:
            page.ref_count_down()


def test_spill_uses_actual_token_database_input_contract(deployment):
    from types import SimpleNamespace as NS

    engine, create, _ = deployment
    store = create()
    source = (
        Path(__file__).resolve().parents[3] / "LMCache-NPU/lmcache/v1/token_database.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_hash_tokens"
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = dict(torch=torch)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, method], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        ns,
    )
    db = NS(hash_func=lambda _: 0, _canonicalize_hash_inputs=lambda *args: args)
    original = engine.token_database.process_tokens

    def tokens(**kwargs):
        ns["_hash_tokens"](db, kwargs["tokens"][:4])
        yield from original(**kwargs)

    engine.token_database.process_tokens = tokens
    record = store.local.records["r", 1]
    assert store.local._spill(record, record.groups)


@pytest.mark.parametrize("prefix", [3, 4, 5, 8, 9])
def test_spill_at_prompt_chunk_boundaries(deployment, prefix):
    engine, create, _ = deployment
    store = create(prefix=prefix)
    spill(store, engine)
    for key, (raw, count) in engine.remote.items():
        assert key.start >= prefix // 4 * 4
        expected = Page((2, 3)[key.group], count, (2, 1) if key.group == 0 else (1,))
        fill(expected, key.start, key.group)
        assert torch.equal(raw, expected.raw_data)


def test_stale_ack_does_not_prevent_spill(deployment):
    engine, create, _ = deployment
    store = create()
    store.release_restore("r", 1, 6)
    assert not store.local.records["r", 1].attempts
    spill(store, engine)


def test_thread_start_failure_falls_back_without_retention_leak(
    deployment, monkeypatch
):
    from types import SimpleNamespace as NS

    engine, create, module = deployment

    def fail():
        raise RuntimeError("no thread resources")

    monkeypatch.setattr(module, "Thread", lambda **kw: NS(start=fail))
    store = create()
    assert not store.local.records and store.local.used == 0
    assert store.local.available("r", 1, 11) == 11
    assert all(p.refs == 1 for p in engine.backend.pages.values())


def test_pressure_monitor_failure_is_reported_not_silent(deployment):
    engine, create, _ = deployment
    store = create()

    def fail():
        raise RuntimeError("allocator capacity unavailable")

    engine.backend.get_allocator_capacity_bytes = fail
    store.local.wake.set()
    wait(lambda: store.local.fatal is not None)
    try:
        with pytest.raises(RuntimeError, match="restart"):
            store.poll()
        assert store.local.used > 0
    finally:
        store.local.thread.join()
        store.local.fatal = None  # No native work was started in this CPU test.


def test_mixed_c8_spill_preserves_per_layer_key_and_scale_planes(deployment):
    from types import SimpleNamespace as NS

    engine, create, _ = deployment
    original_allocate = engine.allocate_checkpoint_fragment
    layouts = ((2,), (1, 4), (2,))

    class MixedPage(Page):
        def __init__(self, count):
            self.valid_tokens, self.num_layers = count, len(layouts)
            self.offsets = [0]
            for widths in layouts:
                self.offsets.append(self.offsets[-1] + sum(widths) * count)
            self.raw_data = torch.empty(self.offsets[-1], dtype=torch.uint8)
            self.refs, self.metadata = 1, NS(fmt="test")

        def layer_tensor(self, layer):
            return self.raw_data[self.offsets[layer] : self.offsets[layer + 1]]

    def allocate(group, count, caches=None):
        if group == 0:
            return original_allocate(group, count, caches)
        page = MixedPage(count)
        engine.allocated.append(page)
        return page, layouts

    def enqueue(plan):
        rows, starts, ends, group = plan
        for layer, row in enumerate(rows):
            widths = (2, 1) if group == 0 else layouts[layer]
            for obj, start, end in zip(row, starts, ends, strict=True):
                offset = 0
                for plane, width in enumerate(widths):
                    for i in range(end - start):
                        obj.tensor[offset + i * width : offset + (i + 1) * width] = (
                            start + i + layer * 31 + plane * 7 + group * 83
                        ) % 256
                    offset += (end - start) * width
        return NS(synchronize=lambda: None)

    engine.allocate_checkpoint_fragment = allocate
    engine.gpu_connector.enqueue_group_capture = enqueue
    store = create()
    spill(store, engine)
    for key, (raw, count) in engine.remote.items():
        if key.group != 1:
            continue
        expected = MixedPage(count)
        enqueue(
            (
                [[NS(tensor=expected.layer_tensor(layer))] for layer in range(3)],
                [key.start],
                [key.end],
                1,
            )
        )
        assert torch.equal(raw, expected.raw_data)
    for key in tuple(engine.backend.pages):
        assert engine.backend.evict(key)
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    try:
        for key, (raw, _) in engine.remote.items():
            assert torch.equal(engine.backend.pages[key].raw_data, raw)
    finally:
        for page in owners:
            page.ref_count_down()


@pytest.mark.parametrize("outcome", ["ok", "missing", "short", "transport", "unknown"])
def test_actual_ascend_remote_page_bridge_owns_terminal_destinations(
    deployment, outcome, monkeypatch
):
    from types import SimpleNamespace as NS

    engine, _, _ = deployment
    unknown = sys.modules[
        "lmcache.v1.remote_fill.native"
    ].NativeExternalPageTransferUnknownError
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"load_checkpoint_page", "persist_checkpoint_page", "probe_checkpoint_pages"},
        object,
        NativeExternalPageTransferUnknownError=unknown,
    )
    obj = cls()
    obj.__dict__.update(
        {
            k: v
            for k, v in vars(engine).items()
            if k
            not in {
                "load_checkpoint_page",
                "persist_checkpoint_page",
                "probe_checkpoint_pages",
            }
        }
    )
    monkeypatch.setattr(
        Page,
        "layer_size_bytes",
        lambda p, layer: p.layer_tensor(layer).numel(),
        raising=False,
    )
    seen = []

    def get(*args):
        seen.append(args)
        if outcome == "ok":
            return
        if outcome == "unknown":
            raise unknown("in flight")
        error = RuntimeError(outcome)
        error.failed_pages = [("key", 1 if outcome == "short" else -1, 100)]
        raise error

    obj.storage_manager = NS(
        batched_get_external_pages=get,
        batched_external_pages_exist=lambda keys, **kw: [outcome != "missing"],
    )
    if outcome == "ok":
        page = obj.load_checkpoint_page("key", 1, 3, "r")
        assert page.refs == 1 and seen
        page.ref_count_down()
    elif outcome == "missing":
        assert obj.load_checkpoint_page("key", 1, 3, "r") is None
        assert engine.allocated[-1].refs == 0
    elif outcome == "unknown":
        with pytest.raises(unknown):
            obj.load_checkpoint_page("key", 1, 3, "r")
        assert obj._checkpoint_quarantined_owners == [engine.allocated[-1]]
        assert engine.allocated[-1].refs == 1
        engine.allocated[-1].ref_count_down()  # Fake test completion only.
    else:
        with pytest.raises(RuntimeError, match=outcome):
            obj.load_checkpoint_page("key", 1, 3, "r")
        assert engine.allocated[-1].refs == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_completed_resume_notifies_retention_only_when_enabled(enabled):
    from types import SimpleNamespace as NS

    source = (
        Path(__file__).resolve().parents[3]
        / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "request.resumed_from_preemption"
        and "owned_groups" in ast.unparse(n)
    )
    calls = []
    request = NS(
        resumed_from_preemption=True,
        req_id="r",
        load_spec=NS(checkpoint_generation=2, dsa_cold_load_generation=8),
    )
    engine = NS(
        register_shared_cpu_sparse_request=lambda *a, **kw: None,
        checkpoint_resume_admitted=lambda *a: calls.append(a),
    )
    worker = NS(
        config=NS(decode_checkpoint_writeback=enabled),
        lmcache_engine=engine,
        _completed_cold_resume=lambda _: True,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
            str(source),
            "exec",
        ),
        dict(self=worker, request=request, resumed_req_ids=set()),
    )
    assert calls == ([("r", 2, 8)] if enabled else [])
