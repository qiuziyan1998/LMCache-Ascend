# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/control tests; native transfers are explicit test boundaries.

Run with --confcutdir=tests/standalone to avoid the NPU bootstrap.
"""

import gc
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(monkeypatch):
    for name in (
        "lmcache_ascend",
        "lmcache_ascend.v1",
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.v1",
        "lmcache.v1.remote_fill",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    native = ModuleType("lmcache.v1.remote_fill.native")
    native.NativeExternalPageTransferUnknownError = type(
        "UnknownDMA", (RuntimeError,), {}
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    control = load(
        "lmcache.integration.vllm.preemption_checkpoint",
        ROOT.parent / "LMCache-NPU/lmcache/integration/vllm/preemption_checkpoint.py",
    )
    load(
        "lmcache_ascend.v1.local_checkpoint",
        ROOT / "lmcache_ascend/v1/local_checkpoint.py",
    )
    worker = load(
        "checkpoint_worker_under_test",
        ROOT / "lmcache_ascend/v1/preemption_checkpoint.py",
    )
    return control, worker


class Page:
    def __init__(self, layers, tokens, widths):
        self.raw_data = torch.arange(
            layers * tokens * sum(widths), dtype=torch.int32
        ).to(torch.uint8)
        self.stride = tokens * sum(widths)
        self.refs = 1
        self.metadata = NS(fmt="test")
        self.valid_tokens, self.num_layers = tokens, layers

    def is_valid(self):
        return self.refs > 0

    def ref_count_up(self):
        self.refs += 1

    def get_dtype(self):
        return torch.uint8

    def layer_data_ptr(self, layer):
        return self.raw_data.data_ptr() + layer * self.stride

    def ref_count_down(self):
        self.refs -= 1
        assert self.refs >= 0

    def layer_tensor(self, layer):
        return self.raw_data[layer * self.stride : (layer + 1) * self.stride]


def test_generation_and_completion_are_not_max_frontiers(api):
    control, _ = api
    capture = control.CaptureSpec("r", 3, 0, 20, 0, ((1,), (2,)))
    state = control.PendingCheckpoint(capture)
    state.accept(control.CheckpointResult("r", 2, "captured", 20))
    assert state.status == "capturing"
    state.accept(control.CheckpointResult("r", 3, "captured", 20))
    state.status, state.end = "persisting", 17
    state.accept(control.CheckpointResult("r", 3, "ready", 18))
    assert state.status == "failed"
    state.accept(control.CheckpointResult("r", 3, "ready", 17))
    assert state.status == "failed"
    assert control.choose_checkpoint_end(18, capture) == 17
    assert control.choose_checkpoint_end(30, capture) == 20


class LocalCPU:
    def __init__(self):
        self.pages = {}
        self.before_put = lambda: None

    def batched_submit_layer_pages(self, keys, pages):
        self.before_put()
        for key, page in zip(keys, pages, strict=True):
            if key in self.pages:
                self.pages[key].ref_count_down()
            page.ref_count_up()
            self.pages[key] = page

    def batched_get_layer_page_prefix(self, keys):
        result = []
        for key in keys:
            if key not in self.pages:
                break
            page = self.pages[key]
            page.ref_count_up()
            result.append(page)
        return result, len(result)

    def evict(self, key):
        page = self.pages[key]
        if page.refs != 1:
            return False
        del self.pages[key]
        page.ref_count_down()
        return True

    def remove(self, key):
        page = self.pages.pop(key, None)
        if page is not None:
            page.ref_count_down()


from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    group: int
    start: int
    end: int
    tokens: tuple

    def split_layers(self, layers):
        return [self] * layers

    def without_layer(self):
        return self


def fill(page, start, group):
    widths = (2, 1) if group == 0 else (1,)
    for layer in range(page.num_layers):
        row, offset = page.layer_tensor(layer), 0
        for plane, width in enumerate(widths):
            for i in range(page.valid_tokens):
                row[offset + i * width : offset + (i + 1) * width] = (
                    start + i + layer * 31 + plane * 7 + group * 83
                ) % 256
            offset += page.valid_tokens * width


def fake_engine():
    backend = LocalCPU()
    allocated, calls = [], []

    def tokens(*, tokens, request_configs=None, kv_group=0):
        for start in range(0, len(tokens), 4):
            end = min(start + 4, len(tokens))
            yield start, end, Key(kv_group, start, end, tuple(tokens[:end]))

    def allocate(group, length, caches=None):
        widths = (2, 1) if group == 0 else (1,)
        page = Page(2, length, widths)
        allocated.append(page)
        return page, widths

    def prefix(tokens, group, configs):
        length = len(tokens) % 4 or 4
        page, _ = allocate(group, length)
        fill(page, len(tokens) - length, group)
        return page

    def cached_prefix(key, group, length):
        pages, count = backend.batched_get_layer_page_prefix([key])
        if count:
            if pages[0].valid_tokens == length:
                return pages[0], (2, 1) if group == 0 else (1,)
            pages[0].ref_count_down()
        return None

    def prepare(rows, starts, ends, **kw):
        calls.append(("prepare", kw["kv_group"], len(starts)))
        return rows, starts, ends, kw["kv_group"]

    def enqueue(plan):
        rows, starts, ends, group = plan
        calls.append(("enqueue", group))
        widths = (2, 1) if group == 0 else (1,)
        for layer, row in enumerate(rows):
            for obj, start, end in zip(row, starts, ends, strict=True):
                offset = 0
                for plane, width in enumerate(widths):
                    for i in range(end - start):
                        obj.tensor[offset + i * width : offset + (i + 1) * width] = (
                            start + i + layer * 31 + plane * 7 + group * 83
                        ) % 256
                    offset += (end - start) * width
        return NS(synchronize=lambda: calls.append(("fence",)))

    return NS(
        config=NS(store_async_max_queue_size=2, blocking_timeout_secs=10, chunk_size=4),
        checkpoint_backend=lambda: backend,
        validate_checkpoint_page=lambda group, page: None,
        is_checkpoint_page_key=lambda key: isinstance(key, tuple),
        checkpoint_page_key=lambda spec, group, start, end: (
            "local",
            spec.req_id,
            spec.generation,
            group,
            start,
            end,
        ),
        token_database=NS(process_tokens=tokens),
        num_layers=2,
        is_frozen=lambda: False,
        allocate_checkpoint_fragment=allocate,
        load_checkpoint_prefix=prefix,
        get_checkpoint_prefix=cached_prefix,
        reclaim_checkpoint_capacity=lambda tokens, groups: False,
        gpu_connector=NS(
            prepare_group_capture=prepare,
            enqueue_group_capture=enqueue,
            finish_checkpoint_capture=lambda: calls.append(("failure-fence",)),
        ),
        allocated=allocated,
        calls=calls,
        backend=backend,
    )


def finish(worker):
    until = time.monotonic() + 5
    results = list(worker.poll())
    while worker.jobs and time.monotonic() < until:
        results.extend(worker.poll())
        time.sleep(0.001)
    assert not worker.jobs
    return results


def start_capture(
    api, monkeypatch, engine=None, end=13, prefix=3, resident=None, state=None
):
    control, module = api
    engine = engine or fake_engine()
    original_tensor = torch.tensor

    def tensor(values, **kw):
        kw.pop("pin_memory", None)
        return original_tensor(values, **kw)

    monkeypatch.setattr(module, "torch", NS(tensor=tensor, long=torch.long))
    store = module.CheckpointWorker(engine)
    blocks = tuple(range(1, (end + 3) // 4 + 1))
    spec = control.CaptureSpec(
        "r",
        1,
        prefix // 4 * 4,
        end,
        prefix if resident is None else resident,
        (blocks, blocks),
        prefix_end=prefix,
    )
    store.capture(spec, {0: [1], 1: [2]}, 4, state)
    return store, spec, engine


def publish(api, store, end):
    control, _ = api
    store.seal(control.SealSpec("r", 1, tuple(range(end))))
    results = finish(store)
    assert [(r.status, r.end) for r in results] == [("ready", end)]


def test_fragmented_capture_uses_chunk_pages_two_submissions_one_fence(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    assert [(r.status, r.end) for r in store.poll()] == [("captured", 13)]
    assert all(p.valid_tokens <= 4 for p in engine.allocated)
    assert [c[0] for c in engine.calls] == [
        "prepare",
        "prepare",
        "enqueue",
        "enqueue",
        "fence",
    ]
    publish(api, store, 12)
    assert all(p.refs in (0, 1) for p in engine.allocated)
    assert store.local.available("r", 1, 12) == 12
    store.close()


@pytest.mark.parametrize("failed_group", [0, 1])
def test_partial_allocation_preserves_completed_pairs(api, monkeypatch, failed_group):
    engine = fake_engine()
    original, reclaims = engine.allocate_checkpoint_fragment, []
    counts = [0, 0]

    def allocate(group, length, caches=None):
        counts[group] += 1
        if group == failed_group and counts[group] >= 3:
            raise MemoryError("pressure")
        return original(group, length, caches)

    engine.allocate_checkpoint_fragment = allocate
    engine.reclaim_checkpoint_capacity = (
        lambda n, g: reclaims.append((n, set(g))) or False
    )
    store, _, engine = start_capture(api, monkeypatch, engine)
    result = store.poll()[0]
    assert result.status == "captured" and result.end == 8
    assert len(reclaims) == 1
    publish(api, store, 8)
    assert store.local.available("r", 1, 8) == 8
    assert all(p.refs <= 1 for p in engine.allocated)
    store.close()


def test_waiting_checkpoint_is_evictable_and_one_group_hole_shortens_frontier(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    key = ("local", "r", 1, 1, 8, 12)
    assert engine.backend.evict(key)
    assert store.local.available("r", 1, 12) == 8
    assert all(p.refs <= 1 for p in engine.allocated)
    store.close()


def test_acquired_sources_cannot_be_evicted_until_restore_releases(api, monkeypatch):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    manifest = store.local.manifest("r", 1)
    end, held = store.local.acquire(manifest, 12)
    assert end == 12
    assert not engine.backend.evict(held[0][0][0].key)
    key = held[0][0][0].key
    store.local.release(held)
    assert engine.backend.evict(key)
    store.close()


def test_local_restore_normalizes_boundary_and_rejected_speculative_tail(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch, end=14)
    store.poll()
    publish(api, store, 11)
    base, owners = store.local.normalize("r", 1, list(range(11)), None)
    assert base == 0
    for group in (0, 1):
        for start, end, key in engine.token_database.process_tokens(
            tokens=list(range(11)), kv_group=group
        ):
            actual = engine.backend.pages[key]
            expected = Page(2, end - start, (2, 1) if group == 0 else (1,))
            fill(expected, start, group)
            assert torch.equal(actual.raw_data, expected.raw_data)
    for page in owners:
        page.ref_count_down()
    store.close()


def test_eviction_between_probe_and_acquire_refuses_restore_without_leaking(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    assert store.local.available("r", 1, 12) == 12
    engine.backend.evict(("local", "r", 1, 0, 4, 8))
    with pytest.raises(ValueError, match="evicted"):
        store.local.normalize("r", 1, list(range(12)), None)
    assert all(p.refs <= 1 for p in engine.allocated)
    store.close()


def test_generation_and_history_mismatch_cannot_restore(api, monkeypatch):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    assert store.local.available("r", 2, 12) == 0
    with pytest.raises(ValueError, match="history"):
        store.local.normalize("r", 1, [99] * 12, None)
    store.close()


def test_cancel_during_local_publication_does_not_release_live_owners(api, monkeypatch):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    started, release = threading.Event(), threading.Event()
    engine.backend.before_put = lambda: (started.set(), release.wait(5))
    store.seal(api[0].SealSpec("r", 1, tuple(range(12))))
    assert started.wait(2)
    store.cancel("r")
    assert any(p.refs for p in engine.allocated)
    release.set()
    assert not finish(store)
    assert store.local.manifest("r", 1) is None
    store.close()


def test_capture_fence_failure_quarantines_sources(api, monkeypatch):
    engine = fake_engine()
    engine.gpu_connector.enqueue_group_capture = lambda plan: NS(
        synchronize=lambda: (_ for _ in ()).throw(RuntimeError("device"))
    )
    engine.gpu_connector.finish_checkpoint_capture = lambda: (_ for _ in ()).throw(
        RuntimeError("still running")
    )
    with pytest.raises(RuntimeError, match="retained"):
        start_capture(api, monkeypatch, engine)
    assert all(p.refs == 1 for p in engine.allocated)


def test_unsealed_capture_deadline_releases_without_normal_decode_polling(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    store.timeout = 0
    store.jobs["r", 1].started -= 1
    results = store.poll()
    assert results[0].status == "failed"
    assert not store.jobs and all(p.refs == 0 for p in engine.allocated)
    store.close()


def test_freeze_before_local_publication_and_gc_disabled_cleanup(api, monkeypatch):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    engine.is_frozen = lambda: True
    enabled = gc.isenabled()
    gc.disable()
    try:
        store.seal(api[0].SealSpec("r", 1, tuple(range(12))))
        assert [r.status for r in finish(store)] == ["failed"]
        assert all(p.refs == 0 for p in engine.allocated)
    finally:
        if enabled:
            gc.enable()
        store.close()


def test_smaller_final_fragment_uses_fragmented_space_without_discarding_progress(
    api, monkeypatch
):
    engine = fake_engine()
    original = engine.allocate_checkpoint_fragment
    engine.allocate_checkpoint_fragment = lambda group, n, caches=None: (
        original(group, n, caches)
        if n <= 2
        else (_ for _ in ()).throw(MemoryError("fragmented"))
    )
    store, _, _ = start_capture(api, monkeypatch, engine, prefix=0)
    assert [(r.status, r.end) for r in store.poll()] == [("captured", 2)]
    publish(api, store, 2)
    assert store.local.available("r", 1, 2) == 2
    store.close()


def test_repeated_preemption_reuses_local_group0_and_recaptures_resident_group1(
    api, monkeypatch
):
    control, module = api
    first, _, engine = start_capture(api, monkeypatch, end=14)
    first.poll()
    publish(api, first, 11)
    _, held = first.local.normalize("r", 1, list(range(11)), None)
    # Model's active Group-0 CPU sources remain held; checkpoint does not evict them.
    sources = list(
        engine.token_database.process_tokens(tokens=list(range(11)), kv_group=0)
    )
    state = NS(
        cached_starts=[a for a, b, k in sources],
        cached_ends=[b for a, b, k in sources],
        cached_keys=[[k for a, b, k in sources]],
    )
    second = module.CheckpointWorker(engine)
    blocks = (tuple(range(1, 5)),) * 2
    second.capture(
        control.CaptureSpec("r", 2, 0, 15, 11, blocks, prefix_end=3),
        {0: [1], 1: [2]},
        4,
        state,
    )
    assert second.poll()[0].status == "captured"
    second.seal(control.SealSpec("r", 2, tuple(range(14))))
    assert [(x.status, x.end) for x in finish(second)] == [("ready", 14)]
    _, restored = second.local.normalize("r", 2, list(range(14)), None)
    for group in (0, 1):
        for a, b, key in engine.token_database.process_tokens(
            tokens=list(range(14)), kv_group=group
        ):
            page = engine.backend.pages[key]
            expected = Page(2, b - a, (2, 1) if group == 0 else (1,))
            fill(expected, a, group)
            assert torch.equal(page.raw_data, expected.raw_data)
    for page in restored + held:
        page.ref_count_down()
    first.close()
    second.close()


def test_invalid_block_table_refuses_before_allocation_or_device_work(api, monkeypatch):
    control, module = api
    engine = fake_engine()
    store = module.CheckpointWorker(engine)
    store.capture(
        control.CaptureSpec("r", 1, 0, 8, 0, ((1, 2), (1, 0))), {0: [1], 1: [2]}, 4
    )
    assert store.poll()[0].status == "failed"
    assert not engine.allocated and not engine.calls
    store.close()


def test_boundary_assembly_reclaims_once_and_keeps_acquired_sources_protected(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    original = engine.allocate_checkpoint_fragment
    attempts = []
    reclaims = []

    def allocate(group, n, caches=None):
        attempts.append((group, n))
        if len(attempts) == 1:
            raise MemoryError("capacity")
        return original(group, n, caches)

    def reclaim(n, groups):
        assert all(p.refs >= 2 for p in engine.backend.pages.values())
        reclaims.append((n, groups))
        return True

    engine.allocate_checkpoint_fragment = allocate
    engine.reclaim_checkpoint_capacity = reclaim
    _, owners = store.local.normalize("r", 1, list(range(12)), None)
    assert len(reclaims) == 1
    for page in owners:
        page.ref_count_down()
    store.close()


@pytest.mark.parametrize("resident", [0, 2, 3])
def test_prefix_remap_boundary_can_precede_original_prompt_without_fabricating_a_storage_key(
    api, monkeypatch, resident
):
    store, _, engine = start_capture(api, monkeypatch, resident=resident)
    store.poll()
    publish(api, store, 11)
    original = engine.load_checkpoint_prefix
    prefix_reads = []

    def prefix(tokens, group, configs):
        prefix_reads.append(len(tokens))
        return original(tokens, group, configs)

    engine.load_checkpoint_prefix = prefix
    _, owners = store.local.normalize("r", 1, list(range(11)), None)
    assert all(n == 3 for n in prefix_reads)
    for group in (0, 1):
        for a, b, key in engine.token_database.process_tokens(
            tokens=list(range(11)), kv_group=group
        ):
            expected = Page(2, b - a, (2, 1) if group == 0 else (1,))
            fill(expected, a, group)
            assert torch.equal(engine.backend.pages[key].raw_data, expected.raw_data)
    for page in owners:
        page.ref_count_down()
    store.close()


def test_duplicate_partial_capture_reports_the_actual_captured_frontier(
    api, monkeypatch
):
    engine = fake_engine()
    original = engine.allocate_checkpoint_fragment
    engine.allocate_checkpoint_fragment = lambda group, n, caches=None: (
        original(group, n, caches)
        if n <= 2
        else (_ for _ in ()).throw(MemoryError("fragmented"))
    )
    store, spec, _ = start_capture(api, monkeypatch, engine, prefix=0)
    assert store.poll()[0].end == 2
    store.capture(spec, {0: [1], 1: [2]}, 4)
    assert store.poll()[0].end == 2
    store.close()


def test_restore_sources_wait_for_all_worker_ack_even_on_cancel_and_failed_local_load(
    api,
):
    _, module = api
    store = module.CheckpointWorker(fake_engine())
    page = Page(2, 4, (2, 1))
    store.begin_restore("r", 1, 7)
    store.hold_restore("r", 1, 7, [page])
    store.cancel("r")
    assert page.refs == 1 and store.restore_owners
    with pytest.raises(RuntimeError, match="acknowledgement"):
        store.close()
    store.release_restore("r", 1, 6)  # Stale completion cannot free the active attempt.
    assert page.refs == 1
    store.release_restore("r", 1, 7)
    assert page.refs == 0 and not store.restore_owners
    store.release_restore("r", 1, 7)  # Duplicate ack is harmless.
    store.close()
