# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/control tests; native transfers are explicit test boundaries.

Run with --confcutdir=tests/standalone to avoid the NPU bootstrap.
"""

import ctypes
import gc
import importlib.util
from pathlib import Path
import sys
import threading
import time
import weakref
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(monkeypatch):
    for name in (
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

    def get_dtype(self):
        return torch.uint8

    def layer_data_ptr(self, layer):
        return self.raw_data.data_ptr() + layer * self.stride

    def ref_count_down(self):
        self.refs -= 1
        assert self.refs >= 0

    def layer_tensor(self, layer):
        return self.raw_data[layer * self.stride : (layer + 1) * self.stride]


def read_vectors(vectors):
    ptrs, sizes = vectors
    return b"".join(ctypes.string_at(p, n) for p, n in zip(ptrs, sizes, strict=True))


def test_crops_rejected_tokens_in_every_plane_and_layer(api):
    _, worker = api
    page = Page(2, 5, (2, 1))
    actual = read_vectors(worker.fragment_vectors([(10, 15, page, (2, 1))], 11, 14, 2))
    # Layer0 K[1:4], V[1:4], then Layer1 K[1:4], V[1:4].
    assert actual == bytes(
        [2, 3, 4, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 21, 22, 26, 27, 28]
    )


def test_assembles_planes_across_cpu_prefix_and_captured_suffix(api):
    _, worker = api
    prefix, suffix = Page(1, 2, (2, 1)), Page(1, 3, (2, 1))
    suffix.raw_data.add_(20)
    actual = read_vectors(
        worker.fragment_vectors(
            [(2, 5, suffix, (2, 1)), (0, 2, prefix, (2, 1))], 0, 4, 1
        )
    )
    assert actual == bytes([0, 1, 2, 3, 20, 21, 22, 23, 4, 5, 26, 27])


@pytest.mark.parametrize("ranges", [[(0, 2), (3, 4)], [(0, 3), (2, 4)], [(0, 2)]])
def test_holes_overlaps_and_short_coverage_rejected(api, ranges):
    _, worker = api
    with pytest.raises(ValueError, match="coverage"):
        worker.fragment_vectors(
            [(a, b, Page(1, b - a, (1,)), (1,)) for a, b in ranges], 0, 4, 1
        )


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


class Storage:
    def __init__(self):
        self.payloads = []
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.error = None

    def batched_get_external_pages(self, keys, ptrs, sizes, owners, req_id):
        for p, n in zip(ptrs[0], sizes[0], strict=True):
            ctypes.memset(p, 123, n)

    def batched_put_external_pages(self, keys, ptrs, sizes, owners, events, req_id):
        from concurrent.futures import Future

        self.started.set()
        self.release.wait(5)
        self.payloads = [
            (key, read_vectors((p, n)))
            for key, p, n in zip(keys, ptrs, sizes, strict=True)
        ]
        f = Future()
        if self.error:
            f.set_exception(self.error)
        else:
            f.set_result(None)
        return f


def fake_engine():
    storage = Storage()

    def tokens(*, tokens, request_configs, kv_group):
        for start in range(0, len(tokens), 4):
            end = min(start + 4, len(tokens))
            yield start, end, (kv_group, start, end, tuple(tokens[:end]))

    def alloc(group, length, caches=None):
        return Page(1, length, (1,)), (1,)

    return NS(
        config=NS(
            store_async_max_queue_size=2,
            blocking_timeout_secs=10,
            chunk_size=4,
            get_extra_config_value=lambda name, default: default,
        ),
        storage_manager=storage,
        token_database=NS(process_tokens=tokens),
        num_layers=1,
        is_frozen=lambda: False,
        allocate_checkpoint_fragment=alloc,
    )


def finish(worker):
    until = time.monotonic() + 5
    results = []
    while worker.jobs and time.monotonic() < until:
        results.extend(worker.poll())
        time.sleep(0.001)
    assert not worker.jobs
    return results


def captured_job(api, engine):
    control, worker = api
    capture = control.CaptureSpec("r", 1, 0, 7, 3, ((1,), (2,)))
    job = worker.CaptureJob(capture)
    job.fragments = {
        0: [(3, 7, Page(1, 4, (1,)), (1,))],
        1: [(0, 7, Page(1, 7, (1,)), (1,))],
    }
    store = worker.CheckpointWorker(engine)
    store.jobs[("r", 1)] = job
    return store, job


def test_persistence_reconstructs_prefix_and_never_exports_speculative_tail(api):
    control, _ = api
    engine = fake_engine()
    store, job = captured_job(api, engine)
    pages = [f[2] for fs in job.fragments.values() for f in fs]
    store.seal(control.SealSpec("r", 1, tuple(range(6))))
    results = finish(store)
    assert [(r.status, r.end) for r in results] == [("ready", 6)]
    assert [(k[:3], v) for k, v in engine.storage_manager.payloads] == [
        ((0, 0, 4), bytes([123, 123, 123, 0])),
        ((0, 4, 6), bytes([1, 2])),
        ((1, 0, 4), bytes([0, 1, 2, 3])),
        ((1, 4, 6), bytes([4, 5])),
    ]
    assert all(p.refs == 0 for p in pages)
    store.close()


def test_cancel_during_persistence_keeps_owners_until_terminal(api):
    control, _ = api
    engine = fake_engine()
    engine.storage_manager.release.clear()
    store, job = captured_job(api, engine)
    page = job.fragments[1][0][2]
    store.seal(control.SealSpec("r", 1, tuple(range(6))))
    assert engine.storage_manager.started.wait(2)
    store.cancel("r", 1)
    assert page.refs == 1 and store.poll() == ()
    engine.storage_manager.release.set()
    assert finish(store) == []
    assert page.refs == 0
    store.close()


def test_persistent_failure_does_not_publish_ready(api):
    control, _ = api
    engine = fake_engine()
    engine.storage_manager.error = ValueError("missing group")
    store, _ = captured_job(api, engine)
    store.seal(control.SealSpec("r", 1, tuple(range(6))))
    assert [r.status for r in finish(store)] == ["failed"]
    store.close()


def test_group_capture_has_one_completion_wait_and_reuses_cpu_buffers(api, monkeypatch):
    control, worker = api
    engine = fake_engine()
    events = []
    allocations = []
    allocate = engine.allocate_checkpoint_fragment

    def alloc(*args, **kwargs):
        result = allocate(*args, **kwargs)
        allocations.append(result[0])
        return result

    engine.allocate_checkpoint_fragment = alloc

    # Native/pinned allocation boundary; all orchestration remains production.
    def tensor(values, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch.tensor(values, **kwargs)

    monkeypatch.setattr(worker, "torch", NS(tensor=tensor, long=torch.long))

    def prepare(rows, starts, ends, **kwargs):
        events.append("prepare")
        return rows

    def enqueue(plan):
        events.append("enqueue")
        return NS(synchronize=lambda: events.append("complete"))

    engine.gpu_connector = NS(
        prepare_group_capture=prepare,
        enqueue_group_capture=enqueue,
        finish_checkpoint_capture=lambda: events.append("exception_fence"),
    )
    store = worker.CheckpointWorker(engine)
    for generation in (1, 2):
        spec = control.CaptureSpec("r", generation, 0, 6, 0, ((1, 2), (3, 4)))
        store.capture(spec, {0: [1], 1: [2]}, 4)
        assert store.poll()[0].status == "captured"
        store.cancel("r", generation)
    assert events == ["prepare", "prepare", "enqueue", "enqueue", "complete"] * 2
    assert len(allocations) == 2
    assert all(page.refs == 1 for page in allocations)
    store.close()
    assert all(page.refs == 0 for page in allocations)


def test_busy_capture_refusal_does_not_synchronize_device(api):
    control, worker = api
    engine = fake_engine()
    engine.gpu_connector = NS(
        finish_checkpoint_capture=lambda: pytest.fail("unexpected device sync")
    )
    store = worker.CheckpointWorker(engine)
    store.max_jobs = 0
    store.capture(
        control.CaptureSpec("r", 1, 0, 6, 0, ((1, 2), (3, 4))), {0: [1], 1: [2]}, 4
    )
    assert store.poll()[0].status == "failed"
    store.close()


def test_unsealed_capture_deadline_progresses_without_model_tokens(api):
    _, _worker = api
    store, job = captured_job(api, fake_engine())
    page = job.fragments[1][0][2]
    job.started -= store.timeout + 1
    assert store.poll()[0].status == "failed"
    assert page.refs == 0 and not store.jobs
    store.close()


def test_failed_checkpoint_retires_job_with_gc_disabled(api):
    control, _ = api
    engine = fake_engine()
    engine.storage_manager.error = ValueError("store failure")
    enabled = gc.isenabled()
    gc.disable()
    try:
        store, job = captured_job(api, engine)
        ref = weakref.ref(job)
        store.seal(control.SealSpec("r", 1, tuple(range(6))))
        assert [r.status for r in finish(store)] == ["failed"]
        del job
        store.close()
        assert ref() is None
    finally:
        if enabled:
            gc.enable()


def test_quarantined_capture_cannot_be_cancelled_away_before_shutdown(api):
    store, job = captured_job(api, fake_engine())
    page = job.fragments[1][0][2]
    job.quarantined = True
    try:
        store.cancel("r", 1)
        assert ("r", 1) in store.jobs
        with pytest.raises(RuntimeError, match="unresolved"):
            store.close()
        assert page.refs == 1
    finally:
        job.quarantined = False
        store.cancel("r", 1)
        store.close()


def test_second_group_oom_refuses_before_any_device_preparation(api, monkeypatch):
    control, worker = api
    engine = fake_engine()
    allocation = engine.allocate_checkpoint_fragment

    def allocate(group, *args, **kwargs):
        if group == 1:
            raise MemoryError("indexer staging full")
        return allocation(group, *args, **kwargs)

    engine.allocate_checkpoint_fragment = allocate
    calls = []
    engine.gpu_connector = NS(
        prepare_group_capture=lambda *a, **kw: calls.append("prepare"),
        finish_checkpoint_capture=lambda: calls.append("sync"),
    )

    def tensor(values, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch.tensor(values, **kwargs)

    monkeypatch.setattr(worker, "torch", NS(tensor=tensor, long=torch.long))
    store = worker.CheckpointWorker(engine)
    store.capture(
        control.CaptureSpec("r", 1, 0, 6, 0, ((1, 2), (3, 4))), {0: [1], 1: [2]}, 4
    )
    assert store.poll()[0].status == "failed"
    assert calls == []
    store.close()


def test_freeze_before_persistence_prevents_checkpoint_writes(api):
    control, _ = api
    engine = fake_engine()
    engine.is_frozen = lambda: True
    store, _ = captured_job(api, engine)
    store.seal(control.SealSpec("r", 1, tuple(range(6))))
    assert [r.status for r in finish(store)] == ["failed"]
    assert not engine.storage_manager.started.is_set()
    store.close()
