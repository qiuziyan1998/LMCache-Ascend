# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of production queue, page leases and backend publication."""

from concurrent.futures import Future
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_prefill_direct_fallback import engine, production_class


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "layerwise_cpu_fill", ROOT / "lmcache_ascend/v1/layerwise_cpu_fill.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Queue = module.LayerwisePutQueue
Lease = module.LayerwiseCPUFillLease


class Page:
    def __init__(self, pointer=1000, device="cpu"):
        self.refs = 1
        self.raw_data = NS(
            device=NS(type=device),
            untyped_storage=lambda: NS(data_ptr=lambda: pointer),
        )
        self.num_layers = 3
        self.layer_size = 16
        self.valid_tokens = 4
        self.metadata = NS(dtypes=["bf16"] * 3)
        self.pointer = pointer

    def is_valid(self):
        return self.refs > 0

    def get_size(self):
        return self.layer_size * self.num_layers

    def layer_data_ptr(self, layer):
        return self.pointer + layer * self.layer_size

    def ref_count_up(self):
        assert self.refs > 0
        self.refs += 1

    def ref_count_down(self):
        assert self.refs > 0
        self.refs -= 1


def test_lease_owns_allocator_refs_not_payload_copies():
    page = Page()
    lease = Lease((page, page))
    assert page.refs == 2
    page.ref_count_down()  # the producer gives up its own reference
    assert page.is_valid()
    lease.release()
    lease.release()
    assert page.refs == 0


def test_partial_lease_failure_rolls_back():
    good, bad = Page(), Page(device="npu")
    with pytest.raises(ValueError, match="CPU pages"):
        Lease((good, bad))
    assert good.refs == bad.refs == 1


def test_poll_and_reserve_below_limits_do_not_wait(monkeypatch):
    monkeypatch.setattr(module, "wait", lambda *a, **k: pytest.fail("unexpected wait"))
    queue = Queue(100, 4, 0)
    future = Future()
    queue.reserve(25)
    queue.add(25, [future])
    queue.poll()
    queue.reserve(25)
    assert queue.pending_bytes == 25
    future.set_result(None)
    queue.poll()
    assert queue.pending_bytes == 0


@pytest.mark.parametrize("max_bytes,max_batches", [(40, 4), (100, 1)])
def test_only_full_queue_applies_backpressure(monkeypatch, max_bytes, max_batches):
    queue = Queue(max_bytes, max_batches, 10)
    old = Future()
    queue.add(25, [old])
    calls = []

    def complete(futures, timeout):
        calls.append(tuple(futures))
        assert timeout > 0
        old.set_result(None)
        return {old}, set()

    monkeypatch.setattr(module, "wait", complete)
    queue.reserve(25)
    assert calls == [(old,)]
    assert queue.pending_bytes == 0


def test_one_oversized_batch_allowed_but_not_two():
    queue = Queue(100, 4, 0)
    queue.reserve(200)
    queue.add(200, [Future()])
    with pytest.raises(TimeoutError):
        queue.reserve(1)


def test_later_failed_put_is_not_hidden_behind_earlier_pending_put():
    queue = Queue(100, 4, 0)
    first, second = Future(), Future()
    queue.add(10, [first])
    queue.add(10, [second])
    error = ValueError("remote failed")
    second.set_exception(error)
    for action in (queue.poll, queue.drain, lambda: queue.reserve(1)):
        with pytest.raises(ValueError) as caught:
            action()
        assert caught.value is error
    assert queue.pending_bytes == 20


def test_submission_error_stays_sticky_even_with_no_future():
    queue = Queue(100, 4, 0)
    error = RuntimeError("submit failed")
    queue.fail(error)
    queue.fail(RuntimeError("later"))
    with pytest.raises(RuntimeError) as caught:
        queue.drain()
    assert caught.value is error


def test_completed_later_put_does_not_consume_capacity_behind_slow_put():
    queue = Queue(50, 2, 0)
    slow, fast = Future(), Future()
    queue.add(25, [slow])
    queue.add(25, [fast])
    fast.set_result(None)
    queue.reserve(25)
    assert list(queue.pending) == [(25, (slow,))]
    assert queue.pending_bytes == 25


def test_final_barrier_waits_all_and_releases_accounting(monkeypatch):
    queue = Queue(100, 4, 1)
    futures = [Future(), Future()]
    for future in futures:
        queue.add(20, [future])

    def complete(batch, timeout):
        for future in batch:
            future.set_result(None)
        return set(batch), set()

    monkeypatch.setattr(module, "wait", complete)
    queue.drain()
    assert queue.pending_bytes == 0
    assert not queue.pending


class UnknownTransfer(RuntimeError):
    pass


def test_request_barrier_does_not_wait_for_other_requests(monkeypatch):
    queue = Queue(100, 4, 1)
    first, other = Future(), Future()
    queue.add(20, [first], req_id="a", keys=("prefix-a",))
    queue.add(30, [other], req_id="b", keys=("prefix-b",))

    def complete(futures, timeout):
        assert set(futures) == {first}
        first.set_result(None)
        return {first}, set()

    monkeypatch.setattr(module, "wait", complete)
    queue.drain_requests(("a",))
    assert not other.done()
    assert queue.pending_bytes == 30
    # A second cancellation/retirement call is nonblocking.
    monkeypatch.setattr(module, "wait", lambda *a, **k: pytest.fail("duplicate wait"))
    queue.drain_requests(("a",))


@pytest.mark.parametrize("own_put", [False, True])
def test_reused_local_prefix_inherits_only_required_remote_puts(monkeypatch, own_put):
    queue = Queue(100, 8, 1)
    prefix, unrelated, own = Future(), Future(), Future()
    queue.add(10, [prefix], req_id="producer", keys=("shared",))
    queue.add(10, [unrelated], req_id="producer", keys=("unshared",))
    queue.track_keys("consumer", ("shared", "already-persisted"))
    queue.track_keys("consumer", ("shared",))  # repeated layer/group reuse
    expected = {prefix}
    if own_put:
        queue.add(10, [own], req_id="consumer", keys=("suffix",))
        expected.add(own)

    def complete(futures, timeout):
        assert set(futures) == expected
        for future in futures:
            future.set_result(None)
        return set(futures), set()

    monkeypatch.setattr(module, "wait", complete)
    queue.drain_requests(("consumer",))
    assert not unrelated.done()
    assert queue.pending_bytes == 10


def test_request_barrier_propagates_reused_prefix_failure():
    queue = Queue(100, 4, 0)
    prefix = Future()
    queue.add(10, [prefix], req_id="producer", keys=("shared",))
    queue.track_keys("consumer", ("shared",))
    prefix.set_exception(ValueError("shared remote put failed"))
    with pytest.raises(ValueError, match="shared remote put failed"):
        queue.drain_requests(("consumer",))


def test_queue_poll_checks_each_batch_future_not_each_page_key():
    class CountedFuture(Future):
        checks = 0

        def done(self):
            self.checks += 1
            return super().done()

    future = CountedFuture()
    queue = Queue(100, 4, 0)
    queue.add(10, [future], req_id="r", keys=range(1024))
    queue.poll()
    assert future.checks == 1
    future.set_result(None)
    queue.poll()
    assert future.checks == 2
    assert not queue.pending


def test_dense_retrieve_tracks_existing_keys_without_rehashing():
    calls = []

    class Base:
        def _dense_retrieve_token_results(self, *args):
            calls.append(args)
            return iter(((0, 4, "shared"), (4, 8, "ready")))

    cls = production_class(
        ROOT / "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"_dense_retrieve_token_results"},
        {},
        base=Base,
    )
    obj = cls()
    queue = obj._layerwise_put_queue = Queue(100, 4, 0)
    prefix = Future()
    queue.add(10, [prefix], req_id="producer", keys=("shared",))
    rows = obj._dense_retrieve_token_results([], None, {}, 1, {"req_id": "consumer"})
    assert list(rows) == [(0, 4, "shared"), (4, 8, "ready")]
    assert len(calls) == 1
    with pytest.raises(TimeoutError):
        queue.drain_requests(("consumer",))


class Local:
    use_hot = True

    def __init__(self):
        self.pages = []

    def batched_submit_layer_pages(self, keys, pages):
        assert not self.pages, "duplicate local publication"
        self.pages.extend(pages)
        for page in pages:
            page.ref_count_up()


class Remote:
    pass


class Key:
    dtype = "bf16"


def storage_manager():
    path = ROOT.parent / "LMCache/lmcache/v1/storage_backend/storage_manager.py"
    cls = production_class(
        path,
        "StorageManager",
        {"batched_put_layer_pages"},
        dict(
            Future=Future,
            LocalCPUBackend=Local,
            RemoteBackend=Remote,
            mooncake_valid_tokens=lambda key, chunk: 4,
            NativeExternalPageTransferUnknownError=UnknownTransfer,
        ),
    )
    obj, local, remote_future = cls(), Local(), Future()
    obj.config = NS(chunk_size=4)
    obj.get_active_storage_backends = lambda **kw: [
        ("local", local),
        ("remote", Remote()),
    ]
    obj._supports_layer_page_backends = lambda backends: True
    obj.batched_put_external_pages = lambda *a: remote_future
    return obj, local, remote_future


@pytest.mark.parametrize("early", [False, True])
def test_local_ready_and_remote_persisted_are_separate(early):
    obj, local, remote = storage_manager()
    page = Page()
    (completion,) = obj.batched_put_layer_pages(
        [Key()], [page], publish_local_early=early
    )
    assert bool(local.pages) is early
    assert not completion.done()
    assert page.refs == (
        2 if early else 1
    )  # remote + optional local, producer released
    remote.set_result(None)
    assert completion.result() is None
    assert local.pages == [page]
    assert page.refs == 1


def test_failed_remote_does_not_turn_early_local_hit_into_success():
    obj, local, remote = storage_manager()
    page = Page()
    futures = obj.batched_put_layer_pages([Key()], [page], publish_local_early=True)
    queue = Queue(100, 4, 0)
    queue.add(page.get_size(), futures)
    remote.set_exception(ValueError("put failed"))
    assert local.pages == [page]
    with pytest.raises(ValueError, match="put failed"):
        queue.drain()
    assert page.refs == 1


@pytest.mark.parametrize("sync", [False, True])
def test_unknown_dma_retains_allocator_source_refs(sync):
    obj, local, remote = storage_manager()
    page = Page()
    error = UnknownTransfer("completion unknown")
    if sync:

        def fail(*args):
            raise error

        obj.batched_put_external_pages = fail
        with pytest.raises(UnknownTransfer):
            obj.batched_put_layer_pages([Key()], [page], publish_local_early=True)
        page.ref_count_down()  # caller's failed-submission cleanup
    else:
        (future,) = obj.batched_put_layer_pages(
            [Key()], [page], publish_local_early=True
        )
        remote.set_exception(error)
        with pytest.raises(UnknownTransfer):
            future.result()
    local.pages.pop().ref_count_down()  # even eviction cannot release live DMA memory
    assert page.refs == 1
    assert error.layerwise_source_pages == (page,)


def cpu_fill_engine(monkeypatch, groups=(0, 1)):
    obj = engine(monkeypatch)
    obj.config.remote_fill_submission_mode = "per_chunk"
    obj._remote_fill_direct_groups = lambda: groups
    events = {group: object() for group in groups}
    obj.gpu_connector = NS(
        layerwise_prefill_store_fences=lambda group: (events[group],)
    )
    # Real source-batch class and merge helper, stripped of NPU imports only.
    path = ROOT / "lmcache_ascend/v1/direct_store_plan.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *(
            node
            for node in tree.body
            if getattr(node, "name", None)
            in {"DirectPageBatch", "merge_deferred_remote_fill_batches"}
        ),
    ]
    ns = {"dataclass": dataclass, "__name__": __name__}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    obj._queue_layerwise_cpu_fill.__func__.__globals__.update(
        LayerwiseCPUFillLease=Lease,
        LayerPageMemoryObj=Page,
        _DirectPageBatch=ns["DirectPageBatch"],
        merge_deferred_remote_fill_batches=ns["merge_deferred_remote_fill_batches"],
        RemoteFillFatalError=UnknownTransfer,
    )
    return obj, events


@pytest.mark.parametrize("groups", [(0,), (0, 1)])
def test_cpu_fill_reuses_exact_completed_pages_and_real_fences(monkeypatch, groups):
    obj, events = cpu_fill_engine(monkeypatch, groups)
    pages = [Page(1000 + 100 * g) for g in groups]
    keys = []
    for group, page in zip(groups, pages):
        key = object()
        keys.append(key)
        obj._queue_layerwise_cpu_fill(
            "r",
            {"lmcache.remote_fill": "transfer"},
            group,
            [NS(without_layer=lambda k=key: k)],
            [page],
            [0],
            [4],
        )
    submitted, future = [], Future()

    def submit(state, batch, end):
        submitted.append(batch)
        state.remote_fill.last_future = future

    obj._schedule_remote_fill_batch = submit
    obj.submit_layerwise_prefill_fills(["r"])
    (batch,) = submitted
    assert batch.keys == keys
    assert batch.ptrs == [[page.pointer] for page in pages]
    assert batch.sizes == [[page.get_size()] for page in pages]
    assert all(a is b.raw_data for a, b in zip(batch.owners, pages))
    assert set(batch.ready_events) == set(events.values())
    assert all(page.refs == 2 for page in pages)
    future.set_result(None)
    assert all(page.refs == 1 for page in pages)
    assert not obj._layerwise_cpu_fill_sources


@pytest.mark.parametrize(
    "failure", ["full_queue", "native_unknown", "native_failed", "incomplete"]
)
def test_cpu_fill_failure_and_fallback_lifetime(monkeypatch, failure):
    obj, _ = cpu_fill_engine(monkeypatch, (0, 1) if failure == "incomplete" else (0,))
    page = Page()
    obj._queue_layerwise_cpu_fill(
        "r",
        {"lmcache.remote_fill": "transfer"},
        0,
        [NS(without_layer=lambda: "key")],
        [page],
        [0],
        [4],
    )
    future = Future()

    def submit(state, batch, end):
        if failure == "full_queue":
            state.remote_fill.disabled_reason = "producer_backpressure"
        else:
            state.remote_fill.last_future = future

    obj._schedule_remote_fill_batch = submit
    obj.submit_layerwise_prefill_fills(["r"])
    if failure in {"native_unknown", "native_failed"}:
        assert page.refs == 2
        future.set_exception(
            UnknownTransfer() if failure == "native_unknown" else ValueError()
        )
    assert page.refs == (2 if failure == "native_unknown" else 1)
    assert bool(obj._layerwise_cpu_fill_quarantine) == (failure == "native_unknown")


def test_bank_fences_are_borrowed_only_from_current_group_generation():
    cls = production_class(
        ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py",
        "VLLMPagedMemLayerwiseNPUConnector",
        {"layerwise_prefill_store_fences"},
        {},
    )
    obj = cls()
    assert obj.layerwise_prefill_store_fences(0) == ()
    obj._layerwise_prefill_transfer_generations = {0: 3, 1: 2}
    first, second = object(), object()
    middle = object()
    obj._layerwise_prefill_save_done_events = {
        (0, 0, 2): (3, middle),
        (0, 0, 0): (3, first),
        (0, 1, 1): (2, object()),
        (1, 0, 0): (2, second),
    }
    assert obj.layerwise_prefill_store_fences(0) == (first, middle)
    assert obj.layerwise_prefill_store_fences(1) == (second,)
