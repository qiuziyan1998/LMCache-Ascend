# SPDX-License-Identifier: Apache-2.0
"""CPU regression for asynchronous D2H surviving preemption and bank reuse."""

import ast
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def production_methods():
    """Load unchanged method bodies without importing the NPU runtime."""
    methods = {}
    sources = (
        (
            "lmcache_ascend/v1/npu_connector/npu_connectors.py",
            "VLLMPagedMemLayerwiseNPUConnector",
            {
                "wait_for_layerwise_prefill_stores",
                "release_layerwise_prefill_dma_cache",
            },
        ),
        (
            "lmcache_ascend/v1/cache_engine.py",
            "AscendLMCacheEngine",
            {"poll_layerwise_prefill_puts", "_layerwise_prefill_async_store_supported"},
        ),
        (
            "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
            "LMCacheAscendConnectorV1Impl",
            {"handle_preemptions", "_forget_layerwise_store_results"},
        ),
    )
    for relative, class_name, names in sources:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        bodies = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert {node.name for node in bodies} == names
        module = ast.parse("from __future__ import annotations")
        module.body.extend(bodies)
        scope = {"logger": logging.getLogger(__name__)}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        methods.update({name: scope[name] for name in names})
    return methods


class Stream:
    def __init__(self):
        self.pending = []
        self.completed = 0

    def submit(self, action):
        self.pending.append(action)

    def event(self):
        return Event(self, len(self.pending))


class Event:
    def __init__(self, stream, end):
        self.stream = stream
        self.end = end
        self.waits = 0
        self.fail = False

    def synchronize(self):
        self.waits += 1
        if self.fail:
            raise RuntimeError("D2H completion unknown")
        while self.stream.completed < self.end:
            self.stream.pending[self.stream.completed]()
            self.stream.completed += 1


class LocalCPUBackend:
    pass


class PutQueue:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def poll(self):
        self.calls.append("poll")

    def drain_requests(self, req_ids):
        ids = tuple(req_ids)
        self.calls.append(ids)
        # Model remote puts retaining their producer events until completion.
        for req_id in ids:
            for event in self.events.get(req_id, ()):
                event.synchronize()

    def drain(self):
        self.drain_requests(self.events)


@pytest.fixture
def make_stack(production_methods):
    def make(*, store_async=False, remote=False):
        def attach(obj, *names):
            for name in names:
                setattr(obj, name, MethodType(production_methods[name], obj))

        source = {
            (group, bank): group * 10 + bank for group in (0, 1) for bank in range(4)
        }
        original = source.copy()
        host, other_done, old_events, other_events = {}, [], {}, []
        for group in (0, 1):
            events = []
            for bank in range(4):
                key = (group, bank)
                stream = Stream()
                stream.submit(lambda key=key: host.__setitem__(key, source[key]))
                events.append(stream.event())
                # Later work on the same stream must not be drained for "old".
                stream.submit(lambda key=key: other_done.append(key))
                other_events.append(stream.event())
            old_events[group] = tuple(events)
        connector = NS(
            _layerwise_prefill_async_store_events={
                "old": old_events,
                "other": {0: other_events},
            },
            _prefill_dma_bound_loads={"old": object()},
            _prefill_dma_plan_cache={"old": object()},
            _prefill_dma_source_cache={"old": object()},
            _layerwise_prefill_load_owners={"old": {}},
            layerwise_prefill_store_fences=lambda group: (),
        )
        attach(
            connector,
            "wait_for_layerwise_prefill_stores",
            "release_layerwise_prefill_dma_cache",
        )
        queue = (
            PutQueue({"old": sum(old_events.values(), ()), "other": other_events})
            if remote
            else None
        )
        engine = NS(
            gpu_connector=connector,
            _layerwise_put_queue=queue,
            checkpoint_worker=None,
            store_location=None,
            storage_manager=NS(
                get_active_storage_backends=lambda **kw: [("cpu", LocalCPUBackend())]
            ),
            wait_for_pending_stores=lambda ids: set(),
            wait_for_direct_stores=lambda ids: None,
            drop_direct_store_states=lambda ids: None,
        )
        attach(
            engine,
            "poll_layerwise_prefill_puts",
            "_layerwise_prefill_async_store_supported",
        )
        dropped = []
        adapter = NS(
            lmcache_engine=engine,
            _force_layerwise_prefill_store=True,
            store_async=store_async,
            kv_role="kv_producer",
            _parent=NS(has_connector_metadata=lambda: False),
            _layerwise_local_store_results={
                ("old", 0): object(),
                ("other", 0): object(),
            },
            _drop_worker_retrieve_state=dropped.append,
            _unfenced_live_stores={},
        )
        attach(adapter, "handle_preemptions", "_forget_layerwise_store_results")
        return NS(
            adapter=adapter,
            engine=engine,
            connector=connector,
            queue=queue,
            source=source,
            original=original,
            host=host,
            other_done=other_done,
            old_events=old_events,
            other_events=other_events,
            dropped=dropped,
        )

    return make


@pytest.mark.parametrize("store_async", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_preemption_preserves_old_kv_before_cross_bank_reuse(
    make_stack, store_async, remote
):
    stack = make_stack(store_async=store_async, remote=remote)
    if not remote:
        assert stack.engine._layerwise_prefill_async_store_supported(
            page_first_store=False
        )
    stack.adapter.handle_preemptions({"old"})
    assert stack.dropped == ["old"]
    assert stack.other_done == []

    # Reallocated blocks can be overwritten by an independent bank stream.
    reuse = Stream()
    reuse.submit(lambda: stack.source.update(dict.fromkeys(stack.source, 999)))
    reuse.event().synchronize()
    for event in stack.other_events:
        event.synchronize()
    assert stack.host == stack.original, "old D2H copied the new request's KV"
    assert "old" not in stack.connector._layerwise_prefill_async_store_events
    assert "other" in stack.connector._layerwise_prefill_async_store_events
    assert "old" in stack.connector._prefill_dma_bound_loads
    assert ("old", 0) not in stack.adapter._layerwise_local_store_results


def test_final_poll_reuses_generator_selection_for_dma_and_remote_puts(make_stack):
    stack = make_stack()
    # Even with no matching remote future, DMA must honor the same selection.
    stack.engine._layerwise_put_queue = PutQueue({})
    stack.engine.poll_layerwise_prefill_puts(final=True, req_ids=iter(["old"]))
    assert stack.engine._layerwise_put_queue.calls == [("old",)]
    assert stack.host == stack.original
    assert not stack.other_done


@pytest.mark.parametrize("remote", [False, True])
def test_nonfinal_poll_keeps_d2h_asynchronous(make_stack, remote):
    stack = make_stack(remote=remote)
    stack.engine.poll_layerwise_prefill_puts()
    assert not stack.host and not stack.other_done
    assert all(event.waits == 0 for event in sum(stack.old_events.values(), ()))
    if remote:
        assert stack.queue.calls == ["poll"]


def test_final_poll_without_selection_drains_all_requests(make_stack):
    stack = make_stack()
    stack.engine.poll_layerwise_prefill_puts(final=True)
    assert stack.host == stack.original
    assert len(stack.other_done) == 8
    assert not stack.connector._layerwise_prefill_async_store_events


@pytest.mark.parametrize("store_async", [False, True])
def test_failed_preemption_fence_preserves_state_and_can_retry(make_stack, store_async):
    stack = make_stack(store_async=store_async)
    failed = stack.old_events[1][2]
    failed.fail = True
    with pytest.raises(RuntimeError, match="D2H completion unknown"):
        stack.adapter.handle_preemptions({"old"})
    assert not stack.dropped
    assert ("old", 0) in stack.adapter._layerwise_local_store_results
    assert (
        stack.connector._layerwise_prefill_async_store_events["old"] is stack.old_events
    )
    assert "old" in stack.connector._prefill_dma_bound_loads
    failed.fail = False
    stack.adapter.handle_preemptions({"old"})
    assert stack.host == stack.original
    assert stack.dropped == ["old"]


def test_release_keeps_owners_until_store_fences_succeed(make_stack):
    stack = make_stack()
    released = []
    stack.connector._layerwise_prefill_load_owners["old"] = {
        1: NS(is_valid=lambda: True, ref_count_down=lambda: released.append(True))
    }
    failed = stack.old_events[1][0]
    failed.fail = True
    with pytest.raises(RuntimeError, match="D2H completion unknown"):
        stack.connector.release_layerwise_prefill_dma_cache("old")
    assert not released
    assert "old" in stack.connector._layerwise_prefill_load_owners
    assert "old" in stack.connector._prefill_dma_bound_loads
    failed.fail = False
    stack.connector.release_layerwise_prefill_dma_cache("old")
    stack.connector.release_layerwise_prefill_dma_cache("old")
    assert released == [True]
    assert "old" not in stack.connector._prefill_dma_bound_loads
    assert "other" in stack.connector._layerwise_prefill_async_store_events


def test_empty_missing_and_legacy_single_event_are_safe(make_stack):
    stack = make_stack()
    stack.engine.poll_layerwise_prefill_puts(final=True, req_ids=())
    stack.engine.poll_layerwise_prefill_puts(final=True, req_ids=("missing",))
    assert not stack.host
    event = stack.old_events[0][0]
    stack.connector._layerwise_prefill_async_store_events["legacy"] = {0: event}
    stack.connector.wait_for_layerwise_prefill_stores(("legacy", "legacy"))
    assert event.waits == 1
    assert "legacy" not in stack.connector._layerwise_prefill_async_store_events
