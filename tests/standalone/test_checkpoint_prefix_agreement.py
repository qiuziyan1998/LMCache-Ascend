# SPDX-License-Identifier: Apache-2.0
"""Exercise ordered TP checkpoint agreement with real mailbox methods."""

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue
from threading import Barrier, Event
from types import SimpleNamespace as NS
import threading
import time
from weakref import proxy

import pytest

ROOT = Path(__file__).resolve().parents[2]


def classes():
    ns = dict(
        Future=Future,
        replace=replace,
        proxy=proxy,
        dataclass=dataclass,
        threading=threading,
        time=time,
        serving_perf_enabled=lambda: False,
        logger=NS(debug=lambda *a, **kw: None),
    )
    ln = ROOT.parent / "LMCache-NPU"
    names = {
        "_shared_envelope_identity",
        "_shared_envelope_mailbox",
        "_receive_matching_shared_envelope",
        "_validate_shared_layerwise_envelope",
        "_shared_layerwise_error_envelope",
    }
    tree = ast.parse((ln / "lmcache/v1/cache_engine.py").read_text(encoding="utf-8"))
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    envelope = next(
        n
        for n in ast.parse(
            (ln / "lmcache/v1/shared_cpu_cache.py").read_text(encoding="utf-8")
        ).body
        if isinstance(n, ast.ClassDef) and n.name == "SharedHandleEnvelope"
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    base = ast.ClassDef(
        name="Base", bases=[], keywords=[], body=nodes, decorator_list=[]
    )
    source = ast.parse(
        (ROOT / "lmcache_ascend/v1/cache_engine.py").read_text(encoding="utf-8")
    )
    engine = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendLMCacheEngine"
    )
    engine.bases = [ast.Name(id="Base", ctx=ast.Load())]
    engine.body = [
        n
        for n in engine.body
        if getattr(n, "name", None)
        in {
            "enable_checkpoint_prefix_agreement",
            "_checkpoint_prefix_result",
            "_receive_checkpoint_envelope",
            "finish_checkpoint_prefix",
            "_ordered_checkpoint_agreement",
            "_checkpoint_remote_fill_agreement",
        }
    ]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, envelope, base, engine], type_ignores=[])
            ),
            "checkpoint_agreement",
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize("ready", [True, False])
def test_marker_cannot_be_overtaken_by_an_unrelated_broadcast(ready):
    ns = classes()
    cls = ns["AscendLMCacheEngine"]
    queues = [Queue() for _ in range(2)]
    reduction = Barrier(2)
    flags = [None, None]
    events = []
    prefix_received = Event()

    def read(obj):
        envelope = queues[obj.rank].get(timeout=3)
        events.append((obj.rank, "receive", envelope.phase))
        if envelope.phase == "checkpoint_index_prefix":
            prefix_received.set()
        return envelope

    ns["Base"]._receive_shared_envelope = read
    engines = [cls(), cls()]
    for rank, engine in enumerate(engines):
        engine.rank = rank
        engine.metadata = NS(is_first_rank=lambda rank=rank: rank == 0)
        engine.shared_cpu_cache_generation = 1
        engine.shared_cpu_cache_strict = True
        engine.config = NS(blocking_timeout_secs=3)
        engine.enable_checkpoint_prefix_agreement()

        def agree(value, rank=rank):
            events.append((rank, "reduce", value))
            flags[rank] = value
            reduction.wait(timeout=3)
            return all(flags)

        engine.collective_all_true_fn = agree
    leader, passive = engines

    def broadcast(envelope):
        events.append((0, "broadcast", envelope.phase))
        queues[1].put(envelope)

    leader._broadcast_shared_envelope = broadcast
    request = NS(req_id="r", load_spec=NS(dsa_cold_load_generation=7))
    other = dict(
        req_id="other", phase="ordinary", request_ordinal=0, layer_id=0, kv_group=0
    )

    def ordinary_send():
        condition, *_ = leader._shared_envelope_mailbox()
        with condition:
            broadcast(
                replace(
                    leader._shared_layerwise_error_envelope(**other, message=""),
                    status="skipped",
                )
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        # An unrelated receiver, not the checkpoint worker, encounters the marker.
        receive = pool.submit(passive._receive_matching_shared_envelope, **other)
        first = pool.submit(leader.finish_checkpoint_prefix, request, True)
        assert prefix_received.wait(2)
        send = pool.submit(ordinary_send)
        second = pool.submit(passive.finish_checkpoint_prefix, request, ready)
        assert first.result(3) == ready and second.result(3) == ready
        send.result(3)
        assert receive.result(3).phase == "ordinary"
    order = [entry[2] for entry in events if entry[1] == "broadcast"]
    assert order == ["checkpoint_index_prefix", "ordinary"]
    assert not passive._checkpoint_prefix_results


def test_unrelated_envelope_does_not_call_consensus():
    ns = classes()
    cls = ns["AscendLMCacheEngine"]
    envelope = NS(phase="ordinary")
    ns["Base"]._receive_shared_envelope = lambda self: envelope
    obj = cls()
    obj.collective_all_true_fn = lambda *a: pytest.fail(
        "ordinary receive entered agreement"
    )
    obj.enable_checkpoint_prefix_agreement()
    assert obj._receive_shared_envelope() is envelope


def test_armed_transport_does_not_create_an_engine_cycle_with_gc_disabled():
    import gc, weakref

    ns = classes()
    cls = ns["AscendLMCacheEngine"]
    ns["Base"]._receive_shared_envelope = lambda self: None
    enabled = gc.isenabled()
    gc.disable()
    try:
        engine = cls()
        engine.enable_checkpoint_prefix_agreement()
        ref = weakref.ref(engine)
        del engine
        assert ref() is None
    finally:
        if enabled:
            gc.enable()


def test_remote_fill_and_checkpoint_reductions_follow_leader_order():
    ns = classes()
    cls = ns["AscendLMCacheEngine"]
    queue = Queue()
    barrier = Barrier(2)
    flags = [None, None]
    sent = Event()
    events = []
    ns["Base"]._receive_shared_envelope = lambda self: queue.get(timeout=3)
    leader, passive = cls(), cls()
    for rank, obj in enumerate((leader, passive)):
        obj.metadata = NS(is_first_rank=lambda rank=rank: rank == 0)
        obj.shared_cpu_cache_generation = 1
        obj.shared_cpu_cache_strict = True
        obj.config = NS(blocking_timeout_secs=3)
        obj.enable_checkpoint_prefix_agreement()

        def reduce(value, rank=rank):
            flags[rank] = value
            barrier.wait(timeout=3)
            result = all(flags)
            barrier.wait(timeout=3)
            return result

        obj.collective_all_true_fn = reduce

    def broadcast(envelope):
        events.append(envelope.phase)
        queue.put(envelope)
        sent.set()

    leader._broadcast_shared_envelope = broadcast
    request = NS(req_id="checkpoint", load_spec=NS(dsa_cold_load_generation=7))
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Passive RemoteFill is ready first; leader checkpoint is ready first.
        rf_passive = pool.submit(
            passive._remote_fill_all_ranks_materialized, True, req_id="fill", kv_group=0
        )
        checkpoint_leader = pool.submit(leader.finish_checkpoint_prefix, request, True)
        assert sent.wait(2)
        rf_leader = pool.submit(
            leader._remote_fill_all_ranks_materialized, True, req_id="fill", kv_group=0
        )
        checkpoint_passive = pool.submit(
            passive.finish_checkpoint_prefix, request, False
        )
        assert (
            checkpoint_leader.result(3) is False
            and checkpoint_passive.result(3) is False
        )
        assert rf_leader.result(3) is True and rf_passive.result(3) is True
    assert events == ["checkpoint_index_prefix", "checkpoint_remote_fill"]


def test_agreement_does_not_expire_before_the_existing_native_read_deadline():
    ns = classes()
    cls = ns["AscendLMCacheEngine"]
    obj = cls()
    obj.shared_cpu_cache_generation = 1
    obj.shared_cpu_cache_strict = True
    obj.config = NS(blocking_timeout_secs=0.001)
    marker = replace(
        obj._shared_layerwise_error_envelope(
            req_id="r",
            phase="checkpoint_index_prefix",
            request_ordinal=7,
            layer_id=0,
            kv_group=1,
            message="",
        ),
        status="skipped",
    )
    ns["Base"]._receive_shared_envelope = lambda self: marker
    obj.collective_all_true_fn = bool
    obj.enable_checkpoint_prefix_agreement()
    result = obj._checkpoint_prefix_result(obj._shared_envelope_identity(marker))
    timer = threading.Timer(0.03, lambda: result.set_result(True))
    timer.start()
    try:
        assert obj._receive_shared_envelope().status == "skipped"
    finally:
        timer.join()
