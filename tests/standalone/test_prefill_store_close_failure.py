# SPDX-License-Identifier: Apache-2.0
"""An unknown D2H completion must not release its destination allocation."""

import ast
import gc
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _method(path, name, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name is not None:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _compile(name, arguments, body):
    function = ast.parse(f"def {name}({arguments}):\n    pass\n").body[0]
    function.body = body
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), name, "exec"), namespace)
    return namespace[name]


def _production_cleanup(backend):
    path = (
        ROOT / "lmcache_ascend/v1/cache_engine.py"
        if backend == "ascend"
        else ROOT.parent / "LMCache/lmcache/v1/cache_engine.py"
    )
    store = _method(path, "store_layer")
    finalizer = next(
        node
        for node in ast.walk(store)
        if isinstance(node, ast.Try)
        and node.finalbody
        and "pending_store_release" in ast.unparse(node.finalbody)
        and "close_fn" in ast.unparse(node.finalbody)
    )
    return _compile(
        "cleanup",
        "self, mem_obj_generator, cached_memory_objs, pending_store_release",
        finalizer.finalbody,
    )


def _attach_connector_methods(connector):
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    cls = "VLLMPagedMemLayerwiseNPUConnector"
    for name in (
        "_retain_unfenced_layerwise_prefill_store",
        "wait_for_layerwise_prefill_stores",
    ):
        method = _method(path, name, cls)
        method.decorator_list = []
        module = ast.parse("from __future__ import annotations")
        module.body.append(method)
        scope = {}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        setattr(connector, name, MethodType(scope[name], connector))


def _production_transfer(
    connector, memory_objs, *, advance_error=False, resources=None
):
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    _attach_connector_methods(connector)
    store = _method(path, "batched_from_gpu", "VLLMPagedMemLayerwiseNPUConnector")
    finalizer = next(
        node
        for node in ast.walk(store)
        if isinstance(node, ast.Try)
        and node.finalbody
        and "store_transfer_pending" in ast.unparse(node.finalbody)
    )
    body = ast.parse(
        "store_transfer_pending = True\n"
        "ordinary_store = False\n"
        "prefill_dma = True\n"
        "kv_group = 0\n"
        "tmp_gpu_buffer_obj = None\n"
        "kwargs = {'req_id': 'request'}\n"
        "kvcaches_snapshot = resources\n"
        "prefill_packets = resources\n"
        "try:\n"
        "    yield\n"
        "    if advance_error:\n"
        "        raise RuntimeError('native enqueue failed')\n"
        "finally:\n"
        "    pass\n"
    ).body
    body[-1].finalbody = finalizer.finalbody
    generator = _compile(
        "transfer",
        "self, memory_objs, advance_error, resources",
        body,
    )(connector, memory_objs, advance_error, resources)
    next(generator)
    return generator


class Destination:
    def __init__(self):
        self.references = 1
        self.value = -1

    def is_valid(self):
        return self.references > 0

    def ref_count_down(self):
        self.references -= 1

    def ref_count_up(self):
        self.references += 1


def _connector(fence):
    stream = NS(synchronize=fence)
    return NS(
        use_gpu=False,
        _layerwise_prefill_dma_streams={(0, 0): stream},
        _synchronize_layerwise_prefill_dma_streams=lambda group: fence(),
    )


@pytest.mark.parametrize("backend", ["ascend", "base"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_failed_store_close_propagates_and_keeps_destination(backend, error_type):
    destination = Destination()
    pending = {id(destination): destination}
    cached = [[destination]]
    removals = []

    def fail_fence():
        raise error_type("D2H completion unknown")

    connector = _connector(fail_fence)
    transfer = _production_transfer(connector, cached)
    engine = NS(
        _remove_pending_layerwise_store_objs=lambda *args: removals.append(args)
    )
    with pytest.raises(error_type, match="D2H completion unknown"):
        _production_cleanup(backend)(engine, transfer, cached, pending)

    # Deferred bytes are deliberately wrong until DMA completes. Recycling this
    # destination now would let a later DMA overwrite a different allocation.
    assert destination.value == -1
    assert destination.references == 2  # Original + quarantined DMA lease.
    assert pending == {id(destination): destination}
    assert cached == [[destination]]
    assert not removals
    assert connector._layerwise_prefill_unfenced_stores["request"]["owners"] == pending


@pytest.mark.parametrize("backend", ["ascend", "base"])
def test_successful_store_close_finishes_dma_before_destination_release(backend):
    destination = Destination()
    pending = {id(destination): destination}

    def finish_fence():
        assert destination.references == 1
        destination.value = 37

    connector = _connector(finish_fence)
    transfer = _production_transfer(connector, [[destination]])

    def remove(cached, pending):
        assert destination.value == 37

    engine = NS(_remove_pending_layerwise_store_objs=remove)
    _production_cleanup(backend)(engine, transfer, [[destination]], pending)
    assert destination.value == 37
    assert destination.references == 0
    assert not pending
    assert not getattr(connector, "_layerwise_prefill_unfenced_stores", {})


@pytest.mark.parametrize("backend", ["ascend", "base"])
@pytest.mark.parametrize("gc_off", [False, True])
def test_failed_advance_survives_exhausted_generator_and_final_reference_drop(
    backend, gc_off
):
    """A second close is a no-op after send/next already failed its finalizer."""
    enabled = gc.isenabled()
    if gc_off:
        gc.disable()
    try:
        destination = Destination()
        reference = weakref.ref(destination)
        pending = {id(destination): destination}
        cached = [[destination], [destination]]  # Merged page appears in every layer.
        state = {"fail": True, "fences": 0}

        def fence():
            state["fences"] += 1
            if state["fail"]:
                raise RuntimeError("D2H completion unknown")
            obj = reference()
            assert obj is not None and obj.references == 1
            obj.value = 37

        connector = _connector(fence)
        transfer = _production_transfer(connector, cached, advance_error=True)
        with pytest.raises(RuntimeError, match="D2H completion unknown"):
            next(transfer)
        assert destination.references == 2
        engine = NS(
            _remove_pending_layerwise_store_objs=lambda rows, pending: rows.clear()
        )
        _production_cleanup(backend)(engine, transfer, cached, pending)
        assert destination.references == 1
        assert not pending
        # Neither an exhausted generator nor the old registry can retain/fence
        # the payload now. Only the independent failure latch can do so.
        cached.clear()
        del destination, transfer
        connector._layerwise_prefill_dma_streams = {}
        gc.collect()
        assert reference() is not None
        with pytest.raises(RuntimeError, match="D2H completion unknown"):
            connector.wait_for_layerwise_prefill_stores(("request",))
        assert reference().references == 1
        assert "request" in connector._layerwise_prefill_unfenced_stores
        state["fail"] = False
        # Temporarily observe the final bytes without keeping the source alive
        # during the failure/retry phase above.
        observed = reference()
        connector.wait_for_layerwise_prefill_stores()
        assert observed.value == 37 and observed.references == 0
        assert state["fences"] == 3
        assert not connector._layerwise_prefill_unfenced_stores
        connector.wait_for_layerwise_prefill_stores()
        assert observed.references == 0  # Idempotent cleanup.
    finally:
        if enabled:
            gc.enable()


def test_missing_request_id_and_partial_owner_release_remain_retryable():
    """A completed earlier lease must not be decremented again on retry."""
    fences = []
    connector = _connector(lambda: fences.append("done"))
    _attach_connector_methods(connector)
    first, second = Destination(), Destination()
    stream = next(iter(connector._layerwise_prefill_dma_streams.values()))
    connector._retain_unfenced_layerwise_prefill_store(
        "None",
        (stream,),
        (first, first, second),
        (),
    )
    assert first.references == second.references == 2
    release_second = second.ref_count_down
    second.ref_count_down = lambda: (_ for _ in ()).throw(
        RuntimeError("release interrupted")
    )
    with pytest.raises(RuntimeError, match="release interrupted"):
        connector.wait_for_layerwise_prefill_stores()
    assert first.references == 1 and second.references == 2
    assert list(connector._layerwise_prefill_unfenced_stores["None"]["owners"]) == [
        id(second)
    ]
    second.ref_count_down = release_second
    connector.wait_for_layerwise_prefill_stores()
    assert first.references == second.references == 1
    assert fences == ["done", "done"]
    assert not connector._layerwise_prefill_unfenced_stores
