# SPDX-License-Identifier: Apache-2.0
"""Run the production publication block with unfinished fallback D2H copies."""

import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class Key:
    def __init__(self, chunk, layer=None):
        self.chunk = chunk
        self.layer = layer

    def without_layer(self):
        return Key(self.chunk)


class Object:
    def __init__(self, value=-1):
        self.value = value

    def get_size(self):
        return 4


class Page(Object):
    pass


class Queue:
    def __init__(self, **kwargs):
        self.reservations = []
        self.batches = []

    def reserve(self, size):
        self.reservations.append(size)

    def add(self, size, futures, **kwargs):
        self.batches.append((size, list(futures), tuple(kwargs["keys"])))

    def fail(self, error):
        raise error


def publish(*, mixed=False, asynchronous=True):
    """Execute the actual page-mode branch, including its flat-object fallback."""
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    store = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "store_layer"
    )
    block = next(
        node
        for node in ast.walk(store)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "page_first_store"
        and isinstance(node.body[0], ast.Assign)
        and node.body[0].targets[0].id == "async_pages"
    )
    flat = [[Object() for _ in range(2)] for _ in range(3)]
    if mixed:
        page = Page()
        for row in flat:
            row[0] = page
    keys = [[Key(chunk, layer) for chunk in range(2)] for layer in range(3)]
    fences = (object(), object()) if asynchronous else ()
    unfenced_reads, fenced_puts, required = [], [], []

    def ordinary_put(keys, objects, **kwargs):
        # A remote backend is allowed to start reading immediately on submit.
        unfenced_reads.extend(obj.value for obj in objects)
        future = Future()
        future.set_result(None)
        return [future]

    def fenced_put(keys, objects, **kwargs):
        fenced_puts.append((keys, objects, kwargs))
        return [Future()]

    engine = NS(
        _force_layerwise_prefill_store=True,
        store_location=None,
        _layerwise_put_queue=None,
        _store_queue_maxsize=2,
        config=NS(
            remote_url="mooncakestore://test",
            remote_fill_max_inflight_bytes=1024,
            blocking_timeout_secs=30,
            enable_remote_lmcache_store=False,
        ),
        storage_manager=NS(
            batched_put=ordinary_put,
            batched_put_layer_pages=fenced_put,
            batched_put_layerwise_objects=fenced_put,
        ),
        _track_sync_store_futures=lambda futures, **kw: required.extend(futures),
    )
    namespace = dict(
        self=engine,
        memory_objs=flat,
        keys=keys,
        page_store=True,
        page_first_store=True,
        async_layerwise_store=asynchronous,
        layerwise_store_fences=fences,
        LayerPageMemoryObj=Page,
        LayerwisePutQueue=Queue,
        pending_store_release={id(obj): obj for row in flat for obj in row},
        req_id="consumer",
        request_configs=None,
        kv_group=0,
    )
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), namespace
    )
    return engine, flat, fences, unfenced_reads, fenced_puts, required


@pytest.mark.parametrize("mixed", [False, True])
def test_flat_fallback_never_reads_host_bytes_before_d2h_fences(mixed):
    engine, flat, fences, reads, puts, required = publish(mixed=mixed)
    assert reads == [], "fallback remote put observed unfinished CPU D2H bytes"
    assert all(put[2]["producer_events"] == fences for put in puts)
    fallback = puts[-1]
    assert len(fallback[1]) == 3  # Preserve the layer-major source matrix.
    assert all(len(row) == (1 if mixed else 2) for row in fallback[1])
    assert not required
    queue = engine._layerwise_put_queue
    assert len(queue.batches) == 1
    assert len(queue.batches[0][2]) == 2
    unique = {id(obj): obj for row in flat for obj in row}
    assert queue.reservations == [sum(obj.get_size() for obj in unique.values())]


def test_synchronous_fallback_keeps_the_existing_ordinary_put():
    _, _, _, reads, puts, _ = publish(asynchronous=False)
    assert reads == [-1] * 6
    assert not puts
