# SPDX-License-Identifier: Apache-2.0
"""Run the production publication block with unfinished fallback D2H copies."""

import ast
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

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


def publish(*, mixed=False, asynchronous=True, trace=None):
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
        starts=[0, 256],
        ends=[256, 512],
        store_debug=trace is not None,
        prefill_store_debug_log=lambda *args, **kwargs: trace.append((args, kwargs)),
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_diagnostics_report_actual_storage_type_and_fences(mixed, asynchronous):
    trace = []
    _, _, fences, reads, _, _ = publish(
        mixed=mixed, asynchronous=asynchronous, trace=trace
    )
    assert [args[0] for args, _ in trace] == (
        ["page_put", "flat_put"] if mixed else ["flat_put"]
    )
    for args, fields in trace:
        assert args[1:] == ("consumer", 0)
        assert fields["async_d2h"] == asynchronous
        assert fields["events"] == len(fences)
    assert reads == ([] if asynchronous else [-1] * (3 if mixed else 6))


@pytest.mark.parametrize(
    "bulk_ok,single_ok", [(True, True), (False, True), (False, False)]
)
def test_allocation_logs_distinguish_bulk_retry_from_actual_single_page_failure(
    bulk_ok, single_ok
):
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    store = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "store_layer"
    )
    bulk = next(
        node
        for node in store.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "pending_chunks and page_store"
    )
    loop = next(
        node
        for node in store.body
        if isinstance(node, ast.For)
        and "enumerate(pending_chunks)" in ast.unparse(node.iter)
    )
    # Run production bulk allocation and per-chunk page retry, before the
    # ordinary-object allocator. A bulk miss alone must not claim a fallback.
    allocations, trace = [], []
    page = Page()

    def allocate(*args, **kwargs):
        allocations.append(kwargs)
        success = bulk_ok if kwargs.get("eviction") is False else single_ok
        return [page] if success else None

    runtime = dict(
        self=NS(
            config=NS(chunk_size=256),
            _force_layerwise_prefill_store=True,
            _num_layers_for_kv_group=lambda group: 3,
            gpu_connector=NS(get_shape=lambda *args, **kwargs: (256,)),
        ),
        pending_chunks=[(0, 256, Key(0), [Key(0, layer) for layer in range(3)])],
        page_store=True,
        page_batch=None,
        legacy_suffix=False,
        local=NS(batched_allocate_layer_pages=allocate),
        page_shape=(256,),
        page_shapes=[(256,)],
        page_dtypes=["bf16"],
        memory_format=None,
        kv_group=0,
        req_id="producer",
        force_store_wait=True,
        store_debug=True,
        prefill_store_debug_log=lambda *args, **kwargs: trace.append((args, kwargs)),
        prefill_store_allocator_snapshot=lambda local: {"free_mb": 12, "pool_mb": 64},
    )
    prefix = ast.For(target=loop.target, iter=loop.iter, body=loop.body[:4], orelse=[])
    module = ast.fix_missing_locations(ast.Module(body=[bulk, prefix], type_ignores=[]))
    exec(compile(module, str(path), "exec"), runtime)
    stages = [args[0] for args, _ in trace]
    assert stages == (
        []
        if bulk_ok
        else ["bulk_retry"]
        if single_ok
        else ["bulk_retry", "page_single_none"]
    )
    assert runtime["legacy_suffix"] is (not bulk_ok and not single_ok)
    assert len(allocations) == (1 if bulk_ok else 2)
    if not bulk_ok:
        assert allocations[-1]["busy_loop"] is False


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_marker_follows_existing_persistence_barriers(enabled, failed):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_layerwise_prefill_batch"
    )
    trace = []

    def poll(**kwargs):
        trace.append("poll")
        if failed:
            raise TimeoutError("pending remote put")

    runtime = dict(
        prefill_store_debug_enabled=lambda: enabled,
        get_tensor_model_parallel_rank=lambda: 0,
        prefill_store_debug_log=lambda stage, *args, **kwargs: trace.append(stage),
        prefill_store_debug_finish=lambda req: trace.append("persist_done"),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), runtime
    )
    req = NS(
        req_id="request",
        token_ids=[1, 2],
        request_configs={},
        is_last_prefill=True,
        is_sparse_decode=False,
    )
    adapter = NS(
        kv_role="kv_producer",
        lmcache_engine=NS(
            submit_layerwise_prefill_fills=lambda ids: trace.append("submit"),
            poll_layerwise_prefill_puts=poll,
            wait_for_pending_sync_stores=lambda: trace.append("wait"),
            adopt_completed_layerwise_store=lambda result: trace.append("adopt"),
            finish_layerwise_prefill_store=lambda *args, **kwargs: trace.append(
                "finish"
            ),
        ),
        _parent=NS(_get_connector_metadata=lambda: NS(requests=[req])),
        _completed_layerwise_stores={("request", 0): object()},
    )
    run = MethodType(runtime[method.name], adapter)
    if failed:
        with pytest.raises(TimeoutError, match="pending remote put"):
            run()
        assert "persist_done" not in trace
        assert ("persist_error" in trace) == enabled
    else:
        run()
        assert trace == (
            (["persist_begin"] if enabled else [])
            + ["submit", "poll", "wait", "adopt", "finish"]
            + (["persist_done"] if enabled else [])
        )
