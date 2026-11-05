# SPDX-License-Identifier: Apache-2.0
"""Abort retirement uses production worker polling and store acknowledgements."""

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS
from weakref import WeakMethod

import pytest

ROOT = Path(__file__).resolve().parents[2]
LN = ROOT.parent / "LMCache-NPU"


def definitions(path, names, namespace):
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("pending_store", [False, True])
@pytest.mark.parametrize(
    "role,async_store,expect_send",
    [("kv_both", True, True), ("kv_consumer", True, False), ("kv_both", False, False)],
)
def test_aborted_cold_load_eventually_acknowledges_send_after_all_owners_retire(
    failed, pending_store, role, async_store, expect_send
):
    ns = dict(
        Future=Future,
        ThreadPoolExecutor=ThreadPoolExecutor,
        WeakMethod=WeakMethod,
        serving_perf_enabled=lambda: False,
        logger=NS(debug=lambda *a, **kw: None, exception=lambda *a, **kw: None),
        _clear_terminal_load_tracebacks=lambda *a: None,
    )
    definitions(
        LN / "lmcache/integration/vllm/cold_load.py", {"ColdLoadCoordinator"}, ns
    )
    names = {
        "get_finished",
        "_publish_completed_cold_load",
        "_fail_completed_cold_load",
        "_record_checkpoint_restore_miss",
        "_finish_aborted_cold_load",
    }
    definitions(LN / "lmcache/integration/vllm/vllm_v1_adapter.py", names, ns)
    base = type("BaseAdapter", (), {name: ns[name] for name in names if name in ns})
    ascend_ns = dict(ns)
    definitions(
        ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        {"_finalize_worker_requests_after_store", "_finish_aborted_cold_load"},
        ascend_ns,
    )
    adapter_type = type(
        "AscendAdapter",
        (base,),
        {
            name: ascend_ns[name]
            for name in (
                "_finalize_worker_requests_after_store",
                "_finish_aborted_cold_load",
            )
            if name in ascend_ns
        },
    )
    definitions(ROOT / "lmcache_ascend/v1/cache_engine.py", {"get_finished_stores"}, ns)
    engine_type = type("Engine", (), {"get_finished_stores": ns["get_finished_stores"]})
    engine = engine_type()
    engine.is_store_async, engine._store_lock = async_store, Lock()
    engine._reported_finished_store_ids, engine._deferred_finished_req_ids = (
        set(),
        set(),
    )
    engine._pending_store_reqs = {"r"} if pending_store and async_store else set()
    engine.drop_direct_store_states = lambda ids: None
    adapter = adapter_type()
    adapter.lmcache_engine, adapter.kv_role, adapter.store_async = (
        engine,
        role,
        async_store,
    )
    adapter._unfenced_live_stores, adapter._worker_retrieve_state = {}, {}
    adapter._late_finished_sending, adapter._invalid_block_ids = set(), set()
    adapter._wait_for_save_done = True
    adapter._release_finished_worker_requests = lambda ids: None
    adapter._drop_worker_retrieve_state = lambda rid: None
    adapter._release_unadopted_shared_request_objects = lambda *a: None
    adapter._release_shared_worker_retrieve_state = lambda *a: None
    adapter._release_request_lookup_pins = lambda rid: None
    adapter._cold_requires_paired_restart = lambda: False

    # Keep weak callback receivers alive through the complete polling sequence.
    class Hooks:
        def restart(self):
            return False

    hooks = Hooks()
    coordinator = ns["ColdLoadCoordinator"](
        adapter._publish_completed_cold_load,
        adapter._fail_completed_cold_load,
        hooks.restart,
    )
    adapter._cold_load_coordinator = coordinator
    adapter._drain_dsa_cold_load_futures = coordinator.poll
    latent, indexer = Future(), Future()
    request = NS(
        load_spec=NS(
            dsa_cold_load_generation=1,
            dsa_group1_direct_hbm=True,
            lmcache_cached_tokens=100,
        )
    )
    coordinator.futures["r"] = (1, latent, request, {7}, 0.0, indexer)
    assert adapter.get_finished({"r"}) == (None, None)
    if failed:
        latent.set_exception(ValueError("missing latent pages"))
    else:
        ready = {"value": False}
        latent.set_result(NS(dense_load_readiness=NS(query=lambda: ready["value"])))
    assert adapter.get_finished(set()) == (None, None)  # Sibling DMA still active.
    indexer.set_result(None)
    if not failed:
        assert adapter.get_finished(set()) == (None, None)  # Metadata DMA not ready.
        ready["value"] = True
    sends, receives = adapter.get_finished(set())
    all_sends, all_receives = set(sends or ()), set(receives or ())
    assert all_receives == {"r"}
    if pending_store:
        assert not all_sends
        engine._pending_store_reqs.clear()
    for _ in range(3):
        sends, receives = adapter.get_finished(set())
        assert not all_sends.intersection(sends or ())
        assert not all_receives.intersection(receives or ())
        all_sends.update(sends or ())
        all_receives.update(receives or ())
    assert all_sends == ({"r"} if expect_send else set()), (
        "Aborted receive never acknowledged connector-owned cleanup"
    )
