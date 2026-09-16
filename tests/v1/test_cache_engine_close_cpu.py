# SPDX-License-Identifier: Apache-2.0
"""CPU constructor/close regressions, with no NPU imports or device allocation.

Execute the unchanged production method ASTs; only the parent engine and
unrelated native/store dependencies are fixtures. This is not NPU validation.
"""

# Standard
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakSet
import __future__
import ast
import logging
import queue
import threading
import time

# Third Party
import pytest


@pytest.fixture
def engine_class():
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "AscendLMCacheEngine"
    )
    methods = {
        "__init__",
        "close",
        "_ensure_store_worker",
        "close_remote_fill_producer",
    }
    cls.body = [
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name in methods
    ]
    assert {item.name for item in cls.body} == methods

    class BaseEngineFixture:
        def __init__(self, config, metadata, *args):
            self.config = config
            self.metadata = metadata
            self.kv_events_enabled = False
            self.events = []

        def _is_passive(self):
            return False

        def wait_for_direct_stores(self, requests):
            assert not requests
            self.events.append("direct_stores_drained")

        def close(self):
            self.events.append("base_close")

    namespace = {
        "LMCacheEngine": BaseEngineFixture,
        "deque": deque,
        "WeakSet": WeakSet,
        "threading": threading,
        "queue": queue,
        "time": time,
        "logger": logging.getLogger(__name__),
        "mooncake_layer_pages_enabled": lambda config: True,
    }
    module = ast.Module(body=[cls], type_ignores=[])
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["AscendLMCacheEngine"]


def make_engine(engine_class, *, store_async=False, direct=False, queue_size=1):
    config = SimpleNamespace(
        store_async=store_async,
        store_async_max_queue_size=queue_size,
        pd_role="sender",
        enable_remote_lmcache_store=direct,
        get_extra_config_value=lambda key, default=None: (
            direct if key == "use_ascend_direct" else default
        ),
    )
    return engine_class(config, SimpleNamespace(world_size=8), None, None, None, None)


@pytest.mark.parametrize(
    "store_async,direct", [(False, False), (True, False), (True, True)]
)
def test_close_without_background_worker_reaches_base_engine(
    engine_class, store_async, direct
):
    engine = make_engine(engine_class, store_async=store_async, direct=direct)
    assert engine._direct_store_enabled == direct
    engine.close()
    assert engine.events == ["direct_stores_drained", "base_close"]
    assert engine._store_queue is None
    assert engine._store_worker_thread is None


def test_sync_mode_does_not_create_queue_or_thread(engine_class, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Sync storage must not create an async worker")

    monkeypatch.setattr(queue, "Queue", unexpected)
    monkeypatch.setattr(threading, "Thread", unexpected)
    engine = make_engine(engine_class)
    engine.close()
    assert engine.events[-1] == "base_close"


@pytest.mark.parametrize("queue_size", [0, 1])
def test_async_close_drains_real_cpu_queue_before_base_close(engine_class, queue_size):
    engine = make_engine(engine_class, store_async=True, queue_size=queue_size)

    def consume():
        while True:
            item = engine._store_queue.get()
            engine._store_queue.task_done()
            if item is None:
                return
            engine.events.append(item)

    engine._store_worker_loop = consume
    engine._ensure_store_worker()
    thread = engine._store_worker_thread
    try:
        engine._store_queue.put("stored")
        engine.close()
        assert not thread.is_alive()
        assert engine.events.index("stored") < engine.events.index("base_close")
        assert engine._store_queue.unfinished_tasks == 0
    finally:
        if thread.is_alive():
            engine._store_queue.put(None, timeout=1)
            thread.join(timeout=2)


@pytest.mark.parametrize(
    "guard", ["_failed_sparse_loads", "_remote_fill_fatal_transfers"]
)
def test_unknown_native_transfer_still_blocks_allocator_teardown(engine_class, guard):
    engine = make_engine(engine_class)
    setattr(engine, guard, ("unfenced-transfer",))
    with pytest.raises(RuntimeError, match="restart"):
        engine.close()
    assert "base_close" not in engine.events
