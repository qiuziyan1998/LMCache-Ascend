# SPDX-License-Identifier: Apache-2.0
"""Exercise producer accounting without loading an accelerator backend."""

import ast
from collections import deque
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_store_has_no_discarded_diagnostic_aggregation():
    path = ROOT / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert not (
                isinstance(node.value.func, ast.Name) and node.value.func.id == "sum"
            )


@pytest.fixture
def implementation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parent / "LMCache"))
    spec = importlib.util.spec_from_file_location(
        "cleanup_test_producer", ROOT / "lmcache_ascend/v1/remote_fill_producer.py"
    )
    producer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, producer)
    spec.loader.exec_module(producer)
    namespace = vars(producer)
    namespace.update(
        deque=deque,
        threading=threading,
        logger=SimpleNamespace(),
        log_remote_fill_diagnostic=lambda *args, **kwargs: None,
    )
    nodes = []
    coordinator = ast.parse(
        (ROOT / "lmcache_ascend/v1/remote_fill_coordinator.py").read_text(
            encoding="utf-8"
        )
    )
    for node in coordinator.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProducerRequestState":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RemoteFillCoordinator":
            node.body = [
                method
                for method in node.body
                if getattr(method, "name", None)
                in {
                    "_acquire_queue_capacity",
                    "_release_queue_capacity",
                    "_record_window_metrics",
                    "finish",
                }
            ]
            nodes.append(node)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(nodes)
    exec(
        compile(ast.fix_missing_locations(module), "producer accounting", "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


def coordinator(implementation):
    instance = implementation.RemoteFillCoordinator()
    instance.config = SimpleNamespace(
        remote_fill_max_inflight_bytes=100,
        remote_fill_max_bytes_per_request=100,
        remote_fill_max_inflight_windows_per_request=2,
        chunk_size=1024,
    )
    metrics = implementation.RemoteFillProducerMetrics()
    instance.get_metrics = lambda: metrics
    return instance, metrics


def test_queue_limits_and_releases_preserve_counter_values(implementation):
    instance, metrics = coordinator(implementation)
    first = implementation.ProducerRequestState()
    second = implementation.ProducerRequestState()
    assert instance._acquire_queue_capacity(first, 60)
    assert not instance._acquire_queue_capacity(second, 50)
    assert metrics.snapshot()["gauges"]["inflight_bytes"] == 60
    instance._release_queue_capacity(first, 60)
    assert instance._acquire_queue_capacity(second, 50)
    instance._release_queue_capacity(second, 50)
    snapshot = metrics.snapshot()
    assert snapshot["gauges"]["inflight_bytes"] == 0
    assert snapshot["gauges"]["inflight_windows"] == 0
    assert "timers" not in snapshot


def test_native_measurements_and_byte_counters_are_preserved(implementation):
    instance, metrics = coordinator(implementation)
    state = implementation.ProducerRequestState()
    result = implementation.RemoteFillWindowResult(
        window_id=1,
        direct_satisfied=True,
        armed=True,
        native_seconds=0.007,
        native_started_monotonic=91.0,
        native_ended_monotonic=91.007,
        submitted_bytes=64,
        existing_pages=2,
    )
    instance._record_window_metrics(state, result)
    assert result.native_seconds == 0.007
    assert state.submitted_bytes == 64
    assert metrics.snapshot()["bytes"]["submitted_bytes"] == 64
    assert metrics.snapshot()["same_key_existing_total"] == 2
    assert {"reserve_seconds", "arm_seconds", "report_seconds"}.isdisjoint(
        asdict(result)
    )


@pytest.mark.parametrize("fatal", [False, True])
def test_finish_preserves_terminal_accounting_and_fatal_handling(implementation, fatal):
    instance, metrics = coordinator(implementation)
    state = implementation.ProducerRequestState(
        handoff=SimpleNamespace(transfer_id="transfer")
    )
    calls = []
    terminal = implementation.RemoteFillTerminalResult(
        "transfer", "LOCAL_FULL", 2048, 2176, True
    )

    def finish(**kwargs):
        calls.append(kwargs)
        if fatal:
            raise implementation.RemoteFillFatalError("terminal ownership unknown")
        return terminal

    state.session = SimpleNamespace(finish=finish)
    state.submitted_bytes = 64
    instance.wait = lambda current: calls.append("wait")
    instance._latch_fatal = lambda current: calls.append("fatal")
    instance.record_success = lambda: calls.append("success")
    if fatal:
        with pytest.raises(implementation.RemoteFillFatalError):
            instance.finish("request", state, 2176, 2048)
        assert calls[-1] == "fatal"
        return
    assert instance.finish("request", state, 2176, 2048) is terminal
    assert calls == [
        "wait",
        dict(
            required_store_end=2176,
            persistent_common_end=2048,
            final_partial_valid_tokens=128,
        ),
        "success",
    ]
    assert metrics.snapshot()["attempts_total"] == {"LOCAL_FULL:none": 1}
    assert metrics.snapshot()["bytes"]["published_bytes"] == 64
