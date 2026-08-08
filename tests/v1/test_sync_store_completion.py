# SPDX-License-Identifier: Apache-2.0
"""Synchronous remote-store completion barrier tests."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace
import threading

# Third Party
import pytest

pytest.importorskip("lmcache")

# First Party
from lmcache_ascend.v1.cache_engine import (
    AscendLMCacheEngine,
    AsyncDecodePersistenceError,
)


def _make_engine(timeout: float = 0.01) -> AscendLMCacheEngine:
    engine = object.__new__(AscendLMCacheEngine)
    engine.is_store_async = False
    engine.config = SimpleNamespace(blocking_timeout_secs=timeout)
    engine._pending_sync_store_futures = set()
    engine._decode_store_future_lock = threading.Lock()
    engine._pending_decode_store_futures = {}
    return engine


def test_sync_store_barrier_propagates_backend_failure() -> None:
    engine = _make_engine()
    future: Future = Future()
    future.set_exception(RuntimeError("remote write failed"))
    engine._track_sync_store_futures([future])

    with pytest.raises(RuntimeError, match="remote write failed"):
        engine.wait_for_pending_sync_stores()

    assert not engine._pending_sync_store_futures


def test_sync_store_barrier_retains_timed_out_operations() -> None:
    engine = _make_engine(timeout=0)
    future: Future = Future()
    engine._track_sync_store_futures([future])

    with pytest.raises(TimeoutError, match="1 remote store operation"):
        engine.wait_for_pending_sync_stores()
    assert engine._pending_sync_store_futures == {future}

    future.set_result(None)
    engine.wait_for_pending_sync_stores()
    assert not engine._pending_sync_store_futures


def test_async_store_mode_does_not_join_operation_futures() -> None:
    engine = _make_engine()
    engine.is_store_async = True
    future: Future = Future()

    engine._track_sync_store_futures([future])
    engine.wait_for_pending_sync_stores()

    assert not engine._pending_sync_store_futures
    assert not future.done()


def test_decode_save_waits_only_for_its_owned_futures() -> None:
    engine = _make_engine()
    first: Future = Future()
    second: Future = Future()
    first.set_result(None)
    second.set_result(None)
    engine._track_sync_store_futures([first], owner="job-a")
    engine._track_sync_store_futures([second], owner="job-b")

    engine.wait_for_pending_decode_save("job-b")

    assert "job-b" not in engine._pending_decode_store_futures
    assert engine._pending_decode_store_futures == {"job-a": {first}}


def test_decode_save_wraps_retryable_backend_failure() -> None:
    engine = _make_engine()
    future: Future = Future()
    future.set_exception(RuntimeError("remote write failed"))
    engine._track_sync_store_futures([future], owner="job")

    with pytest.raises(AsyncDecodePersistenceError, match="job"):
        engine.wait_for_pending_decode_save("job")

    assert "job" not in engine._pending_decode_store_futures


def test_decode_save_retains_timed_out_future_for_retry() -> None:
    engine = _make_engine(timeout=0)
    future: Future = Future()
    engine._track_sync_store_futures([future], owner="job")

    with pytest.raises(AsyncDecodePersistenceError, match="timed out"):
        engine.wait_for_pending_decode_save("job")
    assert engine._pending_decode_store_futures == {"job": {future}}

    future.set_result(None)
    engine.wait_for_pending_decode_save("job")
    assert "job" not in engine._pending_decode_store_futures
