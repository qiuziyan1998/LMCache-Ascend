# SPDX-License-Identifier: Apache-2.0
"""Synchronous remote-store completion barrier tests."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("lmcache")

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


def _make_engine(timeout: float = 0.01) -> AscendLMCacheEngine:
    engine = object.__new__(AscendLMCacheEngine)
    engine.is_store_async = False
    engine._require_store_completion = False
    engine.config = SimpleNamespace(blocking_timeout_secs=timeout)
    engine._pending_sync_store_futures = set()
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


def test_layerwise_async_store_requires_completion() -> None:
    engine = _make_engine()
    engine.is_store_async = True
    future: Future = Future()

    engine._track_sync_store_futures([future], require_completion=True)

    assert engine._pending_sync_store_futures == {future}
