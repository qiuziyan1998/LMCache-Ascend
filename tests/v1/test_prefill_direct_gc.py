# SPDX-License-Identifier: Apache-2.0
"""Known P read failures release inner owners without changing shared readers."""

# Standard
import asyncio
import gc
import threading
import time
import weakref
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache_ascend import prefill_direct
from lmcache_ascend.prefill_direct import PrefillDirectLMCacheEngine


@pytest.mark.parametrize(
    "failure", ["terminal", "terminal_timeout", "short", "chained"]
)
def test_prefill_failure_clears_completed_reader_frames(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()

    class Owner:
        device = SimpleNamespace(type="cpu")

    refs = []

    async def fail(*args: Any) -> None:
        if failure == "terminal_timeout":
            raise TimeoutError("known terminal failure")
        if failure == "chained":
            try:
                raise RuntimeError("inner failure")
            except RuntimeError as error:
                raise TimeoutError("wrapped terminal failure") from error
        raise RuntimeError("known terminal failure")

    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    # Deliberately no LMCache knob on the separate MooncakeStoreConfig.
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector._external_put_lock = asyncio.Lock()
    connector._external_native_deadline = lambda: time.perf_counter() + 2
    connector._inflight_put_tasks = set()
    connector._validate_external_buffer_owners = lambda *args: None
    connector._external_page_key = lambda key, size: key
    connector._register_external_owners = lambda owners: None
    connector.store = SimpleNamespace(batch_get_into_multi_buffers=lambda *args: [-1])
    backend = object.__new__(RemoteBackend)
    backend.loop = loop
    backend._mla_worker_id_as0_mode = False
    backend._external_page_outer_timeout_secs = lambda: 2
    backend.connection = (
        connector
        if failure == "short"
        else SimpleNamespace(batched_get_external_pages=fail)
    )

    engine = object.__new__(PrefillDirectLMCacheEngine)
    engine.storage_manager = backend
    engine._is_passive = lambda: False
    engine._ensure_layerwise_connector_layout = lambda **kwargs: None
    engine.gpu_connector = SimpleNamespace(
        record_dense_load_readiness=lambda **kwargs: None,
        synchronize_dense_load_readiness=lambda ready: None,
    )
    monkeypatch.setattr(prefill_direct, "serving_perf_enabled", lambda: False)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: None), raising=False
    )
    caught_fields = []

    def attempt() -> None:
        def plan(*args: Any) -> Any:
            owner = Owner()
            refs.append(weakref.ref(owner))
            return [[1]], [[8]], (owner,)

        engine.gpu_connector.plan_direct_page_destinations = plan
        try:
            engine._load_prefill_group0_page_plan(
                [(0, 4, "key")], None, [], "request", perf_enabled=False
            )
        except (RuntimeError, TimeoutError) as error:
            caught_fields.append((str(error), getattr(error, "failed_pages", None)))
        else:
            raise AssertionError("The read must fail")

    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            attempt()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(2)
        assert all(ref() is None for ref in refs)
        assert len(caught_fields) == 3
        if failure == "short":
            assert all(fields == [("key", -1, 8)] for _, fields in caught_fields)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()
        if enabled:
            gc.enable()
