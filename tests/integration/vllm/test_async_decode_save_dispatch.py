# SPDX-License-Identifier: Apache-2.0
"""Async decode-save worker ordering and completion tests."""

# Standard
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
import queue
import threading
from unittest.mock import MagicMock, call

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
    _AsyncDecodeSaveTask,
)


def _request(job_id: int, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(
        req_id="request",
        decode_save_generation=4,
        decode_save_job_id=job_id,
        decode_save_is_final=False,
        decode_window_start=start,
        decode_window_end=end,
        save_spec=SimpleNamespace(),
    )


def _adapter_with_two_jobs():
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    engine = MagicMock()
    engine.state_guard.side_effect = nullcontext
    engine.metadata = SimpleNamespace(worker_id=0, world_size=1, use_mla=False)
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter.config = MagicMock()
    adapter.config.get_extra_config_value.return_value = False
    adapter._async_decode_lock = threading.RLock()
    adapter._async_decode_cv = threading.Condition(adapter._async_decode_lock)
    adapter._async_decode_finished_waiting = set()
    adapter._async_decode_finished_ready = set()
    adapter._decode_save_completion_lock = threading.Lock()
    adapter._decode_save_completions = deque()
    adapter._decode_window_save_expected_start = {"request": 0}
    adapter._consume_completed_layerwise_store = MagicMock()
    adapter._clear_decode_window_save_groups_for_window = MagicMock()
    first = _AsyncDecodeSaveTask(_request(1, 0, 256), None)
    second = _AsyncDecodeSaveTask(_request(2, 256, 512), None)
    adapter._async_decode_jobs = {"request": deque((first, second))}
    return adapter, first, second


def test_physical_completion_is_reported_before_ordered_promotion() -> None:
    adapter, first, second = _adapter_with_two_jobs()

    adapter._finish_async_decode_task(second, [(0, "second")])

    [completion] = adapter.get_decode_save_completions()
    assert completion.job_id == 2
    adapter._consume_completed_layerwise_store.assert_not_called()
    assert len(adapter._async_decode_jobs["request"]) == 2

    adapter._finish_async_decode_task(first, [(0, "first")])

    [completion] = adapter.get_decode_save_completions()
    assert completion.job_id == 1
    assert adapter._consume_completed_layerwise_store.call_args_list == [
        call(first.request, 0, True, "first"),
        call(second.request, 0, True, "second"),
    ]
    assert "request" not in adapter._async_decode_jobs


def test_npu_dispatch_stores_each_kv_group_once_with_all_layers() -> None:
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    engine = MagicMock()
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._decode_window_save_required_groups = MagicMock(return_value={0, 1})
    adapter._refresh_kvcaches_list = MagicMock()
    adapter._kvcaches_for_group = MagicMock(
        side_effect=lambda group: [f"group-{group}-layer-0", f"group-{group}-layer-1"]
    )
    store_mask = object()
    slot_mapping = object()
    adapter._prepare_layerwise_store_inputs = MagicMock(
        side_effect=lambda request, save_spec, group: (
            [1, 2],
            slot_mapping,
            store_mask,
            0,
            {"kv_group": group, "decode_window_save": True},
            None,
        )
    )
    adapter._finalize_layerwise_storer = MagicMock(
        side_effect=[
            (True, SimpleNamespace(committed_end=256)),
            (True, SimpleNamespace(committed_end=256)),
        ]
    )
    task = _AsyncDecodeSaveTask(_request(1, 0, 256), ordering_event="event")

    results = adapter._run_async_decode_npu_store(task)

    assert [group for group, _ in results] == [1, 0]
    assert engine.store_layer.call_count == 2
    assert engine.store_layer.call_args_list[0].kwargs["kvcaches"] == [
        "group-1-layer-0",
        "group-1-layer-1",
    ]
    assert engine.store_layer.call_args_list[1].kwargs["kvcaches"] == [
        "group-0-layer-0",
        "group-0-layer-1",
    ]
    for store_call in engine.store_layer.call_args_list:
        assert store_call.kwargs["ordering_event"] == "event"
        assert store_call.kwargs["async_decode_job_key"] == task.owner


def test_dispatcher_round_robins_across_requests() -> None:
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter._async_decode_lock = threading.RLock()
    adapter._async_decode_queue = queue.Queue(maxsize=8)
    adapter._async_decode_ready_reqs = set()
    first_a = _AsyncDecodeSaveTask(_request(1, 0, 256), None)
    second_a = _AsyncDecodeSaveTask(_request(2, 256, 512), None)
    first_b_request = _request(1, 0, 256)
    first_b_request.req_id = "request-b"
    first_b = _AsyncDecodeSaveTask(first_b_request, None)
    adapter._async_decode_jobs = {
        "request": deque((first_a, second_a)),
        "request-b": deque((first_b,)),
    }
    adapter._async_decode_error = None
    adapter._enqueue_async_decode_request("request")
    adapter._enqueue_async_decode_request("request-b")

    dispatched = []
    for _ in range(3):
        req_id = adapter._async_decode_queue.get_nowait()
        task = adapter._take_queued_async_decode_task(req_id)
        assert task is not None
        dispatched.append((task.request.req_id, task.request.decode_save_job_id))

    assert dispatched == [
        ("request", 1),
        ("request-b", 1),
        ("request", 2),
    ]
