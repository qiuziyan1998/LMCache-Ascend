# SPDX-License-Identifier: Apache-2.0
"""Exact slot mapping and host-storage lifetime; CPU pinning is explicitly mocked."""

import gc
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from test_capture_submission import fixture as connector_fixture
from test_preemption_checkpoint import api as checkpoint_api, fake_engine

api = checkpoint_api


@pytest.fixture
def slots(api, monkeypatch):
    def pinned_empty(shape, **kwargs):
        assert kwargs.pop("pin_memory") is True
        assert kwargs["dtype"] == torch.long
        return torch.empty(shape, **kwargs)

    monkeypatch.setattr(api[1], "torch", NS(empty=pinned_empty, long=torch.long))
    return api[1]._checkpoint_slots


@pytest.mark.parametrize("block_size", [1, 3, 4, 128])
def test_partial_boundaries_and_nonmonotonic_blocks(slots, block_size):
    blocks = (9, 2, 9, 5)
    boundaries = sorted(
        {
            0,
            1,
            block_size - 1,
            block_size,
            block_size + 1,
            2 * block_size,
            4 * block_size - 1,
            4 * block_size,
        }
    )
    for start in boundaries:
        for end in boundaries:
            if not 0 <= start < end <= len(blocks) * block_size:
                continue
            result = slots(blocks, start, end, block_size)
            expected = [
                blocks[i // block_size] * block_size + i % block_size
                for i in range(start, end)
            ]
            assert result.tolist() == expected
            assert result.is_contiguous() and result.dtype == torch.long
            assert result.storage_offset() == start % block_size
            assert (
                result.untyped_storage().nbytes() < (end - start + 2 * block_size) * 8
            )


@pytest.mark.parametrize("count", [32, 256, 4096, 100000, 1000000])
def test_large_ranges_under_long_context(slots, count):
    start, block_size = 131, 128
    blocks = tuple(range(9000, 9000 + (start + count + 127) // 128))
    result = slots(blocks, start, start + count, block_size)
    # Consecutive physical blocks give a simple independent vector reference.
    assert torch.equal(result, torch.arange(count) + 9000 * block_size + start)


def test_int64_overflow_is_not_silently_wrapped(slots):
    limit = (1 << 63) - 1
    assert slots([limit // 3], 0, 2, 3).tolist() == [limit - 1, limit]
    for blocks, end in (([limit // 3], 3), ([limit // 3, 1], 4)):
        with pytest.raises(OverflowError):
            slots(blocks, 0, end, 3)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_slot_allocation_failure_preserves_capture_cleanup(
    api, monkeypatch, failure_call
):
    count = 0

    def empty(shape, **kwargs):
        nonlocal count
        count += 1
        if count == failure_call:
            raise RuntimeError("pinned allocation failed")
        kwargs.pop("pin_memory")
        return torch.empty(shape, **kwargs)

    monkeypatch.setattr(api[1], "torch", NS(empty=empty, long=torch.long))
    engine = fake_engine()
    worker = api[1].CheckpointWorker(engine)
    try:
        spec = api[0].CaptureSpec("r", 1, 0, 4, 0, ((1,), (2,)), prefix_end=0)
        worker.capture(spec, {0: [1], 1: [2]}, 4)
        assert worker.poll()[0].status == "failed"
        assert all(page.refs == 0 for page in engine.allocated)
        assert not any(call[0] == "enqueue" for call in engine.calls)
        assert (("failure-fence",) in engine.calls) == (failure_call == 2)
    finally:
        worker.close()


def test_capture_plan_retains_offset_view_until_retirement_with_gc_disabled(slots):
    connector, rows, events, _ = connector_fixture()
    connector._slot_mapping_on_kv_device = lambda host, stream: host.clone()
    host = slots([7, 2], 1, 5, 4)
    owner = weakref.ref(host._base)
    enabled = gc.isenabled()
    gc.disable()
    try:
        plan = connector.prepare_group_capture(
            rows, [1], [5], slot_mapping=host, slot_mapping_base=1, kv_group=0
        )
        assert plan.host_metadata[-1] is host
        del host
        assert owner() is not None
        assert plan.slots.tolist() == [29, 30, 31, 8]
        assert events == []
        del plan
        assert owner() is None
    finally:
        if enabled:
            gc.enable()


def test_pinned_offset_view_npu_upload(api):
    """Run explicitly on NPU to qualify real pinning and nonblocking upload."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    host = api[1]._checkpoint_slots((7, 2), 1, 5, 4)
    assert host.is_pinned() and host.is_contiguous()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        device = host.to("npu", non_blocking=True)
    stream.synchronize()
    assert device.cpu().tolist() == [29, 30, 31, 8]
