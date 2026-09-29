# SPDX-License-Identifier: Apache-2.0
"""Request source leases survive submissions without a completion event."""
import ast
import gc
import importlib.util
from pathlib import Path
from types import MethodType

import pytest
import torch


_PATH = Path(__file__).with_name("test_layerwise_prefill_dma.py")
_SPEC = importlib.util.spec_from_file_location("prefill_dma_failure_fixture", _PATH)
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def _attach_release_methods(connector):
    path = Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "VLLMPagedMemLayerwiseNPUConnector")
    names = {"_retain_unfenced_layerwise_prefill_load", "release_layerwise_prefill_dma_cache"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    for method in methods:
        setattr(connector, method.name, MethodType(scope[method.name], connector))


@pytest.mark.parametrize("packet", [False, True])
@pytest.mark.parametrize("earlier_success", [False, True])
@pytest.mark.parametrize("failure", ["native", "event_record"])
@pytest.mark.parametrize("gc_off", [False, True])
def test_unrecorded_submission_retains_sources_until_fallback_fence(
    packet, earlier_success, failure, gc_off,
):
    enabled = gc.isenabled()
    if gc_off:
        gc.disable()
    try:
        _exercise(packet, earlier_success, failure, fail_fallback=True)
    finally:
        if enabled:
            gc.enable()


def _exercise(packet, earlier_success, failure, fail_fallback):
    npu = _fixture._AsyncNPU()
    state = {"refs": 1, "armed": False, "fail_fallback": fail_fallback}
    stream_fences, event_fences, reads = [], [], []

    class Stream(_fixture._AsyncStream):
        def synchronize(self):
            stream_fences.append(self)
            if state["fail_fallback"]:
                raise RuntimeError("unknown DMA completion")
            if self.tail is not None:
                self.tail.run()

    class Event:
        def record(self, stream):
            if state["armed"] and failure == "event_record":
                raise RuntimeError("record failed after submission")
            self.tail = stream.tail
        def synchronize(self):
            event_fences.append(self)
            if self.tail is not None:
                self.tail.run()

    npu.Event = Event
    banks = [Stream(), Stream()]
    def copy(rows, direction):
        assert not direction
        def read():
            assert state["refs"] > 0, "queued DMA read freed source"
            reads.append(state["refs"])
        npu.current_stream().enqueue(read)
        if state["armed"] and failure == "native":
            raise RuntimeError("native enqueued then raised")

    submit, _, _ = _fixture._reuse_debug_connector(False, npu=npu, copy_hook=copy, layers=4)
    obj = submit.connector
    _attach_release_methods(obj)
    obj._layerwise_prefill_dma_stream = lambda group, bank: banks[bank]
    if packet:
        original_layout = obj._lazy_initialize_buffer_with_staging
        def layout(*args, **kwargs):
            result = original_layout(*args, **kwargs)
            result.indexer_c8 = True
            return result
        obj._lazy_initialize_buffer_with_staging = layout
        obj._prepare_prefill_c8_packets = lambda **kwargs: []
        obj._submit_prefill_c8_packet = lambda prepared, layer, bank, stream, direction: copy(None, direction)
    owner = submit.owners[0]
    owner.ref_count_up = lambda: state.__setitem__("refs", state["refs"] + 1)
    owner.ref_count_down = lambda: state.__setitem__("refs", state["refs"] - 1)
    owner.is_valid = lambda: state["refs"] > 0
    generator = obj.run(
        [0], [4], slot_mapping=torch.empty(0, dtype=torch.long), sync=True,
        kv_group=int(packet), req_id="request", deferred_layerwise_get=True,
        prefill_dma_block_ids_by_bank=((0, 1, 2, 3), (4, 5, 6, 7)),
        prefill_dma_block_size=4,
        prefill_c8_memory_objs=[[owner]] * 4,
    )
    next(generator)
    if earlier_success:
        generator.send([owner])
    state["armed"] = failure is not None
    if failure is not None:
        with pytest.raises(RuntimeError, match="native enqueued|record failed"):
            generator.send([owner])
        assert not reads and state["refs"] == 2
        # Earlier layer events cannot stand in for the failed submission.
        assert len(obj._layerwise_prefill_load_request_events.get("request", [])) == int(earlier_success)
        failed_stream = banks[int(earlier_success)]
        # Even replacing the bank registry must not redirect the fallback fence.
        obj._layerwise_prefill_dma_stream = lambda *args: pytest.fail("lost original stream")
        with pytest.raises(RuntimeError, match="unknown DMA completion"):
            obj.release_layerwise_prefill_dma_cache("request")
        assert len(obj._layerwise_prefill_unfenced_loads["request"]) == 1
        assert state["refs"] == 2 and "request" in obj._layerwise_prefill_unfenced_loads
        assert "request" in obj._layerwise_prefill_load_owners and not reads
        state["fail_fallback"] = False
        obj.release_layerwise_prefill_dma_cache("request")
        assert stream_fences == [failed_stream, failed_stream]
        assert "request" not in obj._layerwise_prefill_unfenced_loads
    else:
        generator.send([owner])
        generator.close()  # Recorded-event GeneratorExit is not an unsafe submission.
        assert not getattr(obj, "_layerwise_prefill_unfenced_loads", {})
        obj.release_layerwise_prefill_dma_cache("request")
        assert not stream_fences and event_fences
    assert len(reads) == 1 + int(earlier_success)
    assert state["refs"] == 1
    obj.release_layerwise_prefill_dma_cache("request")
    assert state["refs"] == 1  # Idempotent teardown.
    owner.ref_count_down()  # Caller finally releases its reused-prefix lease.
    assert state["refs"] == 0


@pytest.mark.parametrize("packet", [False, True])
def test_recorded_event_close_does_not_add_fallback_stream_fence(packet):
    _exercise(packet, False, None, fail_fallback=False)


def test_failed_legacy_deferred_load_retains_original_load_stream():
    submit, _, _ = _fixture._reuse_debug_connector(False)
    obj = submit.connector
    _attach_release_methods(obj)
    calls = []
    original = type("Stream", (), {"synchronize": lambda self: calls.append("old")})()
    obj._retain_unfenced_layerwise_prefill_load("request", 0, None, original)
    obj.load_stream = object()
    obj.release_layerwise_prefill_dma_cache("request")
    assert calls == ["old"]


def test_failed_recorded_event_after_fallback_keeps_request_owners():
    submit, _, _ = _fixture._reuse_debug_connector(False)
    obj = submit.connector
    _attach_release_methods(obj)
    state = {"refs": 1, "fail_event": True}
    owner = submit.owners[0]
    owner.ref_count_down = lambda: state.__setitem__("refs", state["refs"] - 1)
    owner.is_valid = lambda: state["refs"] > 0
    calls = []
    stream = type("Stream", (), {"synchronize": lambda self: calls.append("fallback")})()
    def fence(self):
        if state["fail_event"]:
            raise RuntimeError("recorded fence failed")
    event = type("Event", (), {"synchronize": fence})()
    obj._layerwise_prefill_load_request_events["request"] = [event]
    obj._layerwise_prefill_load_owners["request"] = {id(owner): owner}
    obj._retain_unfenced_layerwise_prefill_load("request", 0, 1, stream)
    with pytest.raises(RuntimeError, match="recorded fence failed"):
        obj.release_layerwise_prefill_dma_cache("request")
    assert state["refs"] == 1 and obj._layerwise_prefill_load_request_events["request"] == [event]
    assert "request" not in obj._layerwise_prefill_unfenced_loads
    state["fail_event"] = False
    obj.release_layerwise_prefill_dma_cache("request")
    assert state["refs"] == 0 and calls == ["fallback"]
