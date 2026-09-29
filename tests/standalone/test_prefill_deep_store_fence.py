# SPDX-License-Identifier: Apache-2.0
"""CPU FIFO proof that deep store fingerprints observe completed host bytes."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class FIFO:
    def __init__(self):
        self.pending = []
        self.completed = 0

    def enqueue(self, action):
        self.pending.append(action)

    def event(self):
        return Event(self, len(self.pending))


class Event:
    def __init__(self, fifo, frontier):
        self.fifo = fifo
        self.frontier = frontier
        self.waits = 0

    def synchronize(self):
        self.waits += 1
        while self.fifo.completed < self.frontier:
            self.fifo.pending[self.fifo.completed]()
            self.fifo.completed += 1


def diagnostic(enabled=True, async_store=True, offset=0, req_id="r"):
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    closure = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "log_completed_store_layer"
    )
    host = [0 if async_store else 23]
    streams = [FIFO(), FIFO()]
    bank = offset % 2
    streams[bank].enqueue(lambda: host.__setitem__(0, 23))
    events = {i: stream.event() for i, stream in enumerate(streams)}
    # A device-wide or full stream drain would execute this unrelated suffix.
    streams[bank].enqueue(lambda: host.__setitem__(0, 99))
    records = []
    namespace = dict(
        _mtp_dw_deep_diag_enabled=lambda: enabled,
        kwargs={"req_id": req_id},
        memory_objs=[[host]],
        starts=[0],
        ends=[1],
        _layer_memory_tensor=lambda obj, layer: obj,
        _bounded_tensor_fingerprint=lambda tensor: tensor[0],
        _mtp_dw_event=lambda *args, **kw: records.append(kw),
        kv_group=1,
        async_layerwise_store=async_store,
        last_store_events=events,
        source_bank_count=2,
        layerwise_prefill_bank_offset=offset,
    )
    exec(
        compile(ast.Module(body=[closure], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[closure.name], events, records, host


@pytest.mark.parametrize("offset", [0, 1])
def test_async_deep_fingerprint_waits_only_relevant_bank_frontier(offset):
    log, events, records, host = diagnostic(offset=offset)
    log(0)
    assert records[0]["chunk_ranges"][0]["fingerprint"] == 23
    assert host == [23]
    assert events[offset].waits == 1
    assert events[1 - offset].waits == 0
    assert events[offset].fifo.completed == events[offset].frontier


@pytest.mark.parametrize(
    "enabled,layer,req_id,async_store",
    [
        (False, 0, "r", True),
        (True, 1, "r", True),
        (True, 0, None, True),
        (True, 0, "r", False),
    ],
)
def test_no_new_wait_without_async_layer_zero_diagnostic(
    enabled, layer, req_id, async_store
):
    log, events, records, host = diagnostic(enabled, async_store, req_id=req_id)
    log(layer)
    assert all(event.waits == 0 for event in events.values())
    if enabled and layer == 0 and req_id is not None:
        assert records[0]["chunk_ranges"][0]["fingerprint"] == 23
    else:
        assert not records
