# SPDX-License-Identifier: Apache-2.0
"""Exercise event attribution with delayed DMA and ignored device waits."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fence_trace_test", ROOT / "lmcache_ascend/v1/kv_fence_debug.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Event:
    def __init__(self, ready=False):
        self.ready = ready
        self.syncs = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.syncs += 1
        self.ready = True


def debug(mode="p_sfa,trace", emit=None, **env):
    records = []
    recorder = MODULE.KVFenceDebug(
        "P",
        0,
        {"LMCACHE_PREFILL_FENCE_DEBUG": mode, **env},
        emit or records.append,
    )
    return recorder, records


def stream_wrapper(stream_id, synchronize):
    class Stream:
        device = "npu:0"

        @property
        def npu_stream(self):
            raise AssertionError("must not drain host task queue for a stream ID")

    result = Stream()
    result.stream_id = stream_id
    result.synchronize = synchronize
    return result


def rows(record):
    return [item for row in record["dependencies"] for item in (row, *row["after"])]


def test_mixed_request_loads_both_groups_and_stable_stream_identity():
    d, records = debug(LMCACHE_PREFILL_FENCE_GROUPS="0")
    sources = [Event(), Event(), Event(True)]
    stream = stream_wrapper(7, lambda: [event.synchronize() for event in sources])
    for index, event in enumerate(sources):
        d.trace_record(
            "load", event, NS(), req=f"r{index}", group=index % 2, layer=3, bank=3
        )
        d.trace_join(stream, event)
    # Real current_stream() calls can create another wrapper for this stream.
    d.before_sfa(0, 3, 3, lambda: stream_wrapper(7, stream.synchronize))
    begin = next(r for r in records if r["event"] == "sync_begin")
    end = records[-1]
    assert {r["req"] for r in begin["dependencies"]} == {"r0", "r1", "r2"}
    assert [r["ready"] for r in begin["dependencies"]] == [False, False, True]
    assert all(r["ready"] for r in end["dependencies"])
    assert end["pending_after"] == []
    assert end["sync_ms"] >= 0
    assert not d.trace.waits[("npu", "npu:0", 7)]


def test_logging_delay_is_not_reported_as_sync_time_or_pending_at_sync():
    event = Event()
    records = []

    def emit(record):
        records.append(record)
        if record["event"] == "sync_begin":
            event.ready = True  # DMA can finish while stdout is being written.

    d, _ = debug(emit=emit)
    stream = NS(synchronize=lambda: None)
    d.trace_record("load", event, NS(), req="r", group=0, layer=0, bank=0)
    d.trace_join(stream, event)
    d.before_sfa(0, 0, 0, lambda: stream)
    assert records[-2]["dependencies"][0]["ready"] is False
    assert records[-1]["before_sync"][0]["ready"] is True
    assert records[-1]["pending_after"] == []


def test_trace_only_observes_without_waiting_and_keeps_old_pending_loads():
    d, records = debug("trace")
    event = Event()
    stream = NS(synchronize=lambda: pytest.fail("trace must never synchronize"))
    d.trace_record("load", event, NS(), group=1, layer=0, bank=0, req="a")
    d.trace_join(stream, event)
    d.begin_step([], (79, 22), 4, 1024)
    d.before_sfa(0, 1, 1, lambda: stream)
    d.begin_step([], (79, 22), 4, 1024)
    d.before_sfa(0, 2, 2, lambda: stream)
    assert records[-1]["event"] == "checkpoint"
    assert records[-1]["dependencies"][0]["req"] == "a"
    assert event.syncs == 0
    assert records[-1]["step"] == 2


def test_input_marker_separates_ready_load_from_unfinished_compute():
    d, records = debug()
    load = Event(True)
    marker = Event()
    marker.record = lambda stream: None
    stream = NS(synchronize=marker.synchronize)
    d.trace_record("load", load, NS(), req="a", group=0, layer=1, bank=1)
    d.trace_join(stream, load)
    d.before_sfa(0, 1, 1, lambda: stream, event_factory=lambda: marker)
    before = rows(records[-2])
    assert next(r for r in before if r["kind"] == "sfa_inputs")["ready"] is False
    assert next(r for r in before if r["kind"] == "load")["ready"] is True
    assert records[-1]["pending_after"] == []


def test_ignored_runtime_wait_is_visible_after_successful_stream_sync():
    d, records = debug()
    load = Event()
    marker = Event()
    marker.record = lambda stream: None
    # Fault injection: the runtime completes compute without honoring its wait.
    stream = NS(synchronize=marker.synchronize)
    d.trace_record("load", load, NS(), req="bad", group=0, layer=1, bank=1)
    d.trace_join(stream, load)
    d.before_sfa(0, 1, 1, lambda: stream, event_factory=lambda: marker)
    assert records[-1]["pending_after"] == ["e1"]


def test_trace_only_detects_completed_input_marker_with_incomplete_load():
    d, records = debug("trace")
    load, marker = Event(), Event(True)
    marker.record = lambda stream: None
    stream = NS(synchronize=lambda: pytest.fail("no diagnostic sync"))
    d.trace_record("load", load, NS(), req="bad", group=0, layer=1, bank=1)
    d.trace_join(stream, load)
    d.before_sfa(0, 1, 1, lambda: stream, event_factory=lambda: marker)
    assert records[-1]["ordering_violation"] == ["e1"]


def test_p_use_retains_load_provenance_through_input_marker():
    d, records = debug("p_use,trace")
    load, marker, use = Event(), Event(), Event()
    marker.record = lambda stream: None
    stream = NS()
    d.trace_record("load", load, NS(), req="a", group=0, layer=4, bank=0)
    d.trace_join(stream, load)
    d.before_sfa(0, 4, 0, lambda: stream, event_factory=lambda: marker)
    d.trace_record("use", use, stream, group=0, layer=4, bank=0)
    use.synchronize = lambda: [setattr(e, "ready", True) for e in (load, marker, use)]
    d.wait("p_use", 0, 4, use, bank=0)
    assert any(r.get("req") == "a" for r in rows(records[-2]))
    assert records[-1]["pending_after"] == []


def test_sync_failure_has_matching_begin_error_and_original_exception():
    d, records = debug()

    def failed():
        raise RuntimeError("injected sync error")

    with pytest.raises(RuntimeError, match="device_fence_failed"):
        d.before_sfa(0, 2, 2, lambda: NS(synchronize=failed))
    begin = next(r for r in records if r["event"] == "sync_begin")
    error = next(r for r in records if r["event"] == "sync_error")
    assert begin["sync"] == error["sync"]
    assert "injected sync error" in error["error"]


def test_query_failure_is_logged_not_silently_marked_ready():
    d, records = debug()
    event = Event()
    event.query = lambda: (_ for _ in ()).throw(RuntimeError("query failed"))
    d.trace_record("load", event, NS(), req="r", group=0, layer=0, bank=0)
    stream = NS(synchronize=lambda: None)
    d.trace_join(stream, event)
    d.before_sfa(0, 0, 0, lambda: stream)
    row = records[-1]["dependencies"][0]
    assert row["ready"] is None and "query failed" in row["query_error"]


def test_filter_does_not_create_marker_or_touch_stream():
    d, _ = debug(LMCACHE_PREFILL_FENCE_LAYERS="4:5")
    fail = lambda: pytest.fail("filtered boundary touched NPU")
    d.before_sfa(0, 0, 0, fail, event_factory=fail)


def test_bank_save_and_load_labels_keep_upstream_use():
    d, records = debug()
    compute, bank = NS(synchronize=lambda: None), NS()
    use, save, load = Event(True), Event(True), Event()
    d.trace_record("use", use, compute, group=0, layer=0, bank=0)
    d.trace_join(bank, use)
    d.trace_record("save", save, bank, group=0, layer=0, bank=0, req="a")
    d.trace_record("load", load, bank, group=0, layer=4, bank=0, req="a")
    d.trace_join(compute, load)
    compute.synchronize = load.synchronize
    d.before_sfa(0, 4, 0, lambda: compute)
    row = next(r for r in records[-2]["dependencies"] if r["kind"] == "load")
    assert row["layer"] == 4 and row["after"][0]["kind"] == "save"


def test_history_is_bounded_between_completed_batches():
    d, _ = debug()
    compute = stream_wrapper(0, lambda: None)
    bank = stream_wrapper(1, lambda: None)
    for _ in range(100):
        d.begin_step([], (79, 22), 4, 1024)
        for layer in range(5):
            event = Event(True)
            d.trace_record("load", event, bank, group=0, layer=layer, bank=0)
            d.trace_join(compute, event)
            d.before_sfa(0, layer, 0, lambda: compute)
    d.begin_step([], (79, 22), 4, 1024)
    assert len(d.trace.events) <= 2
    assert len(d.trace.streams) == 2


def test_final_cleanup_still_identifies_loads_from_an_earlier_chunk():
    d, records = debug("trace")
    bank = NS()
    old = Event(True)
    d.trace_record("load", old, bank, req="r", group=0, layer=3, bank=3)
    original_id = d.trace.events[id(old)].fields["id"]
    d.trace_record("load", Event(True), bank, req="r", group=0, layer=7, bank=3)
    d.begin_step([], (79, 22), 4, 1024)
    d.observe_existing(old, stage="load_finish", req="r")
    row = records[-1]["dependencies"][0]
    assert row["kind"] == "load" and row["layer"] == 3
    assert row["id"] == original_id
    d.trace.forget_request("r")
    d.begin_step([], (79, 22), 4, 1024)
    assert id(old) not in d.trace.events


@pytest.mark.parametrize("group", [0, 1, "c8"])
def test_real_load_and_use_callbacks_trace_the_actual_dma_events(monkeypatch, group):
    from test_layerwise_prefill_dma import _AsyncNPU, _async_bank_connector

    monkeypatch.setattr(
        _AsyncNPU.Event, "query", lambda e: e.tail is None or e.tail.done, raising=False
    )
    monkeypatch.setattr(
        _AsyncNPU.Event,
        "synchronize",
        lambda e: e.tail.run() if e.tail is not None else None,
    )
    api = _async_bank_connector(group, bank_count=4, layers=6)
    d, records = debug("p_use,trace")
    api.connector.fence_debug = d
    physical_group = 1 if group == "c8" else group
    for chunk in range(2):
        d.begin_step([], (6, 6), 4, 4)
        generator, load = api.start(chunk)
        for layer in range(4):
            load(layer)
        for layer in range(6):
            api.read(chunk, layer)
            api.save(layer)
            if layer + 4 < 6:
                load(layer + 4)
        generator.close()
    starts = [r for r in records if r["event"] == "sync_begin"]
    assert len(starts) == 12
    for record in starts:
        assert any(
            row.get("kind") == "load"
            and row.get("req") == "request"
            and row.get("group") == physical_group
            and row.get("layer") == record["layer"]
            for row in rows(record)
        )
    assert all(not r["pending_after"] for r in records if r["event"] == "sync_end")
    assert api.observed == [(chunk, layer) for chunk in range(2) for layer in range(6)]


def test_summary_reports_pending_load_compute_only_and_unfinished(tmp_path):
    import json

    spec = importlib.util.spec_from_file_location(
        "fence_summary_test", ROOT / "tools/summarize_kv_fences.py"
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    path = tmp_path / "P.log"
    common = dict(host="P", pid=1, stage="p_sfa", step=2, layer=0, group=0, bank=0)
    load = dict(id="e1", kind="load", req="r", ready=False)
    records = [
        dict(common, event="sync_begin", sync=1, dependencies=[load]),
        dict(
            common,
            event="sync_end",
            sync=1,
            sync_ms=2,
            before_sync=[load],
            dependencies=[dict(load, ready=True)],
            pending_after=[],
        ),
        dict(common, event="sync_begin", sync=2, dependencies=[dict(load, ready=True)]),
    ]
    path.write_text("\n".join("[KVFENCE] " + json.dumps(r) for r in records))
    result = summary.summarize([path], "r", 3)
    assert result["stages"]["p_sfa"]["load_pending"] == 1
    assert result["unfinished_syncs"] == 1
    assert result["pending_after_count"] == 0
    assert "dependencies" not in result["first_unfinished"]
