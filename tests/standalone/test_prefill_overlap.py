# SPDX-License-Identifier: Apache-2.0
"""CPU timelines prove metric arithmetic, nonblocking collection and gating."""

import importlib.util
import ast
import json
from pathlib import Path
import sys
from threading import Event, get_ident
from types import SimpleNamespace as NS

import pytest


_PATH = Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/prefill_overlap.py"
_SPEC = importlib.util.spec_from_file_location("prefill_overlap_test_module", _PATH)
overlap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = overlap
_SPEC.loader.exec_module(overlap)


class DeviceEvent:
    def __init__(self, time=0.0, ready=True):
        self.time = time
        self.ready = ready
        self.queries = 0

    def record(self, stream):
        self.time = stream.time
        self.ready = stream.ready

    def query(self):
        self.queries += 1
        return self.ready

    def elapsed_time(self, end):
        assert self.ready and end.ready, "read before completion"
        return end.time - self.time

    def synchronize(self):
        pytest.fail("statistics must never synchronize")


def event_elapsed(start, end):
    return start.elapsed_time(end)


def test_batch_timestamp_is_captured_at_submission_not_collector_emission():
    emitted = []
    recorder = overlap.PrefillOverlapRecorder(
        DeviceEvent,
        emitted.append,
        worker={},
        elapsed_time=event_elapsed,
        start_worker=False,
    )
    recorder.begin_step((("r", 0, 4096),))
    timestamp = recorder.active.start_time_ns
    assert timestamp > 0
    stream = NS(time=0.0, ready=True)
    ticket = recorder.start_interval("load", stream, 0, 0, 0)
    stream.time = 3.0
    recorder.end_interval(ticket, stream)
    recorder.finish_step()
    assert not emitted
    assert recorder.poll()
    result = json.loads(emitted[0].split(overlap.LOG_MARKER)[1])
    assert result["batch_start_time_ns"] == timestamp
    assert result["time_basis"] == "host_batch_submission"
    assert result["load_ms"] == 3.0


def test_acl_timer_uses_raw_handles_and_never_torch_elapsed_or_sync():
    calls = []

    class Function:
        def __call__(self, output, start, end):
            calls.append((start.value, end.value))
            output._obj.value = 12.5
            return 0

    function = Function()
    timer = overlap.AclEventTimer(
        library_loader=lambda: NS(aclrtEventElapsedTime=function),
    )

    class NpuEvent:
        def __init__(self, handle):
            self.npu_event = handle

        def elapsed_time(self, end):
            pytest.fail("torch-npu elapsed_time implicitly drains queues")

        def synchronize(self):
            pytest.fail("must not synchronize an already completed event")

    assert timer(NpuEvent(123), NpuEvent(456)) == 12.5
    assert calls == [(123, 456)]


def test_acl_timer_reports_runtime_error_instead_of_inventing_zero_time():
    class Function:
        def __call__(self, *args):
            return 42

    timer = overlap.AclEventTimer(
        library_loader=lambda: NS(aclrtEventElapsedTime=Function()),
    )
    with pytest.raises(RuntimeError, match="ACL error 42"):
        timer(NS(npu_event=1), NS(npu_event=2))


def interval(kind, group, layer, start, end, model_layer=None, bank=...):
    return overlap.TimingInterval(
        kind,
        group,
        layer % 2 if bank is ... else bank,
        layer,
        layer if model_layer is None else model_layer,
        DeviceEvent(start),
        DeviceEvent(end),
    )


def test_same_bank_mean_of_layer_ratios_and_all_ready_layer_starts():
    batch = overlap.StepTimings(1, (("r", 4096, 8192),))
    batch.intervals = [
        interval("wait", 0, 0, 0, 1, bank=0),
        interval("wait", 0, 1, 9, 11, bank=0),
        interval("wait", 0, 2, 21, 29, bank=0),
        # Sparse indexer row 1 belongs to MODEL layer 2, not model layer 1.
        interval("wait", 1, 1, 29, 31, model_layer=2),
        interval("load", 0, 0, 0, 3),
        interval("save", 0, 1, 10, 15),
        interval("load", 1, 1, 16, 20),
    ]
    result = overlap.summarize_step(batch, event_elapsed)
    assert result["layer_samples"] == 2
    assert result["schema_version"] == 2
    assert result["coverage_definition"] == "same_bank_reuse_v1"
    assert result["coverage"] == pytest.approx((0.8 + 0.6) / 2)
    assert result["coverage"] != pytest.approx(1 - 10 / 30)
    assert result["first_layer_wait_ms"] == 1
    assert result["wait_ms"] == 10
    assert result["initial_bank_wait_ms"] == 3
    assert result["all_wait_ms"] == 13
    assert result["load_ms"] == 7
    assert result["save_ms"] == 5
    assert result["dma_ms"] == 12
    assert result["requests"][0]["token_end"] == 8192


def test_different_banks_and_unknown_bank_have_no_invented_reuse_sample():
    batch = overlap.StepTimings(1, (("r", 0, 17),))
    batch.intervals = [interval("wait", 0, 0, 0, 1)]
    assert overlap.summarize_step(batch, event_elapsed)["coverage"] is None
    batch.intervals.append(interval("wait", 0, 2, 10, 11, bank=1))
    assert overlap.summarize_step(batch, event_elapsed)["layer_samples"] == 0
    batch.intervals.append(interval("wait", 0, 3, 20, 21, bank=None))
    result = overlap.summarize_step(batch, event_elapsed)
    assert result["layer_samples"] == 0 and result["unknown_bank_waits"] == 1
    assert result["all_wait_ms"] == 3


@pytest.mark.parametrize("bank_count", [2, 4, 8])
def test_real_same_bank_predecessor_tracks_rotation_and_chunk_phase(bank_count):
    batch = overlap.StepTimings(1, (("r", 4096, 8192),))
    expected = []
    for layer in range(bank_count * 2):
        waited = 1 if layer < bank_count else layer - bank_count + 2
        end = layer * 10 + waited
        batch.intervals.append(
            interval(
                "wait",
                0,
                layer,
                layer * 10,
                end,
                bank=(layer + 1) % bank_count,
            )
        )
        if layer >= bank_count:
            previous_start = (layer - bank_count) * 10 + 1
            expected.append(1 - waited / (end - previous_start))
    result = overlap.summarize_step(batch, event_elapsed)
    assert result["layer_samples"] == bank_count
    assert result["bank_reuse_samples"] == bank_count
    assert result["coverage"] == pytest.approx(sum(expected) / bank_count)
    assert result["initial_bank_wait_ms"] == bank_count
    assert all(bank["reuse_samples"] == 1 for bank in result["banks"])


def test_sparse_indexer_pairs_its_own_bank_across_nonadjacent_model_layers():
    batch = overlap.StepTimings(1, (("r", 4096, 8192),))
    for ordinal, model_layer in enumerate([3, 7, 11, 15, 19]):
        batch.intervals.append(
            interval(
                "wait",
                1,
                ordinal,
                ordinal * 10,
                ordinal * 10 + 1,
                model_layer=model_layer,
                bank=(ordinal + 2) % 4,
            )
        )
    result = overlap.summarize_step(batch, event_elapsed)
    assert result["bank_reuse_samples"] == result["layer_samples"] == 1
    assert result["coverage"] == pytest.approx(1 - 1 / 40)


def test_kv_groups_have_independent_banks_and_model_layers_have_equal_weight():
    batch = overlap.StepTimings(1, (("r", 4096, 8192),))
    batch.intervals = [
        interval("wait", 0, 0, 0, 1),
        interval("wait", 1, 0, 1, 2),
        interval("wait", 0, 1, 10, 11),
        interval("wait", 1, 1, 11, 12),
        interval("wait", 0, 2, 20, 24),
        interval("wait", 1, 2, 24, 26),
        interval("wait", 0, 3, 30, 31),
    ]
    result = overlap.summarize_step(batch, event_elapsed)
    layer2 = ((1 - 4 / 24) + (1 - 2 / 24)) / 2
    layer3 = 1 - 1 / 19
    assert result["coverage"] == pytest.approx((layer2 + layer3) / 2)
    assert result["layer_samples"] == 2 and result["bank_reuse_samples"] == 3
    assert result["initial_bank_wait_ms"] == 4
    assert result["wait_ms"] == 7 and result["all_wait_ms"] == 11


def test_out_of_range_ratios_are_not_silently_clamped():
    batch = overlap.StepTimings(1, (("r", 0, 4096),))
    batch.intervals = [
        interval("wait", 0, 0, 0, 8),
        interval("wait", 0, 1, 2, 9, bank=0),
    ]
    result = overlap.summarize_step(batch, event_elapsed)
    assert result["coverage"] == -6
    assert result["out_of_range_layers"] == 1


def test_no_submission_queries_and_pending_events_are_not_recycled():
    created, emitted = [], []

    def factory():
        event = DeviceEvent()
        created.append(event)
        return event

    recorder = overlap.PrefillOverlapRecorder(
        factory,
        emitted.append,
        elapsed_time=event_elapsed,
        worker={"tp_rank": 1, "dp_rank": 0},
        start_worker=False,
    )
    compute = NS(time=0, ready=False)
    recorder.begin_step((("r", 0, 4096),))
    ticket = recorder.start_interval("wait", compute, 0, 0, 0)
    compute.time = 1
    recorder.end_interval(ticket, compute)
    recorder.finish_step()
    assert not any(event.queries for event in created)
    assert not recorder.poll()
    assert recorder.pool.empty()
    assert not emitted

    recorder.begin_step((("r", 4096, 8192),))
    second = recorder.start_interval("wait", compute, 0, 1, 0)
    assert second.start not in (ticket.start, ticket.end)
    for event in (ticket.start, ticket.end):
        event.ready = True
    assert recorder.poll()
    result = json.loads(emitted[0].split(overlap.LOG_MARKER)[1])
    assert result["tp_rank"] == 1
    assert result["requests"][0]["token_start"] == 0
    assert not recorder.pool.empty()


@pytest.mark.parametrize("requests", [()])
def test_skipped_batch_records_no_events(requests):
    recorder = overlap.PrefillOverlapRecorder(
        lambda: pytest.fail("created event for skipped batch"),
        lambda line: pytest.fail("logged skipped batch"),
        elapsed_time=event_elapsed,
        worker={},
        start_worker=False,
    )
    recorder.begin_step(requests)
    assert recorder.start_interval("load", None, 0, 0, 0) is None
    recorder.end_interval(None, None)
    recorder.finish_step()
    assert not recorder.poll()


def test_queue_overflow_drops_measurements_without_blocking_inference():
    recorder = overlap.PrefillOverlapRecorder(
        DeviceEvent,
        lambda line: None,
        elapsed_time=event_elapsed,
        worker={},
        start_worker=False,
    )
    for _ in range(overlap.MAX_PENDING_STEPS):
        recorder.begin_step((("r", 0, 4),))
        recorder.finish_step()
    recorder.begin_step((("r", 4, 8),))
    assert recorder.active is None
    assert recorder.dropped_steps == 1


def test_real_collector_thread_owns_queries_elapsed_time_and_logging():
    submit_thread = get_ident()
    emitted, initialized = [], []
    done = Event()

    class ThreadEvent(DeviceEvent):
        def record(self, stream):
            assert get_ident() == submit_thread
            super().record(stream)

        def query(self):
            assert get_ident() != submit_thread
            return super().query()

        def elapsed_time(self, end):
            assert get_ident() != submit_thread
            return super().elapsed_time(end)

    def emit(line):
        assert get_ident() != submit_thread
        emitted.append(line)
        done.set()

    recorder = overlap.PrefillOverlapRecorder(
        ThreadEvent,
        emit,
        elapsed_time=event_elapsed,
        worker={},
        initialize_thread=lambda: initialized.append(get_ident()),
    )
    try:
        stream = NS(time=0, ready=True)
        recorder.begin_step((("r", 0, 4096),))
        for layer in range(2):
            stream.time = layer * 10
            ticket = recorder.start_interval("wait", stream, 0, 0, layer)
            stream.time += 1
            recorder.end_interval(ticket, stream)
        recorder.finish_step()
        assert done.wait(timeout=2), "collector did not emit completed work"
        assert len(emitted) == 1
        assert initialized and initialized[0] != submit_thread
        result = json.loads(emitted[0].split(overlap.LOG_MARKER)[1])
        assert result["coverage"] == pytest.approx(0.9)
    finally:
        recorder.close()


def test_collector_initialization_failure_disables_measurement():
    done, output = Event(), []

    def fail():
        raise RuntimeError("device timing unavailable")

    def emit(line):
        output.append(line)
        done.set()

    recorder = overlap.PrefillOverlapRecorder(
        lambda: pytest.fail("failed collector must not create events"),
        emit,
        elapsed_time=event_elapsed,
        worker={},
        initialize_thread=fail,
    )
    try:
        assert done.wait(timeout=2)
        recorder.begin_step((("r", 0, 4096),))
        assert recorder.active is None
        assert "collector_error" in output[0]
    finally:
        recorder.close()


@pytest.mark.parametrize("actual_ids", [["r"], ["r", "r"]])
def test_single_actual_request_uses_exact_partial_or_cached_span(actual_ids):
    request = NS(req_id="r", prefill_overlap_span=(1234, 1777))
    metadata = {"layer": NS(req_ids=actual_ids)}
    assert overlap.request_spans([request], metadata) == (("r", 1234, 1777),)


@pytest.mark.parametrize("actual_ids", [["r", "other"], [], ["other"]])
def test_attention_batch_can_reject_request_omitted_by_connector(actual_ids):
    request = NS(req_id="r", prefill_overlap_span=(0, 4096))
    assert overlap.request_spans([request], {"layer": NS(req_ids=actual_ids)}) == ()


def test_missing_ids_require_cpu_proof_of_one_query():
    request = NS(req_id="r", prefill_overlap_span=(0, 4096))
    assert overlap.request_spans([request], {"layer": NS()}) == ()
    assert overlap.request_spans(
        [request], {"layer": NS(query_start_loc_cpu=[0, 4096])}
    )
    assert (
        overlap.request_spans(
            [request], {"layer": NS(query_start_loc_cpu=[0, 1, 4096])}
        )
        == ()
    )


def test_multi_request_batch_keeps_unequal_spans_and_counts_timing_once():
    requests = [
        NS(req_id="a", prefill_overlap_span=(0, 1024)),
        NS(req_id="b", prefill_overlap_span=(8192, 11264)),
    ]
    spans = overlap.request_spans(requests, {"layer": NS(req_ids=["a", "b"])})
    assert spans == (("a", 0, 1024), ("b", 8192, 11264))
    emitted = []
    recorder = overlap.PrefillOverlapRecorder(
        DeviceEvent,
        emitted.append,
        worker={},
        elapsed_time=event_elapsed,
        start_worker=False,
    )
    recorder.begin_step(spans)
    stream = NS(time=0, ready=True)
    for layer in range(8):
        stream.time = layer * 10
        ticket = recorder.start_interval(
            "wait", stream, 0, None, layer, rotation_slot=layer % 4
        )
        stream.time += 2
        recorder.end_interval(ticket, stream)
    recorder.finish_step()
    assert recorder.poll()
    result = json.loads(emitted[0].split(overlap.LOG_MARKER)[1])
    assert result["scope"] == "batch_step" and len(result["requests"]) == 2
    assert result["coverage"] == pytest.approx(0.95)  # 1 - 2 / 40, NOT 1 - 2 / 10.
    assert result["layer_samples"] == 4 and result["bank_reuse_samples"] == 4
    assert result["all_wait_ms"] == 16 and result["wait_ms"] == 8
    assert result["unknown_bank_waits"] == 0
    assert len(result["batch_rotation_slots"]) == 4


@pytest.mark.parametrize("group", [0, 1, "c8"])
def test_timing_events_do_not_change_real_bank_fifo_dependencies(group):
    fixture_path = Path(__file__).with_name("test_layerwise_prefill_dma.py")
    spec = importlib.util.spec_from_file_location("overlap_fifo_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    api = fixture._async_bank_connector(group, 0)
    events, output = [], []

    class TimingEvent:
        def __init__(self):
            events.append(self)
            self.queries = 0

        def record(self, stream):
            self.tail = stream.enqueue()

        def query(self):
            self.queries += 1
            return self.tail.done

        def elapsed_time(self, end):
            assert self.tail.done and end.tail.done
            return 1.0

        def synchronize(self):
            pytest.fail("timing must not wait on the host")

    recorder = overlap.PrefillOverlapRecorder(
        TimingEvent,
        output.append,
        elapsed_time=event_elapsed,
        worker={},
        start_worker=False,
    )
    api.connector.prefill_overlap_recorder = recorder
    recorder.begin_step((("request", 0, 4096),))
    generator, load = api.start(0)
    load(0)
    load(1)
    for layer in range(4):
        api.read(0, layer)
        api.save(layer)
        if layer + 2 < 4:
            load(layer + 2)
    next(generator)
    generator.close()
    recorder.finish_step()
    assert not api.observed and not api.saved
    assert not any(event.queries for event in events)
    assert not recorder.poll()
    for bank in api.banks:
        bank.tail.run()
    api.npu.compute.tail.run()
    assert api.observed == [(0, layer) for layer in range(4)]
    assert sorted(api.saved) == [(0, layer) for layer in range(4)]
    assert recorder.poll()
    assert json.loads(output[0].split(overlap.LOG_MARKER)[1])["dma_ms"] == 8


@pytest.mark.parametrize(
    "enabled,decode", [(True, False), (False, False), (True, True)]
)
def test_scheduler_range_comes_from_actual_query_not_saved_prefix(enabled, decode):
    path = _PATH.parents[2] / "integration/vllm/vllm_v1_adapter.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheAscendConnectorV1Impl"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_request_meta"
    )

    class Base:
        def _build_request_meta(self, *args, **kwargs):
            return NS(prefill_overlap_span=None)

    unit = ast.parse("from __future__ import annotations\nclass Adapter(Base): pass")
    unit.body[1].body = [method]
    scope = {"Base": Base}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), scope)
    adapter = scope["Adapter"]()
    adapter._prefill_overlap_enabled = enabled
    adapter._unfinished_requests = {"r": NS(num_computed_tokens=4096)}
    tracker = NS(req_id="r", token_ids=range(5043), prompt_len=5043)
    result = adapter._build_request_meta(tracker, None, is_sparse_decode=decode)
    assert result.prefill_overlap_span == (
        (4096, 5043) if enabled and not decode else None
    )
