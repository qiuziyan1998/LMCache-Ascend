# SPDX-License-Identifier: Apache-2.0
"""Tests for clock-safe RemoteFill stage attribution."""

# Standard
from pathlib import Path
import importlib.util
import sys

# Third Party
import pytest


def _module():
    path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "v1"
        / "kv_transfer"
        / "remote_fill_trace.py"
    )
    spec = importlib.util.spec_from_file_location("remote_fill_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _event(name, timestamp, **fields):
    return {
        "event": name,
        "role": "proxy",
        "host": "host",
        "clock_domain": "host:boot",
        "wall_time_ns": int(timestamp * 1_000_000),
        "monotonic_ms": timestamp,
        **fields,
    }


def test_trace_uses_only_local_clock_differences() -> None:
    module = _module()
    events = [
        _event("proxy_request_received", 0),
        _event("proxy_prefiller_dispatch", 10),
        _event("proxy_prefill_response_received", 100),
        _event("proxy_decoder_send_start", 110),
        _event("proxy_decoder_first_byte_received", 150),
    ]
    trace = module.build_trace(events, trace="request")
    assert trace["proxy_local_phases_ms"] == {
        "routing_before_prefill": 10.0,
        "prefiller_round_trip": 90.0,
        "handoff_before_decoder_send": 10.0,
        "decoder_send_to_first_byte": 40.0,
    }
    events[-1]["clock_domain"] = "other:boot"
    with pytest.raises(ValueError, match="different clock domains"):
        module.build_trace(events, trace="request")


def test_window_components_are_explanatory_not_double_counted() -> None:
    module = _module()
    windows = [
        {
            **_event("remote_fill_window_complete", 0),
            "role": "prefiller",
            "window_id": index,
            "reserve_ms": 1,
            "arm_ms": 2,
            "source_event_wait_ms": 3,
            "source_registration_ms": 1,
            "native_slot_wait_ms": 4,
            "native_ms": 5,
            "report_ms": 6,
            "native_started_monotonic_ms": 20 + index * 10,
            "native_ended_monotonic_ms": 25 + index * 10,
            "full_window": True,
            "direct_satisfied": True,
        }
        for index in range(2)
    ]
    chunks = [
        {
            **_event("prefiller_model_chunk_complete", 20 + index * 20),
            "role": "prefiller",
            "tp_rank": 0,
            "model_started_monotonic_ms": 10 + index * 20,
            "model_ended_monotonic_ms": 20 + index * 20,
            "model_forward_cpu_ms": 10,
        }
        for index in range(2)
    ]
    trace = module.build_trace(windows + chunks, trace="request")
    assert trace["remote_fill_window_totals_ms"]["native_ms"] == 10
    assert trace["native_gap_ms"] == [5.0]
    assert trace["prefiller_model"]["model_forward_cpu_ms"] == 20
    assert trace["prefiller_model"]["model_prefill_span_ms"] == 30
    assert trace["full_window_native_overlap_percent"] == 100
    assert "not additional" in trace["double_counting_note"]
