# SPDX-License-Identifier: Apache-2.0
"""The opt-in first-bank timer must not synchronize the compute stream."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace


def test_first_bank_diagnostics_only_read_completed_events():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {
        "_layerwise_prefill_transfer_state",
        "_layerwise_prefill_bank",
        "wait_for_layerwise_prefill_load",
        "_flush_prefill_first_bank_timing_events",
    }
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    assert {node.name for node in cls.body} == names

    records = []

    class Event:
        def __init__(self, **kwargs):
            self.options = kwargs

        def record(self, stream):
            self.stream = stream

        def query(self):
            return True

        def elapsed_time(self, other):
            return 0.0

        def synchronize(self):
            raise AssertionError("diagnostic event must not synchronize")

    class Stream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event)

    stream = Stream()
    namespace = {
        "torch": SimpleNamespace(
            npu=SimpleNamespace(current_stream=lambda: stream, Event=Event)
        ),
        "time": time,
        "logger": object(),
        "prefill_start_timing_enabled": lambda: True,
        "prefill_start_timing_log": lambda logger, stage, started, **fields: (
            records.append((stage, fields))
        ),
    }
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    connector = namespace["VLLMPagedMemLayerwiseNPUConnector"]()
    save, load = Event(), Event()
    connector._layerwise_prefill_save_done_events = {(0, 0, 0): (0, save)}
    connector._layerwise_prefill_load_done_events = {(0, 0): (0, load)}

    connector.wait_for_layerwise_prefill_load(layer_id=0, kv_group=0)
    # A completed H2D load already carries the exact layer's save dependency;
    # joining the save again would re-expose the store-stream backlog.
    assert stream.waited == [load]
    assert [stage for stage, _ in records] == ["first_bank_wait_enqueue"]

    connector._flush_prefill_first_bank_timing_events()
    assert [stage for stage, _ in records] == [
        "first_bank_wait_enqueue", "first_bank_wait_device"
    ]
    assert records[-1][1]["device_wait_ms"] == 0.0
