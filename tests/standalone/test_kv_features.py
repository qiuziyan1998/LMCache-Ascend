# SPDX-License-Identifier: Apache-2.0
"""Sample-byte persistence, address mapping, lifetime and bounded-failure tests."""

import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import MethodType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def features(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1"
    dependency = "lmcache_ascend.v1.kv_trace"
    spec = importlib.util.spec_from_file_location(dependency, root / "kv_trace.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, dependency, helper)
    spec = importlib.util.spec_from_file_location(
        "kv_features_test", root / "kv_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(root):
    path = next(root.rglob("index.jsonl"))
    return path, [json.loads(line) for line in path.read_text().splitlines()]


def read(path, record):
    width = record.get("byte_width", 1)
    raw = (path.parent / "features.bin").read_bytes()
    return b"".join(
        raw[offset : offset + (end - start) * width]
        for start, end, offset in record["pieces"]
    )


def test_sample_is_exactly_one_raw_byte_per_token(features):
    value = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    assert torch.equal(features.token_bytes(value), value.view(torch.uint8)[:, 0])
    assert features.token_bytes(value).numel() == 10
    with pytest.raises(ValueError):
        features.token_bytes(value.T)


def test_unchanged_prefix_reuses_extents_and_changes_preserve_old_bytes(
    features, tmp_path
):
    file = features.FeatureFile(tmp_path / "worker", {})
    original = torch.arange(8192).to(torch.uint8)
    base = dict(stage="P_COMPUTE", group=0, layer=0, component=0, start=0, end=8192)
    file.write(base, original)
    original_size = file.payload.tell()
    file.write(dict(base, stage="P_LOAD"), original)
    assert file.payload.tell() == original_size == 8192
    changed = original.clone()
    changed[5000] ^= 1
    file.write(base, changed)
    assert file.payload.tell() == 8192 + 4096
    file.close()
    path, records = rows(tmp_path)
    observations = [r for r in records if "pieces" in r]
    assert read(path, observations[0]) == original.numpy().tobytes()
    assert read(path, observations[1]) == original.numpy().tobytes()
    assert read(path, observations[2]) == changed.numpy().tobytes()


def test_paged_sampling_uses_real_logical_order_and_never_hashes(
    features, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys.modules["lmcache_ascend.v1.kv_trace"],
        "content_checksum",
        lambda *args: pytest.fail("hash must not run"),
    )
    trace = features.KVTrace(0, "P", root=tmp_path, emit=lambda r: None)
    plane = torch.arange(64, dtype=torch.uint8).reshape(4, 4, 1, 4)
    trace.paged(
        "req", "P_LOAD", 0, 0, 8, [plane], torch.tensor([2, 0]), 0, 8, block_size=4
    )
    trace.finish("req")
    assert trace.drain(5)
    path, records = rows(tmp_path)
    record = next(r for r in records if r.get("stage") == "P_LOAD")
    assert list(read(path, record)) == [32, 36, 40, 44, 0, 4, 8, 12]
    assert records[-1]["stage"] == "END"


def test_bad_table_is_reported_without_submitting_an_out_of_bounds_gather(
    features, tmp_path
):
    logs = []
    trace = features.KVTrace(0, "P", root=tmp_path, emit=logs.append)
    plane = torch.arange(16, dtype=torch.uint8).reshape(2, 4, 1, 2)
    trace.paged(
        "req", "P_LOAD", 0, 0, 8, [plane], torch.tensor([0]), 0, 8, block_size=4
    )
    trace.paged(
        "req", "P_LOAD", 0, 1, 8, [plane], torch.tensor([0]), 0, 8, block_size=4
    )
    assert len(logs) == 1 and logs[0]["error"] == "feature_table_range"
    assert trace.failed


def test_queue_full_does_not_launch_producer_or_lose_async_buffers(features, tmp_path):
    logs, calls = [], []
    trace = features.KVTrace(0, "P", root=tmp_path, emit=logs.append)
    trace.slots = threading.BoundedSemaphore(1)
    trace.slots.acquire()
    trace.capture(dict(req="req", stage="P_LOAD"), lambda: calls.append(True))
    assert calls == []
    assert logs[0]["error"] == "feature_queue_full"


def test_failed_event_record_keeps_pending_host_storage_alive(
    features, tmp_path, monkeypatch
):
    trace = features.KVTrace(0, "P", root=tmp_path, emit=lambda r: None)
    value = SimpleNamespace(
        shape=(4,), dtype=torch.uint8, device=SimpleNamespace(type="npu")
    )
    host = SimpleNamespace(copy_=lambda *args, **kwargs: None)
    event = SimpleNamespace(
        record=lambda stream: (_ for _ in ()).throw(RuntimeError("device failed"))
    )
    monkeypatch.setattr(
        features,
        "torch",
        SimpleNamespace(
            empty=lambda *args, **kwargs: host,
            npu=SimpleNamespace(
                Event=lambda: event, current_stream=lambda device: None
            ),
        ),
    )
    trace.capture(dict(req="req", stage="P_LOAD"), lambda: value)
    assert trace.failed
    assert trace.quarantine == [(host, value, event)]
    assert not trace.drain(5)


def test_cpu_reader_waits_real_event_and_releases_only_after_capture(
    features, tmp_path
):
    trace = features.KVTrace(0, "P", root=tmp_path, emit=lambda r: None)
    gate, calls = threading.Event(), []
    obj = SimpleNamespace(refs=1)
    obj.ref_count_up = lambda: setattr(obj, "refs", obj.refs + 1)
    obj.ref_count_down = lambda: setattr(obj, "refs", obj.refs - 1)
    trace.cpu_work(
        "req",
        lambda: calls.append(threading.get_ident()),
        [obj],
        [SimpleNamespace(synchronize=lambda: gate.wait(5))],
    )
    assert obj.refs == 2 and calls == []
    assert not trace.drain(0.01)
    gate.set()
    assert trace.drain(5)
    assert obj.refs == 1 and calls == [trace.thread.ident]


def bind_connector(obj, features):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "trace_attention_kv",
            "trace_cpu_kv",
            "trace_selected_kv",
            "finish_kv_trace",
            "trace_direct_kv",
        )
    ]
    scope = dict(
        torch=torch,
        diagnostic=features.diagnostic,
        kv_trace_enabled=lambda: True,
        KVTrace=features.KVTrace,
        _layer_memory_tensor=lambda obj, layer: obj.tensor,
        _layer_source_memory_objs=lambda objects, layer: objects,
    )
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    for method in methods:
        setattr(obj, method.name, MethodType(scope[method.name], obj))


@pytest.mark.parametrize("prefix", [3, 4])
def test_partial_history_produced_and_bank_bytes_align(features, tmp_path, prefix):
    trace = features.KVTrace(0, "P", root=tmp_path, emit=lambda r: None)
    plane = torch.arange(16).to(torch.uint8).reshape(2, 4, 1, 2)
    obj = SimpleNamespace(
        kv_trace=trace,
        _layerwise_prefill_dma=True,
        lmcache_chunk_size=4,
        kv_trace_requests={
            "req": SimpleNamespace(
                block_ids_by_bank=[[[1, 0]], [[0, 1]]],
                layerwise_prefill_bank_offset=0,
                is_last_prefill=True,
                token_ids=list(range(7)),
            )
        },
        _group_layouts={0: SimpleNamespace(indexer_c8=False)},
    )
    bind_connector(obj, features)
    meta = SimpleNamespace(
        req_ids=["req"], seq_lens_cpu=[7], query_start_loc_cpu=[0, 7 - prefix]
    )
    table = torch.tensor([[1, 0]])
    obj.trace_attention_kv(0, 0, "P_LOAD", [plane], table, meta)
    logical = torch.cat((plane[1], plane[0]))
    obj.trace_attention_kv(
        0, 0, "P_COMPUTE", [plane], table, meta, produced=(logical[prefix:7],)
    )
    trace.finish("req")
    assert trace.drain(5)
    path, records = rows(tmp_path)
    values = {r["stage"]: read(path, r) for r in records if "pieces" in r}
    assert values["P_LOAD"] == values["P_COMPUTE"][:prefix]
    assert values["P_PRODUCED"] == values["P_COMPUTE"][prefix:]


def test_cpu_packet_c8_components_skip_allocator_padding(features, tmp_path):
    trace = features.KVTrace(0, "D", root=tmp_path, emit=lambda r: None)
    raw = torch.arange(24, dtype=torch.uint8)
    batches = {}
    trace.cpu_work(
        "req",
        lambda: (
            trace.cpu_packet(
                "req", "D_CPU", 1, 0, 3, 0, 3, raw, 4, (4, 2), batches=batches
            ),
            trace.flush_packets("req", 1, 0, 3, batches),
        ),
        (),
        (),
    )
    trace.finish("req")
    assert trace.drain(5)
    path, records = rows(tmp_path)
    values = {r["component"]: read(path, r) for r in records if "pieces" in r}
    assert values == {0: bytes([0, 4, 8]), 1: bytes([16, 18, 20])}


def test_nested_cpu_save_captures_before_END_and_keeps_sources_leased(
    features, tmp_path
):
    trace = features.KVTrace(0, "P", root=tmp_path, emit=lambda r: None)
    obj = SimpleNamespace(
        kv_trace=trace,
        _layerwise_prefill_dma=True,
        dtype=torch.float16,
        _group_layouts={0: SimpleNamespace(indexer_c8=False)},
        checkpoint_plane_widths=lambda group: (2, 1),
        _sparse_lmc_host_interleaved=lambda group: False,
    )
    bind_connector(obj, features)
    packet = SimpleNamespace(refs=1, tensor=torch.arange(12, dtype=torch.float16))
    packet.ref_count_up = lambda: setattr(packet, "refs", packet.refs + 1)
    packet.ref_count_down = lambda: setattr(packet, "refs", packet.refs - 1)
    trace.cpu_work(
        "req",
        lambda: obj.trace_cpu_kv("P_CPU_STORE", "req", 0, [0], [4], [[packet]]),
        [packet],
        (),
    )
    trace.finish("req")
    assert trace.drain(5)
    _, records = rows(tmp_path)
    assert len([r for r in records if r.get("stage") == "P_CPU_STORE"]) == 2
    assert records[-1]["stage"] == "END"
    assert packet.refs == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_actual_sparse_features_align_by_logical_position(features, tmp_path, corrupt):
    trace = features.KVTrace(0, "D", root=tmp_path, emit=lambda r: None)
    obj = SimpleNamespace(
        kv_trace=trace,
        _layerwise_prefill_dma=False,
        kv_trace_sources={("req", 0, 0): ((0,), (8,), (), (2,), False)},
    )
    bind_connector(obj, features)
    plane = torch.zeros(2, 4, 1, 2, dtype=torch.uint8)
    plane.reshape(8, 2)[2, 0] = 55
    plane.reshape(8, 2)[3, 0] = 11 if not corrupt else 12
    obj.trace_selected_kv(
        "req",
        0,
        0,
        [plane],
        torch.tensor([[2, 3, -1]]),
        torch.tensor([[5, 1, -1]]),
        torch.tensor([2]),
        8,
    )
    trace.finish("req")
    assert trace.drain(5)
    path, records = rows(tmp_path)
    record = next(r for r in records if r.get("stage") == "D_SELECTED")
    assert read(path, record)[:2] == bytes([55, 11 if not corrupt else 12])
    positions = next(r for r in records if r.get("stage") == "MAP_SELECTED_POS")
    assert torch.frombuffer(
        bytearray(read(path, positions)), dtype=torch.int64
    ).tolist() == [5, 1, -1]


def test_direct_mixed_c8_features_read_actual_physical_destinations(features, tmp_path):
    trace = features.KVTrace(0, "D", root=tmp_path, emit=lambda r: None)
    obj = SimpleNamespace(
        kv_trace=trace,
        _layerwise_prefill_dma=False,
        _group_layouts={
            1: SimpleNamespace(
                indexer_c8=True, kv_device="cpu", layer_slot_factors=(2,)
            )
        },
        _stream_context_or_null=lambda stream: nullcontext(),
        load_stream=None,
    )
    bind_connector(obj, features)
    key = torch.arange(1024).to(torch.int8).reshape(4, 128, 1, 2)
    scale = torch.arange(512, dtype=torch.float16).reshape(4, 128, 1, 1)
    obj.trace_direct_kv("req", 1, [(key, scale)], torch.arange(128, 256), [0], [128])
    trace.finish("req")
    assert trace.drain(5)
    path, records = rows(tmp_path)
    actual = [r for r in records if r.get("stage") == "D_DIRECT"]
    assert len(actual) == 2
    for record, plane in zip(actual, (key, scale), strict=True):
        assert read(path, record) == features.token_bytes(plane[2]).numpy().tobytes()
