# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for byte checksums, canonical segments and real slot mapping."""

import importlib.util
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def trace_module():
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/kv_trace.py"
    spec = importlib.util.spec_from_file_location("kv_trace_standalone", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hash_checks_every_byte_and_preserves_order(trace_module):
    raw = torch.arange(8192, dtype=torch.int64).to(torch.uint8)
    baseline = trace_module.content_checksum(raw)
    for position in (0, 7, 4097, 8191):
        changed = raw.clone()
        changed[position] ^= 1
        assert not torch.equal(baseline, trace_module.content_checksum(changed))
    assert not torch.equal(baseline, trace_module.content_checksum(raw.flip(0)))
    assert torch.equal(baseline, trace_module.content_checksum(raw.reshape(64, 128)))


def test_checksum_is_byte_exact_for_bfloat16_payload(trace_module):
    values = torch.tensor([0.0, -0.0, float("nan"), 1.5], dtype=torch.bfloat16)
    assert torch.equal(
        trace_module.content_checksum(values),
        trace_module.content_checksum(values.view(torch.uint8)),
    )
    assert not torch.equal(
        trace_module.content_checksum(values),
        trace_module.content_checksum(values[:-1]),
    )


def test_partial_segments_and_packet_padding(trace_module):
    assert list(trace_module.segments(999, 2050)) == [
        (999, 1024),
        (1024, 2048),
        (2048, 2050),
    ]
    packet = torch.arange(24, dtype=torch.uint8)
    first, second = trace_module.packet_planes(packet, 4, (4, 2))
    assert first.tolist() == packet[:16].reshape(4, 4).tolist()
    assert second.tolist() == packet[16:].reshape(4, 2).tolist()
    with pytest.raises(ValueError):
        trace_module.packet_planes(packet, 3, (4, 2))


def test_paged_reads_actual_table_instead_of_dma_plan(trace_module):
    records = []
    tracer = trace_module.KVTrace(0, "P", emit=records.append)
    plane = torch.arange(32, dtype=torch.uint8).reshape(4, 4, 1, 2)
    table = torch.tensor([2, 0])
    tracer.paged("req", "P_LOAD", 0, 3, 8, [plane], table, 0, 8, block_size=4)
    expected = torch.cat((plane[2], plane[0])).reshape(8, 2)
    expected_hash = "".join(
        f"{int(x) & 0xFFFFFFFF:08x}"
        for x in trace_module.content_checksum(expected).tolist()
    )
    assert records[0]["stage"] == "PLAN"
    assert records[1]["hashes"] == [[0, 8, expected_hash]]
    assert "tensor" not in records[1] and "values" not in records[1]


def test_independent_mirror_does_not_modify_source(trace_module):
    records = []
    tracer = trace_module.KVTrace(0, "D", emit=records.append)
    packet = torch.arange(24, dtype=torch.uint8)
    saved = packet.clone()
    tracer.cpu_packet(
        "req", "D_CPU", 0, 0, 3, 0, 3, packet, 4, (4, 2), mirror_device="cpu"
    )
    assert torch.equal(packet, saved)
    for component in (0, 1):
        rows = [r for r in records if r.get("component") == component]
        assert len(rows) == 2 and rows[0]["hash"] == rows[1]["hash"]


def test_background_source_is_retained_until_real_producer_completion(trace_module):
    import threading

    done, release = threading.Event(), threading.Event()
    owner = type("Owner", (), {"refs": 1})()
    owner.ref_count_up = lambda: setattr(owner, "refs", owner.refs + 1)
    owner.ref_count_down = lambda: setattr(owner, "refs", owner.refs - 1)
    producer = type("Producer", (), {"synchronize": lambda self: release.wait(5)})()
    tracer = trace_module.KVTrace(0, "P", emit=lambda value: None)
    tracer.cpu_work("req", done.set, (owner,), (producer,))
    assert owner.refs == 2 and not done.is_set()
    release.set()
    tracer.queue.join()
    assert done.is_set() and owner.refs == 1


def test_disabled_switch(trace_module, monkeypatch):
    monkeypatch.delenv("LMCACHE_PREFILL_STORE_DEBUG", raising=False)
    assert not trace_module.enabled()
    monkeypatch.setenv("LMCACHE_PREFILL_STORE_DEBUG", "1")
    assert trace_module.enabled()


def test_checksum_matches_unsigned_scalar_reference(trace_module):
    raw = bytes(range(61))
    padded = raw + bytes(len(raw) % 2)
    words = [
        int.from_bytes(padded[i : i + 2], "little") for i in range(0, len(padded), 2)
    ]
    expected = [
        (sum(words) & 0xFFFFFFFF) ^ len(raw),
        sum((i + 1) * word for i, word in enumerate(words)) & 0xFFFFFFFF,
    ]
    value = torch.tensor(list(raw), dtype=torch.uint8)
    assert [
        int(x) & 0xFFFFFFFF for x in trace_module.content_checksum(value)
    ] == expected


def test_batched_checksums_equal_each_independent_segment(trace_module):
    values = torch.arange(7 * 61, dtype=torch.int32).to(torch.uint8).reshape(7, 61)
    actual = trace_module.content_checksums(values)
    expected = torch.stack([trace_module.content_checksum(row) for row in values])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 31, 61, 1024])
def test_simple_checksum_matches_scalar_for_batched_odd_and_even_rows(
    trace_module, width
):
    raw = torch.randint(
        0,
        256,
        (3, width),
        dtype=torch.int32,
        generator=torch.Generator().manual_seed(width),
    ).to(torch.uint8)
    actual = trace_module.content_checksums(raw)
    for row, checksums in zip(raw.tolist(), actual.tolist(), strict=True):
        data = bytes(row) + bytes(width % 2)
        words = [
            int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)
        ]
        assert [int(x) & 0xFFFFFFFF for x in checksums] == [
            (sum(words) & 0xFFFFFFFF) ^ width,
            sum((i + 1) * word for i, word in enumerate(words)) & 0xFFFFFFFF,
        ]


def test_position_weight_catches_reordering_with_same_plain_sum(trace_module):
    values = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
    before = trace_module.content_checksum(values)
    after = trace_module.content_checksum(values[[1, 0, 2, 3]])
    assert before[0] == after[0] and before[1] != after[1]
    assert before.shape == (2,)


def test_checksum_wraps_int32_and_supports_odd_storage_offset(trace_module):
    words = 200000
    values = torch.full((words * 2,), 255, dtype=torch.uint8)
    actual = trace_module.content_checksum(values)
    assert [int(x) & 0xFFFFFFFF for x in actual] == [
        ((65535 * words) & 0xFFFFFFFF) ^ (words * 2),
        (65535 * words * (words + 1) // 2) & 0xFFFFFFFF,
    ]
    odd = values[1:9]
    assert odd.storage_offset() == 1
    assert torch.equal(
        trace_module.content_checksum(odd), trace_module.content_checksum(odd.clone())
    )


def test_unaligned_half_scale_slice_checksums_without_host_readback(trace_module):
    values = torch.arange(9, dtype=torch.float16)[1:7]
    assert values.storage_offset() == 1
    assert torch.equal(
        trace_module.content_checksum(values),
        trace_module.content_checksum(values.clone()),
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4, 128, 130])
def test_all_partial_boundary_hashes_match_independent_prefixes(trace_module, width):
    values = (
        torch.arange(31 * width, dtype=torch.int32).to(torch.uint8).reshape(31, width)
    )
    actual = trace_module.prefix_checksums(values)
    expected = torch.stack(
        [trace_module.content_checksum(values[:end]) for end in range(1, 32)]
    )
    assert torch.equal(actual, expected)


def test_interleaved_cpu_layout_normalizes_to_model_planes(trace_module):
    packet = torch.arange(24, dtype=torch.uint8).reshape(4, 6)
    planes = trace_module.packet_planes(packet, 4, (4, 2), interleaved=True)
    assert torch.equal(planes[0], packet[:, :4])
    assert torch.equal(planes[1], packet[:, 4:])


def test_batched_paged_segments_do_not_change_logical_hashes(trace_module):
    output = []
    tracer = trace_module.KVTrace(0, "P", emit=output.append, segment_tokens=4)
    plane = torch.arange(64, dtype=torch.int32).to(torch.uint8).reshape(8, 4, 1, 2)
    table = torch.tensor([3, 0, 7])
    tracer.paged("req", "P_LOAD", 0, 0, 11, [plane], table, 0, 11, block_size=4)
    actual = output[1]["hashes"]
    assert [(r[0], r[1]) for r in actual] == [(0, 4), (4, 8), (8, 11)]
    for (begin, end, digest), block in zip(actual, table, strict=True):
        expected = plane[int(block), : end - begin].reshape(end - begin, 2)
        words = trace_module.content_checksum(expected)
        assert digest == "".join(f"{int(x) & 0xFFFFFFFF:08x}" for x in words)


def test_long_prefix_hash_lines_are_bounded_without_dropping_segments(trace_module):
    import json

    output = []
    tracer = trace_module.KVTrace(0, "P", emit=output.append, segment_tokens=4)
    plane = torch.arange(400, dtype=torch.int32).to(torch.uint8).reshape(100, 4, 1, 1)
    tracer.paged(
        "request",
        "P_COMPUTE",
        0,
        0,
        400,
        [plane],
        torch.arange(100),
        0,
        400,
        block_size=4,
    )
    data = [row for row in output if row["stage"] == "P_COMPUTE"]
    assert len(data) == 4
    assert sum(len(row["hashes"]) for row in data) == 100
    assert max(len(json.dumps(row)) for row in data) < 4096


def load_connector_methods(obj, trace_module):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "trace_cpu_kv",
            "trace_selected_kv",
            "finish_kv_trace",
            "trace_attention_kv",
            "trace_direct_kv",
            "enqueue_load_kv_trace",
        )
    ]
    scope = dict(
        torch=torch,
        KVTrace=trace_module.KVTrace,
        kv_trace_enabled=lambda: True,
        diagnostic=trace_module.diagnostic,
        content_checksum=trace_module.content_checksum,
        REFERENCE_SEGMENT_TOKENS=trace_module.REFERENCE_SEGMENT_TOKENS,
        segments=trace_module.segments,
        packet_planes=trace_module.packet_planes,
        _layer_source_memory_objs=lambda objects, layer: objects,
        _layer_memory_tensor=lambda obj, layer: obj.tensor,
    )
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    for method in methods:
        setattr(obj, method.name, MethodType(scope[method.name], obj))


def test_cpu_load_hash_waits_real_h2d_on_worker_and_freezes_metadata(trace_module):
    import threading

    released = threading.Event()
    records, calls = [], []
    tracer = trace_module.KVTrace(0, "P", emit=records.append)
    owner = SimpleNamespace(refs=1, value="not ready")
    owner.ref_count_up = lambda: setattr(owner, "refs", owner.refs + 1)
    owner.ref_count_down = lambda: setattr(owner, "refs", owner.refs - 1)
    ready = SimpleNamespace(synchronize=lambda: released.wait(5))
    connector = SimpleNamespace(kv_trace=tracer)
    load_connector_methods(connector, trace_module)
    connector.trace_cpu_kv = lambda *args, **kwargs: calls.append(
        (threading.get_ident(), args, kwargs, owner.value)
    )
    starts, ends, objects = [0], [4], [owner]
    connector.enqueue_load_kv_trace("request", 1, starts, ends, objects, 3, ready)
    assert owner.refs == 2 and calls == []
    starts.append(4)
    ends.append(8)
    objects.clear()
    owner.value = "DMA completed"
    released.set()
    tracer.queue.join()
    assert owner.refs == 1 and len(calls) == 1
    ident, args, kwargs, value = calls[0]
    assert ident == tracer.thread.ident
    assert args == ("P_CPU_LOAD", "request", 1, (0,), (4,), ((owner,),))
    assert kwargs == dict(layer_ids=(3,))
    assert value == "DMA completed"


def test_missing_cpu_load_fence_reports_coverage_error_without_reading(trace_module):
    records = []
    connector = SimpleNamespace(
        kv_trace=trace_module.KVTrace(0, "P", emit=records.append)
    )
    load_connector_methods(connector, trace_module)
    connector.enqueue_load_kv_trace("request", 0, [0], [4], [], 0, None)
    assert records[0]["error"] == "missing_load_completion"


def test_stall_summary_is_bounded_once_per_job_and_does_not_touch_device(trace_module):
    records = []
    trace = trace_module.KVTrace(0, "P", emit=records.append)
    record = dict(req="request", stage="P_CPU_LOAD", group=1, layer=3)
    trace.pending_job = (100.0, record, False)
    trace.check_stall(159.0)
    assert records == []
    trace.check_stall(161.0)
    trace.check_stall(180.0)
    assert len(records) == 1
    result = records[0]
    assert result["stage"] == "STALL" and result["blocked_stage"] == "P_CPU_LOAD"
    assert result["req"] == "request" and result["elapsed_s"] == 61.0
    assert result["layer"] == 3 and "error" not in result
    assert len(result["main_stack"]) <= 8 and len(result["hash_stack"]) <= 8
    # A different stalled job may emit its own summary.
    trace.pending_job = (200.0, record, False)
    trace.check_stall(261.0)
    assert len(records) == 2
    trace.pending_job = None
    trace.check_stall(999.0)
    assert len(records) == 2


def test_shutdown_barrier_cannot_release_storage_ahead_of_cpu_hash(trace_module):
    import threading

    release = threading.Event()
    completed = threading.Event()
    trace = trace_module.KVTrace(0, "P", emit=lambda row: None)
    trace.cpu_work("request", lambda: release.wait(5) and completed.set(), (), ())
    assert trace.drain(timeout=0.01) is False
    assert not completed.is_set()
    release.set()
    assert trace.drain(timeout=5) is True
    assert completed.is_set()


@pytest.mark.parametrize("corrupt", [False, True])
def test_real_sparse_transfer_probe_compares_every_selected_byte(
    trace_module, monkeypatch, corrupt
):
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(
            *args,
            **{key: value for key, value in kwargs.items() if key != "pin_memory"},
        ),
    )
    event = type(
        "Event",
        (),
        {"record": lambda self, stream: None, "synchronize": lambda self: None},
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=event, current_stream=lambda *args: None),
        raising=False,
    )
    records = []
    tracer = trace_module.KVTrace(0, "D", emit=records.append, segment_tokens=4)
    connector = SimpleNamespace(
        kv_trace=tracer,
        _layerwise_prefill_dma=False,
        _prefill_worker_id=0,
        dtype=torch.float16,
        _group_layouts={0: SimpleNamespace(indexer_c8=False, kv_device="cpu")},
        checkpoint_plane_widths=lambda group: (2, 1),
        _sparse_lmc_host_interleaved=lambda group: False,
    )
    load_connector_methods(connector, trace_module)
    keys = torch.arange(16, dtype=torch.float16).reshape(8, 2)
    rope = torch.arange(8, dtype=torch.float16).reshape(8, 1)
    objects = []
    for begin in (0, 4):
        obj = SimpleNamespace(
            tensor=torch.cat(
                (keys[begin : begin + 4].flatten(), rope[begin : begin + 4].flatten())
            ),
            refs=1,
        )
        obj.ref_count_up = lambda obj=obj: setattr(obj, "refs", obj.refs + 1)
        obj.ref_count_down = lambda obj=obj: setattr(obj, "refs", obj.refs - 1)
        objects.append(obj)
    connector.trace_cpu_kv("D_CPU", "req", 0, [0, 4], [4, 8], [objects], mirror=True)
    assert len([row for row in records if row["stage"] == "D_MIRROR"]) == 2
    actual_keys = torch.zeros(2, 4, 1, 2, dtype=torch.float16)
    actual_rope = torch.zeros(2, 4, 1, 1, dtype=torch.float16)
    actual_keys.reshape(8, 2)[[2, 3]] = keys[[5, 1]]
    actual_rope.reshape(8, 1)[[2, 3]] = rope[[5, 1]]
    if corrupt:
        actual_keys.reshape(-1).view(torch.uint8)[3 * 4] ^= 1
    connector.trace_selected_kv(
        "req",
        0,
        0,
        (actual_keys, actual_rope),
        torch.tensor([[2, 3, 0, 0]]),
        torch.tensor([[5, 1, -1, 0]]),
        torch.tensor([2]),
        8,
    )
    tracer.queue.join()
    selected = [row for row in records if row["stage"] == "D_SELECTED"]
    assert len(selected) == 2
    assert selected[0]["match"] is (not corrupt)
    assert selected[1]["match"] is True
    connector.finish_kv_trace("req")
    tracer.queue.join()
    assert all(obj.refs == 1 for obj in objects)


@pytest.mark.parametrize("prefix", [3, 4])
def test_attention_probe_observes_history_tail_and_current_scatter(
    trace_module, prefix
):
    records = []
    tracer = trace_module.KVTrace(0, "P", emit=records.append, segment_tokens=4)
    plane = torch.arange(16, dtype=torch.float16).reshape(2, 4, 1, 2)
    connector = SimpleNamespace(
        kv_trace=tracer,
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
    load_connector_methods(connector, trace_module)
    metadata = SimpleNamespace(
        req_ids=["req"], seq_lens_cpu=[7], query_start_loc_cpu=[0, 7 - prefix]
    )
    table = torch.tensor([[1, 0]])
    connector.trace_attention_kv(0, 0, "P_LOAD", [plane], table, metadata)
    assert not any(row.get("error") for row in records)
    # Current KV occupies the first physical block after the loaded history.
    logical = torch.cat((plane[1], plane[0]))
    connector.trace_attention_kv(
        0, 0, "P_COMPUTE", [plane], table, metadata, produced=(logical[prefix:7],)
    )
    assert not any(row.get("error") for row in records)
    loaded = next(row for row in records if row["stage"] == "P_LOAD")["hashes"]
    after = next(
        row
        for row in records
        if row["stage"] == ("P_HISTORY_TAIL" if prefix == 3 else "P_COMPUTE")
    )["hashes"]
    assert loaded[0] == after[0]
    produced = {
        (row["start"], row["end"]): row["hash"]
        for row in records
        if row["stage"] == "P_PRODUCED"
    }
    scattered = next(row for row in records if row["stage"] == "P_SCATTER")["hashes"]
    assert all(digest == produced[start, end] for start, end, digest in scattered)
    assert any(row["stage"] == "P_FINAL" and row["final"] for row in records)


def test_direct_indexer_import_observes_real_mixed_c8_destinations(trace_module):
    records = []
    tracer = trace_module.KVTrace(0, "D", emit=records.append, segment_tokens=128)
    connector = SimpleNamespace(
        kv_trace=tracer,
        _layerwise_prefill_dma=False,
        _group_layouts={
            1: SimpleNamespace(
                indexer_c8=True, kv_device="cpu", layer_slot_factors=(2,)
            )
        },
        _stream_context_or_null=lambda stream: nullcontext(),
        load_stream=None,
    )
    load_connector_methods(connector, trace_module)
    key = torch.arange(512 * 2, dtype=torch.int32).to(torch.int8).reshape(4, 128, 1, 2)
    scale = torch.arange(512, dtype=torch.float16).reshape(4, 128, 1, 1)
    # Logical slots 128..255 map to physical slots 256..383 for factor 2.
    connector.trace_direct_kv(
        "req", 1, [(key, scale)], torch.arange(128, 256), [0], [128]
    )
    assert not any(row.get("error") for row in records)
    hashes = [row for row in records if row["stage"] == "D_DIRECT"]
    assert len(hashes) == 2
    for component, plane in enumerate((key, scale)):
        expected = trace_module.content_checksum(plane[2])
        assert hashes[component]["hashes"][0][2] == "".join(
            f"{int(x) & 0xFFFFFFFF:08x}" for x in expected
        )
