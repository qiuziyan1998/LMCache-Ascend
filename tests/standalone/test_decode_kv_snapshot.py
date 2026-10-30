# SPDX-License-Identifier: Apache-2.0
"""Lossless CPU fixtures and the actual adapter/connector completion hooks."""

import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import Any, Sequence

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
NAME = "lmcache_ascend.v1.decode_kv_snapshot"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "lmcache_ascend/v1/decode_kv_snapshot.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_full_500k_history_across_16k_chunks(tmp_path):
    count = 500_000
    values = torch.arange(count * 4).reshape(count, 4).to(torch.bfloat16)
    counts = [min(16384, count - i) for i in range(0, count, 16384)]
    packets = [v.contiguous() for v in values.split(counts)]
    snapshot = M.RawSnapshot(
        str(tmp_path),
        "request",
        range(count),
        dict(dp_rank=0, tp_rank=0, tp_size=1),
        [1],
    )
    snapshot.write_layer(
        0,
        0,
        M.cpu_batches(packets, counts, (8,), False),
        ["torch.bfloat16"],
        "received_cpu",
    )
    snapshot.finish()
    assert (snapshot.directory / "g0-l0-c0.bin").read_bytes() == values.view(
        torch.uint8
    ).numpy().tobytes()
    assert (
        json.loads((snapshot.directory / "manifest.json").read_text())["status"]
        == "complete"
    )


@pytest.mark.parametrize("interleaved", [False, True])
def test_planes_tail_capacity_and_unusual_float_bits(tmp_path, interleaved):
    # Includes NaN payloads, negative zero, and a subnormal: preserve bits, not values.
    a = torch.tensor([[0x7FC1, -32768], [1, 0x3F80], [3, 4]], dtype=torch.int16).view(
        torch.bfloat16
    )
    b = torch.tensor([[11], [12], [13]], dtype=torch.bfloat16)
    packed = (
        torch.cat((a, b), dim=1)
        if interleaved
        else torch.cat((a.flatten(), b.flatten()))
    )
    snapshot = M.RawSnapshot(
        str(tmp_path), "request", [1, 2], dict(dp_rank=0, tp_rank=0), [1]
    )
    snapshot.write_layer(
        0,
        0,
        M.cpu_batches([packed], [2], (4, 2), interleaved),
        ["torch.bfloat16"] * 2,
        "received_cpu",
    )
    snapshot.finish()
    for component, expected in enumerate((a[:2], b[:2])):
        assert (
            snapshot.directory / f"g0-l0-c{component}.bin"
        ).read_bytes() == expected.contiguous().view(torch.uint8).numpy().tobytes()


def test_c8_mapping_keeps_key_and_scale(tmp_path):
    logical = torch.tensor([0, 1, 128, 129])
    slots = M.physical_indexer_slots(
        logical, c8=True, factor=2, block_map=torch.tensor([2, 0])
    )
    assert slots.tolist() == [256, 257, 0, 1]
    key = torch.arange(512 * 3).to(torch.int8).reshape(4, 128, 1, 3)
    scale = torch.arange(512).to(torch.float16).reshape(4, 128, 1, 1)
    s = M.RawSnapshot(
        str(tmp_path), "r", [1, 2, 3, 4], dict(dp_rank=0, tp_rank=0), [0, 1]
    )
    s.write_layer(
        1,
        0,
        M.paged_batches([key, scale], slots),
        ["torch.int8", "torch.float16"],
        "received_npu",
    )
    s.finish()
    assert (s.directory / "g1-l0-c0.bin").read_bytes() == key.reshape(512, 3)[
        slots
    ].numpy().tobytes()
    assert (s.directory / "g1-l0-c1.bin").read_bytes() == scale.reshape(512, 1)[
        slots
    ].view(torch.uint8).numpy().tobytes()
    assert M.physical_indexer_slots(
        logical, c8=True, factor=2, block_map=None
    ).tolist() == [0, 1, 256, 257]


@pytest.mark.parametrize("slots", [torch.tensor([-1]), torch.tensor([1024])])
def test_invalid_slots_never_silently_clamped(slots):
    with pytest.raises(ValueError):
        M.physical_indexer_slots(slots, c8=True, factor=1, block_map=torch.tensor([0]))


def test_incomplete_layer_does_not_publish_success(tmp_path):
    s = M.RawSnapshot(str(tmp_path), "r", [1, 2, 3], dict(dp_rank=0, tp_rank=0), [1])
    try:
        s.write_layer(
            0, 0, [[torch.ones(2, 1, dtype=torch.uint8)]], ["torch.uint8"], "cpu"
        )
    except ValueError as error:
        s.fail(error)
    meta = json.loads((s.directory / "manifest.json").read_text())
    assert meta["status"] == "error"
    assert meta["errors"] and not meta["files"]


def test_strided_device_plane_only_gathers_requested_rows():
    storage = torch.arange(256 * 8).reshape(2, 128, 1, 8)
    plane = storage[..., ::2]
    assert not plane.is_contiguous()
    slots = torch.tensor([129, 1, 64])
    batches = list(M.paged_batches([plane], slots))
    assert torch.equal(batches[0][0], storage[slots // 128, slots % 128, :, ::2])


def test_same_request_capture_cannot_overwrite_previous_snapshot(tmp_path):
    first = M.RawSnapshot(str(tmp_path), "r", [1], dict(dp_rank=0, tp_rank=0), [1])
    with pytest.raises(FileExistsError):
        M.RawSnapshot(str(tmp_path), "r", [9], dict(dp_rank=0, tp_rank=0), [1])
    assert json.loads((first.directory / "tokens.json").read_text()) == [1]


def load_method(path, cls, method):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method
    )


def test_actual_completion_hook_waits_then_writes(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, NAME, M)
    messages = []
    logger = NS(
        info=lambda *a: messages.append(a), exception=lambda *a: messages.append(a)
    )
    method = load_method(
        ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py",
        "VLLMPagedMemLayerwiseNPUConnector",
        "save_decode_kv_snapshot",
    )
    scope = dict(
        Any=Any,
        Sequence=Sequence,
        torch=torch,
        logger=logger,
        _layer_memory_tensor=lambda obj, layer: obj.tensor,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "connector",
            "exec",
        ),
        scope,
    )
    connector = NS(
        _layerwise_prefill_dma=False,
        dtype=torch.bfloat16,
        load_stream=None,
        _group_layouts={
            0: NS(k_hidden_dims=2, v_hidden_dims=0),
            1: NS(indexer_c8=False),
        },
        _sparse_lmc_host_interleaved=lambda group: False,
        _stream_context_or_null=lambda stream: nullcontext(),
    )
    connector.save_decode_kv_snapshot = scope[method.name].__get__(connector)
    latent = torch.full((3, 2), -999, dtype=torch.bfloat16)
    indexer = torch.full((1, 128, 1, 2), -999, dtype=torch.bfloat16)
    source = NS(
        total_tokens=3,
        chunk_token_counts=[3],
        layers=[NS(memory_objs=[NS(tensor=latent)], tensors=[])],
    )
    state = NS(prepared_sparse_sources={0: source}, dense_load_readiness=object())
    events = []

    class Base:
        def _run_dsa_cold_compact_load(self, *args):
            events.append("receive_submitted")
            return state

    adapter_method = load_method(
        ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        "_run_dsa_cold_compact_load",
    )
    cls = ast.ClassDef(
        name="Adapter",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[adapter_method],
        decorator_list=[],
    )
    env = dict(Base=Base)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "adapter",
            "exec",
        ),
        env,
    )
    adapter = env["Adapter"]()
    adapter._decode_kv_snapshot_dir = str(tmp_path)
    adapter._decode_kv_snapshot_metadata = dict(dp_rank=0, tp_rank=0, tp_size=1)
    adapter._latent_layer_names = ["layer0"]
    adapter._indexer_layer_names = ["layer0"]
    adapter.lmcache_engine = NS(gpu_connector=connector)

    def complete(event):
        assert event is state.dense_load_readiness
        events.append("receive_completed")
        latent.fill_(17)
        indexer.fill_(29)

    adapter._synchronize_dsa_cold_dense_readiness = complete
    plan = dict(
        request=NS(req_id="request"),
        tokens=[1, 2, 3],
        indexer_kvcaches=[indexer],
        indexer_slots_cpu=torch.tensor([2, 0, 1]),
    )
    assert adapter._run_dsa_cold_compact_load(plan, None, None) is state
    assert events == ["receive_submitted", "receive_completed"]
    manifest = next(tmp_path.rglob("manifest.json"))
    assert json.loads(manifest.read_text())["status"] == "complete"
    assert (manifest.parent / "g0-l0-c0.bin").read_bytes() == latent.view(
        torch.uint8
    ).numpy().tobytes()
    assert (manifest.parent / "g1-l0-c0.bin").read_bytes() == indexer[:, :3].view(
        torch.uint8
    ).numpy().tobytes()
    # Disabled mode executes only the original load and does not touch the fence.
    events.clear()
    adapter._decode_kv_snapshot_dir = ""
    adapter._run_dsa_cold_compact_load(plan, None, None)
    assert events == ["receive_submitted"]
    # Disk failures mark only the diagnostic and cannot re-enter transfer fallback.
    adapter._decode_kv_snapshot_dir = str(tmp_path)
    plan["request"] = NS(req_id="disk-error")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(M.RawSnapshot, "write_layer", fail_write)
    assert adapter._run_dsa_cold_compact_load(plan, None, None) is state
    error_manifest = next((tmp_path / "disk-error").rglob("manifest.json"))
    assert json.loads(error_manifest.read_text())["status"] == "error"
