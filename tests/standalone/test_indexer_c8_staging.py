# SPDX-License-Identifier: Apache-2.0
"""Exercise byte packet staging; native scatter is qualified separately."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def stage(monkeypatch):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_c8_staging_packets"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    ns = {"torch": torch}
    exec(compile(tree, str(source), "exec"), ns)
    retained = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: retained.append(tensor)
    )
    impl = SimpleNamespace(_stream_context_or_null=lambda _: nullcontext())
    return lambda *args: ns[method.name](impl, *args), retained


@pytest.mark.parametrize("from_npu", [False, True])
def test_staging_preserves_planar_packets_and_exact_offsets(stage, from_npu):
    submit, retained = stage
    sizes = [7, 128, 1]
    packets = [torch.arange(n * 130).to(torch.uint8) for n in sizes]
    originals = [p.clone() for p in packets]
    staging = torch.full((sum(sizes) * 130 + 32,), 17, dtype=torch.uint8)
    views, pointers = submit(packets, staging, sizes, None, from_npu)
    assert len(views) == 3
    assert retained[0] is staging
    offset = 0
    for size, view, pointer, packet in zip(
        sizes, views, pointers.tolist(), packets, strict=True
    ):
        assert pointer == staging.data_ptr() + offset
        assert view.numel() == size * 130
        if from_npu:
            assert torch.all(view == 17)  # No readback before the scatter completes.
        else:
            assert torch.equal(view, packet)
        offset += size * 130
    assert torch.all(staging[-32:] == 17)
    assert all(torch.equal(a, b) for a, b in zip(originals, packets, strict=True))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(7 * 130 - 1, dtype=torch.uint8),
        torch.zeros(7 * 130, dtype=torch.bfloat16),
    ],
)
def test_staging_rejects_short_or_wrongly_typed_packets(stage, tensor):
    submit, _ = stage
    with pytest.raises(ValueError, match="complete contiguous byte packets"):
        submit([tensor], torch.empty(7 * 130, dtype=torch.uint8), [7], None, False)
