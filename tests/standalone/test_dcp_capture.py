# SPDX-License-Identifier: Apache-2.0
"""Actual canonical CPU bytes from independently owned DCP source stripes."""

import ast
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from test_checkpoint_capture_pages import Page, packet

ROOT = Path(__file__).resolve().parents[2]


def api():
    ns = dict(dataclass=dataclass, Any=Any, torch=torch, LayerPageMemoryObj=Page)
    for path in (
        ROOT.parent / "LMCache-NPU/lmcache/integration/vllm/dcp.py",
        ROOT / "lmcache_ascend/v1/npu_connector/dcp_capture.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, *nodes], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            ns,
        )
    return ns


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("origin", [0, 4])
@pytest.mark.parametrize("start,end", [(0, 1), (127, 129), (5, 1301)])
def test_shards_write_exact_canonical_planes(world, origin, start, end):
    ns = api()
    layers, block = 2, 128
    ids = tuple(range(1, 1 + origin + (end + world * block - 1) // (world * block)))
    layout = ns["DcpWriteLayout"](world, block, ids, origin)
    bounds = [start, *range((start // 1024 + 1) * 1024, end, 1024), end]
    pages = [packet(b - a, (1152, 1152)) for a, b in zip(bounds, bounds[1:])]
    for page in pages:
        page.meta.dtypes = [torch.bfloat16] * layers
        page.meta.shapes = [torch.Size([page.valid_tokens * 576])] * layers
    for rank in range(world):
        caches = [
            (
                torch.full((max(ids) + 1, block, 1, 512), -10, dtype=torch.bfloat16),
                torch.full((max(ids) + 1, block, 1, 64), -10, dtype=torch.bfloat16),
            )
            for _ in range(layers)
        ]
        for token, slot, count in layout.runs(start, end, rank):
            for layer, planes in enumerate(caches):
                for plane, tensor in enumerate(planes):
                    values = (
                        torch.arange(token, token + count) % 97
                        + layer * 100
                        + plane * 20
                    ).bfloat16()
                    tensor.view(-1, tensor.shape[-1])[slot : slot + count] = values[
                        :, None
                    ]
        plan = ns["prepare_dcp_capture"](
            pages, bounds[:-1], bounds[1:], caches, layout, rank
        )
        for dst, src, size in plan.copies:
            ctypes.memmove(dst, src, size)
    for page, a, b in zip(pages, bounds, bounds[1:]):
        for layer in range(layers):
            raw = page.raw_data[
                page.group_prefix_sum[layer] : page.group_prefix_sum[layer + 1]
            ].view(torch.bfloat16)
            cursor = 0
            for plane, width in enumerate((512, 64)):
                expected = (
                    torch.arange(a, b) % 97 + layer * 100 + plane * 20
                ).bfloat16()
                actual = raw[cursor : cursor + (b - a) * width].view(b - a, width)
                assert torch.equal(actual, expected[:, None].expand_as(actual))
                cursor += (b - a) * width


@pytest.mark.parametrize("prefix", [1, 127, 128, 129, 257])
def test_reuse_canonical_prefix_before_capturing_missing_resident_suffix(prefix):
    ns = api()
    path = ROOT / "lmcache_ascend/v1/dcp_save.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seed = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "seed_prefix"
    )
    from bisect import bisect_right

    ns["bisect_right"] = bisect_right
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[seed], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    starts, ends = [0, 128, 256], [128, 256, 300]
    destination = [
        [NS(tensor=torch.full((128 * 576,), -1, dtype=torch.bfloat16)) for _ in starts]
    ]
    rows = []
    for start, end in zip(starts, ends):
        raw = torch.zeros(128 * 576, dtype=torch.bfloat16)
        raw[: 128 * 512].view(128, 512)[:] = (torch.arange(start, start + 128) % 97)[
            :, None
        ]
        raw[128 * 512 :].view(128, 64)[:] = (
            torch.arange(start, start + 128) % 97 + 100
        )[:, None]
        rows.append(raw)
    released = []
    importer = NS(
        source_manifest=lambda _: NS(group_id=0, chunk_starts=starts, chunk_ends=ends),
        borrow_source_tensors=lambda _: ((tuple(rows),), NS(release=released.append)),
    )
    assert (
        ns["seed_prefix"](
            NS(compact_source_importer=importer),
            NS(total_tokens=prefix),
            destination,
            starts,
            ends,
        )
        == prefix
    )
    assert released == ["never_imported"]
    for obj, start, end in zip(destination[0], starts, ends):
        for width, base, extra in ((512, 0, 0), (64, 128 * 512, 100)):
            actual = obj.tensor[base : base + 128 * width].view(128, width)
            count = max(0, min(prefix, end) - start)
            expected = ((torch.arange(start, start + count) % 97) + extra).bfloat16()
            assert torch.equal(actual[:count], expected[:, None].expand(count, width))
            assert (actual[count:] == -1).all()


def test_capture_skips_nonresident_prefix_and_rejects_missing_tail():
    ns = api()
    layout = ns["DcpWriteLayout"](2, 128, (0, 2))
    page = packet(600, (1152,))
    page.meta.dtypes = [torch.bfloat16]
    page.meta.shapes = [torch.Size([600 * 576])]
    caches = [
        (torch.zeros(3, 128, 1, width, dtype=torch.bfloat16) for width in (512, 64))
    ]
    caches = [tuple(caches[0])]
    for rank in (0, 1):
        plan = ns["prepare_dcp_capture"](
            [page], [0], [512], caches, layout, rank, capture_start=256
        )
        assert sum(n for _, _, n in plan.copies) == 128 * 1152
        with pytest.raises(ValueError, match="null"):
            ns["prepare_dcp_capture"]([page], [0], [512], caches, layout, rank)
