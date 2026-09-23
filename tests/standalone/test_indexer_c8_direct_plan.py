# SPDX-License-Identifier: Apache-2.0
"""Exercise production byte-vector plans with CPU cache tensors."""

import ast
import ctypes
import subprocess
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def connector():
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {
        "_direct_page_tensor_layout",
        "_plan_direct_page_buffers",
        "_reject_direct_page_plan",
        "plan_direct_page_sources",
        "plan_direct_page_destinations",
        "get_shape",
        "plan_compact_page_layout",
        "checkpoint_plane_widths",
        "_allocate_layerwise_staging_buffer",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(
        ast.ClassDef(
            name="Connector", bases=[], keywords=[], body=methods, decorator_list=[]
        )
    )
    formats = SimpleNamespace(MLA_KV=0, MLA_LATENT=1, DSA_INDEX=2, DSA_KV=3)
    ns = dict(
        torch=torch,
        pairwise=pairwise,
        KVCacheFormat=formats,
        _DENSE_DIRECT_LOAD_DISABLE=False,
    )
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    conn = ns["Connector"]()
    layout = SimpleNamespace(
        kv_format=formats.DSA_INDEX,
        indexer_c8=True,
        k_hidden_dims=130,
        v_hidden_dims=0,
        dsa_hidden_dims=130,
    )
    conn._group_layouts = {1: layout}
    conn._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout
    conn._expected_group_layers = lambda _: 2
    conn.kv_format = formats.DSA_INDEX
    conn.dtype = torch.bfloat16
    return conn


@pytest.mark.parametrize("tokens", [1, 7, 127, 128, 129, 255])
@pytest.mark.parametrize("layerwise", [False, True])
def test_plan_encodes_partial_keys_then_scales_and_restores(
    connector, tokens, layerwise
):
    caches = [
        (
            torch.arange(4 * 128 * 128).to(torch.int8).reshape(4, 128, 1, 128),
            torch.arange(4 * 128, dtype=torch.float16).reshape(4, 128, 1, 1),
        )
        for _ in range(2)
    ]
    # Discontiguous slots exercise run splitting; final scale is exactly 2 bytes.
    slots = torch.arange(tokens, dtype=torch.long) * 2
    plan = connector.plan_direct_page_sources(
        caches, slots, [17], [17 + tokens], 1, layerwise, slot_mapping_base=17
    )
    assert plan is not None, connector._direct_page_plan_rejections
    ptrs, sizes, owners = plan
    assert all(
        a is b
        for a, b in zip(owners, (t for pair in caches for t in pair), strict=True)
    )
    packets = [
        b"".join(ctypes.string_at(p, n) for p, n in zip(ps, ns, strict=True))
        for ps, ns in zip(ptrs, sizes, strict=True)
    ]
    expected = []
    for key, scale in caches:
        key_bytes = (
            key.view(512, 128)[slots].contiguous().view(torch.uint8).numpy().tobytes()
        )
        scale_bytes = (
            scale.view(512, 1)[slots].contiguous().view(torch.uint8).numpy().tobytes()
        )
        expected.append(key_bytes + scale_bytes)
    assert packets == (expected if layerwise else [b"".join(expected)])
    assert sum(map(len, packets)) == 2 * tokens * 130
    destination = [(torch.full_like(k, 42), torch.full_like(s, 42)) for k, s in caches]
    dest_ptrs, dest_sizes, _ = connector.plan_direct_page_destinations(
        destination, slots, [0], [tokens], 1, layerwise
    )
    for packet, ps, ns in zip(packets, dest_ptrs, dest_sizes, strict=True):
        offset = 0
        for p, n in zip(ps, ns, strict=True):
            ctypes.memmove(p, packet[offset : offset + n], n)
            offset += n
        assert offset == len(packet)
    untouched = torch.ones(512, dtype=torch.bool)
    untouched[slots] = False
    for original, restored in zip(caches, destination, strict=True):
        for a, b in zip(original, restored, strict=True):
            assert torch.equal(a.flatten(0, 1)[slots], b.flatten(0, 1)[slots])
            assert torch.all(b.flatten(0, 1)[untouched] == 42)


def test_reject_mismatched_scale_capacity(connector):
    caches = [
        (
            torch.empty(4, 128, 1, 128, dtype=torch.int8),
            torch.empty(3, 128, 1, 1, dtype=torch.float16),
        )
    ] * 2
    assert (
        connector.plan_direct_page_sources(caches, torch.tensor([0]), [0], [1], 1)
        is None
    )
    assert connector._direct_page_plan_rejections[1] == "c8_key_scale_layout_mismatch"


def test_c8_compact_plan_does_not_treat_scale_as_another_layer(connector):
    caches = [
        (
            torch.empty(4, 128, 1, 128, dtype=torch.int8),
            torch.empty(4, 128, 1, 1, dtype=torch.float16),
        )
    ] * 2
    assert (
        connector.plan_compact_page_layout(caches, torch.tensor([0]), [0], [1], 1)
        is None
    )
    assert (
        connector._direct_page_plan_rejections[1]
        == "compact_layout_requires_single_plane"
    )
    assert connector.checkpoint_plane_widths(1) == (128, 2)


def test_c8_staging_reserves_bytes_not_bfloat16_elements(connector):
    layout = connector._group_layouts[1]
    allocated = []

    def allocate(shape, dtype, fmt):
        tensor = torch.empty(shape, dtype=dtype)
        allocated.append(tensor.numel() * tensor.element_size())
        return SimpleNamespace(tensor=tensor)

    layout.storage_dtype = torch.uint8
    layout.gpu_buffer_allocator = SimpleNamespace(allocate=allocate)
    connector._check_staging_transfer_tokens = lambda *args: None
    _, tensor = connector._allocate_layerwise_staging_buffer(
        num_tokens=7, kv_group=1, layout=layout, expected_fmt=None
    )
    assert tensor.dtype == torch.uint8
    assert allocated == [7 * 130]


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("layerwise", [False, True])
def test_disabled_byte_plan_matches_baseline(connector, group, layerwise):
    root = Path(__file__).parents[2]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            "e18b939e8f8c2d20ba0a9ecd1692a6f3e5bb1dab:lmcache_ascend/v1/npu_connector/npu_connectors.py",
        ]
    ).decode()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {
        "_direct_page_tensor_layout",
        "_plan_direct_page_buffers",
        "_reject_direct_page_plan",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(methods)
    ns = dict(connector._plan_direct_page_buffers.__globals__)
    exec(compile(tree, "baseline_direct_plan", "exec"), ns)
    layout = SimpleNamespace(
        kv_format=group + 1,
        indexer_c8=False,
        k_hidden_dims=512,
        v_hidden_dims=64,
        dsa_hidden_dims=128,
    )
    connector._group_layouts = {group: layout}
    connector._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout
    widths = [512, 64] if group == 0 else [128]
    caches = [
        tuple(torch.empty(4, 128, 1, w, dtype=torch.bfloat16) for w in widths)
        for _ in range(2)
    ]
    slots = torch.tensor([2, 3, 256, 300, 301])
    args = caches, slots, [0, 3], [3, 5], group, layerwise
    actual = connector._plan_direct_page_buffers(*args)
    assert actual is not None
    # Bind the original validation too, not merely the old outer planner.
    connector._direct_page_tensor_layout = ns["_direct_page_tensor_layout"].__get__(
        connector
    )
    expected = ns["_plan_direct_page_buffers"](connector, *args)
    assert expected is not None
    assert actual[:2] == expected[:2]
    assert all(a is b for a, b in zip(actual[2], expected[2], strict=True))
