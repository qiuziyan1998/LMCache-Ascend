# SPDX-License-Identifier: Apache-2.0
"""Exercise D metadata publication without importing the NPU runtime."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


class Page:
    """No tensor accessor: native group-store pointers are already resolved."""


def engine(prefill, append):
    path = ROOT / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_append_group_store_tensors", "_append_group_store_tensors_legacy",
    }
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in names]
    runtime = {"torch": torch, "LayerPageMemoryObj": Page}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *methods], type_ignores=[],
    )), str(path), "exec"), runtime)
    obj = NS(
        _force_layerwise_prefill_store=prefill,
        gpu_connector=NS(append_sparse_chunk_ptr_cache_for_layer=append),
    )
    for name in names:
        setattr(obj, name, MethodType(runtime[name], obj))
    return obj


@pytest.mark.parametrize("layers", [2, 3])
def test_decoder_uses_native_layer_addresses_without_reresolving_pages(layers):
    def unexpected(*args, **kwargs):
        pytest.fail("D must preserve native group-store pointer publication")

    obj = engine(False, unexpected)
    tensors, host, device = [], [], []
    page = Page()
    sources = [[page] for _ in range(layers)]
    rows = [[100 + 10 * layer] for layer in range(layers)]
    # Deliberately different from host rows: the native device table is the
    # authoritative device address and must not be reconstructed from a page.
    native = torch.tensor([[1000 + 10 * layer] for layer in range(layers)])
    obj._append_group_store_tensors(sources, tensors, host, device, rows, native)
    old_rows = tuple(device)
    assert tensors == []  # No per-layer tensor views of a merged page.
    assert host == rows
    assert [row.data_ptr() for row in device] == [row.data_ptr() for row in native]

    following = native + 1
    obj._append_group_store_tensors(
        sources, tensors, host, device,
        [[row[0] + 1] for row in rows], following,
    )
    assert [row.tolist() for row in old_rows] == native.tolist()
    expected = torch.cat((native, following), 1).tolist()
    assert [row.tolist() for row in device] == expected
    assert host == [[100 + 10 * layer, 101 + 10 * layer] for layer in range(layers)]


def test_prefill_keeps_incremental_publication_for_each_layer():
    calls = []

    def append(layer, sources, host, device, *, kv_group):
        calls.append((layer, tuple(sources), kv_group))
        host[layer].append(500 + layer)
        device[layer] = None

    obj = engine(True, append)
    page = Page()
    tensors, host, device = [], [], []
    obj._append_group_store_tensors(
        [[page], [page]], tensors, host, device,
        [[100], [200]], torch.tensor([[1000], [2000]]), kv_group=1,
    )
    assert calls == [(0, (page,), 1), (1, (page,), 1)]
    assert host == [[500], [501]] and device == [None, None]
