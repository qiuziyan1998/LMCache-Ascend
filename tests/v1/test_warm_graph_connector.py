# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the deployed Ascend connector's optional warm entry."""

import ast
from pathlib import Path
from types import SimpleNamespace


def method():
    source = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheAscendConnectorV1Dynamic"
    )
    entry = next(
        node
        for node in cls.body
        if getattr(node, "name", "") == "prepare_warm_sparse_graph_step"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(entry)
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[entry.name]


def test_dynamic_ascend_connector_forwards_the_same_step_metadata():
    arguments = (object(), ("layer.0", "layer.1"), ("request",), (4096,))
    step = object()
    calls = []
    engine = SimpleNamespace(
        prepare_warm_sparse_graph_step=lambda *args: (calls.append(args), step)[1]
    )
    assert method()(SimpleNamespace(_lmcache_engine=engine), *arguments) is step
    assert calls == [arguments]


def test_older_engine_keeps_the_ordinary_preparation_path():
    assert (
        method()(SimpleNamespace(_lmcache_engine=SimpleNamespace()), None, (), (), ())
        is None
    )
