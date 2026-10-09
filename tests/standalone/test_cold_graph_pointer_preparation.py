# SPDX-License-Identifier: Apache-2.0
"""Execute the derived cold-load hook with explicit stream/fence collaborators."""

import ast
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("enabled", [False, True])
def test_warm_deferral_uses_existing_full_graph_setting(monkeypatch, enabled):
    path = Path(__file__).resolve().parents[2] / (
        "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    cls = next(
        n for n in ast.parse(path.read_text(encoding="utf8")).body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheAscendConnectorV1Impl"
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "__init__"]

    class BeforeBaseInitialization(Exception):
        pass

    class Base:
        def __init__(self, *args, **kwargs):
            assert self._defer_prepared_sparse_setup is enabled
            raise BeforeBaseInitialization

    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", str(int(enabled)))
    namespace = dict(
        LMCacheConnectorV1Impl=Base, os=os, logger=NS(debug=lambda *a: None)
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    exec(compile(module, str(path), "exec"), namespace)
    with pytest.raises(BeforeBaseInitialization):
        namespace[cls.name](None, None, None)


@pytest.mark.parametrize(
    "enabled,resident,explicit,unbundled",
    [
        (True, False, False, True),
        (False, False, False, True),
        (True, True, False, True),
        (True, False, True, True),
        (True, False, False, False),
    ],
)
def test_derived_preparation_precedes_final_fence(
    monkeypatch,
    enabled,
    resident,
    explicit,
    unbundled,
):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheAscendConnectorV1Impl"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_record_dsa_cold_dense_load_readiness"
    )
    cls.body = [method]
    trace = []
    source, stream, owners = object(), object(), (object(),)
    event = object() if explicit else None

    @contextmanager
    def context(value):
        assert value is stream
        trace.append("enter")
        yield
        trace.append("exit")

    def prepare(value, chunk, width):
        assert value is source and chunk == 1024 and width == 1024
        trace.append("prepare")

    class Base:
        def _record_dsa_cold_dense_load_readiness(self, value, readiness, additional):
            assert value is state and readiness is event and additional is owners
            trace.append("fence")

    module = ModuleType("lmcache_ascend.v1.npu_connector.sparse_graph")
    module.prepare_source_pointer_pairs = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    scope = {"LMCacheConnectorV1Impl": Base, "torch": NS(npu=NS(stream=context))}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    adapter = scope[cls.name]()
    adapter._prepare_cold_graph_pointers = enabled
    adapter._lmcache_chunk_size = 1024
    adapter.lmcache_engine = NS(gpu_connector=NS(load_stream=stream))
    key = torch.empty((1, 128, 1, 512), dtype=torch.bfloat16)
    adapter._kvcaches_for_group = lambda group: [(key, key)] if unbundled else [key]
    state = NS(
        prepared_sparse_sources={0: source}, dense_prefix_resident_tokens=int(resident)
    )
    adapter._record_dsa_cold_dense_load_readiness(state, event, owners)
    expected = (
        ["enter", "prepare", "exit", "fence"]
        if enabled and not resident and not explicit and unbundled
        else ["fence"]
    )
    assert trace == expected
