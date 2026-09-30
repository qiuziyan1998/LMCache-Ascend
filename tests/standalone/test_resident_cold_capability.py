# SPDX-License-Identifier: Apache-2.0
"""Scheduler admission must agree with the actual Ascend dense-load capability."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "LMCACHE_ASCEND_DENSE_DIRECT_DISABLE",
        "LMCACHE_ASCEND_DENSE_DIRECT_LOAD_DISABLE",
        "LMCACHE_ASCEND_DENSE_DIRECT_STORE_DISABLE",
    ],
)
@pytest.mark.parametrize("value", ["0", "1", "TRUE", "on", "false"])
def test_resident_admission_matches_dense_connector(name, value):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    adapter = ast.parse(
        (root / "integration/vllm/vllm_v1_adapter.py").read_text(encoding="utf-8")
    )
    enabled = next(
        n
        for n in ast.walk(adapter)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_resident_cold_load_enabled"
            for t in n.targets
        )
    )
    connector = ast.parse(
        (root / "v1/npu_connector/npu_connectors.py").read_text(encoding="utf-8")
    )
    flags = [
        n
        for n in connector.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id in {"_DENSE_DIRECT_DISABLE", "_DENSE_DIRECT_LOAD_DISABLE"}
            for t in n.targets
        )
    ]
    capability = next(
        n
        for n in ast.walk(connector)
        if isinstance(n, ast.FunctionDef)
        and n.name == "supports_dense_sparse_cache_retention"
    )
    scope = dict(
        self=NS(), os=NS(getenv=lambda key, default: value if key == name else default)
    )
    exec(
        compile(
            ast.Module(body=[*flags, capability, enabled], type_ignores=[]),
            "resident_capability",
            "exec",
        ),
        scope,
    )
    actual = scope["supports_dense_sparse_cache_retention"](scope["self"])
    assert scope["self"]._resident_cold_load_enabled == actual
    assert actual == (name.endswith("STORE_DISABLE") or value in ("0", "false"))
