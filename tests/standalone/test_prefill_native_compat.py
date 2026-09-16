# SPDX-License-Identifier: Apache-2.0
"""Run native-call contract tests on CPU; actual device kernels are mocked."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "test_name",
    [
        "test_dense_direct_fast_state_cache_separates_load_and_store",
        "test_prepared_dense_load_bypasses_shape_cache_and_validates_once",
    ],
)
def test_prefill_preserves_production_fast_paths(monkeypatch, test_name):
    root = Path(__file__).resolve().parents[2]
    path = root / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {
        "_run_dense_direct_kv_transfer_layer",
        "_stream_context_or_null",
        "_dense_direct_pointer_cache_signature",
        "_tensor_layout_signature",
        "_get_or_create_sparse_direct_layer_state",
        "_sparse_direct_state_key",
        "_vllm_layer_cache_identity_signature",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    assert {n.name for n in cls.body} == names
    cls.bases = []
    plan = next(
        n for n in tree.body if getattr(n, "name", None) == "_SparseDestinationPlan"
    )
    module = ModuleType("native_contract")
    module.__dict__.update(torch=torch, nullcontext=nullcontext)
    for name in (
        "prepare_sparse_direct_layer_state",
        "dense_mla_dsa_batched_direct_kv_transfer",
        "dense_mla_dsa_batched_direct_kv_transfer_fast",
        "dense_mla_dsa_batched_direct_kv_transfer_prepared",
    ):
        setattr(module, name, lambda *a, **kw: None)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    unit = ast.Module(body=[future, plan, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), module.__dict__)

    test_path = root / "tests/v1/test_npu_connector.py"
    test_tree = ast.parse(test_path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in test_tree.body
        if getattr(n, "name", None)
        in {"_TrackingStream", "_RecordableTensor", test_name}
    ]
    namespace = dict(
        torch=torch,
        pytest=pytest,
        nullcontext=nullcontext,
        SimpleNamespace=SimpleNamespace,
        npu_connectors=module,
        VLLMPagedMemLayerwiseNPUConnector=module.VLLMPagedMemLayerwiseNPUConnector,
    )
    exec(
        compile(
            ast.Module(body=[future, *nodes], type_ignores=[]), str(test_path), "exec"
        ),
        namespace,
    )
    namespace[test_name](monkeypatch)
