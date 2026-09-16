# SPDX-License-Identifier: Apache-2.0
"""Check the actual priming call against the real connector signature on CPU."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import create_autospec
import logging

import pytest
import torch


@pytest.mark.parametrize("kv_group,layers", [(0, 79), (1, 22)])
def test_deferred_store_priming_passes_its_own_kv_group(kv_group, layers):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "append_sparse_chunk_ptr_cache_for_layers"
    )
    # No hardware body executes, but the production signature is unmodified.
    method.body = [ast.Pass()]
    ns = {}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    connector = SimpleNamespace()
    append = create_autospec(ns[method.name].__get__(connector))
    setattr(connector, method.name, append)
    generator = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "batched_from_gpu"
    )
    calls = [
        n
        for n in ast.walk(generator)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == method.name
    ]
    assert len(calls) == 1
    memory_objs = [[object()] for _ in range(layers)]
    host_rows, device_rows = [], []
    module = ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        dict(
            self=connector,
            memory_objs=memory_objs,
            kv_group=kv_group,
            deferred_dense_chunk_dev_ptrs=host_rows,
            deferred_dense_chunk_ptrs_npu=device_rows,
        ),
    )
    append.assert_called_once_with(
        memory_objs, host_rows, device_rows, kv_group=kv_group
    )


@pytest.mark.parametrize("kv_group", [0, 1])
def test_real_deferred_store_generator_with_cpu_streams(monkeypatch, kv_group):
    """Run the repository's complete bank-rotation test without NPU imports.

    Only native transfer preparation/kernels and streams are mocked by the
    existing test. Generator control flow, pointer-call signature, completion
    events, group identity and reset handling execute from production code.
    """
    root = Path(__file__).resolve().parents[2]
    path = root / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    actual = {
        "batched_from_gpu",
        "_prepare_dense_direct_chunk_metadata",
        "_dense_direct_dummy_metadata_tensor",
        "_run_dense_direct_kv_transfer_layer",
        "_dense_direct_pointer_cache_signature",
        "_tensor_layout_signature",
        "_expected_group_layers",
        "get_num_layers",
        "_stream_context_or_null",
        "_slot_mapping_on_kv_device",
        "_layerwise_prefill_transfer_state",
        "_layerwise_prefill_transfer_generation",
        "_check_layerwise_prefill_transfer_generation",
        "_set_layerwise_prefill_bank_count",
        "reset_layerwise_prefill_transfer_state",
    }
    mocked = {
        "initialize_kvcaches_ptr",
        "_lazy_initialize_buffer_with_staging",
        "_is_mla_dsa_format",
        "_expected_memory_format",
        "_layerwise_token_major",
        "_sparse_lmc_host_interleaved",
        "_check_layerwise_transfer_invariants",
        "append_sparse_chunk_ptr_cache_for_layers",
        "_resolve_sparse_chunk_ptrs_npu",
        "_get_or_create_sparse_direct_layer_state",
    }
    methods = [n for n in cls.body if getattr(n, "name", None) in actual | mocked]
    found = {n.name for n in methods}
    assert actual <= found
    cls.bases, cls.decorator_list, cls.body = [], [], methods
    # These are immediately monkeypatched by the original test; use inert
    # declarations only for inherited methods absent from this class.
    missing = mocked - found
    cls.body += ast.parse(
        "\n".join(f"def {name}(self, *args, **kwargs): pass" for name in missing)
    ).body
    functions = [
        n
        for n in tree.body
        if getattr(n, "name", None)
        in {
            "_resolve_layerwise_slot_mapping",
            "_slice_layerwise_slot_mapping",
            "_layer_memory_tensor",
        }
    ]
    module = ModuleType("cpu_npu_connector")
    module.__dict__.update(
        torch=torch,
        nullcontext=nullcontext,
        logger=logging.getLogger(__name__),
        LayerPageMemoryObj=type("LayerPageMemoryObj", (), {}),
        _DENSE_DIRECT_STORE_DISABLE=False,
        _mtp_dw_deep_diag_enabled=lambda: False,
        dense_mla_dsa_batched_direct_kv_transfer_fast=lambda *args, **kwargs: None,
        dense_mla_dsa_batched_direct_kv_transfer=lambda *args, **kwargs: None,
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), module.__dict__)
    tests_path = root / "tests/v1/test_npu_connector.py"
    test_tree = ast.parse(tests_path.read_text(encoding="utf-8"))
    test_name = (
        "test_deferred_batched_from_gpu_rotates_two_banks_and_reports_completion"
    )
    definitions = [
        n
        for n in test_tree.body
        if getattr(n, "name", None)
        in {
            "_TrackingStream",
            "_TrackingEvent",
            "_DenseLayout",
            "_MemoryObj",
            test_name,
        }
    ]
    namespace = dict(
        torch=torch,
        pytest=pytest,
        nullcontext=nullcontext,
        SimpleNamespace=SimpleNamespace,
        npu_connectors=module,
        MemoryFormat=SimpleNamespace(KV_MLA_LATENT_FMT="latent"),
        VLLMPagedMemLayerwiseNPUConnector=module.VLLMPagedMemLayerwiseNPUConnector,
    )
    unit = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(tests_path), "exec"), namespace)
    recorded = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append((tensor, stream)),
    )
    namespace[test_name](monkeypatch, kv_group)
    assert recorded, "native transfer pointers must retain their input tensors"
