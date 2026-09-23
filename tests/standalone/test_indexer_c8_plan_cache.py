# SPDX-License-Identifier: Apache-2.0
"""C8 plans follow storage identity, not transient outer cache containers."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("validation", [False, True])
def test_plan_reuses_rewrapped_buffers_but_not_rebound_scales(validation):
    source = (
        Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    plan = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_SparseDestinationPlan"
    )
    names = {"_get_or_create_sparse_destination_plan", "_tensor_layout_signature"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    module = ast.parse("from __future__ import annotations")
    module.body += [
        plan,
        ast.ClassDef(
            name="Connector", bases=[], keywords=[], body=methods, decorator_list=[]
        ),
    ]
    state_factory = Mock(side_effect=lambda *tensors: tensors)
    group_factory = Mock(side_effect=lambda states: SimpleNamespace(states=states))
    ns = dict(
        torch=torch,
        _SPARSE_DESTINATION_PLAN_CACHE_SIZE=4,
        lmc_ops=SimpleNamespace(
            IndexerC8State=state_factory, IndexerC8GroupState=group_factory
        ),
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    connector = ns["Connector"]()
    connector.indexer_c8_layout = object()
    connector.enable_npu_transfer_validation = validation
    connector._expected_group_layers = lambda _: 2
    keys = [torch.empty(2, 128, 1, 128, dtype=torch.int8) for _ in range(2)]
    scales = [torch.empty(2, 128, 1, 1, dtype=torch.float16) for _ in range(2)]

    def get():
        return connector._get_or_create_sparse_destination_plan(
            kvcaches_ref=list(zip(keys, scales, strict=True)),
            kv_group=1,
            slot_mapping_ref=torch.arange(4),
            sparse_kv_format=1,
            sparse_k_hidden_dims=130,
            sparse_v_hidden_dims=0,
            sparse_dsa_hidden_dims=130,
            expected_device=torch.device("cpu"),
        )

    original = get()
    assert get() is original
    assert state_factory.call_count == 2
    assert group_factory.call_count == 1
    scales[1] = torch.empty_like(scales[1])
    rebound = get()
    assert rebound is not original
    assert get() is rebound
    assert group_factory.call_count == 2
