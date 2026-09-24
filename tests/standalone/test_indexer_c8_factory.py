# SPDX-License-Identifier: Apache-2.0
"""Execute the factory and derived metadata constructor without NPU allocation."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


def factory_api(native=True):
    root = Path(__file__).parents[2] / "lmcache_ascend/v1/npu_connector"
    source = root / "npu_connectors.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "from_metadata"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(
        ast.ClassDef(
            name="Derived", bases=[], keywords=[], body=[method], decorator_list=[]
        )
    )
    symbols = (
        "IndexerC8State",
        "IndexerC8GroupState",
        "indexer_c8_transfer_prepared",
        "indexer_c8_group_transfer_prepared",
    )
    ns = dict(lmc_ops=NS(**{name: object() for name in symbols}) if native else NS())
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    calls = []

    def init(self, **kwargs):
        calls.append(kwargs)

    derived = ns["Derived"]
    derived.__init__ = init
    source = root / "__init__.py"
    method = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "CreateNPUConnector"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    device_probe = Mock(return_value=8)
    ns.update(
        _build_info=NS(__framework_name__="pytorch"),
        EngineType=NS(VLLM="vllm"),
        need_gpu_interm_buffer=lambda _: False,
        configure_npu_content_diagnostics=Mock(),
        torch=NS(
            npu=NS(device_count=device_probe, set_device=Mock()), device=lambda x: x
        ),
        VLLMPagedMemLayerwiseNPUConnector=derived,
    )
    exec(compile(tree, str(source), "exec"), ns)
    config = NS(use_layerwise=True, enable_blending=False, dsa_two_groups=True)
    descriptor = object()
    metadata = NS(
        indexer_c8_layout=descriptor,
        use_mla=True,
        worker_id=6,
        kv_shape=(3, 1, 1024, 1, 576),
        kv_dtype=torch.bfloat16,
        max_model_len=178000,
        runtime_kv_group_layer_counts=(3, 1),
    )
    return ns[method.name], config, metadata, derived, calls, device_probe


def test_c8_factory_enters_derived_connector_with_same_descriptor():
    create, config, metadata, derived, calls, _ = factory_api()
    connector = create(config, metadata, "vllm")
    assert isinstance(connector, derived)
    assert calls[0]["indexer_c8_layout"] is metadata.indexer_c8_layout
    assert calls[0]["device"] == "npu:6"
    assert calls[0]["dtype"] == torch.bfloat16
    assert connector.runtime_kv_group_layer_counts == (3, 1)
    assert connector.dsa_two_groups is True


def test_missing_c8_native_capability_fails_before_connector_allocation():
    create, config, metadata, _, calls, _ = factory_api(native=False)
    with pytest.raises(RuntimeError, match="rebuilding"):
        create(config, metadata, "vllm")
    assert not calls


@pytest.mark.parametrize("invalid", ["dsa_two_groups", "use_layerwise"])
def test_incompatible_c8_connector_rejected_before_device_setup(invalid):
    create, config, metadata, _, calls, probe = factory_api()
    setattr(config, invalid, False)
    with pytest.raises(ValueError, match="two-group layerwise"):
        create(config, metadata, "vllm")
    probe.assert_not_called()
    assert not calls
