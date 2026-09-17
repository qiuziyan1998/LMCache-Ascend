# SPDX-License-Identifier: Apache-2.0
"""Check platform patch dispatch without initializing device resources."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_preimported_dynamic_wrapper_reaches_ascend_with_runtime_topology(monkeypatch):
    # Import the generic wrapper first, as happens with plugin discovery.
    from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
    from lmcache.integration.vllm import vllm_v1_adapter as base
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_connector as outer
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    base_class = base.LMCacheConnectorV1Impl
    # Register undo records for the actual patch's module/class mutations.
    monkeypatch.setattr(base, "LMCacheConnectorV1Impl", base_class)
    for name in (
        "supports_dsa_index_lmcache",
        "uses_layerwise_model_callbacks",
        "supports_staged_sfa_sparse_load",
        "handle_preemptions",
    ):
        monkeypatch.setattr(
            outer.LMCacheConnectorV1,
            name,
            getattr(outer.LMCacheConnectorV1, name, None),
            raising=False,
        )
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/__init__.py"
    patch = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_patch_vllm_v1_adapter"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[patch], type_ignores=[]), str(path), "exec"), namespace
    )
    namespace[patch.name]()
    assert base.LMCacheConnectorV1Impl is LMCacheAscendConnectorV1Impl

    reached = []

    class BeforeResources(RuntimeError):
        pass

    def capture(self, config, role, parent, kv_cache_config=None):
        reached.append((type(self), kv_cache_config, type(parent)))
        raise BeforeResources()

    monkeypatch.setattr(base_class, "__init__", capture)
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda self, **kw: None)
    config = SimpleNamespace(kv_transfer_config=None, parallel_config=None)
    topology = SimpleNamespace(kv_cache_groups=[object(), object()])
    for wrapper in (LMCacheConnectorV1Dynamic, LMCacheAscendConnectorV1Dynamic):
        with pytest.raises(BeforeResources):
            wrapper(config, object(), topology)
    assert reached == [
        (LMCacheAscendConnectorV1Impl, topology, wrapper)
        for wrapper in (LMCacheConnectorV1Dynamic, LMCacheAscendConnectorV1Dynamic)
    ]
