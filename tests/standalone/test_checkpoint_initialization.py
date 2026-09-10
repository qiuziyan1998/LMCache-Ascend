# SPDX-License-Identifier: Apache-2.0
"""Validate checkpoint admission before resource initialization, using real init code."""

import ast
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Role(Enum):
    SCHEDULER = 1
    WORKER = 2


class Config(NS):
    def get_extra_config_value(self, key, default):
        return default


def init_api(config):
    base_path = ROOT.parent / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py"
    ascend_path = ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    base_tree, ascend_tree = [
        ast.parse(p.read_text(encoding="utf-8")) for p in (base_path, ascend_path)
    ]
    init = next(
        n
        for n in base_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl"
    )
    init.body = [
        n
        for n in init.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"__init__", "_validate_preemption_checkpoint_setup"}
    ]
    validate = next(
        n
        for n in ast.walk(ascend_tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_validate_preemption_checkpoint_setup"
    )
    calls = []

    class ResourcesStarted(RuntimeError):
        pass

    def factory(*args, **kwargs):
        calls.append("factory")
        raise ResourcesStarted()

    ns = dict(
        lmcache_get_or_create_config=lambda: config,
        LMCacheEngineConfig=Config,
        VllmServiceFactory=factory,
        KVConnectorRole=Role,
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, init, validate], type_ignores=[])
            ),
            str(base_path),
            "exec",
        ),
        ns,
    )
    cls = type(
        "AscendAdapter",
        (ns[init.name],),
        {
            "_apply_extra_config": lambda *a: None,
            "_validate_preemption_checkpoint_setup": ns[validate.name],
        },
    )
    return cls, calls, ResourcesStarted


def configs():
    config = Config(
        decode_preemption_checkpoint=True,
        pd_role="receiver",
        store_async=True,
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        enable_sparse_attention=True,
        dsa_two_groups=True,
        enable_dsa_cold_compact_load=True,
        dsa_group1_load_mode="persistent_direct_hbm",
    )
    vllm = NS(
        device_config=NS(device="cpu"),
        kv_transfer_config=NS(kv_role="kv_both"),
        parallel_config=NS(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        cache_config=NS(enable_prefix_caching=False),
    )
    parent = type(
        "CheckpointConnector", (), {"handle_preemptions_with_metadata": lambda *a: None}
    )()
    return config, vllm, parent


def test_invalid_checkpoint_config_is_rejected_before_services_start():
    config, vllm, parent = configs()
    config.pd_role = "sender"
    cls, calls, _ = init_api(config)
    with pytest.raises(ValueError, match="requires"):
        cls(vllm, Role.SCHEDULER, parent)
    assert calls == []


def test_scheduler_does_not_import_native_ops():
    config, vllm, parent = configs()
    cls, calls, started = init_api(config)
    with pytest.raises(started):
        cls(vllm, Role.SCHEDULER, parent)
    assert calls == ["factory"]


def test_missing_native_binding_is_not_swallowed_by_manager(monkeypatch):
    config, vllm, parent = configs()
    package = ModuleType("lmcache_ascend")
    package.c_ops = NS()
    monkeypatch.setitem(sys.modules, "lmcache_ascend", package)
    cls, calls, _ = init_api(config)
    with pytest.raises(ValueError, match="Rebuild"):
        cls(vllm, Role.WORKER, parent)
    assert calls == []


def test_connector_without_checkpoint_delegation_is_rejected():
    config, vllm, _ = configs()
    cls, calls, _ = init_api(config)
    with pytest.raises(ValueError, match="dynamic"):
        cls(vllm, Role.SCHEDULER, object())
    assert calls == []
