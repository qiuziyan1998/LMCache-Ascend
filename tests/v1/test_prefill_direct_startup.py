# SPDX-License-Identifier: Apache-2.0
"""Feature selection occurs after configuration and before starting services."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.v1 import cache_engine as engine_module
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
from lmcache_ascend.prefill_direct import PrefillDirectConnector


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("cached_mode", [None, False, True])
def test_adapter_selects_once_after_extra_config(
    monkeypatch: pytest.MonkeyPatch, initial: bool, override: Any, cached_mode: Any
) -> None:
    expected = initial if override is None else override

    class Config(SimpleNamespace):
        pass

    config = Config(prefill_group0_direct_hbm=initial, extra_config={}, validate=Mock())
    base = LMCacheAscendConnectorV1Impl.__mro__[1]
    observed = []

    class Manager:
        def __init__(self, config: Any, factory: Any, connector: Any) -> None:
            self.connector = connector
            self.lmcache_engine = (
                None
                if cached_mode is None
                else type("CachedEngine", (), {"prefill_direct_active": cached_mode})()
            )

        def start_services(self) -> None:
            observed.append(type(self.connector))

    monkeypatch.setattr(adapter_module, "LMCacheEngineConfig", Config)
    monkeypatch.setattr(adapter_module, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(adapter_module, "VllmServiceFactory", lambda *args: None)
    monkeypatch.setattr(adapter_module, "LMCacheManager", Manager)
    monkeypatch.setattr(base, "_init_connector_state", lambda *args: None)
    monkeypatch.setattr(base, "_setup_metrics", lambda *args: None)
    monkeypatch.setattr(adapter_module, "VLLM_VERSION", "test")
    monkeypatch.setattr(
        adapter_module, "utils", SimpleNamespace(get_version=lambda: "test")
    )
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config=(
                {}
                if override is None
                else {"lmcache.prefill_group0_direct_hbm": override}
            ),
        ),
    )
    instance = object.__new__(LMCacheAscendConnectorV1Impl)
    if cached_mode is not None and cached_mode != expected:
        with pytest.raises(ValueError, match="fixed at engine startup"):
            base.__init__(
                instance, vllm_config, SimpleNamespace(name="WORKER"), object()
            )
        assert observed == []
        return
    base.__init__(instance, vllm_config, SimpleNamespace(name="WORKER"), object())
    assert type(instance) is (
        PrefillDirectConnector if expected else LMCacheAscendConnectorV1Impl
    )
    assert observed == [type(instance)]
    assert config.validate.call_count == int(expected)
    assert instance.config is config
    if not expected:
        assert type(instance)._start_load_kv is base._start_load_kv
        assert not hasattr(instance, "_prefill_group0_direct_hbm")


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_factory_preserves_type_and_rejects_live_mode_change(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    created = []

    class Engine:
        def __init__(self, *args: Any) -> None:
            created.append((type(self), args))

        @classmethod
        def prefill_direct_type(cls) -> type:
            return SpecializedEngine

    class SpecializedEngine(Engine):
        prefill_direct_active = True

    builder = engine_module.LMCacheEngineBuilder
    for name in ("_instances", "_cfgs", "_metadatas", "_stat_loggers"):
        monkeypatch.setattr(builder, name, {})
    monkeypatch.setattr(builder, "_Create_token_database", lambda *args: "tokens")
    monkeypatch.setattr(engine_module, "LMCacheEngine", Engine)
    monkeypatch.setattr(
        engine_module,
        "NUMADetector",
        SimpleNamespace(get_numa_mapping=lambda config: None),
    )
    monkeypatch.setattr(engine_module, "LMCacheStatsLogger", Mock())
    config = SimpleNamespace(prefill_group0_direct_hbm=enabled, validate=Mock())
    metadata = object()
    args = ("test", config, metadata, None, None, None, None)
    engine = builder.get_or_create(*args)
    assert type(engine) is (SpecializedEngine if enabled else Engine)
    assert builder.get_or_create(*args) is engine
    assert len(created) == 1
    assert created[0][1] == (config, metadata, "tokens", None, None, None, None)
    config.prefill_group0_direct_hbm = not enabled
    with pytest.raises(ValueError, match="fixed at engine startup"):
        builder.get_or_create(*args)
    assert len(created) == 1
