# SPDX-License-Identifier: Apache-2.0
"""Decoder-side direct remote LocalCPU fill contract tests."""

# Standard
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import logging

# Third Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.remote_fill import (
    ControlPage,
    PageDisposition,
    ReservedPageView,
    UnsafePageLifecycleError,
    content_digest,
)
from lmcache.v1.shared_cpu_cache import SharedHandleBatch
from lmcache.v1.storage_backend.local_cpu_backend import (
    LayerPageAdmissionRollbackError,
)
import pytest
import torch

# First Party
from lmcache_ascend.v1.cache_engine import (
    AscendLMCacheEngine,
    _remote_fill_advertise_host_from_session,
    _validate_remote_fill_control_port,
)
from lmcache_ascend.v1.remote_fill import (
    AscendRemoteFillPageLifecycle,
    DecoderRemoteFillRuntime,
    DecoderRemoteFillServiceHost,
    RemoteFillDecoderLayout,
    RemoteFillGroupLayout,
    build_decoder_layout,
    create_decoder_remote_fill_runtime,
    remote_fill_token_hash_identity,
)
from lmcache_ascend.v1.remote_fill_producer import RemoteFillFatalError


@dataclass
class _FakePage:
    size: int
    data_ptr: int
    refs: int = 1
    pins: int = 0
    ref_failures: int = 0

    def get_size(self) -> int:
        return self.size

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        if self.ref_failures:
            self.ref_failures -= 1
            raise RuntimeError("injected ref release failure")
        if self.refs <= 0:
            raise RuntimeError("page reference underflow")
        self.refs -= 1

    def unpin(self) -> None:
        if self.pins <= 0:
            raise RuntimeError("page pin underflow")
        self.pins -= 1


@pytest.mark.parametrize(
    ("algorithm", "hash_type", "hash_bytes", "expected"),
    (
        ("builtin", int, None, "builtin"),
        ("sha256", int, None, "sha256:int"),
        ("sha256_cbor", bytes, 32, "sha256_cbor:bytes:32"),
        ("xxhash", bytes, 16, "xxhash:bytes:16"),
    ),
)
def test_token_hash_identity_includes_actual_output_abi(
    algorithm: str,
    hash_type: type[int] | type[bytes],
    hash_bytes: int | None,
    expected: str,
) -> None:
    assert remote_fill_token_hash_identity(algorithm, hash_type, hash_bytes) == expected


@pytest.mark.parametrize(
    ("algorithm", "hash_type", "hash_bytes"),
    (
        ("builtin", bytes, 32),
        ("sha256", bytes, None),
        ("sha256", int, 32),
    ),
)
def test_token_hash_identity_rejects_inconsistent_contract(
    algorithm: str,
    hash_type: type[int] | type[bytes],
    hash_bytes: int | None,
) -> None:
    with pytest.raises(ValueError):
        remote_fill_token_hash_identity(algorithm, hash_type, hash_bytes)


def test_decoder_layout_infers_dsa_dimensions_without_manual_config(
    tmp_path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"kv_lora_rank":512,"qk_rope_head_dim":64,"dsa_head_dim":128}',
        encoding="utf-8",
    )
    config = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")
    config.dsa_two_groups = True
    config.remote_fill_cache_namespace = "test-deployment"
    config.remote_fill_model_artifact_id = "test-serving-bundle"
    config.extra_config = {"save_only_first_rank": True}
    metadata = LMCacheMetadata(
        model_name=str(model_dir),
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(79, 1, 1024, 1, 576),
        use_mla=True,
        chunk_size=1024,
    )

    layout = build_decoder_layout(
        config,
        metadata,
        layout_tag="payload-v3",
        num_layers=79,
    )

    assert layout.group_dimensions == (576, 128)


def test_decoder_layout_validation_uses_registered_groups_before_lazy_connector(
    tmp_path,
) -> None:
    """Startup must not use the connector's stale pre-forward group layout."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"kv_lora_rank":512,"qk_rope_head_dim":64,"dsa_head_dim":128}',
        encoding="utf-8",
    )
    config = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")
    config.dsa_two_groups = True
    config.remote_fill_cache_namespace = "test-deployment"
    config.remote_fill_model_artifact_id = "test-serving-bundle"
    config.extra_config = {"save_only_first_rank": True}
    metadata = LMCacheMetadata(
        model_name=str(model_dir),
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(79, 1, 1024, 1, 576),
        use_mla=True,
        chunk_size=1024,
    )
    metadata.kv_layer_groups_manager.kv_layer_groups = [
        KVLayerGroupInfo(
            [f"model.layers.{index}.self_attn.attn" for index in range(79)],
            list(range(79)),
            torch.Size([1946, 128, 576]),
            torch.bfloat16,
        ),
        KVLayerGroupInfo(
            [
                f"model.layers.{index}.self_attn.indexer.k_cache"
                for index in range(79)
            ],
            list(range(79)),
            torch.Size([8757, 128, 128]),
            torch.bfloat16,
        ),
    ]
    layout = build_decoder_layout(
        config,
        metadata,
        layout_tag="payload-v3",
        num_layers=79,
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = metadata
    engine.config = config
    engine.dsa_two_groups = True
    engine.gpu_connector = SimpleNamespace(
        get_shape=Mock(side_effect=AssertionError("lazy layout is unavailable"))
    )

    engine._validate_remote_fill_decoder_layout(layout)

    assert engine._expected_shared_cpu_chunk_metadata(
        kv_group=0, num_tokens=1
    ) == (
        torch.Size([576]),
        torch.bfloat16,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert engine._expected_shared_cpu_chunk_metadata(
        kv_group=1, num_tokens=1
    ) == (
        torch.Size([128]),
        torch.bfloat16,
        MemoryFormat.KV_DSA_INDEX_FMT,
    )


def test_decoder_layout_validation_rejects_unequal_group_layer_coverage(
    tmp_path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"kv_lora_rank":512,"qk_rope_head_dim":64,"dsa_head_dim":128}',
        encoding="utf-8",
    )
    config = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")
    config.dsa_two_groups = True
    config.remote_fill_cache_namespace = "test-deployment"
    config.remote_fill_model_artifact_id = "test-serving-bundle"
    config.extra_config = {"save_only_first_rank": True}
    metadata = LMCacheMetadata(
        model_name=str(model_dir),
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(81, 1, 1024, 1, 576),
        use_mla=True,
        chunk_size=1024,
    )
    metadata.kv_layer_groups_manager.kv_layer_groups = [
        KVLayerGroupInfo(
            [f"latent.{index}" for index in range(81)],
            list(range(81)),
            torch.Size([1946, 128, 576]),
            torch.bfloat16,
        ),
        KVLayerGroupInfo(
            [f"indexer.{index}" for index in range(79)],
            list(range(79)),
            torch.Size([8757, 128, 128]),
            torch.bfloat16,
        ),
    ]
    layout = build_decoder_layout(
        config,
        metadata,
        layout_tag="payload-v3",
        num_layers=81,
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = metadata
    engine.config = config
    engine.dsa_two_groups = True
    engine.gpu_connector = SimpleNamespace()

    with pytest.raises(ValueError, match="registered_layers=79"):
        engine._validate_remote_fill_decoder_layout(layout)


class _FakeLocalBackend:
    def __init__(self) -> None:
        self.hot: dict[CacheEngineKey, _FakePage] = {}
        self.allocations: list[dict[str, Any]] = []
        self.next_ptr = 0x100000
        self.commit_error: Exception | None = None

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        assert pin is False
        return key in self.hot

    def batched_allocate_layer_pages(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        batch_size: int,
        num_layers: int,
        fmt: MemoryFormat,
        busy_loop: bool = True,
        valid_tokens: int | list[int] | None = None,
        full_tokens: int | None = None,
        eviction: bool = True,
    ) -> list[_FakePage]:
        assert full_tokens is not None
        counts = (
            [full_tokens] * batch_size
            if valid_tokens is None
            else [valid_tokens] * batch_size
            if isinstance(valid_tokens, int)
            else valid_tokens
        )
        token_dim = int(shapes[0].numel()) // full_tokens
        pages = []
        for count in counts:
            size = count * token_dim * dtypes[0].itemsize * num_layers
            pages.append(_FakePage(size=size, data_ptr=self.next_ptr))
            self.next_ptr += size + 0x1000
        self.allocations.append(
            {
                "fmt": fmt,
                "busy_loop": busy_loop,
                "eviction": eviction,
                "valid_tokens": tuple(counts),
                "pages": tuple(pages),
            }
        )
        return pages

    def commit_external_two_group_prefix_if_absent(
        self,
        required_group0_keys: tuple[CacheEngineKey, ...],
        required_group1_keys: tuple[CacheEngineKey, ...],
        ready_reservations: dict[CacheEngineKey, _FakePage],
    ) -> SimpleNamespace:
        if self.commit_error is not None:
            raise self.commit_error
        required = tuple(
            key
            for pair in zip(
                required_group0_keys,
                required_group1_keys,
                strict=True,
            )
            for key in pair
        )
        missing = tuple(
            key
            for key in required
            if key not in self.hot and key not in ready_reservations
        )
        if missing:
            return SimpleNamespace(
                committed=False,
                inserted_keys=(),
                existing_keys=(),
                missing_keys=missing,
                lock_wait_seconds=0.0,
                lock_hold_seconds=0.0,
            )
        inserted = []
        existing = []
        for key in required:
            if key in self.hot:
                existing.append(key)
                continue
            page = ready_reservations[key]
            page.ref_count_up()
            self.hot[key] = page
            inserted.append(key)
        return SimpleNamespace(
            committed=True,
            inserted_keys=tuple(inserted),
            existing_keys=tuple(existing),
            missing_keys=(),
            lock_wait_seconds=0.0,
            lock_hold_seconds=0.0,
            retention_trace_id=17,
        )


class _FakeServer:
    def __init__(self) -> None:
        self.closed = False

    def serve_once(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _FailingServer(_FakeServer):
    def serve_once(self) -> bool:
        raise RuntimeError("control receive failed")


class _FakeService:
    def __init__(
        self,
        *,
        maintenance: tuple[str, ...] = (),
        shutdown: tuple[str, ...] = (),
    ) -> None:
        self.maintenance = maintenance
        self.shutdown_result = shutdown
        self.shutdown_calls = 0
        self.maintenance_called = Event()
        self.metrics = _FakeMetricsSnapshot()

    def run_maintenance(self) -> tuple[str, ...]:
        self.maintenance_called.set()
        return self.maintenance

    def shutdown(self) -> tuple[str, ...]:
        self.shutdown_calls += 1
        return self.shutdown_result

    def metrics_snapshot(self) -> "_FakeMetricsSnapshot":
        return self.metrics


@dataclass(frozen=True)
class _FakeMetricsSnapshot:
    active_transactions: int = 0
    submitted_bytes_total: int = 0


@dataclass
class _FakeCapabilityPage:
    size: int
    layer_size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    address: int = 64
    physical_size: int = 64
    num_layers: int = 2
    valid_tokens: int = 1
    refs: int = 1
    pins: int = 0

    @property
    def metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            address=self.address,
            phy_size=self.physical_size,
        )

    @property
    def is_pinned(self) -> bool:
        return self.pins > 0

    def is_valid(self) -> bool:
        return self.refs > 0

    def get_size(self) -> int:
        return self.size

    def get_shape(self) -> torch.Size:
        return self.shape

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_memory_format(self) -> MemoryFormat:
        return self.fmt

    def unpin(self) -> None:
        self.pins -= 1

    def ref_count_down(self) -> None:
        self.refs -= 1


class _FakeCapabilityBackend:
    def __init__(self, page: _FakeCapabilityPage) -> None:
        self.page = page

    def batched_allocate_layer_pages(self, *args: Any, **kwargs: Any) -> list[Any]:
        assert kwargs["busy_loop"] is False
        assert kwargs["eviction"] is False
        assert kwargs["valid_tokens"] == [1]
        return [self.page]


class _FakeCapabilityEngine:
    _remote_fill_group1_startup_identity = (
        AscendLMCacheEngine._remote_fill_group1_startup_identity
    )
    _validate_remote_fill_decoder_layout = (
        AscendLMCacheEngine._validate_remote_fill_decoder_layout
    )
    _preflight_remote_fill_shared_group1 = (
        AscendLMCacheEngine._preflight_remote_fill_shared_group1
    )

    def __init__(
        self,
        *,
        rank: int,
        broadcast: Any,
        page: _FakeCapabilityPage,
        passive_page: _FakeCapabilityPage | None = None,
        collective_all_true: Any = bool,
    ) -> None:
        self.metadata = SimpleNamespace(
            world_size=2,
            first_rank=0,
            worker_id=rank,
            is_first_rank=lambda: rank == 0,
        )
        self.shared_cpu_cache_generation = 11
        self._remote_fill_shared_group1_supported = False
        self._broadcast = broadcast
        self._page = page
        self._passive_page = passive_page
        self.collective_all_true_fn = collective_all_true

    def broadcast_object_fn(self, payload: Any, source_rank: int) -> Any:
        return self._broadcast(payload, source_rank)

    def _expected_shared_cpu_chunk_metadata(
        self, *, kv_group: int, num_tokens: int
    ) -> tuple[torch.Size, torch.dtype, MemoryFormat]:
        assert num_tokens == 1
        if kv_group == 0:
            return (
                torch.Size([1, 1, 1, 4]),
                torch.bfloat16,
                MemoryFormat.KV_MLA_LATENT_FMT,
            )
        assert kv_group == 1
        return (
            torch.Size([1, 1, 1, 2]),
            torch.bfloat16,
            MemoryFormat.KV_DSA_INDEX_FMT,
        )

    def _shared_local_cpu_backend(self) -> _FakeCapabilityBackend:
        return _FakeCapabilityBackend(self._page)

    def _validate_rank0_shared_mem_obj(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _make_shared_handle_batch(
        self, memory_objs: list[list[Any]], keys: list[list[CacheEngineKey]]
    ) -> SharedHandleBatch:
        assert all(layer == [self._page] for layer in memory_objs)
        return SharedHandleBatch(
            shm_name="remote-fill-test",
            producer_rank=0,
            num_layers=2,
            num_chunks=1,
            physical_sizes=[self._page.layer_size],
            chunk_hashes=[keys[0][0].chunk_hash],
            offsets=[],
            page_offsets=[self._page.address],
            page_physical_sizes=[self._page.physical_size],
        )

    def _make_passive_layer_page_views(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        if self._passive_page is None:
            raise AssertionError("rank0 must not construct a passive view")
        return (self._passive_page,)

    @staticmethod
    def _release_shared_retrieve_objs(pages: list[Any], *, unpin: bool) -> None:
        assert unpin is False
        while pages:
            pages.pop().ref_count_down()


def _layout() -> RemoteFillDecoderLayout:
    return RemoteFillDecoderLayout(
        layout_tag="layout-v3",
        cache_namespace_tag="namespace-v3",
        model_artifact_id="weights-v1",
        model_name="model",
        key_world_size=1,
        chunk_size=4,
        num_layers=2,
        groups=(
            RemoteFillGroupLayout(
                kv_group=0,
                raw_token_dim=4,
                dtype=torch.bfloat16,
                fmt=MemoryFormat.KV_MLA_LATENT_FMT,
            ),
            RemoteFillGroupLayout(
                kv_group=1,
                raw_token_dim=2,
                dtype=torch.bfloat16,
                fmt=MemoryFormat.KV_DSA_INDEX_FMT,
            ),
        ),
    )


def _capability_page() -> _FakeCapabilityPage:
    return _FakeCapabilityPage(
        size=8,
        layer_size=4,
        shape=torch.Size([1, 1, 1, 2]),
        dtype=torch.bfloat16,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
    )


def test_group1_shared_startup_uses_rank0_broadcast_and_tp_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def pin_many(pages: list[_FakeCapabilityPage]) -> bool:
        for page in pages:
            page.pins += 1
        return True

    monkeypatch.setattr(LayerPageMemoryObj, "pin_many", staticmethod(pin_many))
    rank0_page = _capability_page()

    def rank0_broadcast(payload: Any, source_rank: int) -> Any:
        assert source_rank == 0
        if isinstance(payload, dict) and "batch" in payload:
            captured["payload"] = payload
            return payload
        return payload

    rank0 = _FakeCapabilityEngine(
        rank=0,
        broadcast=rank0_broadcast,
        page=rank0_page,
    )
    descriptor = {"version": 3, "group1_schema_version": "dsa-index-v2"}
    rank0._preflight_remote_fill_shared_group1(_layout(), descriptor)

    assert rank0._remote_fill_shared_group1_supported is True
    assert rank0_page.pins == 0
    assert rank0_page.refs == 0
    assert "payload" in captured

    passive_page = _capability_page()

    def passive_broadcast(payload: Any, source_rank: int) -> Any:
        assert source_rank == 0
        if source_rank == 0:
            return captured["payload"]
        return payload

    passive = _FakeCapabilityEngine(
        rank=1,
        broadcast=passive_broadcast,
        page=_capability_page(),
        passive_page=passive_page,
    )
    passive._preflight_remote_fill_shared_group1(_layout(), descriptor)

    assert passive._remote_fill_shared_group1_supported is True
    assert passive_page.refs == 0


def test_startup_rejects_group_layout_not_matching_decoder_metadata() -> None:
    engine = _FakeCapabilityEngine(
        rank=0,
        broadcast=lambda payload, source_rank: payload,
        page=_capability_page(),
    )
    layout = _layout()
    incompatible = RemoteFillDecoderLayout(
        layout_tag=layout.layout_tag,
        cache_namespace_tag=layout.cache_namespace_tag,
        model_artifact_id=layout.model_artifact_id,
        model_name=layout.model_name,
        key_world_size=layout.key_world_size,
        chunk_size=layout.chunk_size,
        num_layers=layout.num_layers,
        groups=(
            RemoteFillGroupLayout(
                kv_group=0,
                raw_token_dim=5,
                dtype=torch.bfloat16,
                fmt=MemoryFormat.KV_MLA_LATENT_FMT,
            ),
            layout.groups[1],
        ),
    )

    with pytest.raises(ValueError, match="kv_group=0"):
        engine._validate_remote_fill_decoder_layout(incompatible)


def test_group1_shared_startup_fails_closed_on_passive_rank_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pin_many(pages: list[_FakeCapabilityPage]) -> bool:
        for page in pages:
            page.pins += 1
        return True

    monkeypatch.setattr(LayerPageMemoryObj, "pin_many", staticmethod(pin_many))
    rank0_page = _capability_page()

    engine = _FakeCapabilityEngine(
        rank=0,
        broadcast=lambda payload, source_rank: payload,
        page=rank0_page,
        collective_all_true=lambda local_ready: False,
    )
    with pytest.raises(ValueError, match="another TP rank rejected"):
        engine._preflight_remote_fill_shared_group1(
            _layout(),
            {"version": 3, "group1_schema_version": "dsa-index-v2"},
        )

    assert engine._remote_fill_shared_group1_supported is False
    assert rank0_page.pins == 0
    assert rank0_page.refs == 0


def _key(
    chunk_index: int,
    kv_group: int,
    valid_tokens: int = 4,
    *,
    chunk_hash: int | bytes | None = None,
) -> CacheEngineKey:
    request_configs = {"lmcache.tag.payload_v3": "payload-v3"}
    if kv_group == 1:
        request_configs["lmcache.tag.dsa_idx"] = "v3"
    if valid_tokens != 4:
        request_configs["lmcache.tag.internal.valid_tokens"] = valid_tokens
    return CacheEngineKey(
        "model",
        1,
        0,
        100 + chunk_index if chunk_hash is None else chunk_hash,
        torch.bfloat16,
        request_configs,
        kv_group=kv_group,
    )


def _control_page(
    chunk_index: int,
    kv_group: int,
    *,
    valid_tokens: int = 4,
    expected_bytes: int | None = None,
    chunk_hash: int | bytes | None = None,
) -> ControlPage:
    layout = _layout()
    key = _key(
        chunk_index,
        kv_group,
        valid_tokens,
        chunk_hash=chunk_hash,
    )
    return ControlPage(
        canonical_key=key.to_string(),
        kv_group=kv_group,
        chunk_index=chunk_index,
        chunk_start=chunk_index * layout.chunk_size,
        chunk_end=chunk_index * layout.chunk_size + valid_tokens,
        valid_tokens=valid_tokens,
        destination_tp_rank=0,
        expected_bytes=(
            layout.group(kv_group).expected_bytes(valid_tokens, layout.num_layers)
            if expected_bytes is None
            else expected_bytes
        ),
        layer_count=layout.num_layers,
        layout_tag=layout.layout_tag,
    )


def _lifecycle(
    local: _FakeLocalBackend,
    *,
    capacity: bool = True,
    chunk_hash_type: type[int] | type[bytes] = int,
    chunk_hash_bytes: int | None = None,
) -> AscendRemoteFillPageLifecycle:
    def pin_pages(pages: list[_FakePage]) -> bool:
        for page in pages:
            page.pins += 1
        return True

    return AscendRemoteFillPageLifecycle(
        local_backend=local,
        layout=_layout(),
        capacity_available=lambda requested: capacity and requested >= 0,
        chunk_hash_type=chunk_hash_type,
        chunk_hash_bytes=chunk_hash_bytes,
        pin_pages=pin_pages,
    )


def _finish(required_store_end: int, partial: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        required_store_end=required_store_end,
        persistent_common_end=required_store_end,
        final_partial_valid_tokens=partial,
    )


def _views(
    controls: tuple[ControlPage, ...],
    prepared: tuple[Any, ...],
) -> tuple[ReservedPageView, ...]:
    return tuple(
        ReservedPageView(
            control_page=control,
            prepared=result,
            reservation_id=f"reservation-{index}",
        )
        for index, (control, result) in enumerate(zip(controls, prepared, strict=True))
        if result.disposition is not PageDisposition.MISSING
    )


def test_full_and_partial_pages_publish_only_after_exact_atomic_finish(
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = tuple(
        _control_page(chunk, group, valid_tokens=4 if chunk == 0 else 2)
        for chunk in range(2)
        for group in (0, 1)
    )

    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)

    assert all(item.disposition is PageDisposition.ALLOCATED for item in prepared)
    assert local.hot == {}
    assert [call["valid_tokens"] for call in local.allocations] == [(4, 2), (4, 2)]
    assert all(
        not call["busy_loop"] and not call["eviction"] for call in local.allocations
    )
    assert [item.destination_length for item in prepared] == [64, 32, 32, 16]

    with caplog.at_level(logging.INFO):
        committed = lifecycle.commit_pages(
            "transfer",
            controls,
            _views(controls, prepared),
            _finish(6, partial=2),
        )

    assert committed is True
    assert set(local.hot) == {
        _key(chunk, group, valid_tokens=4 if chunk == 0 else 2)
        for chunk in range(2)
        for group in (0, 1)
    }
    for call in local.allocations:
        for page in call["pages"]:
            assert page.refs == 1
            assert page.pins == 0
    group0 = tuple(
        CacheEngineKey.from_string(controls[index].canonical_key)
        for index in range(0, len(controls), 2)
    )
    group1 = tuple(
        CacheEngineKey.from_string(controls[index].canonical_key)
        for index in range(1, len(controls), 2)
    )
    expected_digest = content_digest(
        (
            tuple(key.to_string() for key in group0),
            tuple(key.to_string() for key in group1),
        )
    )
    assert '"inserted_keys":4' in caplog.text
    assert '"existing_keys":0' in caplog.text
    assert '"retention_trace_id":17' in caplog.text
    assert f'"required_key_digest":"{expected_digest}"' in caplog.text


@pytest.mark.parametrize(
    "chunk_hash",
    (
        bytes.fromhex("ab" * 32),
        bytes.fromhex("00" + "ab" * 31),
    ),
)
@pytest.mark.parametrize("valid_tokens", (4, 2))
def test_digest_control_keys_match_independent_decoder_lookup(
    chunk_hash: bytes,
    valid_tokens: int,
) -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(
        local,
        chunk_hash_type=bytes,
        chunk_hash_bytes=32,
    )
    controls = (
        _control_page(0, 0, valid_tokens=valid_tokens, chunk_hash=chunk_hash),
        _control_page(0, 1, valid_tokens=valid_tokens, chunk_hash=chunk_hash),
    )

    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    assert lifecycle.commit_pages(
        "transfer",
        controls,
        _views(controls, prepared),
        _finish(valid_tokens, partial=valid_tokens if valid_tokens < 4 else 0),
    )

    lookup_keys = (
        _key(0, 0, valid_tokens=valid_tokens, chunk_hash=chunk_hash),
        _key(0, 1, valid_tokens=valid_tokens, chunk_hash=chunk_hash),
    )
    assert all(isinstance(key.chunk_hash, bytes) for key in local.hot)
    assert all(local.contains(key) for key in lookup_keys)
    assert {key.to_string() for key in local.hot} == {
        page.canonical_key for page in controls
    }


@pytest.mark.parametrize("digest_bytes", (b"", b"\xab" * 31, b"\xab" * 33))
def test_digest_control_keys_reject_unexpected_width(
    digest_bytes: bytes,
) -> None:
    lifecycle = _lifecycle(
        _FakeLocalBackend(),
        chunk_hash_type=bytes,
        chunk_hash_bytes=32,
    )
    controls = (
        _control_page(0, 0, chunk_hash=digest_bytes),
        _control_page(0, 1, chunk_hash=digest_bytes),
    )

    with pytest.raises(ValueError, match="unexpected digest width"):
        lifecycle.prepare_pages("transfer", 0, controls, True)


@pytest.mark.parametrize(
    ("commit_error", "expected_error"),
    (
        (RuntimeError("rollback completed"), RuntimeError),
        (
            LayerPageAdmissionRollbackError("rollback incomplete"),
            UnsafePageLifecycleError,
        ),
    ),
)
def test_local_commit_distinguishes_safe_and_unsafe_rollback(
    commit_error: Exception,
    expected_error: type[Exception],
) -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    views = _views(controls, prepared)
    local.commit_error = commit_error

    with pytest.raises(expected_error):
        lifecycle.commit_pages(
            "transfer",
            controls,
            views,
            _finish(4),
        )

    assert local.hot == {}
    lifecycle.release_pages(views, "test cleanup")


def test_post_commit_release_is_retry_safe_and_fatal() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    handle = prepared[0].handle
    handle.page.ref_failures = 1
    with pytest.raises(UnsafePageLifecycleError, match="after publish"):
        lifecycle.commit_pages(
            "transfer",
            controls,
            _views(controls, prepared),
            _finish(4),
        )

    assert set(local.hot) == {_key(0, 0), _key(0, 1)}
    assert handle.unpinned is True
    assert handle.reference_released is False
    assert handle.released is False
    handle.release()
    assert handle.page.pins == 0
    assert handle.page.refs == 1
    assert handle.released is True


def test_missing_group_manifest_cannot_publish_any_page() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)

    with pytest.raises(ValueError, match="manifest is incomplete"):
        lifecycle.commit_pages(
            "transfer",
            (controls[0],),
            _views(controls, prepared),
            _finish(4),
        )

    assert local.hot == {}
    lifecycle.release_pages(_views(controls, prepared), "test cleanup")


def test_prefix_starting_after_first_chunk_cannot_publish() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(1, 0), _control_page(1, 1))
    prepared = lifecycle.prepare_pages("transfer", 1, controls, True)

    with pytest.raises(ValueError, match="manifest has a hole"):
        lifecycle.commit_pages(
            "transfer",
            controls,
            _views(controls, prepared),
            _finish(4),
        )

    assert local.hot == {}
    lifecycle.release_pages(_views(controls, prepared), "test cleanup")


def test_existing_page_wins_race_and_hidden_duplicate_is_reclaimed() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    winners = {
        _key(0, group): _FakePage(size=controls[group].expected_bytes, data_ptr=0xF000)
        for group in (0, 1)
    }
    local.hot.update(winners)

    assert lifecycle.commit_pages(
        "transfer",
        controls,
        _views(controls, prepared),
        _finish(4),
    )
    assert local.hot == winners
    for result in prepared:
        assert result.handle.page.refs == 0
        assert result.handle.page.pins == 0


def test_probe_only_and_capacity_rejection_never_allocate() -> None:
    controls = (_control_page(0, 0), _control_page(0, 1))
    for reserve_missing, capacity in ((False, True), (True, False)):
        local = _FakeLocalBackend()
        prepared = _lifecycle(local, capacity=capacity).prepare_pages(
            "transfer",
            0,
            controls,
            reserve_missing,
        )
        assert all(item.disposition is PageDisposition.MISSING for item in prepared)
        assert local.allocations == []


def test_manifest_rejects_duplicates_and_prefiller_byte_claims() -> None:
    lifecycle = _lifecycle(_FakeLocalBackend())
    control = _control_page(0, 0)
    with pytest.raises(ValueError, match="duplicate pages"):
        lifecycle.prepare_pages("transfer", 0, (control, control), True)
    for group, valid_tokens in ((0, 4), (1, 4), (0, 2), (1, 2)):
        incorrect = _control_page(
            0,
            group,
            valid_tokens=valid_tokens,
            expected_bytes=1,
        )
        with pytest.raises(ValueError, match="expected byte count mismatch"):
            lifecycle.prepare_pages("transfer", 0, (incorrect,), True)


def test_release_is_idempotent_for_unarmed_hidden_pages() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    views = _views(controls, prepared)

    lifecycle.release_pages(views, "unarmed timeout")
    lifecycle.release_pages(views, "idempotent abort")

    for result in prepared:
        assert result.handle.page.refs == 0
        assert result.handle.page.pins == 0


def test_invalid_prepared_handle_does_not_leak_other_hidden_pages() -> None:
    local = _FakeLocalBackend()
    lifecycle = _lifecycle(local)
    controls = (_control_page(0, 0), _control_page(0, 1))
    prepared = lifecycle.prepare_pages("transfer", 0, controls, True)
    invalid = SimpleNamespace(
        disposition=PageDisposition.ALLOCATED,
        handle=object(),
    )

    with pytest.raises(TypeError, match="1 invalid allocated handles"):
        lifecycle.release_prepared_pages(
            (invalid, prepared[0], prepared[1]),
            "invalid core result",
        )

    for result in prepared:
        assert result.handle.page.refs == 0
        assert result.handle.page.pins == 0


def test_service_host_propagates_only_maintenance_fatal_evidence() -> None:
    service = _FakeService(maintenance=("armed-transfer",))
    server = _FakeServer()
    fatal: list[tuple[str, ...]] = []
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=server,
        fatal_restart=fatal.append,
    )

    host.start()
    assert service.maintenance_called.wait(timeout=1.0)
    host.close()

    assert fatal == [("armed-transfer",)]
    assert service.shutdown_calls == 1
    assert server.closed


@pytest.mark.parametrize(
    ("shutdown_result", "expected_fatal"),
    (
        ((), []),
        (("armed-transfer",), [("armed-transfer",)]),
    ),
)
def test_service_loop_failure_reports_only_real_shutdown_fatal_evidence(
    shutdown_result: tuple[str, ...],
    expected_fatal: list[tuple[str, ...]],
    caplog,
) -> None:
    service = _FakeService(shutdown=shutdown_result)
    server = _FailingServer()
    fatal: list[tuple[str, ...]] = []
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=server,
        fatal_restart=fatal.append,
    )

    with caplog.at_level(logging.ERROR):
        host.start()
        assert host._closed.wait(timeout=1.0)
    host.close()

    assert fatal == expected_fatal
    assert service.shutdown_calls == 1
    assert server.closed
    assert '"code":"RF-D-005"' in caplog.text
    assert '"action":"DISABLE_DIRECT_SERVICE_AND_AUDIT_ARMED_STATE"' in caplog.text


def test_unarmed_service_failure_stops_decoder_placement_advertisement() -> None:
    service = _FakeService()
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=_FailingServer(),
        fatal_restart=lambda transfers: None,
    )
    runtime = DecoderRemoteFillRuntime(
        lifecycle=SimpleNamespace(),
        state=SimpleNamespace(),
        service=service,
        host=host,
        placement=SimpleNamespace(to_dict=lambda: {"enabled": True}),
    )
    engine = object.__new__(AscendLMCacheEngine)
    engine._remote_fill_runtime = runtime
    engine._init_failed = False
    engine._health_monitor = None

    host.start()
    assert host._closed.wait(timeout=1.0)

    assert engine.get_remote_fill_placement_info() is None
    assert engine.is_healthy()
    host.close()


def test_close_before_start_drains_safe_service_without_fatal_restart() -> None:
    service = _FakeService()
    server = _FakeServer()
    fatal: list[tuple[str, ...]] = []
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=server,
        fatal_restart=fatal.append,
    )

    host.close()

    assert fatal == []
    assert service.shutdown_calls == 1
    assert server.closed


def test_service_thread_start_failure_closes_control_transport() -> None:
    service = _FakeService()
    server = _FakeServer()
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=server,
        fatal_restart=lambda transfers: None,
    )
    host._thread = SimpleNamespace(
        start=Mock(side_effect=RuntimeError("thread start failed")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        host.start()

    assert service.shutdown_calls == 1
    assert server.closed
    assert host._closed.is_set()
    assert host.available is False


def test_service_host_emits_only_changed_fixed_cardinality_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeService()
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=_FakeServer(),
        fatal_restart=lambda transfers: None,
    )
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "lmcache_ascend.v1.remote_fill._log_remote_fill_event",
        lambda event, **fields: events.append((event, fields)),
    )

    host._emit_metrics_if_changed(force=True)
    assert events == []

    service.metrics = _FakeMetricsSnapshot(
        active_transactions=1,
        submitted_bytes_total=4096,
    )
    host._emit_metrics_if_changed(force=True)

    assert events == [
        (
            "remote_fill_metrics",
            {
                "active_transactions": 1,
                "submitted_bytes_total_delta": 4096,
            },
        ),
        ("remote_fill_arm", {"submitted_bytes": 4096}),
    ]


def test_engine_close_retains_allocator_when_native_destination_is_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    base_close_calls = 0

    class _FatalRuntime:
        def close(self) -> None:
            engine._remote_fill_fatal_transfers = ("armed-transfer",)

    def base_close(self: Any) -> None:
        nonlocal base_close_calls
        base_close_calls += 1

    monkeypatch.setattr(LMCacheEngine, "close", base_close)
    engine._remote_fill_runtime = _FatalRuntime()
    engine._remote_fill_fatal_transfers = ()

    with pytest.raises(RuntimeError, match="deliberately retained"):
        engine.close()

    assert base_close_calls == 0
    assert engine._remote_fill_runtime is not None


def test_producer_and_decoder_fatal_paths_share_one_supervisor_latch() -> None:
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic,
    )

    engine = object.__new__(AscendLMCacheEngine)
    engine._remote_fill_fatal_transfers = ()
    failures: list[str] = []
    engine.mark_init_failed = failures.append

    future: Future = Future()
    future.set_exception(RemoteFillFatalError("native completion unknown"))
    producer_state = SimpleNamespace(
        remote_fill_handoff=SimpleNamespace(transfer_id="producer-transfer"),
        remote_fill_futures=deque((future,)),
    )
    with pytest.raises(RemoteFillFatalError):
        engine._wait_remote_fill_windows(producer_state)
    engine._remote_fill_require_paired_restart(("decoder-transfer", ""))

    assert engine.remote_fill_requires_paired_restart()
    assert engine._remote_fill_fatal_transfers == (
        "decoder-transfer",
        "producer-transfer",
    )
    assert len(failures) == 1

    connector = object.__new__(LMCacheAscendConnectorV1Dynamic)
    connector._lmcache_engine = SimpleNamespace(lmcache_engine=engine)
    assert connector.remote_fill_requires_paired_restart()


def test_engine_close_reaches_allocator_after_safe_remote_fill_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    base_close_calls = 0

    class _SafeRuntime:
        def close(self) -> None:
            return None

    def base_close(self: Any) -> None:
        nonlocal base_close_calls
        base_close_calls += 1

    monkeypatch.setattr(LMCacheEngine, "close", base_close)
    monkeypatch.setattr(
        AscendLMCacheEngine,
        "close_remote_fill_producer",
        lambda self: None,
    )
    monkeypatch.setattr(
        AscendLMCacheEngine,
        "wait_for_direct_stores",
        lambda self, states: None,
    )
    engine._remote_fill_runtime = _SafeRuntime()
    engine._remote_fill_fatal_transfers = ()
    engine._direct_store_states = {}
    engine._live_source_builders = {}
    engine._completed_live_sources = {}
    engine._pending_live_source_diagnostics = {}
    engine._store_queue = None

    engine.close()

    assert base_close_calls == 1
    assert engine._remote_fill_runtime is None


def test_prefiller_fatal_close_retains_native_owned_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(AscendLMCacheEngine)
    base_close = Mock()
    monkeypatch.setattr(LMCacheEngine, "close", base_close)
    engine._remote_fill_runtime = None
    engine._remote_fill_fatal_transfers = ("producer-transfer",)

    with pytest.raises(RuntimeError, match="paired P\+D restart"):
        engine.close()

    base_close.assert_not_called()


def test_decoder_remote_fill_startup_is_idempotent_after_success() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine._remote_fill_decoder_initialized = True

    # The success guard must precede every config, topology, collective, and
    # endpoint access. LMCache's base post_init is idempotent and a repeated
    # connector callback must not bind a second decoder control service.
    engine._initialize_decoder_remote_fill()


def _config(*, shared: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enable_remote_lmcache_store=True,
        local_cpu=True,
        use_layerwise=True,
        dsa_two_groups=True,
        enable_sparse_attention=True,
        enable_shared_cpu_cache=shared,
        pre_caching_hash_algorithm="builtin",
        remote_fill_max_rpc_message_bytes=65536,
        remote_fill_max_control_pages_per_window=8,
        remote_fill_max_inflight_bytes=1024,
        remote_fill_max_active_transactions=2,
        remote_fill_max_inflight_windows_per_request=2,
        remote_fill_max_reserved_bytes=4096,
        remote_fill_max_bytes_per_request=4096,
        remote_fill_reservation_ttl_sec=30,
        remote_fill_descriptor_ttl_sec=10,
        remote_fill_native_hard_timeout_ms=120000,
        remote_fill_terminal_record_ttl_sec=300,
    )


def _metadata(*, world_size: int, first: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        worker_id=0 if first else 1,
        use_mla=True,
        world_size=world_size,
        engine_id="decoder",
        is_first_rank=lambda: first,
    )


def test_runtime_requires_tp0_and_strict_shared_storage_for_tp_gt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    common = dict(
        local_backend=_FakeLocalBackend(),
        layout=_layout(),
        destination_engine_epoch=7,
        destination_dp_rank=0,
        destination_dp_size=1,
        destination_remote_session="decoder:1234",
        control_host="127.0.0.1",
        control_advertise_host="10.0.0.2",
        control_port=19000,
        shared_cache_generation=11,
        capacity_available=lambda size: True,
        fatal_restart=lambda transfers: None,
        global_te_push=True,
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        shared_cpu_cache_strict=True,
        chunk_hash_type=int,
        chunk_hash_bytes=None,
        descriptor_verification_capability=b"x" * 32,
        server_factory=lambda service: _FakeServer(),
    )
    with pytest.raises(ValueError, match="physical TP0"):
        create_decoder_remote_fill_runtime(
            config=_config(shared=True),
            metadata=_metadata(world_size=2, first=False),
            **common,
        )
    with pytest.raises(ValueError, match="strict shared"):
        create_decoder_remote_fill_runtime(
            config=_config(shared=False),
            metadata=_metadata(world_size=2),
            **common,
        )


def test_single_rank_runtime_advertises_pointer_free_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    server = _FakeServer()
    runtime = create_decoder_remote_fill_runtime(
        config=_config(shared=False),
        metadata=_metadata(world_size=1),
        local_backend=_FakeLocalBackend(),
        layout=_layout(),
        destination_engine_epoch=7,
        destination_dp_rank=0,
        destination_dp_size=1,
        destination_remote_session="decoder:1234",
        control_host="127.0.0.1",
        control_advertise_host="10.0.0.2",
        control_port=19000,
        shared_cache_generation=0,
        capacity_available=lambda size: True,
        fatal_restart=lambda transfers: None,
        global_te_push=True,
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        shared_cpu_cache_strict=False,
        chunk_hash_type=int,
        chunk_hash_bytes=None,
        descriptor_verification_capability=b"x" * 32,
        server_factory=lambda service: server,
    )

    placement = runtime.placement.to_dict()
    assert placement["destination_engine_id"] == "decoder"
    assert placement["destination_engine_epoch"] == 7
    assert placement["destination_tp_size"] == 1
    assert placement["destination_dp_size"] == 1
    assert placement["global_te_push"] is True
    assert placement["control_endpoint"] == "tcp://10.0.0.2:19000"
    assert placement["token_hash_algorithm"] == "builtin"
    assert placement["python_hash_seed"] == "0"
    assert placement["descriptor_verification_capability"] == (b"x" * 32).hex()
    assert "descriptor_verification_capability" not in repr(runtime.placement)
    assert all("ptr" not in key and "secret" not in key for key in placement)
    runtime.close()
    assert server.closed


def test_nonbuiltin_hash_ignores_unrelated_python_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _config(shared=False)
    config.pre_caching_hash_algorithm = "sha256_cbor_64bit"
    runtime = create_decoder_remote_fill_runtime(
        config=config,
        metadata=_metadata(world_size=1),
        local_backend=_FakeLocalBackend(),
        layout=_layout(),
        destination_engine_epoch=7,
        destination_dp_rank=0,
        destination_dp_size=1,
        destination_remote_session="decoder:1234",
        control_host="127.0.0.1",
        control_advertise_host="10.0.0.2",
        control_port=19000,
        shared_cache_generation=0,
        capacity_available=lambda required: True,
        fatal_restart=lambda transfers: None,
        global_te_push=True,
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        shared_cpu_cache_strict=False,
        chunk_hash_type=bytes,
        chunk_hash_bytes=32,
        descriptor_verification_capability=b"x" * 32,
        server_factory=lambda service: _FakeServer(),
    )

    assert (
        runtime.placement.token_hash_algorithm
        == "sha256_cbor_64bit:bytes:32"
    )
    assert runtime.placement.python_hash_seed == ""
    runtime.close()


def test_runtime_rotates_verification_capability_per_incarnation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    capabilities = iter((b"a" * 32, b"b" * 32))
    monkeypatch.setattr(
        "lmcache_ascend.v1.remote_fill.secrets.token_bytes",
        lambda size: next(capabilities),
    )
    common = dict(
        config=_config(shared=False),
        metadata=_metadata(world_size=1),
        local_backend=_FakeLocalBackend(),
        layout=_layout(),
        destination_engine_epoch=7,
        destination_dp_rank=0,
        destination_dp_size=1,
        destination_remote_session="decoder:1234",
        control_host="127.0.0.1",
        control_advertise_host="10.0.0.2",
        control_port=19000,
        shared_cache_generation=0,
        capacity_available=lambda size: True,
        fatal_restart=lambda transfers: None,
        global_te_push=True,
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        shared_cpu_cache_strict=False,
        chunk_hash_type=int,
        chunk_hash_bytes=None,
        server_factory=lambda service: _FakeServer(),
    )

    first = create_decoder_remote_fill_runtime(**common)
    second = create_decoder_remote_fill_runtime(**common)

    assert first.placement.descriptor_verification_capability == (b"a" * 32).hex()
    assert second.placement.descriptor_verification_capability == (b"b" * 32).hex()
    first.close()
    second.close()


def test_runtime_separates_bind_and_ipv6_advertised_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    runtime = create_decoder_remote_fill_runtime(
        config=_config(shared=False),
        metadata=_metadata(world_size=1),
        local_backend=_FakeLocalBackend(),
        layout=_layout(),
        destination_engine_epoch=7,
        destination_dp_rank=0,
        destination_dp_size=1,
        destination_remote_session="decoder:1234",
        control_host="0.0.0.0",
        control_advertise_host="2001:db8::7",
        control_port=19000,
        shared_cache_generation=0,
        capacity_available=lambda size: True,
        fatal_restart=lambda transfers: None,
        global_te_push=True,
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        shared_cpu_cache_strict=False,
        chunk_hash_type=int,
        chunk_hash_bytes=None,
        descriptor_verification_capability=b"x" * 32,
        server_factory=lambda service: _FakeServer(),
    )

    assert runtime.placement.control_endpoint == "tcp://[2001:db8::7]:19000"
    runtime.close()


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "[::]", "127.0.0.1", "localhost"]
)
def test_runtime_rejects_nonroutable_advertised_host(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(ValueError, match="advertised control host"):
        create_decoder_remote_fill_runtime(
            config=_config(shared=False),
            metadata=_metadata(world_size=1),
            local_backend=_FakeLocalBackend(),
            layout=_layout(),
            destination_engine_epoch=7,
            destination_dp_rank=0,
            destination_dp_size=1,
            destination_remote_session="decoder:1234",
            control_host=host,
            control_port=19000,
            shared_cache_generation=0,
            capacity_available=lambda size: True,
            fatal_restart=lambda transfers: None,
            global_te_push=True,
            save_only_first_rank=True,
            save_indexer_only_first_rank=True,
            shared_cpu_cache_strict=False,
            chunk_hash_type=int,
            chunk_hash_bytes=None,
            descriptor_verification_capability=b"x" * 32,
            server_factory=lambda service: _FakeServer(),
        )


def test_remote_fill_control_port_range_rejects_overflow_and_collision() -> None:
    assert (
        _validate_remote_fill_control_port(
            19000,
            1,
            2,
            internal_api_enabled=True,
            internal_api_port_start=6999,
        )
        == 19001
    )
    with pytest.raises(ValueError, match="control port"):
        _validate_remote_fill_control_port(
            65535,
            0,
            2,
            internal_api_enabled=False,
            internal_api_port_start=6999,
        )
    with pytest.raises(ValueError, match="overlap"):
        _validate_remote_fill_control_port(
            6998,
            0,
            2,
            internal_api_enabled=True,
            internal_api_port_start=6999,
        )


@pytest.mark.parametrize(
    ("session", "host"),
    (("decoder.example:1234", "decoder.example"), ("tcp://10.0.0.8:9", "10.0.0.8")),
)
def test_remote_fill_control_host_uses_existing_global_te_identity(
    session: str, host: str
) -> None:
    assert _remote_fill_advertise_host_from_session(session) == host
