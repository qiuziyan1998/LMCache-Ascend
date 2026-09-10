# SPDX-License-Identifier: Apache-2.0
"""Sender direct-prefix dispatch, ownership and TP ordering contracts."""

# Standard
from typing import Any
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch
import gc
import ctypes
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.mooncake_layout import mooncake_valid_tokens
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache_ascend.v1 import cache_engine as engine_module
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.remote_fill_producer import RemoteFillFatalError
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    KVCacheFormat,
    VLLMPagedMemLayerwiseNPUConnector,
)


def _engine(monkeypatch: pytest.MonkeyPatch, rank: int = 0, world: int = 4) -> Any:
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(
        pd_role="sender",
        prefill_latent_direct_load=True,
        dsa_index_transfer_mode="persistent_direct_hbm",
        chunk_size=4,
    )
    engine.metadata = SimpleNamespace(worker_id=rank, world_size=world, first_rank=0)
    engine._init_failed = False
    engine._health_monitor = None
    engine._is_passive = lambda: rank != 0
    engine._remote_fill_runtime = None
    engine._external_page_reader = None
    engine._remote_fill_decoder_initialized = False
    events = []
    process_calls = []

    def process_tokens(
        tokens: Any = None,
        mask: Any = None,
        hashes: Any = None,
        offsets: Any = None,
        request_configs: Any = None,
        kv_group: Any = 0,
    ) -> Any:
        process_calls.append(kv_group)
        if hashes is None:
            ranges = [
                (start, min(start + 4, len(tokens)), start)
                for start in range(0, len(tokens), 4)
                if mask is None or bool(mask[start])
            ]
        else:
            ranges, start = [], 0
            for value, length in zip(hashes, offsets, strict=True):
                ranges.append((start, start + length, value))
                start += length
        for start, end, value in ranges:
            tags = dict(request_configs or {})
            if end - start < 4:
                tags["lmcache.tag.internal.valid_tokens"] = end - start
            yield (
                start,
                end,
                CacheEngineKey(
                    "model",
                    1,
                    0,
                    value,
                    torch.float16,
                    tags,
                    kv_group=kv_group,
                ),
            )

    engine.token_database = SimpleNamespace(process_tokens=process_tokens)
    engine._remote_fill_retrieve_plan = Mock(
        side_effect=lambda req, plans, group: [("RemoteBackend", True)] * len(plans)
    )
    engine._ensure_layerwise_connector_layout = Mock()

    def plan(caches: Any, slots: Any, starts: Any, ends: Any, group: Any) -> Any:
        events.append("plan")
        return (
            [[100 + start] for start in starts],
            [[end - start] for start, end in zip(starts, ends, strict=True)],
            (object(),),
        )

    engine.gpu_connector = SimpleNamespace(
        plan_direct_page_destinations=Mock(side_effect=plan),
        direct_page_layout_supported=Mock(return_value=True),
        direct_page_load_supported=Mock(return_value=True),
        record_dense_load_readiness=Mock(
            side_effect=lambda **kw: events.append("record") or "fence"
        ),
        synchronize_dense_load_readiness=Mock(
            side_effect=lambda ready: events.append("fence")
        ),
    )
    read = Mock(side_effect=lambda *args: events.append("read"))
    reader = SimpleNamespace(batched_get_external_pages=read, close=Mock())
    if rank:
        engine._external_page_reader = reader
        engine.storage_manager = None
    else:
        engine.storage_manager = reader
    engine.collective_all_true_fn = Mock(
        side_effect=lambda ready: events.append(("tp", ready)) or ready
    )
    engine._remote_fill_require_paired_restart = Mock()
    engine.num_layers = 2
    monkeypatch.setattr(engine_module, "serving_perf_enabled", lambda: False)
    monkeypatch.setattr(
        engine_module,
        "serving_perf_now",
        Mock(side_effect=AssertionError("perf disabled")),
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: "compute"), raising=False
    )
    return engine, events, read, process_calls


def _retrieve(engine: Any, count: Any = 11, skip: Any = 4, state: Any = None) -> Any:
    mask = torch.ones(count, dtype=torch.bool)
    mask[:skip] = False
    slots = torch.arange(count)
    return engine.retrieve_prefill_group0_direct(
        list(range(count)),
        mask,
        kvcaches=[object()],
        slot_mapping=slots,
        req_id="request",
        request_configs=None,
        shared_cpu_request_preflight_state={} if state is None else state,
    )


@pytest.mark.parametrize("rank,world", [(0, 1), (0, 4), (3, 4), (7, 8)])
@pytest.mark.parametrize("count,skip", [(0, 0), (4, 4), (8, 0), (11, 4), (9, 8)])
def test_direct_prefix_preserves_ranges_protocol_and_group1_hashes(
    monkeypatch: Any, rank: Any, world: Any, count: Any, skip: Any
) -> None:
    engine, events, read, process_calls = _engine(monkeypatch, rank, world)
    state = {}
    retriever = _retrieve(engine, count, skip, state)
    values = list(retriever)
    assert values[:-1] == [None] * (engine.num_layers + 1)
    assert values[-1].tolist() == [False] * skip + [True] * (count - skip)
    assert events == (["plan", "record", "fence", "read"] if count > skip else []) + (
        [("tp", True)] if world > 1 else []
    )
    if count > skip:
        keys, ptrs, sizes, owners, req = read.call_args.args
        assert [key.kv_group for key in keys] == [0] * len(keys)
        assert [mooncake_valid_tokens(key, 4) for key in keys] == [
            min(4, count - start) for start in range(skip, count, 4)
        ]
        assert ptrs == [[100 + start] for start in range(skip, count, 4)]
        assert sum(map(sum, sizes)) == count - skip
        assert owners and req == "request"
        assert engine.gpu_connector.record_dense_load_readiness.call_args.kwargs == {
            "stream": "compute"
        }
        if rank:
            engine._remote_fill_retrieve_plan.assert_not_called()
    else:
        read.assert_not_called()
    group1 = list(
        engine._dense_retrieve_token_results(
            list(range(count)),
            None,
            None,
            1,
            {"shared_cpu_request_preflight_state": state},
        )
    )
    assert [(start, end) for start, end, _ in group1] == [
        (start, min(start + 4, count)) for start in range(skip, count, 4)
    ]
    assert process_calls == [0, 1]


@pytest.mark.parametrize(
    "failure", ["proof", "plan", "fence", "read", "peer", "unknown"]
)
def test_failure_is_decided_before_first_yield(monkeypatch: Any, failure: Any) -> None:
    engine, events, read, _ = _engine(monkeypatch)
    if failure == "proof":
        engine._remote_fill_retrieve_plan.return_value = None
        engine._remote_fill_retrieve_plan.side_effect = None
    elif failure == "plan":
        engine.gpu_connector.plan_direct_page_destinations.return_value = None
        engine.gpu_connector.plan_direct_page_destinations.side_effect = None
    elif failure == "fence":
        engine.gpu_connector.synchronize_dense_load_readiness.side_effect = (
            RuntimeError("fence failed")
        )
    elif failure == "read":
        read.side_effect = RuntimeError("read failed")
    elif failure == "unknown":
        read.side_effect = NativeExternalPageTransferUnknownError("get", Future())
    else:
        engine.collective_all_true_fn.side_effect = (
            lambda ready: events.append(("tp", ready)) or False
        )
    with pytest.raises((RuntimeError, ValueError)):
        next(_retrieve(engine))
    assert engine.collective_all_true_fn.call_count == 1
    assert engine.collective_all_true_fn.call_args.args == (failure == "peer",)
    if failure in {"proof", "plan", "fence"}:
        read.assert_not_called()
    assert engine._remote_fill_require_paired_restart.call_count == int(
        failure == "unknown"
    )


@pytest.mark.parametrize("bad_mask", [[True, False, True, True], [True, True, True]])
def test_invalid_mask_never_writes(monkeypatch: Any, bad_mask: Any) -> None:
    engine, _, read, _ = _engine(monkeypatch)
    gen = engine.retrieve_prefill_group0_direct(
        list(range(4)),
        torch.tensor(bad_mask),
        kvcaches=[object()],
        slot_mapping=torch.arange(4),
        req_id="request",
        request_configs=None,
        shared_cpu_request_preflight_state={},
    )
    with pytest.raises(ValueError):
        next(gen)
    read.assert_not_called()
    engine.collective_all_true_fn.assert_called_once_with(False)


@pytest.mark.parametrize(
    "ranges,count,group",
    [
        ([(0, 4)], 4, 1),
        ([(0, 4), (2, 6)], 6, 0),
        ([(0, 4), (8, 12)], 12, 0),
        ([(0, 3)], 4, 0),
        ([(-4, 0)], 4, 0),
        ([(0, 8)], 4, 0),
        ([(0, 4)], 8, 0),
    ],
)
def test_invalid_mask_or_page_ranges_never_submit(
    monkeypatch: pytest.MonkeyPatch,
    ranges: list[tuple[int, int]],
    count: int,
    group: int,
) -> None:
    engine, events, read, _ = _engine(monkeypatch)
    key = CacheEngineKey("model", 1, 0, 0, torch.float16, kv_group=group)
    engine.token_database.process_tokens = lambda **kwargs: iter(
        (start, end, key) for start, end in ranges
    )
    with pytest.raises(ValueError):
        next(_retrieve(engine, count=count, skip=0))
    read.assert_not_called()
    assert events == [("tp", False)]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_sender_reader_startup_and_preflight(
    monkeypatch: Any, enabled: Any, rank: Any
) -> None:
    engine, _, read, _ = _engine(monkeypatch, rank)
    engine.config.prefill_latent_direct_load = enabled
    engine.is_store_async = False
    engine._direct_store_enabled = False
    engine._initialize_decoder_remote_fill = Mock()
    reader = SimpleNamespace(batched_get_external_pages=read, close=Mock())
    factory = Mock(return_value=reader)
    monkeypatch.setattr(engine_module, "RemoteExternalPageReader", factory)
    with patch.object(LMCacheEngine, "post_init"):
        engine.post_init()
    assert factory.call_count == int(enabled and rank != 0)
    caches = [(SimpleNamespace(device=SimpleNamespace(type="npu")),)]
    engine.preflight_prefill_group0_direct_hbm(caches)
    assert engine.gpu_connector.direct_page_load_supported.call_count == int(enabled)


def test_sender_preflight_failure_closes_reader(monkeypatch: Any) -> None:
    engine, _, _, _ = _engine(monkeypatch, 1)
    reader = engine._external_page_reader
    engine.gpu_connector.direct_page_load_supported.return_value = False
    with pytest.raises(RuntimeError, match="Group-0 direct-HBM preflight failed"):
        engine.preflight_prefill_group0_direct_hbm(
            [(SimpleNamespace(device=SimpleNamespace(type="npu")),)]
        )
    reader.close.assert_called_once()
    assert engine._external_page_reader is None


def test_unknown_native_state_blocks_later_loads(monkeypatch: Any) -> None:
    engine, _, read, _ = _engine(monkeypatch)
    del engine._remote_fill_require_paired_restart
    engine._remote_fill_fatal_transfers = ()
    read.side_effect = NativeExternalPageTransferUnknownError("get", Future())
    with pytest.raises(RemoteFillFatalError):
        next(_retrieve(engine))
    assert not engine.is_healthy()
    with pytest.raises(RuntimeError, match="unavailable"):
        next(_retrieve(engine))
    assert read.call_count == 1


@pytest.mark.parametrize("failed_rank", [None, 0, 3])
def test_all_tp_ranks_decide_before_group1(monkeypatch: Any, failed_rank: Any) -> None:
    engines = [_engine(monkeypatch, rank)[0] for rank in range(4)]
    barrier = Barrier(4, timeout=5)
    flags = [None] * 4
    entered_group1 = []
    for rank, engine in enumerate(engines):

        def decision(ready: Any, rank: Any = rank) -> Any:
            flags[rank] = ready
            barrier.wait()
            return all(flags)

        engine.collective_all_true_fn = decision
        if rank == failed_rank:
            reader = (
                engine.storage_manager if rank == 0 else engine._external_page_reader
            )
            reader.batched_get_external_pages.side_effect = RuntimeError(
                "one rank failed"
            )

    def run(rank: Any) -> Any:
        generator = _retrieve(engines[rank])
        try:
            next(generator)
            entered_group1.append(rank)
        except RuntimeError:
            return False
        finally:
            generator.close()
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(4)))
    assert results == [failed_rank is None] * 4
    assert sorted(entered_group1) == (list(range(4)) if failed_rank is None else [])


def test_direct_prefix_slow_summary_is_gated_and_includes_ordering(
    monkeypatch: Any,
) -> None:
    engine, _, _, _ = _engine(monkeypatch)
    times = iter(i / 10 for i in range(40))
    monkeypatch.setattr(engine_module, "serving_perf_enabled", lambda: True)
    monkeypatch.setattr(engine_module, "serving_perf_now", lambda: next(times))
    log = Mock()
    monkeypatch.setattr(engine_module, "serving_perf_log", log)
    list(_retrieve(engine))
    assert log.call_args.args[1] == "prefill_group0_direct_load_slow"
    fields = log.call_args.kwargs
    assert fields["bytes"] == 7
    assert fields["pages"] == 2
    assert fields["predecessor_wait_ms"] > 0
    assert fields["tp_decision_ms"] > 0


def test_failed_generator_does_not_retain_itself_with_gc_disabled(
    monkeypatch: Any,
) -> None:
    engine, _, _, _ = _engine(monkeypatch)

    class Owner:
        pass

    refs = []

    def fail(*args: Any) -> Any:
        owner = Owner()
        refs.append(weakref.ref(owner))
        raise RuntimeError("known read failure")

    engine.storage_manager.batched_get_external_pages = fail
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(5):
            try:
                next(_retrieve(engine))
            except RuntimeError:
                pass
        assert all(ref() is None for ref in refs)
    finally:
        if was_enabled:
            gc.enable()


def test_direct_prefix_copies_both_planes_to_fragmented_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real destination planner through the new engine's native API."""
    engine, events, _, _ = _engine(monkeypatch)
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    layout = SimpleNamespace(kv_format=KVCacheFormat.MLA_LATENT)
    connector._lazy_initialize_buffer_with_staging = lambda *args, **kwargs: layout
    connector.get_shape = lambda n, group: torch.Size([n * 576])
    caches = [
        tuple(torch.zeros((6, 4, 1, width), dtype=torch.float16) for width in (512, 64))
        for _ in range(2)
    ]
    for layer in caches:
        for owner in layer:
            ctypes.memset(owner.data_ptr(), 0xA5, owner.numel() * owner.element_size())
    engine.gpu_connector.plan_direct_page_destinations = (
        connector.plan_direct_page_destinations
    )
    slots = torch.tensor([0, 1, 2, 3, 8, 9, 12, 13, 16, 18, 19])
    mask = torch.ones(11, dtype=torch.bool)
    mask[:4] = False

    def read(keys: Any, ptrs: Any, sizes: Any, owners: Any, req: str) -> None:
        events.append("read")
        for key, page_ptrs, page_sizes in zip(keys, ptrs, sizes, strict=True):
            start = key.chunk_hash
            end = start + mooncake_valid_tokens(key, 4)
            source = b"".join(
                bytes(
                    (ctypes.c_uint16 * ((end - start) * width))(
                        *[
                            layer * 10000 + plane * 1000 + token * 64 + col
                            for token in range(start, end)
                            for col in range(width)
                        ]
                    )
                )
                for layer in range(2)
                for plane, width in enumerate((512, 64))
            )
            assert len(source) == sum(page_sizes)
            offset = 0
            for ptr, size in zip(page_ptrs, page_sizes, strict=True):
                ctypes.memmove(ptr, source[offset : offset + size], size)
                offset += size

    engine.storage_manager.batched_get_external_pages = read
    result = list(
        engine.retrieve_prefill_group0_direct(
            list(range(11)),
            mask,
            kvcaches=caches,
            slot_mapping=slots,
            req_id="request",
            request_configs=None,
            shared_cpu_request_preflight_state={},
        )
    )[-1]
    assert torch.equal(result, mask)
    selected = {int(slots[token]): token for token in range(4, 11)}
    for layer, planes in enumerate(caches):
        for plane, (owner, width) in enumerate(zip(planes, (512, 64), strict=True)):
            data = (ctypes.c_uint16 * owner.numel()).from_address(owner.data_ptr())
            for slot in range(24):
                for col in range(width):
                    expected = (
                        layer * 10000 + plane * 1000 + selected[slot] * 64 + col
                        if slot in selected
                        else 0xA5A5
                    )
                    assert data[slot * width + col] == expected


def test_direct_load_capability_honors_disable_without_disabling_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lmcache_ascend.v1.npu_connector import npu_connectors

    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector._direct_page_tensor_layout = lambda *args: ([], (), 0)
    monkeypatch.setattr(npu_connectors, "_DENSE_DIRECT_LOAD_DISABLE", True)
    assert not connector.direct_page_load_supported([], 0)
    assert connector.direct_page_layout_supported([], 0)
