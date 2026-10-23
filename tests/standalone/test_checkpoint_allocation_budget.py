# SPDX-License-Identifier: Apache-2.0
"""Check exact staging byte plans through the actual Ascend allocation helpers."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def engine_fixture(address_manager):
    path = ROOT / "lmcache_ascend/v1/cache_engine.py"
    names = {
        "reclaim_checkpoint_capacity",
        "allocate_checkpoint_fragment",
        "prepare_checkpoint_allocator",
    }
    nodes = [
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"_CHECKPOINT_RECLAIM_SCAN_ENTRIES": 4096}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    calls = []
    allocator = (
        NS(address_manager=NS(compute_aligned_size=lambda n: (n + 63) // 64 * 64))
        if address_manager
        else NS(align_bytes=64)
    )
    local = NS(
        get_memory_allocator=lambda: allocator,
        reclaim_evictable_capacity=lambda n, **kw: calls.append((n, kw)) or True,
    )
    obj = NS(
        num_layers=2,
        _num_layers_for_kv_group=lambda group: 2,
        metadata=NS(runtime_kv_group_layer_counts=None),
        save_only_first_rank=True,
        save_indexer_only_first_rank=True,
        _shared_local_cpu_backend=lambda: local,
        _ensure_layerwise_connector_layout=lambda **kw: None,
        _shared_cpu_dtype_for_kv_group=lambda g: torch.bfloat16
        if g == 0
        else torch.uint8,
        _memory_format_for_kv_group=lambda g: f"group{g}",
        gpu_connector=NS(
            get_shape=lambda n, kv_group: torch.Size([n * (3 if kv_group == 0 else 1)]),
            supports_batched_from_gpu_group=lambda g: True,
            checkpoint_plane_widths=lambda g: (3,) if g == 0 else (1,),
        ),
    )
    for name in names:
        setattr(obj, name, MethodType(ns[name], obj))
    return obj, local, calls


@pytest.mark.parametrize("address_manager", [False, True])
def test_capacity_plan_rounds_each_group_allocation_independently(address_manager):
    engine, _, calls = engine_fixture(address_manager)
    assert engine.reclaim_checkpoint_capacity(8, {0: ["latent"], 1: ["index"]})
    # Group0: 8*3*2*2=96 ->128; Group1: 8*1*1*2=16 ->64.
    assert calls == [
        (
            192,
            dict(
                min_free_bytes=0,
                min_free_ratio=0,
                num_layers=2,
                cause="checkpoint_capacity_reclaim",
                max_scan_entries=4096,
            ),
        )
    ]


def test_single_missing_group_does_not_request_capacity_for_the_other():
    engine, _, calls = engine_fixture(False)
    assert engine.reclaim_checkpoint_capacity(8, {1: ["index"]})
    assert calls[0][0] == 64
    assert not engine.reclaim_checkpoint_capacity(8, {})


def test_allocation_remains_single_attempt_without_implicit_eviction_or_wait():
    engine, local, calls = engine_fixture(False)
    attempts = []
    local.batched_allocate_layer_pages = (
        lambda *args, **kwargs: attempts.append(kwargs) or None
    )
    with pytest.raises(MemoryError, match="staging allocation refused"):
        engine.allocate_checkpoint_fragment(0, 8, ["latent"])
    assert len(attempts) == 1 and not calls
    assert attempts[0]["busy_loop"] is False and attempts[0]["eviction"] is False


@pytest.mark.parametrize("mixed", [False, True])
def test_capture_allocator_prepares_invariants_once_and_keeps_group_identity(mixed):
    engine, local, _ = engine_fixture(False)
    layouts, widths, shapes, allocations = [], [], [], []
    engine._num_layers_for_kv_group = lambda g: 2 if g == 0 else 3
    engine._ensure_layerwise_connector_layout = lambda **kw: layouts.append(
        kw["kv_group"]
    )
    engine.gpu_connector.checkpoint_plane_widths = lambda g: (
        widths.append(g),
        (3,) if g == 0 else (1,),
    )[1]
    shape = engine.gpu_connector.get_shape
    engine.gpu_connector.get_shape = lambda n, kv_group: (
        shapes.append(kv_group),
        shape(n, kv_group=kv_group),
    )[1]
    if mixed:
        engine.metadata.indexer_c8_layout = NS(mixed=True)
        engine.metadata.indexer_layer_shapes = lambda n: [
            torch.Size([n * w]) for w in (130, 256, 130)
        ]
    local.batched_allocate_layer_pages = lambda *a, **kw: (
        allocations.append((a, kw)),
        [object()],
    )[1]
    allocators = [engine.prepare_checkpoint_allocator(g, [f"group{g}"]) for g in (0, 1)]
    for n in (4,) * 63 + (1,):
        for group, allocate in enumerate(allocators):
            _, planes = allocate(n)
            args, kw = allocations[-1]
            assert args[3:5] == (2 if group == 0 else 3, f"group{group}")
            assert all(
                d == (torch.bfloat16 if group == 0 else torch.uint8) for d in args[1]
            )
            assert planes == ((3,) if group == 0 else (1,))
            assert kw == dict(
                busy_loop=False, eviction=False, valid_tokens=n, full_tokens=n
            )
    assert layouts == widths == [0, 1]
    assert shapes.count(1) == (0 if mixed else 64)


def test_mixed_reclaim_does_not_compute_discarded_generic_shape():
    engine, _, calls = engine_fixture(True)
    engine.metadata.indexer_c8_layout = NS(
        mixed=True, layer_bytes=lambda n, i: n * (130, 256)[i]
    )
    engine.gpu_connector.get_shape = lambda *a, **kw: pytest.fail(
        "unused generic shape"
    )
    assert engine.reclaim_checkpoint_capacity(3, {1: None})
    assert calls[0][0] == 1216  # ceil(3 * (130 + 256) / 64) * 64.


def test_prepared_allocator_keeps_size_checks_and_does_not_cache_across_operations():
    engine, local, reclaims = engine_fixture(False)
    calls = []
    local.batched_allocate_layer_pages = lambda *a, **kw: (
        calls.append((a, kw)),
        [object()],
    )[1]
    first = engine.prepare_checkpoint_allocator(0)
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="unavailable"):
            first(invalid)
    assert not calls
    first(3)
    assert calls[-1][1]["valid_tokens"] == 3
    # A later operation observes its own metadata; there is no engine-global plan.
    engine._shared_cpu_dtype_for_kv_group = lambda g: torch.uint8
    engine.allocate_checkpoint_fragment(0, 1)
    assert calls[-1][0][1] == [torch.uint8]
    assert not reclaims
    local.batched_allocate_layer_pages = lambda *a, **kw: None
    with pytest.raises(MemoryError, match="staging allocation refused"):
        first(2)
