# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the layerwise prefill DMA planner."""

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import sys

import pytest
import torch


def _load_dma_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend"
        / "v1"
        / "npu_connector"
        / "layerwise_dma.py"
    )
    spec = importlib.util.spec_from_file_location("layerwise_dma_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_block_id_plan_and_incremental_binding():
    dma = _load_dma_module()
    starts = [0, 4, 8, 12]
    ends = [4, 8, 12, 16]
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    plan = cycle.plan_block_id_ranges([10, 11, 12, 13], 4, starts, ends)

    assert int(plan.tokens.sum()) == 16
    assert plan.slot.tolist() == [40, 44, 48, 52]

    rows = dma.bind_copy_addresses(
        plan,
        [100, 200, 300, 400],
        [1000],
        [4, 4, 4, 4],
        [2],
        2,
        device_to_host=False,
        host_chunk_tokens=[4, 4, 4, 4],
    )
    assert len(rows) == len(plan)

    objects = [object() for _ in starts]
    bound = dma.bind_incremental_copy_addresses(
        plan,
        objects,
        starts,
        ends,
        [1000],
        [2],
        2,
        lambda _obj: 100,
        lambda _obj: 4,
        None,
        slot_prefix_unchanged=False,
    )
    assert len(bound.rows) == len(rows)


def test_empty_block_range_is_allocation_free():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    plan = cycle.plan_block_id_ranges([], 4, [], [])
    assert plan.chunk.numel() == 0
    assert plan.tokens.numel() == 0
    assert torch.equal(plan.slot, torch.empty(0, dtype=torch.long))


def test_reused_chunks_counts_stable_prefix_and_replaced_tail():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    owners = [object(), object(), object()]
    resolved = []

    def bind(count, previous, *, final_tokens=4, stable=True):
        starts = [4 * index for index in range(count)]
        ends = [start + 4 for start in starts]
        ends[-1] = starts[-1] + final_tokens
        return dma.bind_incremental_copy_addresses(
            cycle.plan_block_id_ranges([0, 1, 2], 4, starts, ends),
            owners[:count], starts, ends, [1000], [2], 2,
            lambda owner: resolved.append(owner) or 100,
            lambda owner: 4, previous, slot_prefix_unchanged=stable,
        )

    first = bind(2, None, final_tokens=2)
    assert first.reused_chunks == 0 and resolved == owners[:2]
    resolved.clear()
    grown_tail = bind(2, first)
    assert grown_tail.reused_chunks == 1 and resolved == owners[1:2]
    resolved.clear()
    extended = bind(3, grown_tail)
    assert extended.reused_chunks == 2 and resolved == owners[2:]
    resolved.clear()
    rebound = bind(3, extended, stable=False)
    assert rebound.reused_chunks == 0 and resolved == owners


def _reuse_debug_connector(enabled):
    dma = _load_dma_module()
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    run = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "batched_to_gpu"
    )
    planner = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prefill_dma_plans"
    )
    logs = []
    copies = []
    fmt = NS(MLA_LATENT=NS(value=1), DSA_INDEX=NS(value=2))
    ticks = iter(range(100))

    def debug_clock():
        assert enabled, "Disabled reuse debugging must not read the clock"
        return next(ticks) / 1000

    class TorchWithoutNPU:
        npu = NS(
            stream=lambda stream: nullcontext(),
            Event=lambda: NS(record=lambda stream: None),
        )

        def __getattr__(self, name):
            return getattr(torch, name)

    ns = dict(
        torch=TorchWithoutNPU(),
        time=NS(perf_counter=debug_clock),
        _DENSE_DIRECT_LOAD_DISABLE=False,
        KVCacheFormat=fmt,
        LayerPageSource=type("LayerPageSource", (), {}),
        LayerPageMemoryObj=type("LayerPageMemoryObj", (), {}),
        _layer_source_memory_objs=lambda source, layer: source,
        _layer_memory_tensor=lambda obj, layer: obj.tensor,
        bind_incremental_copy_addresses=dma.bind_incremental_copy_addresses,
        plan_block_id_ranges_incremental=dma.plan_block_id_ranges_incremental,
        prepare_source_addresses=dma.prepare_source_addresses,
        prefill_start_timing_enabled=lambda: False,
        prefill_reuse_debug_enabled=lambda rank: enabled and rank == 1,
        prefill_reuse_debug_log=lambda rank, stage, **fields: logs.append(
            (rank, stage, fields)
        ),
        logger=NS(isEnabledFor=lambda level: False),
        lmc_ops=NS(
            layerwise_prefill_dma_copy=lambda rows, direction: copies.append(
                [list(row) for row in rows]
            )
        ),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, planner, run], type_ignores=[])
            ),
            str(path), "exec",
        ),
        ns,
    )
    caches = [(torch.zeros((8, 4, 2)),) for _ in range(2)]
    state = ({}, {}, {}, {})
    obj = NS(
        _prefill_worker_id=1,
        initialize_kvcaches_ptr=lambda **kwargs: None,
        kvcaches=caches,
        _lazy_initialize_buffer_with_staging=lambda *args, **kwargs: NS(
            kv_format=fmt.MLA_LATENT, vllm_two_major=False,
            k_hidden_dims=2, v_hidden_dims=0, dsa_hidden_dims=0,
        ),
        _is_mla_dsa_format=lambda group: True,
        _set_layerwise_prefill_bank_count=lambda *args: None,
        _layerwise_prefill_transfer_generation=lambda group: 0,
        _check_layerwise_transfer_generation=lambda *args: None,
        _check_layerwise_prefill_transfer_generation=lambda *args: None,
        _check_layerwise_transfer_invariants=lambda **kwargs: None,
        _layerwise_token_major=lambda group: True,
        _expected_memory_format=lambda group: "fmt",
        _sparse_lmc_host_interleaved=lambda group: True,
        _expected_group_layers=lambda group: 2,
        prefill_dma_cycles={0: dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)},
        _prefill_dma_bound_loads={},
        _layerwise_prefill_load_owners={},
        _layerwise_prefill_load_request_events={},
        _layerwise_prefill_bank=lambda layer, group, offset: (layer + offset) % 2,
        _layerwise_prefill_dma_stream=lambda group, bank: bank,
        _layerwise_prefill_transfer_state=lambda: state,
        checkpoint_plane_widths=lambda group: [2],
        _lmc_plane_num_tokens=lambda tensor, group: 4,
        load_stream=object(),
        use_gpu=False,
    )
    obj.run = MethodType(ns["batched_to_gpu"], obj)
    owners = [
        NS(
            data_ptr=100 + index * 100, tensor=torch.empty(4),
            metadata=NS(fmt="fmt"), is_valid=lambda: True, ref_count_up=lambda: None,
        )
        for index in range(4)
    ]

    def submit(count, *, offset=0, block_ids=(0, 1, 2, 3), block_ids_by_bank=None):
        starts = [4 * index for index in range(count)]
        generator = obj.run(
            starts, [start + 4 for start in starts],
            slot_mapping=torch.empty(0, dtype=torch.long), sync=True,
            kv_group=0, req_id="request", deferred_layerwise_get=True,
            prefill_dma_block_ids_by_bank=(
                (block_ids, block_ids)
                if block_ids_by_bank is None else block_ids_by_bank
            ),
            prefill_dma_block_size=4, layerwise_prefill_bank_offset=offset,
        )
        next(generator)
        generator.send(owners[:count])
        generator.send(owners[:count])
        next(generator)
        generator.close()

    submit.connector = obj
    return submit, logs, copies


def test_reuse_debug_aggregates_actual_bind_and_phase_map_drops_once_per_group():
    submit, logs, _ = _reuse_debug_connector(True)
    submit(2)
    submit(3)
    submit(4, offset=1)
    submit(4)
    submit(4, offset=1, block_ids=(1, 0, 2, 3))

    assert len(logs) == 5
    assert [fields["x"] for _, _, fields in logs] == [
        "0/4", "4/2", "0/8", "6/2", "0/8"
    ]
    assert [fields["drop"] for _, _, fields in logs] == [
        "0/0", "0/0", "0/0", "0/0", "0/4"
    ]
    assert [fields["p"] for _, _, fields in logs] == [8, 12, 16, 16, 16]
    assert [fields["plan"] for _, _, fields in logs] == [
        "full", "reuse", "reuse", "reuse", "full"
    ]
    assert all(
        rank == 1 and stage == "dma" and fields["l"] == 2
        and fields["g"] == 0
        for rank, stage, fields in logs
    )
    assert all(fields["ms"] >= 0 for _, _, fields in logs)
    assert all(abs(fields["ms"] - 2.0) < 1e-6 for _, _, fields in logs)


def test_reuse_debug_disabled_does_not_emit_or_change_dma_rows():
    enabled_submit, _, enabled_copies = _reuse_debug_connector(True)
    disabled_submit, logs, disabled_copies = _reuse_debug_connector(False)
    enabled_submit(2)
    enabled_submit(3)
    disabled_submit(2)
    disabled_submit(3)
    # Each connector has its own NPU mock allocation; source addresses, byte
    # counts and the number of submitted rows must remain identical.
    assert [[row[1:] for row in rows] for rows in disabled_copies] == [
        [row[1:] for row in rows] for rows in enabled_copies
    ]
    assert logs == []


def _assert_plan_equal(actual, expected):
    for name in ("chunk", "slot", "chunk_token", "tokens"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_incremental_planner_append_work_is_bounded_and_storage_amortized(monkeypatch):
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=6, chunk_tokens=4, internal_offsets=[2])
    full_plan = dma.DmaCycle.plan_block_id_ranges
    planned = []

    def record_plan(self, block_ids, block_size, starts, ends, *args, **kwargs):
        planned.append((len(starts), len(block_ids)))
        return full_plan(self, block_ids, block_size, starts, ends, *args, **kwargs)

    monkeypatch.setattr(dma.DmaCycle, "plan_block_id_ranges", record_plan)
    blocks = list(range(64))
    previous = None
    copied = 0
    for count in range(1, 65):
        starts = list(range(0, count * 4, 4))
        ends = [start + 4 for start in starts]
        state = dma.plan_block_id_ranges_incremental(
            cycle, blocks, 4, starts, ends, previous
        )
        assert planned[-1] == (1, 1)
        _assert_plan_equal(
            state.plan, full_plan(cycle, blocks, 4, starts, ends)
        )
        assert state.planned_chunks == 1
        assert state.planned_blocks == 1
        assert state.reused_chunks == count - 1
        copied += state.copied_segments
        if previous is not None and state.segment_count <= previous.table.shape[1]:
            assert state.table.data_ptr() == previous.table.data_ptr()
            assert state.copied_segments == 0
        previous = state
    # Doubling copies the historical table only at capacity boundaries.
    assert copied < 2 * state.segment_count
    unchanged = dma.plan_block_id_ranges_incremental(
        cycle, blocks, 4, starts, ends, state
    )
    assert unchanged.planned_chunks == unchanged.planned_blocks == 0
    assert unchanged.table is state.table
    assert len(planned) == 64


def test_incremental_planner_partial_tail_holes_and_nonperiod_aligned_block():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=6, chunk_tokens=4, internal_offsets=[2])
    blocks = [9, 10, 15, 16, 17, 2, 3]
    previous = None
    steps = [
        ([3], [5]),
        ([3], [7]),
        ([3, 11], [7, 13]),
        ([3, 11], [7, 15]),
        ([3, 11, 15], [7, 15, 19]),
        ([3, 11, 15, 23], [7, 15, 19, 27]),
    ]
    for starts, ends in steps:
        state = dma.plan_block_id_ranges_incremental(
            cycle, blocks, 4, starts, ends, previous, slot_mapping_base=3
        )
        _assert_plan_equal(
            state.plan,
            cycle.plan_block_id_ranges(blocks, 4, starts, ends, slot_mapping_base=3),
        )
        assert state.planned_chunks == state.planned_blocks == 1
        previous = state


def test_incremental_planner_rebuilds_changed_map_but_ignores_unused_blocks():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    first = dma.plan_block_id_ranges_incremental(
        cycle, [0, 1, 2, 3], 4, [0, 4], [4, 8]
    )
    unused_changed = dma.plan_block_id_ranges_incremental(
        cycle, [0, 1, 7, 6], 4, [0, 4], [4, 8], first
    )
    assert not unused_changed.mapping_changed
    assert unused_changed.planned_chunks == 0
    extended = dma.plan_block_id_ranges_incremental(
        cycle, [0, 1, 7, 6], 4, [0, 4, 8], [4, 8, 12], unused_changed
    )
    assert extended.reused_chunks == 2 and extended.planned_blocks == 1
    changed = dma.plan_block_id_ranges_incremental(
        cycle, [1, 0, 7, 6], 4, [0, 4, 8], [4, 8, 12], extended
    )
    assert changed.mapping_changed and changed.reused_chunks == 0
    assert changed.table.data_ptr() != extended.table.data_ptr()
    _assert_plan_equal(
        changed.plan,
        cycle.plan_block_id_ranges([1, 0, 7, 6], 4, [0, 4, 8], [4, 8, 12]),
    )
    old_table = changed.table.clone()
    with pytest.raises(ValueError, match="exceeds NPU bank"):
        dma.plan_block_id_ranges_incremental(
            cycle, [1, 0, 7, 6], 4, [0, 4, 8, 12], [4, 8, 12, 16], changed,
            slot_capacity=8,
        )
    assert torch.equal(changed.table, old_table)


def test_source_metadata_and_binding_only_resolve_new_chunks_across_banks(monkeypatch):
    dma = _load_dma_module()
    full_bind = dma.bind_copy_addresses
    bound_sizes = []

    def record_bind(plan, host_ptrs, npu_ptrs, chunk_sizes, *args, **kwargs):
        bound_sizes.append(len(chunk_sizes))
        return full_bind(plan, host_ptrs, npu_ptrs, chunk_sizes, *args, **kwargs)

    monkeypatch.setattr(dma, "bind_copy_addresses", record_bind)
    cycle = dma.DmaCycle.build(bundle_tokens=6, chunk_tokens=4, internal_offsets=[2])
    owners = [NS(ptr=1000 + index * 100, tokens=4) for index in range(32)]
    blocks = list(range(32))
    plan = source = None
    bindings = {}
    resolved = []
    copied = 0
    for count in range(1, 33):
        starts = list(range(0, count * 4, 4))
        ends = [start + 4 for start in starts]
        plan = dma.plan_block_id_ranges_incremental(
            cycle, blocks, 4, starts, ends, plan
        )
        resolved.clear()
        source = dma.prepare_source_addresses(
            owners[:count], starts, ends,
            lambda obj: resolved.append(obj) or obj.ptr,
            lambda obj: obj.tokens, source,
        )
        assert resolved == owners[count - 1:count]
        assert source.rebuilt_chunks == 1
        copied += source.copied_chunks
        bank = count % 2
        prior = bindings.get(bank)
        bound = dma.bind_incremental_copy_addresses(
            plan.plan, owners[:count], starts, ends,
            [10000 + bank * 10000, 50000 + bank * 10000], [2, 3], 2,
            lambda _: pytest.fail("Cached source pointers must be used"),
            lambda _: pytest.fail("Cached physical lengths must be used"),
            prior, slot_prefix_unchanged=True, source_metadata=source,
            reuse_rows_in_place=True,
        )
        assert bound_sizes[-1] <= 2
        expected = full_bind(
            cycle.plan_block_id_ranges(blocks, 4, starts, ends),
            [obj.ptr for obj in owners[:count]],
            [10000 + bank * 10000, 50000 + bank * 10000],
            [4] * count, [2, 3], 2, device_to_host=False,
        )
        assert bound.rows == expected
        if prior is not None:
            assert bound.rows is prior.rows
            assert bound.reused_chunks == count - 2
        bindings[bank] = bound
    assert copied < 2 * len(owners)


def test_grown_and_replaced_tail_binding_preserves_earlier_rows():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=6, chunk_tokens=4, internal_offsets=[2])
    owners = [NS(ptr=1000 + index * 100, tokens=4) for index in range(3)]
    plan = source = bound = None
    for count, last_size, replace in [(2, 1, False), (2, 3, False), (2, 4, True),
                                      (3, 2, False), (3, 4, False)]:
        if replace:
            owners[count - 1] = NS(ptr=8000, tokens=4)
        starts = list(range(0, count * 4, 4))
        ends = [start + 4 for start in starts]
        ends[-1] = starts[-1] + last_size
        prior = bound
        prefix_rows = [row[:] for row in prior.rows[:1]] if prior else []
        plan = dma.plan_block_id_ranges_incremental(
            cycle, [0, 3, 4], 4, starts, ends, plan
        )
        source = dma.prepare_source_addresses(
            owners[:count], starts, ends, lambda obj: obj.ptr, lambda obj: obj.tokens,
            source,
        )
        bound = dma.bind_incremental_copy_addresses(
            plan.plan, owners[:count], starts, ends, [10000], [2], 2,
            lambda obj: obj.ptr, lambda obj: obj.tokens,
            prior, slot_prefix_unchanged=True, source_metadata=source,
            reuse_rows_in_place=True,
        )
        assert bound.rows == dma.bind_copy_addresses(
            cycle.plan_block_id_ranges([0, 3, 4], 4, starts, ends),
            [obj.ptr for obj in owners[:count]], [10000],
            [end - start for start, end in zip(starts, ends, strict=True)], [2], 2,
            device_to_host=False, host_chunk_tokens=[4] * count,
        )
        if prior:
            assert bound.rows[:1] == prefix_rows
            assert source.rebuilt_chunks == 1


def test_replaced_source_generation_rebinds_every_address_with_same_slots():
    dma = _load_dma_module()
    cycle = dma.DmaCycle.build(bundle_tokens=8, chunk_tokens=4)
    starts, ends = [0, 4, 8], [4, 8, 12]
    plan = dma.plan_block_id_ranges_incremental(
        cycle, [0, 1, 2], 4, starts, ends
    )
    old_owners = [NS(ptr=1000 + index * 100) for index in range(3)]
    source = dma.prepare_source_addresses(
        old_owners, starts, ends, lambda obj: obj.ptr, lambda obj: 4
    )
    bound = dma.bind_incremental_copy_addresses(
        plan.plan, old_owners, starts, ends, [10000], [2], 2,
        lambda obj: obj.ptr, lambda obj: 4, None,
        slot_prefix_unchanged=True, source_metadata=source, reuse_rows_in_place=True,
    )
    old_rows = [row[:] for row in bound.rows]
    new_owners = [NS(ptr=5000 + index * 100) for index in range(3)]
    same_plan = dma.plan_block_id_ranges_incremental(
        cycle, [0, 1, 2], 4, starts, ends, plan
    )
    new_source = dma.prepare_source_addresses(
        new_owners, starts, ends, lambda obj: obj.ptr, lambda obj: 4, source
    )
    new_bound = dma.bind_incremental_copy_addresses(
        same_plan.plan, new_owners, starts, ends, [10000], [2], 2,
        lambda obj: obj.ptr, lambda obj: 4, bound,
        slot_prefix_unchanged=True, source_metadata=new_source,
        reuse_rows_in_place=True,
    )
    assert same_plan.reused_chunks == 3 and same_plan.planned_chunks == 0
    assert new_source.rebuilt_chunks == 3 and new_bound.reused_chunks == 0
    assert new_source.table.data_ptr() != source.table.data_ptr()
    assert new_bound.rows == dma.bind_copy_addresses(
        same_plan.plan, [obj.ptr for obj in new_owners], [10000], [4] * 3, [2], 2,
        device_to_host=False,
    )
    assert [row[1] for row in new_bound.rows] == [5000, 5100, 5200]
    assert bound.rows == old_rows
    assert source.owners == tuple(old_owners)


def test_changed_bank_map_invalidates_only_its_bound_rows():
    submit, logs, _ = _reuse_debug_connector(True)
    submit(2)
    submit(2, offset=1)
    obj = submit.connector
    cache = obj._prefill_dma_bound_loads["request"]
    bank_one = {key: value for key, value in cache.items() if key[2] == 1}
    submit(2, block_ids_by_bank=((1, 0, 2, 3), (0, 1, 2, 3)))
    assert logs[-1][2]["drop"] == "0/2"
    assert logs[-1][2]["plan"] == "mix"
    for key, value in bank_one.items():
        assert cache[key].rows is value.rows


def test_invalid_second_bank_preserves_first_bank_invalidation_for_retry():
    submit, logs, _ = _reuse_debug_connector(True)
    submit(2)
    submit(2, offset=1)
    cache = submit.connector._prefill_dma_plan_cache["request"]
    first_bank = cache[(0, 0)]
    with pytest.raises(ValueError, match="exceeds NPU bank"):
        submit(2, block_ids_by_bank=((1, 0, 2, 3), (9, 1, 2, 3)))
    assert cache[(0, 0)] is first_bank
    submit(2, block_ids_by_bank=((1, 0, 2, 3), (0, 1, 2, 3)))
    assert logs[-1][2]["drop"] == "0/2"
    assert logs[-1][2]["plan"] == "mix"


def test_request_release_and_generation_reset_clear_incremental_metadata():
    submit, _, _ = _reuse_debug_connector(False)
    submit(2)
    obj = submit.connector
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "VLLMPagedMemLayerwiseNPUConnector")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("release_layerwise_prefill_dma_cache",
                                 "reset_layerwise_prefill_transfer_state")]
    ns = {}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *methods], type_ignores=[]
    )), str(path), "exec"), ns)
    release = MethodType(ns["release_layerwise_prefill_dma_cache"], obj)
    reset = MethodType(ns["reset_layerwise_prefill_transfer_state"], obj)
    released = []
    for owner in obj._layerwise_prefill_load_owners["request"].values():
        owner.ref_count_down = lambda: released.append(True)
    for event in obj._layerwise_prefill_load_request_events["request"]:
        event.synchronize = lambda: None
    release("request")
    assert len(released) == 2
    for name in ("_prefill_dma_bound_loads", "_prefill_dma_plan_cache",
                 "_prefill_dma_source_cache"):
        assert getattr(obj, name) == {}
    submit(2)
    state = obj._layerwise_prefill_transfer_state()
    state[0][0] = 2
    reset(0, synchronize=False)
    assert state[1][0] == 1
    for name in ("_prefill_dma_bound_loads", "_prefill_dma_plan_cache",
                 "_prefill_dma_source_cache"):
        assert getattr(obj, name) == {}
