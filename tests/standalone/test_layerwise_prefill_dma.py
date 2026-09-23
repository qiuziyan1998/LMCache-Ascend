# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the layerwise prefill DMA planner."""

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import sys

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
        prefill_start_timing_enabled=lambda: False,
        prefill_reuse_debug_enabled=lambda rank: enabled and rank == 1,
        prefill_reuse_debug_log=lambda rank, stage, **fields: logs.append(
            (rank, stage, fields)
        ),
        logger=NS(isEnabledFor=lambda level: False),
        lmc_ops=NS(
            layerwise_prefill_dma_copy=lambda rows, direction: copies.append(rows)
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
        _prefill_dma_bank_offsets={},
        _prefill_dma_slot_snapshots={},
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

    def submit(count, *, offset=0, block_ids=(0, 1, 2, 3)):
        starts = [4 * index for index in range(count)]
        generator = obj.run(
            starts, [start + 4 for start in starts],
            slot_mapping=torch.empty(0, dtype=torch.long), sync=True,
            kv_group=0, req_id="request", deferred_layerwise_get=True,
            prefill_dma_block_ids_by_bank=(block_ids, block_ids),
            prefill_dma_block_size=4, layerwise_prefill_bank_offset=offset,
        )
        next(generator)
        generator.send(owners[:count])
        generator.send(owners[:count])
        next(generator)
        generator.close()

    return submit, logs, copies


def test_reuse_debug_aggregates_actual_bind_and_phase_map_drops_once_per_group():
    submit, logs, _ = _reuse_debug_connector(True)
    submit(2)
    submit(3)
    submit(4, offset=1)
    submit(4, offset=1, block_ids=(1, 0, 2, 3))

    assert len(logs) == 4
    assert [fields["x"] for _, _, fields in logs] == ["0/4", "4/2", "0/8", "0/8"]
    assert [fields["drop"] for _, _, fields in logs] == ["0/0", "0/0", "2/0", "0/2"]
    assert [fields["p"] for _, _, fields in logs] == [8, 12, 16, 16]
    assert all(
        rank == 1 and stage == "dma" and fields["l"] == 2
        and fields["g"] == 0 and fields["plan"] == "full"
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
