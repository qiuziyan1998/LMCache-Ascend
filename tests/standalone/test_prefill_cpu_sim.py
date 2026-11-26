# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU interleavings for the production prefill transfer protocol."""

import importlib.util
from pathlib import Path
import sys

import pytest


def simulator():
    path = Path(__file__).resolve().parents[2] / "tools/prefill_cpu_sim.py"
    spec = importlib.util.spec_from_file_location("prefill_cpu_sim_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sim():
    return simulator()


def arguments(sim, *extra):
    return sim.parser().parse_args(
        [
            "--requests",
            "4",
            "--batch-size",
            "2",
            "--tp",
            "2",
            "--layers",
            "9",
            "--indexer-layers",
            "3",
            "--tokens",
            "257",
            "--prefill-chunk",
            "256",
            *extra,
        ]
    )


@pytest.mark.parametrize("banks", [2, 4, 8])
@pytest.mark.parametrize("schedule", ["random", "dma-lag", "compute-lag"])
@pytest.mark.parametrize("indexer_format", ["bf16", "mixed-c8"])
@pytest.mark.parametrize("cpu_objects", ["merged", "flat"])
def test_complete_production_pipeline(
    sim, banks, schedule, indexer_format, cpu_objects
):
    report = sim.Simulation(
        arguments(
            sim,
            "--banks",
            str(banks),
            "--schedule",
            schedule,
            "--indexer-format",
            indexer_format,
            "--cpu-objects",
            cpu_objects,
        ),
        0,
    ).run()
    assert report["status"] == "passed"
    assert report["operations"]["load"] > 0
    assert report["operations"]["publish"] > 4
    assert report["batches"] > 2
    assert report["checked_tokens"] > 4 * 257
    assert "adapter.save_kv_layer" in report["production_methods"]
    assert "sfa._submit_sfa_layerwise_transfer_window" in report["production_methods"]
    assert "runner._get_block_table_and_slot_mapping" in report["production_methods"]


def test_partial_chunk_replacement_and_batch_row_changes(sim):
    run = sim.Simulation(
        arguments(
            sim,
            "--requests",
            "5",
            "--batch-size",
            "3",
            "--tokens",
            "640",
            "--prefill-chunk",
            "512",
            "--schedule",
            "dma-lag",
        ),
        0,
    )
    report = run.run()
    assert report["status"] == "passed"
    assert any(request.retired_pages for request in run.requests)
    assert all(
        not page.is_valid()
        for request in run.requests
        for page in request.retired_pages
    )


@pytest.mark.parametrize(
    "fault", ["skip-load-wait", "skip-bank-handoff", "stale-publication"]
)
def test_missing_dependency_is_detected(sim, fault):
    args = arguments(sim, "--fault", fault, "--schedule", "dma-lag")
    with pytest.raises(sim.OrderingError, match="actual="):
        sim.Simulation(args, 0).run()


def test_synchronous_diagnostics_mask_old_publication_race(sim):
    args = arguments(
        sim, "--fault", "stale-publication", "--schedule", "dma-lag", "--debug-sync"
    )
    assert sim.Simulation(args, 0).run()["status"] == "passed"


def test_real_model_layer_counts_include_final_mtp_row(sim):
    args = arguments(
        sim, "--requests", "1", "--tp", "1", "--layers", "79", "--indexer-layers", "22"
    )
    run = sim.Simulation(args, 5)
    report = run.run()
    assert report["status"] == "passed"
    assert (0, 78) in run.requests[0].remote
    assert (1, 21) in run.requests[0].remote
    assert all(
        not page.is_valid()
        for pages in run.requests[0].pages.values()
        for page in pages
    )


def test_sharded_writer_uses_complete_view_and_physical_tail_stride(sim):
    from types import SimpleNamespace

    import torch

    model = sim.Simulation(arguments(sim, "--layers", "6", "--banks", "4", "--schedule", "dma-lag"), 7)
    worker = model.workers[0]
    model.device.current = worker.compute
    planes = tuple(torch.empty((5, 128, 1, w), dtype=torch.int16) for w in (8, 1))
    for plane in planes:
        model.device.memory(plane, plane)
    workspace = SimpleNamespace(planes=planes, save_events={})
    full = sim.MergedPage(128, 0, 6, model.device, model.is_c8)
    tail = sim.MergedPage(8, 0, 6, model.device, model.is_c8)
    tail.valid_tokens = 3
    # The physical G0 bank has one local column. Only the complete view covers both blocks.
    gen = worker.connector.batched_from_gpu(
        [[full, tail] for _ in range(6)], [0, 128], [128, 131], kv_group=0,
        kvcaches=worker.caches[0], slot_mapping=None, sync=True, req_id="r",
        prefill_dma_block_ids_by_bank=((1,),) * 4, prefill_dma_block_size=128,
        layerwise_prefill_bank_count=4, deferred_layerwise_put=True,
        layerwise_prefill_async_store=True, prefill_dcp_save=(workspace, [3, 1]),
    )
    next(gen)
    for layer in range(6):
        for event in workspace.save_events.values():
            worker.compute.wait_event(event)
        workspace.save_events.clear()

        def fill(layer=layer):
            for i, plane in enumerate(planes):
                plane[3].fill_(100 * layer + i)
                plane[1].fill_(100 * layer + i + 7)

        worker.compute.submit("write complete layer", fill)
        ready = sim.Event()
        ready.record(worker.compute)
        worker.connector.record_layerwise_prefill_bank_use(layer, 0, ready)
        gen.send(None)
        assert layer % 4 in workspace.save_events
    list(gen)
    model.device.drain()
    for layer in range(6):
        expected = 100 * layer
        assert (full.views[layer][:128*8] == expected).all()
        assert (full.views[layer][128*8:] == expected + 1).all()
        assert (tail.views[layer][:3*8] == expected + 7).all()
        assert (tail.views[layer][8*8:8*8+3] == expected + 8).all()
        assert (tail.views[layer][3*8:8*8].view(torch.uint8) == 0xA5).all()
        assert (tail.views[layer][8*8+3:].view(torch.uint8) == 0xA5).all()

    # Reload those exact canonical pages on the writer and the passive rank.
    for peer in model.workers:
        model.device.current = peer.compute
        caches = peer.caches[0]
        for row in caches:
            for plane in row:
                plane.fill_(-1)
        loader = peer.connector.batched_to_gpu(
            [0, 128], [128, 131], kvcaches=caches, slot_mapping=None, sync=True, kv_group=0,
            req_id="reload", deferred_layerwise_get=True, layerwise_prefill_bank_count=4,
            prefill_dma_block_ids_by_bank=((1,), (2,), (3,), (4,)), prefill_dma_block_size=128,
            prefill_dma_shard=(2, peer.rank),
        )
        next(loader)
        for layer in range(6):
            loader.send([full, tail])
            peer.connector.wait_for_layerwise_prefill_load(layer, 0, request_ids=("reload",))

            def check(layer=layer, peer=peer, row=caches[layer]):
                count = 128 if peer.rank == 0 else 3
                for i, plane in enumerate(row):
                    page = plane[layer % 4 + 1]
                    assert (page[:count] == 100 * layer + i + 7 * peer.rank).all()
                    assert (page[count:] == -1).all()

            peer.compute.submit("check historical stripe", check)
            ready = sim.Event()
            ready.record(peer.compute)
            peer.connector.record_layerwise_prefill_bank_use(layer, 0, ready)
        list(loader)
    model.device.drain()
