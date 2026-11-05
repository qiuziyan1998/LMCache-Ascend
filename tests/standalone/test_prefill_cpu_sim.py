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
