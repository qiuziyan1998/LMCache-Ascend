# SPDX-License-Identifier: Apache-2.0
"""Execute the real RemoteFill page lifecycle on a CPU fixed-slot allocator."""

# Standard
import importlib.util
from pathlib import Path
import sys

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.page_slot_pool import PageSlotSpec
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.remote_fill import ControlPage, PageDisposition
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
import pytest
import torch


@pytest.fixture
def remote_fill(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/remote_fill.py"
    spec = importlib.util.spec_from_file_location("fixed_slot_remote_fill", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "admit,fail_group1", [(False, False), (True, False), (True, True)]
)
def test_partial_remote_fill_charges_full_slots_but_transfers_only_payload(
    remote_fill, admit, fail_group1, monkeypatch
):
    # A one-slot pool verifies that RemoteFill can take the last free slot too.
    specs = (PageSlotSpec((8, 8), 4, 1), PageSlotSpec((4, 4), 4, 1))
    allocator = TensorMemoryAllocator(
        torch.zeros(32768, dtype=torch.uint8), page_slot_specs=specs
    )
    config = LMCacheEngineConfig.from_defaults(local_cpu=True)
    PinMonitor.GetOrCreate(config)
    local = LocalCPUBackend(config, memory_allocator=allocator)
    groups = (
        remote_fill.RemoteFillGroupLayout(
            0, 4, torch.bfloat16, MemoryFormat.KV_MLA_LATENT_FMT
        ),
        remote_fill.RemoteFillGroupLayout(
            1, 2, torch.bfloat16, MemoryFormat.KV_DSA_INDEX_FMT
        ),
    )
    layout = remote_fill.RemoteFillDecoderLayout(
        "layout", "namespace", "artifact", "model", 1, 4, 2, groups
    )
    capacity_checks = []
    lifecycle = remote_fill.AscendRemoteFillPageLifecycle(
        local_backend=local,
        layout=layout,
        capacity_available=lambda size: capacity_checks.append(size) or admit,
        chunk_hash_type=int,
        chunk_hash_bytes=None,
    )
    controls = []
    for group in (0, 1):
        key = CacheEngineKey(
            "model",
            1,
            0,
            1,
            torch.bfloat16,
            {"lmcache.tag.internal.valid_tokens": 1},
            kv_group=group,
        )
        controls.append(
            ControlPage(
                canonical_key=key.to_string(),
                kv_group=group,
                chunk_index=0,
                chunk_start=0,
                chunk_end=1,
                valid_tokens=1,
                destination_tp_rank=0,
                expected_bytes=groups[group].expected_bytes(1, 2),
                layer_count=2,
                layout_tag="layout",
            )
        )
    if fail_group1:
        original_allocate = local.batched_allocate_layer_pages
        monkeypatch.setattr(
            local,
            "batched_allocate_layer_pages",
            lambda *a, **kw: (
                None
                if a[4] == MemoryFormat.KV_DSA_INDEX_FMT
                else original_allocate(*a, **kw)
            ),
        )
        with pytest.raises(MemoryError, match="hidden page allocation failed"):
            lifecycle.prepare_pages("transfer", 0, tuple(controls), True)
        assert allocator.total_allocated_size == 0 and allocator.memcheck()
        return
    prepared = lifecycle.prepare_pages("transfer", 0, tuple(controls), True)
    assert capacity_checks == [8192]
    if admit:
        assert [p.destination_length for p in prepared] == [16, 8]
        assert [p.reservation_length for p in prepared] == [16, 8]
        assert all(p.disposition is PageDisposition.ALLOCATED for p in prepared)
        assert allocator.total_allocated_size == 8192
        lifecycle.release_prepared_pages(prepared, "test abort")
    else:
        assert all(p.disposition is PageDisposition.MISSING for p in prepared)
    assert allocator.total_allocated_size == 0
    assert allocator.memcheck()
