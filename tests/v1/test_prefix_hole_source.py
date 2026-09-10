# SPDX-License-Identifier: Apache-2.0
"""Prefix-hole source planning requires no optional direct-load implementation."""

# Standard
import ctypes
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.remote_fill import ControlPage
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    KVCacheFormat,
    VLLMPagedMemLayerwiseNPUConnector,
)


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("bad", [None, "short", "negative", "bytes", "fence"])
def test_direct_prefix_repair_source_uses_exact_full_mapping(monkeypatch, group, bad):
    engine = object.__new__(AscendLMCacheEngine)
    engine.gpu_connector = SimpleNamespace()
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = 2
    widths = (512, 64) if group == 0 else (128,)
    fmt = KVCacheFormat.MLA_LATENT if group == 0 else KVCacheFormat.DSA_INDEX
    connector._lazy_initialize_buffer_with_staging = lambda *a, **k: SimpleNamespace(
        kv_format=fmt
    )
    connector.get_shape = lambda n, g: torch.Size([n * sum(widths)])
    caches = [
        tuple(torch.zeros((6, 4, 1, width), dtype=torch.float16) for width in widths)
        for _ in range(2)
    ]
    for layer, planes in enumerate(caches):
        for plane, owner in enumerate(planes):
            data = (ctypes.c_uint16 * owner.numel()).from_address(owner.data_ptr())
            for i in range(len(data)):
                data[i] = (layer * 17000 + plane * 9000 + i) % 64000
    slots = torch.tensor([8, 2, 15, 1, 20, 7])
    pages = tuple(
        ControlPage(
            canonical_key=f"group{group}-chunk{i}",
            kv_group=group,
            chunk_index=i,
            chunk_start=start,
            chunk_end=end,
            valid_tokens=end - start,
            destination_tp_rank=0,
            expected_bytes=2 * (end - start) * sum(widths) * 2,
            layer_count=2,
            layout_tag="layout",
        )
        for i, (start, end) in enumerate(((0, 4), (4, 6)))
    )
    engine.gpu_connector.plan_direct_page_sources = connector.plan_direct_page_sources
    events = (object(),)
    if bad == "short":
        slots = slots[:5]
    elif bad == "negative":
        slots[0] = -1
    elif bad == "bytes":
        import msgspec

        pages = (msgspec.structs.replace(pages[0], expected_bytes=1), pages[1])
    elif bad == "fence":
        events = ()
    if bad:
        with pytest.raises(ValueError):
            engine._remote_fill_prefix_source_plan(
                pages, ({group: caches}, {group: slots}), events
            )
        return
    plan = engine._remote_fill_prefix_source_plan(
        pages, ({group: caches}, {group: slots}), events
    )
    assert plan.producer_events is events
    for page, source in zip(pages, plan.pages, strict=True):
        actual = b"".join(
            ctypes.string_at(ptr, count)
            for ptr, count in zip(
                source.source_ptrs, source.source_lengths, strict=True
            )
        )
        expected = b"".join(
            ctypes.string_at(
                owner.data_ptr() + int(slots[token]) * width * 2, width * 2
            )
            for planes in caches
            for owner, width in zip(planes, widths, strict=True)
            for token in range(page.chunk_start, page.chunk_end)
        )
        assert actual == expected
    assert {id(owner) for owner in plan.owners} == {
        id(owner) for planes in caches for owner in planes
    }
