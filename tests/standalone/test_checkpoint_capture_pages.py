# SPDX-License-Identifier: Apache-2.0
"""Physical capture pages versus the existing per-layer path, with CPU transfers."""

import gc
from itertools import accumulate
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from test_capture_submission import fixture
from test_local_checkpoint_restore import implementation
from test_preemption_checkpoint import Page


def packet(tokens, widths):
    # Exercise production byte-offset and view methods without importing NPU code.
    base = implementation(
        "../LMCache-NPU/lmcache/v1/memory_management.py",
        "TensorMemoryObj",
        {"get_tensor", "get_dtypes", "get_size"},
        Page,
    )
    cls = implementation(
        "../LMCache-NPU/lmcache/v1/memory_management.py",
        "LayerPageMemoryObj",
        {"layer_tensor", "layer_data_ptr", "layer_size_bytes", "layer_layout_is_valid"},
        base,
    )
    p = cls(len(widths), tokens, (1,))
    p.group_prefix_sum = tuple(accumulate((tokens * w for w in widths), initial=0))
    p.raw_data = torch.zeros(p.group_prefix_sum[-1], dtype=torch.uint8)
    p.valid = True
    p.layer_size = None
    p.metadata = p.meta = NS(
        fmt="fmt",
        shapes=[torch.Size([tokens * w]) for w in widths],
        dtypes=[torch.uint8] * len(widths),
    )
    p._base_data_ptr = p.raw_data.data_ptr()
    return p


def setup(pages, c8=False, widths=None):
    obj, _, events, _ = fixture()
    layers = pages[0].num_layers
    obj.kvcaches = [object()] * layers
    obj._expected_group_layers = lambda g: layers
    ns = obj.prepare_group_capture.__func__.__globals__
    calls, samples = [], []
    ns["lmc_ops"].get_device_ptr = lambda p, n: calls.append((p, n)) or p + 4096
    ns["prepare_sparse_direct_layer_state"] = (
        lambda sample, *a: samples.append(sample) or object()
    )
    layout = obj._lazy_initialize_buffer_with_staging()
    layout.indexer_c8 = c8
    layout.layer_token_bytes = widths
    layout.kv_device = obj.kv_device
    obj._get_or_create_sparse_destination_plan = lambda **kw: NS(
        states=[object()] * layers, c8_group=object()
    )
    return obj, events, calls, samples


def prepare(obj, pages, shared=True):
    rows = [pages] * pages[0].num_layers
    if not shared:
        rows = [list(row) for row in rows]
    sizes = [p.valid_tokens for p in pages]
    ends = list(accumulate(sizes))
    return obj.prepare_group_capture(
        rows, [0, *ends[:-1]], ends, slot_mapping=torch.arange(ends[-1]), kv_group=0
    )


@pytest.mark.parametrize("c8,widths", [(False, (8, 8)), (True, (256, 130, 256))])
def test_physical_pages_match_generic_pointers_and_plan_lifetime(c8, widths):
    pages = [packet(n, widths) for n in (4, 2)]
    if not c8:
        for page in pages:
            page.meta.dtypes = [torch.bfloat16] * len(widths)
            page.meta.shapes = [
                torch.Size([page.valid_tokens * w // 2]) for w in widths
            ]
    obj, events, calls, samples = setup(pages, c8, widths)
    plan = prepare(obj, pages)
    assert calls == [(p.raw_data.data_ptr(), p.group_prefix_sum[-1]) for p in pages]
    assert len(samples) == (0 if c8 else len(widths))
    assert all(sample.dtype == torch.bfloat16 for sample in samples)
    expected = [[p.layer_data_ptr(i) + 4096 for p in pages] for i in range(len(widths))]
    assert plan.host_rows == expected
    calls.clear()
    generic = prepare(obj, pages, shared=False)
    assert generic.host_rows == plan.host_rows
    assert torch.equal(generic.pointers, plan.pointers)
    assert torch.equal(generic.sizes, plan.sizes)
    assert len(calls) == len(pages) * len(widths)
    assert not events  # No extra fence or native launch during preparation.
    refs = [weakref.ref(p.raw_data) for p in pages]
    if not c8:
        del page  # The last loop variable must not retain a page during this check.
    enabled = gc.isenabled()
    gc.disable()
    try:
        del pages, generic
        samples.clear()
        assert all(ref() is not None for ref in refs)
        del plan
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("layers,c8", [(79, False), (22, True)])
def test_capture_metadata_scales_with_pages_not_layers(layers, c8):
    pages = [Page(layers, 4, (8,)) for _ in range(98)]
    for p in pages:
        p.metadata.fmt = "fmt"
    obj, _, calls, samples = setup(pages, c8, (8,) * layers)
    prepare(obj, pages)
    assert len(calls) == 98
    assert len(samples) == (0 if c8 else layers)


@pytest.mark.parametrize(
    "mutation", ["format", "invalid", "extent", "c8_dtype", "c8_short"]
)
def test_invalid_physical_metadata_rejected_before_upload(mutation):
    pages = [packet(4, (256, 130)), packet(2, (256, 130))]
    obj, _, _, _ = setup(pages, True, (256, 130))
    obj._slot_mapping_on_kv_device = lambda *a: pytest.fail("uploaded invalid metadata")
    if mutation == "format":
        pages[1].metadata.fmt = "wrong"
    elif mutation == "invalid":
        pages[1].refs = 0
    elif mutation == "extent":
        pages[1].raw_data = pages[1].raw_data[:1]
    elif mutation == "c8_dtype":
        pages[1].meta.dtypes[0] = torch.bfloat16
        pages[1].meta.shapes[0] = torch.Size([256])
    else:
        obj._lazy_initialize_buffer_with_staging().layer_token_bytes = (257, 130)
    with pytest.raises(ValueError):
        prepare(obj, pages)


def test_separately_registered_layers_keep_generic_fallback():
    pages = [packet(4, (8, 8))]
    obj, _, calls, _ = setup(pages)
    ns = obj.prepare_group_capture.__func__.__globals__

    def mapped(p, size):
        calls.append((p, size))
        return p + 4096 if size <= 32 else 0

    ns["lmc_ops"].get_device_ptr = mapped
    plan = prepare(obj, pages)
    assert [size for _, size in calls] == [64, 32, 32]
    assert plan.host_rows == [[pages[0].layer_data_ptr(i) + 4096] for i in range(2)]
