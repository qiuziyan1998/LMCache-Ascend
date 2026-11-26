# SPDX-License-Identifier: Apache-2.0
"""Boundary-prefix GET descriptors use existing physical page metadata."""

from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

from test_checkpoint_capture_pages import packet
from test_local_checkpoint_restore import implementation


@pytest.mark.parametrize("group,layers", [(0, 79), (1, 22)])
def test_prefix_get_does_not_build_layer_views(group, layers):
    widths = (256,) * layers if group == 0 else (256, 130) * (layers // 2)
    page = packet(3, widths)
    if group == 0:
        page.meta.dtypes = [torch.bfloat16] * layers
        page.meta.shapes = [torch.Size([3 * width // 2]) for width in widths]
        page.get_dtype = lambda: torch.bfloat16
    # The allocator may provide more storage than the logical page extent.
    page.raw_data = torch.zeros(page.group_prefix_sum[-1] + 64, dtype=torch.uint8)
    page._base_data_ptr = page.raw_data.data_ptr()
    reads, calls = [], []
    layer_tensor = page.layer_tensor
    page.layer_tensor = lambda layer: (reads.append(layer), layer_tensor(layer))[1]
    cls = implementation(
        "lmcache_ascend/v1/cpu_cache_operations.py",
        "AscendCPUCacheOperations",
        {"load_checkpoint_prefix"},
        object,
        deque=deque,
        NativeExternalPageTransferUnknownError=type("UnknownDMA", (RuntimeError,), {}),
    )
    engine = cls()
    engine.token_database = NS(process_tokens=lambda **kw: iter([(4, 7, "key")]))
    engine.get_checkpoint_prefix = lambda *a, **kw: None
    engine.allocate_checkpoint_fragment = lambda *a: (page, ())
    engine._num_layers_for_kv_group = lambda g: layers
    engine.storage_manager = NS(batched_get_external_pages=lambda *a: calls.append(a))

    assert engine.load_checkpoint_prefix(tuple(range(7)), group, None) is page
    assert calls == [
        (
            ["key"],
            [[page.layer_data_ptr(i) for i in range(layers)]],
            [[3 * width for width in widths]],
            (page.raw_data,),
            "checkpoint-prefix",
        )
    ]
    assert not reads
    assert page.refs == 1
