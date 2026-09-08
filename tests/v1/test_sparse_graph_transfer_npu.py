# SPDX-License-Identifier: Apache-2.0
"""Native ACL capture: real changing top-k, source pointers and partial tails.

Run on the server before loading the model:
pytest -q tests/v1/test_sparse_graph_transfer_npu.py
"""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


def test_one_capture_replays_live_topk_and_growing_cpu_history():
    from lmcache.v1.gpu_connector.sparse import (
        PreparedSparseSource,
        PreparedSparseSourceLayer,
    )
    from lmcache.v1.memory_management import PinMemoryAllocator
    from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

    helpers = Path(__file__).resolve().parents[2] / "benchmark/v1/kv_transfer"
    sys.path.insert(0, str(helpers))
    from load_benchmark_utils import (
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )

    ensure_ascend_host_memory_registered()
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    k_width, pe_width, chunk_size = 512, 64, 256
    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    owners = []

    def source_for(counts, offset):
        chunks = []
        for chunk_id, count in enumerate(counts):
            obj = allocator.allocate(torch.Size([count * (k_width + pe_width)]), dtype)
            assert obj is not None and obj.tensor is not None
            owners.append(obj)
            chunk = obj.tensor
            for i in range(count):
                token = chunk_id * chunk_size + i
                chunk[: count * k_width].view(count, k_width)[i].fill_(offset + token)
                chunk[count * k_width :].view(count, pe_width)[i].fill_(
                    offset + token + 100
                )
            chunks.append(chunk)
        return PreparedSparseSource(
            layers=(
                PreparedSparseSourceLayer(
                    tuple(chunks), build_chunk_ptrs_npu(chunks, device)
                ),
            ),
            total_tokens=sum(counts),
            chunk_token_counts=tuple(counts),
            pointer_device=device,
        )

    try:
        cases = [
            (source_for((256,), 1), [0, 17, 128, 255], 1),
            (source_for((256, 256), 1), [255, 256, 300, 511], 1),
            (source_for((256, 256, 17), 1), [0, 511, 512, 528], 1),
            (source_for((13,), 1000), [0, 3, 11, 12], 1000),
        ]
        caches = tuple(
            torch.full((4, 16, 1, width), -7, dtype=dtype, device=device)
            for width in (k_width, pe_width)
        )
        slots = torch.arange(4, dtype=torch.int64, device=device).view(1, 4)
        counts = torch.full((1,), 4, dtype=torch.int32, device=device)
        scores = torch.zeros(1024, device=device)
        transfer = SparseGraphTransfer(caches, slots, chunk_size, 1024)
        transfer.load(torch.topk(scores, 4).indices.view(1, 4), counts, slots)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            selected = torch.topk(scores, 4).indices.view(1, 4)
            transfer.load(selected, counts, slots)
        addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
        for source, tokens, offset in cases:
            transfer.bind(source, 0)
            scores.fill_(-1000)
            for rank, token in enumerate(tokens):
                scores[token] = 10 - rank
            for cache in caches:
                cache.fill_(-7)
            graph.replay()
            torch.npu.synchronize()
            assert selected.cpu().tolist() == [tokens]
            for plane, cache in enumerate(caches):
                flat = cache.view(-1, cache.shape[-1])
                for slot, token in enumerate(tokens):
                    expected = torch.full_like(flat[slot], offset + token + plane * 100)
                    torch.testing.assert_close(flat[slot], expected, rtol=0, atol=0)
                assert flat[4:].eq(-7).all().item()
            assert addresses == (
                transfer.ptrs.data_ptr(),
                transfer.valid_tokens.data_ptr(),
            )
        # No-offload and zero-count steps must not read stale request pointers.
        for clear_source in (True, False):
            if clear_source:
                transfer.clear_source()
            else:
                transfer.bind(cases[0][0], 0)
                counts.zero_()
            for cache in caches:
                cache.fill_(-7)
            graph.replay()
            torch.npu.synchronize()
            assert all(cache.eq(-7).all().item() for cache in caches)
    finally:
        torch.npu.synchronize()
        for obj in owners:
            obj.ref_count_down()
