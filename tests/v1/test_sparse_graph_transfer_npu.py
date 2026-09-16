# SPDX-License-Identifier: Apache-2.0
"""Native ACL capture: real changing top-k, source pointers and partial tails.

Run from the repository root before loading the model:
python -m pytest --confcutdir=tests/v1 tests/v1/test_sparse_graph_transfer_npu.py
"""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("request_capacity", [1, 4, 16])
@pytest.mark.parametrize(
    "payload",
    [
        (torch.bfloat16, torch.int64, torch.int64, False),
        (torch.bfloat16, torch.int32, torch.int64, True),
        (torch.float16, torch.int64, torch.int32, True),
        (torch.float16, torch.int32, torch.int32, False),
    ],
)
def test_one_capture_replays_live_topk_and_growing_cpu_history(
    request_capacity, payload, incremental
):
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
    # Isolated pytest runs skip conftest.py's NPU bootstrap. A device
    # descriptor alone does not create the context required by aclrtMallocHost.
    torch.npu.set_device(device)
    _ = torch.zeros(1, device=device)
    torch.npu.synchronize()
    dtype = torch.bfloat16
    k_width, pe_width, chunk_size = 512, 64, 256
    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    dtype, selected_dtype, slot_dtype, count_matrix = payload
    owners = []

    def source_for(counts, offset, total=None):
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
            total_tokens=sum(counts) if total is None else total,
            chunk_token_counts=tuple(counts),
            pointer_device=device,
        )

    try:
        cases = [
            (source_for((256,), 1), [0, 17, 128, 255], 1),
            (source_for((256, 256), 1), [255, 256, 300, 511], 1),
            (source_for((256, 256, 17), 1), [0, 511, 512, 528], 1),
            (source_for((13,), 1000), [0, 3, 11, 12], 1000),
            # Logical history is shorter than the physical tail: the PE base
            # must still use 13, and selected tokens >= 10 must not be read.
            (source_for((13,), 100, total=10), [0, 9, 10, 12], 100),
        ]
        caches = tuple(
            torch.full(
                ((request_capacity * 4 + 15) // 16 + 1, 16, 1, width),
                -7,
                dtype=dtype,
                device=device,
            )
            for width in (k_width, pe_width)
        )
        slots = torch.arange(
            request_capacity * 4, dtype=slot_dtype, device=device
        ).view(request_capacity, 4)
        count_storage = torch.full(
            (request_capacity, 16), 99, dtype=torch.int32, device=device
        )
        count_storage[:, 0] = 4
        # Both [R,16] and a noncontiguous [R] column view are production layouts.
        counts = count_storage if count_matrix else count_storage[:, 0]
        scores = torch.zeros((request_capacity, 1024), device=device)
        transfer = SparseGraphTransfer(
            caches, slots, chunk_size, 1024, request_capacity=request_capacity
        )
        transfer.load(torch.topk(scores, 4).indices.to(selected_dtype), counts, slots)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            selected = torch.topk(scores, 4).indices.to(selected_dtype)
            transfer.load(selected, counts, slots)
        addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
        previous = None
        lane_cases = [cases[0]] * request_capacity
        for step in range(len(cases)):
            if incremental:
                lane_cases[step % max(1, request_capacity - 1)] = cases[step]
            else:
                lane_cases = [
                    cases[(step + lane) % len(cases)]
                    for lane in range(request_capacity)
                ]
            sources = [case[0] for case in lane_cases]
            if request_capacity > 1:
                sources[-1] = None
            lanes = None
            if incremental and previous is not None:
                changed = tuple(
                    i
                    for i, (old, new) in enumerate(zip(previous, sources, strict=True))
                    if old is not new
                )
                lanes = transfer.plan_bind_update(sources, changed)
            transfer.bind_batch(sources, 0, lanes=lanes)
            previous = tuple(sources)
            scores.fill_(-1000)
            for lane, (_, tokens, _) in enumerate(lane_cases):
                for rank, token in enumerate(tokens):
                    scores[lane, token] = 10 - rank
            for cache in caches:
                cache.fill_(-7)
            graph.replay()
            torch.npu.synchronize()
            assert selected.cpu().tolist() == [case[1] for case in lane_cases]
            for plane, cache in enumerate(caches):
                flat = cache.view(-1, cache.shape[-1])
                for lane, (_, tokens, offset) in enumerate(lane_cases):
                    for column, token in enumerate(tokens):
                        slot = lane * 4 + column
                        value = (
                            -7
                            if sources[lane] is None
                            or token >= sources[lane].total_tokens
                            else offset + token + plane * 100
                        )
                        expected = torch.full_like(flat[slot], value)
                        torch.testing.assert_close(flat[slot], expected, rtol=0, atol=0)
                assert flat[request_capacity * 4 :].eq(-7).all().item()
            assert addresses == (
                transfer.ptrs.data_ptr(),
                transfer.valid_tokens.data_ptr(),
            )
        # Direct native execution also checks negative/huge selections and both
        # negative/too-large destination slots without accessing invalid memory.
        transfer.bind_batch([cases[3][0]] * request_capacity, 0)
        invalid = torch.tensor(
            [[0, 12, 13, -1]] * request_capacity, dtype=selected_dtype, device=device
        )
        for cache in caches:
            cache.fill_(-7)
        slots.fill_(-1)
        transfer.load(invalid, counts, slots)
        slots.fill_(caches[0].shape[0] * 16)
        transfer.load(invalid, counts, slots)
        torch.npu.synchronize()
        assert all(cache.eq(-7).all().item() for cache in caches)
        slots.copy_(
            torch.arange(request_capacity * 4, dtype=slot_dtype, device=device).view_as(
                slots
            )
        )
        invalid[:, 2] = 2**40 if selected_dtype == torch.int64 else 2**30
        for active in (-1, 0, 1, 2, 4, 99):
            count_storage[:, 0] = active
            for cache in caches:
                cache.fill_(-7)
            transfer.load(invalid, counts, slots)
            torch.npu.synchronize()
            for plane, cache in enumerate(caches):
                flat = cache.view(-1, cache.shape[-1])
                for lane in range(request_capacity):
                    for column, value in enumerate((1000, 1012, -7, -7)):
                        expected = (
                            value + plane * 100 if column < min(2, active) else -7
                        )
                        torch.testing.assert_close(
                            flat[lane * 4 + column],
                            torch.full_like(flat[lane * 4 + column], expected),
                            rtol=0,
                            atol=0,
                        )
        # No-offload and zero-count steps must not read stale request pointers.
        for clear_source in (True, False):
            if clear_source:
                transfer.clear_source()
            else:
                transfer.bind_batch([cases[0][0]] * request_capacity, 0)
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
