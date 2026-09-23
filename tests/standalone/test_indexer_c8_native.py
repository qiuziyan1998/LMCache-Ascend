# SPDX-License-Identifier: Apache-2.0
"""Device qualification for the byte-exact C8 transfer kernel.

Run after rebuilding the extension: pytest -q tests/standalone/test_indexer_c8_native.py
"""

import ctypes

import pytest
import torch

pytest.importorskip("torch_npu")
from lmcache_ascend import c_ops


@pytest.mark.parametrize("strided_slots", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_native_c8_packets_round_trip_with_short_scales(
    strided_slots, fixed, metadata_dtype
):
    counts = [1024, 1024, 7] if fixed else [1, 7, 127, 128, 129, 754, 1024]
    tokens = sum(counts)
    slots = torch.arange(tokens, dtype=torch.int64) * (2 if strided_slots else 1)
    blocks = (int(slots[-1]) + 128) // 128
    keys_cpu = (
        torch.arange(blocks * 128 * 128).to(torch.int8).reshape(blocks, 128, 1, 128)
    )
    # Include sign bits, subnormals, infinities and NaN payloads without doing
    # floating-point arithmetic on the scales during verification.
    scales_cpu = (
        (torch.arange(blocks * 128, dtype=torch.int32) * 97)
        .to(torch.int16)
        .view(torch.float16)
        .reshape(blocks, 128, 1, 1)
    )
    keys, scales = keys_cpu.npu(), scales_cpu.npu()
    state = c_ops.IndexerC8State(keys, scales)
    # Guard bytes surround each packet: a 2-byte tail must not write 32 bytes.
    buffers = [
        torch.full((130 * n + 64,), 173, dtype=torch.uint8, device=keys.device)
        for n in counts
    ]
    ptrs = torch.tensor(
        [b.data_ptr() + 32 for b in buffers], dtype=torch.int64, device=keys.device
    )
    lengths = torch.tensor(counts, dtype=metadata_dtype, device=keys.device)
    starts = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()[:-1]],
        dtype=metadata_dtype,
        device=keys.device,
    )
    slot_map = slots.to(device=keys.device, dtype=metadata_dtype)
    if fixed:
        # Fixed layout reuses the connector's dummy metadata, never read by the kernel.
        starts = lengths = torch.empty(1, dtype=metadata_dtype, device=keys.device)
    c_ops.indexer_c8_transfer_prepared(
        state, ptrs, starts, lengths, slot_map, 1024, True, fixed_chunks=fixed
    )
    torch.npu.synchronize()
    offset = 0
    for n, buffer in zip(counts, buffers, strict=True):
        selected = slots[offset : offset + n]
        expected = torch.cat(
            (
                keys_cpu.view(-1, 128)[selected]
                .contiguous()
                .view(torch.uint8)
                .flatten(),
                scales_cpu.view(-1, 1)[selected]
                .contiguous()
                .view(torch.uint8)
                .flatten(),
            )
        )
        actual = buffer.cpu()
        assert torch.equal(actual[32:-32], expected)
        assert torch.all(actual[:32] == 173) and torch.all(actual[-32:] == 173)
        offset += n
    keys.fill_(42)
    scales.fill_(-1)
    c_ops.indexer_c8_transfer_prepared(
        state, ptrs, starts, lengths, slot_map, 1024, False, fixed_chunks=fixed
    )
    actual_keys, actual_scales = keys.cpu().view(-1, 128), scales.cpu().view(-1, 1)
    assert torch.equal(actual_keys[slots], keys_cpu.view(-1, 128)[slots])
    assert torch.equal(
        actual_scales.view(torch.int16)[slots],
        scales_cpu.view(-1, 1).view(torch.int16)[slots],
    )
    untouched = torch.ones(blocks * 128, dtype=torch.bool)
    untouched[slots] = False
    assert torch.all(actual_keys[untouched] == 42)
    assert torch.all(actual_scales[untouched] == -1)


def test_native_c8_rejects_wrong_scale_dtype_and_metadata():
    keys = torch.empty(2, 128, 1, 128, dtype=torch.int8, device="npu")
    scales = torch.empty(2, 128, 1, 1, dtype=torch.float16, device="npu")
    with pytest.raises(RuntimeError, match="float16 scales"):
        c_ops.IndexerC8State(keys, scales.bfloat16())
    state = c_ops.IndexerC8State(keys, scales)
    meta = torch.empty(0, dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="int32/int64 vectors"):
        c_ops.indexer_c8_transfer_prepared(
            state, meta.cpu(), meta, meta, meta, 1024, False
        )


@pytest.mark.parametrize("layers", [1, 3])
def test_native_c8_group_keeps_layers_and_planes_separate(layers):
    keys = [
        torch.full((2, 128, 1, 128), i + 1, dtype=torch.int8, device="npu")
        for i in range(layers)
    ]
    scales = [
        torch.full((2, 128, 1, 1), (i + 1) / 16, dtype=torch.float16, device="npu")
        for i in range(layers)
    ]
    state = c_ops.IndexerC8GroupState(
        [c_ops.IndexerC8State(k, s) for k, s in zip(keys, scales, strict=True)]
    )
    # Adjacent layer packets have deliberately unaligned starts after layer 0.
    packet = torch.full((layers * 7 * 130,), 255, dtype=torch.uint8, device="npu")
    pointers = torch.tensor(
        [[packet.data_ptr() + i * 7 * 130] for i in range(layers)],
        dtype=torch.int64,
        device="npu",
    )
    offsets = torch.tensor([0], dtype=torch.int32, device="npu")
    counts = torch.tensor([7], dtype=torch.int32, device="npu")
    slots = torch.arange(7, dtype=torch.int64, device="npu")
    c_ops.indexer_c8_group_transfer_prepared(
        state, pointers, offsets, counts, slots, 7, True
    )
    actual = packet.cpu().view(layers, 7 * 130)
    for i in range(layers):
        assert torch.all(actual[i, : 7 * 128].view(torch.int8) == i + 1)
        assert torch.all(actual[i, 7 * 128 :].view(torch.float16) == (i + 1) / 16)


@pytest.mark.parametrize("tokens", [1, 7, 129])
def test_native_c8_registered_cpu_packet_round_trip(tokens):
    keys = torch.arange(4 * 128 * 128).to(torch.int8).reshape(4, 128, 1, 128).npu()
    scales = (
        (torch.arange(4 * 128, dtype=torch.int32) * 97)
        .to(torch.int16)
        .view(torch.float16)
        .reshape(4, 128, 1, 1)
        .npu()
    )
    state = c_ops.IndexerC8State(keys, scales)
    slots_cpu = torch.arange(tokens, dtype=torch.int64) * 2
    slots = slots_cpu.npu()
    expected_keys = keys.cpu().view(-1, 128)[slots_cpu]
    expected_scales = scales.cpu().view(-1, 1)[slots_cpu]
    payload = torch.cat(
        (
            expected_keys.contiguous().view(torch.uint8).flatten(),
            expected_scales.contiguous().view(torch.uint8).flatten(),
        )
    )
    allocation = int(c_ops.alloc_pinned_ptr(tokens * 130 + 64, 0))
    try:
        ctypes.memset(allocation, 173, tokens * 130 + 64)
        device_pointer = int(c_ops.get_device_ptr(allocation + 32, tokens * 130))
        assert device_pointer
        pointers = torch.tensor([device_pointer], dtype=torch.int64, device=keys.device)
        offsets = torch.tensor([0], dtype=torch.int32, device=keys.device)
        counts = torch.tensor([tokens], dtype=torch.int32, device=keys.device)
        c_ops.indexer_c8_transfer_prepared(
            state, pointers, offsets, counts, slots, tokens, True
        )
        torch.npu.synchronize()
        assert (
            ctypes.string_at(allocation + 32, tokens * 130) == payload.numpy().tobytes()
        )
        assert ctypes.string_at(allocation, 32) == bytes([173]) * 32
        assert ctypes.string_at(allocation + 32 + tokens * 130, 32) == bytes([173]) * 32
        keys.zero_()
        scales.zero_()
        c_ops.indexer_c8_transfer_prepared(
            state, pointers, offsets, counts, slots, tokens, False
        )
        assert torch.equal(keys.cpu().view(-1, 128)[slots_cpu], expected_keys)
        assert torch.equal(
            scales.cpu().view(-1, 1).view(torch.int16)[slots_cpu],
            expected_scales.view(torch.int16),
        )
    finally:
        # A failed fence must not release memory still visible to native DMA.
        torch.npu.synchronize()
        c_ops.free_pinned_ptr(allocation)


def test_native_c8_repeated_copy_on_two_streams_keeps_disjoint_slots():
    tokens = 257
    keys = torch.full((5, 128, 1, 128), -7, dtype=torch.int8, device="npu")
    scales = torch.full((5, 128, 1, 1), -1, dtype=torch.float16, device="npu")
    state = c_ops.IndexerC8State(keys, scales)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    owners = []
    for rank, stream in enumerate(streams):
        key_bits = ((torch.arange(tokens * 128) + rank * 37) % 256).to(torch.uint8)
        scale_bits = ((torch.arange(tokens) + rank * 991) * 97).to(torch.int16)
        packet_cpu = torch.cat((key_bits, scale_bits.view(torch.uint8)))
        packet = packet_cpu.npu()
        pointers = torch.tensor([packet.data_ptr()], dtype=torch.int64, device="npu")
        offsets = torch.zeros(1, dtype=torch.int32, device="npu")
        counts = torch.tensor([tokens], dtype=torch.int32, device="npu")
        slots = (torch.arange(tokens, dtype=torch.int64) * 2 + rank).npu()
        stream.wait_stream(torch.npu.current_stream())
        owners.append((packet, pointers, offsets, counts, slots, key_bits, scale_bits))
        with torch.npu.stream(stream):
            for _ in range(16):
                c_ops.indexer_c8_transfer_prepared(
                    state, pointers, offsets, counts, slots, tokens, False
                )
    torch.npu.synchronize()
    key_result = keys.cpu().view(-1, 128).view(torch.uint8)
    scale_result = scales.cpu().view(-1).view(torch.int16)
    for _, _, _, _, slots, expected_keys, expected_scales in owners:
        indices = slots.cpu()
        assert torch.equal(key_result[indices].flatten(), expected_keys)
        assert torch.equal(scale_result[indices], expected_scales)
    assert torch.all(keys.cpu().view(-1, 128)[tokens * 2 :] == -7)
    assert torch.all(scales.cpu().view(-1)[tokens * 2 :] == -1)
