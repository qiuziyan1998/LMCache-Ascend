# SPDX-License-Identifier: Apache-2.0
"""NPU regression: recycle queued C8 handlers after request metadata release.

Run after rebuilding LMCache-Ascend. Each case runs in a subprocess so the old
queue-mutex deadlock fails with a timeout instead of hanging the pytest process.
"""

import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytest.importorskip("torch_npu")
from lmcache_ascend import c_ops


def _worker(group):
    gc.disable()
    tokens = 7
    keys = torch.zeros((2, 128, 1, 128), dtype=torch.int8, device="npu")
    scales = torch.zeros((2, 128, 1, 1), dtype=torch.float16, device="npu")
    state = c_ops.IndexerC8State(keys, scales)
    key_bytes = torch.arange(tokens * 128).to(torch.uint8)
    scale_bits = (torch.arange(tokens) * 97).to(torch.int16)
    packets = [torch.cat((key_bytes, scale_bits.view(torch.uint8))).npu()]
    if group:
        bf16 = torch.zeros_like(keys, dtype=torch.bfloat16)
        bf16_bits = torch.arange(tokens * 128).to(torch.int16)
        packets.insert(0, bf16_bits.view(torch.uint8).npu())
        state = c_ops.IndexerC8GroupState([c_ops.IndexerC8State(bf16), state])
    compute = torch.npu.current_stream()
    transfer = torch.npu.Stream()
    copy = (
        c_ops.indexer_c8_group_transfer_prepared
        if group
        else c_ops.indexer_c8_transfer_prepared
    )
    # The torch-npu queue has 4096 slots. Metadata/event operations additionally
    # exercise reuse; retain data/cache owners but drop request metadata promptly.
    for _ in range(8192):
        pointers = torch.tensor(
            [p.data_ptr() for p in packets], dtype=torch.int64, device="npu"
        )
        if group:
            pointers = pointers.view(2, 1)
        offsets = torch.zeros(1, dtype=torch.int32, device="npu")
        counts = torch.full((1,), tokens, dtype=torch.int32, device="npu")
        slots = torch.arange(tokens, dtype=torch.int64, device="npu")
        with torch.npu.stream(transfer):
            transfer.wait_stream(compute)
            for tensor in (pointers, offsets, counts, slots):
                tensor.record_stream(transfer)
            copy(state, pointers, offsets, counts, slots, tokens, False)
        del tensor, pointers, offsets, counts, slots
        # A separate uncaptured tensor reproduces the outer request-cleanup
        # event that can recycle a slot containing an old tensor-owning handler.
        cleanup = torch.empty(8, device="npu")
        cleanup.record_stream(transfer)
        del cleanup
    torch.npu.synchronize()
    assert torch.equal(
        keys.cpu().view(-1, 128)[:tokens].view(torch.uint8).flatten(), key_bytes
    )
    assert torch.equal(scales.cpu().view(torch.int16).view(-1)[:tokens], scale_bits)
    if group:
        assert torch.equal(
            bf16.cpu().view(torch.int16).view(-1, 128)[:tokens].flatten(), bf16_bits
        )
    assert torch.count_nonzero(keys.cpu().view(-1, 128)[tokens:]) == 0
    assert torch.count_nonzero(scales.cpu().view(-1)[tokens:]) == 0


@pytest.mark.parametrize("mode", ["single", "group"])
def test_c8_queue_recycling_does_not_deadlock(mode):
    env = dict(os.environ, TASK_QUEUE_ENABLE="1", ASCEND_LAUNCH_BLOCKING="0")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _worker(sys.argv[1] == "group")
