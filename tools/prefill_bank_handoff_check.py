# SPDX-License-Identifier: Apache-2.0
"""Check the production P-bank event handoff with real NPU DMA, without a model.

Run inside the serving environment on one NPU. All buffers are allocated before
the measured handoffs. Payload checks happen only after the final device fence.
This isolates event/DMA ordering; it does not validate SFA or PD model outputs.
"""

from __future__ import annotations

# Standard
import argparse
import json
import os
import time


def compare_payload(actual, expected, boundary: str) -> dict:
    """Report the first exact mismatch after completion, never in the hot loop."""
    different = actual != expected
    count = int(different.sum())
    result = {"boundary": boundary, "different": count}
    if count:
        index = tuple(int(value) for value in different.nonzero()[0])
        result.update(
            index=list(index),
            actual=float(actual[index]),
            expected=float(expected[index]),
        )
    return result


def run_case(torch, ops, connector_cls, fence_cls, args, mode: str) -> dict:
    """Exercise compute -> event -> save -> future load on fixed bank streams."""
    shape = (args.layers, args.requests, args.tokens, args.width)
    source = torch.empty(shape, dtype=torch.float32, pin_memory=True)
    # Distinguish every layer, request and position, using exact FP32 values.
    source.copy_((torch.arange(source.numel()) % 1048576).reshape(shape))
    saved = torch.empty_like(source, pin_memory=True)
    saved.fill_(-1)
    bank_data = torch.empty((args.banks, *shape[1:]), device="npu", dtype=torch.float32)
    observed = torch.empty(shape, device="npu", dtype=torch.float32)
    left = torch.ones(
        (args.matmul_size, args.matmul_size), device="npu", dtype=torch.float32
    )
    right = torch.ones_like(left)
    scratch = torch.empty_like(left)
    compute = torch.npu.current_stream()
    streams = [
        compute if mode == "same_stream" else torch.npu.Stream()
        for _ in range(args.banks)
    ]
    torch.mm(left, right, out=scratch)
    torch.npu.synchronize()

    # Use the actual connector's wait and final-access methods. Only bypass
    # engine construction: this test has preallocated buffers and no LMCache.
    connector = object.__new__(connector_cls)
    connector._layerwise_prefill_dma = True
    connector._layerwise_prefill_bank_counts = {0: args.banks}
    connector._layerwise_prefill_transfer_generations = {0: 0}
    connector._layerwise_prefill_dma_streams = {
        (0, bank): stream for bank, stream in enumerate(streams)
    }
    connector.load_stream = streams[0]
    connector.fence_debug = (
        fence_cls("P", 0, {"LMCACHE_PREFILL_FENCE_DEBUG": "p_use"})
        if mode == "p_use"
        else None
    )
    _, generations, _, load_done = connector._layerwise_prefill_transfer_state()
    generation = generations.setdefault(0, 0)
    request_ids = tuple(f"request-{index}" for index in range(args.requests))
    events = []  # Retain all events until the end, independently of the registry.
    row_bytes = source[0, 0].numel() * source.element_size()

    def load(layer: int) -> None:
        bank = layer % args.banks
        # A separate completion per request, as in the serving connector.
        for request in reversed(range(args.requests)):
            with torch.npu.stream(streams[bank]):
                ops.layerwise_prefill_dma_copy(
                    [
                        (
                            bank_data[bank, request].data_ptr(),
                            source[layer, request].data_ptr(),
                            row_bytes,
                        )
                    ],
                    False,
                )
            done = torch.npu.Event()
            done.record(streams[bank])
            events.append(done)
            load_done.setdefault((0, layer), {})[request_ids[request]] = (
                generation,
                done,
            )

    started = time.perf_counter()
    try:
        for layer in range(min(args.banks, args.layers)):
            load(layer)
        for layer in range(args.layers):
            bank = layer % args.banks
            # Keep the compute queue busy before the bank read/write. Waiting
            # on the load is a device dependency, not a CPU synchronization.
            for _ in range(args.compute_repeats):
                torch.mm(left, right, out=scratch)
            connector.wait_for_layerwise_prefill_load(layer, 0, request_ids=request_ids)
            observed[layer].copy_(bank_data[bank])
            bank_data[bank].add_(0.25)
            used = torch.npu.Event()
            used.record(compute)
            events.append(used)
            connector.record_layerwise_prefill_bank_use(layer, 0, used)
            with torch.npu.stream(streams[bank]):
                ops.layerwise_prefill_dma_copy(
                    [
                        (
                            saved[layer, request].data_ptr(),
                            bank_data[bank, request].data_ptr(),
                            row_bytes,
                        )
                        for request in range(args.requests)
                    ],
                    True,
                )
            future = layer + args.banks
            if future < args.layers:
                load(future)
    finally:
        # Keep operands alive even if submission fails part way through.
        torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    reads = compare_payload(observed.cpu(), source, "load_to_compute")
    stores = compare_payload(saved, source + 0.25, "compute_to_save")
    return {
        "mode": mode,
        "seconds": round(elapsed, 3),
        "passed": reads["different"] == stores["different"] == 0,
        "checks": [reads, stores],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=79)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--matmul-size", type=int, default=2048)
    parser.add_argument("--compute-repeats", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["event", "p_use", "same_stream"],
        choices=["event", "p_use", "same_stream"],
    )
    args = parser.parse_args()
    if (
        min(
            args.banks,
            args.layers,
            args.requests,
            args.tokens,
            args.width,
            args.matmul_size,
            args.compute_repeats,
            args.repeat,
        )
        <= 0
    ):
        parser.error("sizes and repetition counts must be positive")
    if args.layers <= args.banks:
        parser.error("layers must exceed banks to test actual bank reuse")

    # Import after argument parsing so --help also works without NPU packages.
    import torch
    import torch_npu
    import lmcache_ascend.c_ops as ops
    from lmcache_ascend.v1.kv_fence_debug import KVFenceDebug
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )

    torch.npu.set_device(args.device)
    print(
        "[BANK_HANDOFF] "
        + json.dumps(
            {
                "event": "start",
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "task_queue": os.getenv("TASK_QUEUE_ENABLE"),
                "launch_blocking": os.getenv("ASCEND_LAUNCH_BLOCKING"),
                "config": vars(args),
                "scope": "raw DMA and bank handoff; no SFA or PD",
            }
        ),
        flush=True,
    )
    passed = True
    for repeat in range(args.repeat):
        for mode in args.modes:
            result = run_case(
                torch, ops, VLLMPagedMemLayerwiseNPUConnector, KVFenceDebug, args, mode
            )
            result["repeat"] = repeat
            passed = passed and result["passed"]
            print("[BANK_HANDOFF] " + json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
