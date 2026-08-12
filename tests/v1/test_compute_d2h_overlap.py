# SPDX-License-Identifier: Apache-2.0
"""Standalone NPU compute/D2H overlap measurement.

This test intentionally does not start vLLM or LMCache.  It compares the same
matrix multiplications and pinned-host D2H copy under strict serial scheduling
and independent NPU streams.
"""

# Standard
import math
import os
import statistics
import time
from typing import Callable

# Third Party
import pytest
import torch


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _median_wall_ms(fn: Callable[[], None], samples: int) -> float:
    timings = []
    for _ in range(samples):
        torch.npu.synchronize()
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
def test_independent_npu_compute_and_d2h_overlap() -> None:
    """Parallel streams must beat strict serialization by a visible margin."""
    matmul_size = _env_int("LMCACHE_OVERLAP_MATMUL_SIZE", 4096, minimum=256)
    transfer_mb = _env_int("LMCACHE_OVERLAP_TRANSFER_MB", 512, minimum=1)
    samples = _env_int("LMCACHE_OVERLAP_SAMPLES", 7, minimum=3)
    warmups = _env_int("LMCACHE_OVERLAP_WARMUPS", 2, minimum=1)
    configured_compute_iters = _env_int(
        "LMCACHE_OVERLAP_COMPUTE_ITERS", 0, minimum=0
    )
    min_overlap_ratio = _env_float(
        "LMCACHE_OVERLAP_MIN_RATIO", 0.10, minimum=0.0
    )

    compute_stream = torch.npu.Stream()
    transfer_stream = torch.npu.Stream()

    left = torch.full(
        (matmul_size, matmul_size),
        0.01,
        dtype=torch.bfloat16,
        device="npu",
    )
    right = torch.full_like(left, 0.02)
    output = torch.empty_like(left)

    transfer_elements = transfer_mb * 1024 * 1024 // 2
    transfer_source = torch.full(
        (transfer_elements,),
        1.0,
        dtype=torch.bfloat16,
        device="npu",
    )
    transfer_target = torch.empty(
        (transfer_elements,),
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    torch.npu.synchronize()

    def launch_compute(iterations: int) -> None:
        with torch.npu.stream(compute_stream):
            for _ in range(iterations):
                torch.mm(left, right, out=output)

    def launch_transfer() -> None:
        with torch.npu.stream(transfer_stream):
            transfer_target.copy_(transfer_source, non_blocking=True)

    # Warm up kernels and allocators before using an isolated probe to balance
    # the compute duration against one D2H copy.
    for _ in range(warmups):
        launch_compute(1)
        launch_transfer()
        torch.npu.synchronize()

    probe_samples = max(3, min(samples, 5))
    one_compute_ms = _median_wall_ms(lambda: launch_compute(1), probe_samples)
    one_transfer_ms = _median_wall_ms(launch_transfer, probe_samples)
    if configured_compute_iters:
        compute_iters = configured_compute_iters
    else:
        compute_iters = max(
            1,
            min(512, math.ceil(one_transfer_ms / max(one_compute_ms, 1e-6))),
        )

    def compute_only() -> None:
        launch_compute(compute_iters)

    def serial() -> None:
        launch_compute(compute_iters)
        compute_stream.synchronize()
        launch_transfer()
        transfer_stream.synchronize()

    def parallel() -> None:
        launch_compute(compute_iters)
        launch_transfer()
        compute_stream.synchronize()
        transfer_stream.synchronize()

    for _ in range(warmups):
        serial()
        parallel()

    compute_ms = _median_wall_ms(compute_only, samples)
    transfer_ms = _median_wall_ms(launch_transfer, samples)

    # Alternate order to reduce thermal/frequency drift bias between modes.
    serial_samples = []
    parallel_samples = []
    for sample in range(samples):
        ordered = (serial, parallel) if sample % 2 == 0 else (parallel, serial)
        for fn in ordered:
            torch.npu.synchronize()
            started = time.perf_counter()
            fn()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if fn is serial:
                serial_samples.append(elapsed_ms)
            else:
                parallel_samples.append(elapsed_ms)

    serial_ms = statistics.median(serial_samples)
    parallel_ms = statistics.median(parallel_samples)
    saved_ms = serial_ms - parallel_ms
    overlap_capacity_ms = min(compute_ms, transfer_ms)
    overlap_ratio = saved_ms / max(overlap_capacity_ms, 1e-6)
    speedup = serial_ms / max(parallel_ms, 1e-6)

    print("\nNPU compute/D2H overlap result")
    print(
        f"matmul={matmul_size}x{matmul_size} compute_iters={compute_iters} "
        f"transfer={transfer_mb} MiB samples={samples}"
    )
    print(f"compute-only median: {compute_ms:.3f} ms")
    print(f"D2H-only median:     {transfer_ms:.3f} ms")
    print(f"strict serial:       {serial_ms:.3f} ms")
    print(f"two-stream parallel: {parallel_ms:.3f} ms")
    print(f"saved:               {saved_ms:.3f} ms")
    print(f"speedup:             {speedup:.4f}x")
    print(f"overlap ratio:       {overlap_ratio:.2%}")

    assert float(transfer_target[0]) == pytest.approx(1.0)
    assert overlap_ratio >= min_overlap_ratio, (
        "Independent NPU compute and pinned-host D2H did not overlap enough: "
        f"required={min_overlap_ratio:.2%}, observed={overlap_ratio:.2%}, "
        f"serial={serial_ms:.3f} ms, parallel={parallel_ms:.3f} ms."
    )
