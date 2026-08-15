# SPDX-License-Identifier: Apache-2.0
"""Compare the dense single-layer D2H operator with all AIVs and eight AIVs.

This is an Ascend hardware performance test, not a CPU unit test.  It uses the
same two-group MLA-latent and DSA-index calls as layerwise prefill and verifies
that limiting the launch to eight AIV blocks preserves the exact output.

Run with::

    pytest -q -s tests/v1/test_single_layer_core_limit.py
"""

from __future__ import annotations

# Standard
from contextlib import contextmanager
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Iterator

# Third Party
import pytest
import torch


_CORE_LIMIT_ENV = "LMCACHE_ASCEND_DENSE_DIRECT_AIV_CORE_LIMIT"


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _add_benchmark_helpers_to_path() -> None:
    benchmark_dir = (
        Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
    )
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))


@contextmanager
def _direct_aiv_core_limit(limit: int | None) -> Iterator[None]:
    previous = os.environ.get(_CORE_LIMIT_ENV)
    if limit is None:
        os.environ.pop(_CORE_LIMIT_ENV, None)
    else:
        os.environ[_CORE_LIMIT_ENV] = str(limit)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_CORE_LIMIT_ENV, None)
        else:
            os.environ[_CORE_LIMIT_ENV] = previous


def _block_fragmented_slots(total_tokens: int, block_size: int) -> list[int]:
    """Map adjacent logical blocks to every other physical cache block."""
    slots = []
    for token in range(total_tokens):
        logical_block, block_offset = divmod(token, block_size)
        physical_block = logical_block * 2
        slots.append(physical_block * block_size + block_offset)
    return slots


@pytest.mark.skipif(
    not _npu_available(),
    reason="single-layer AIV core-count comparison requires an Ascend NPU",
)
@pytest.mark.parametrize(
    "total_tokens",
    (2048, 16384),
    ids=("tokens-2048", "tokens-16384"),
)
def test_single_layer_save_all_aivs_vs_eight_aivs(total_tokens: int) -> None:
    """Measure both launch sizes and require bit-exact two-group D2H output."""
    import torch_npu  # noqa: F401

    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    _add_benchmark_helpers_to_path()
    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        KV_FORMAT_DSA_INDEX,
        KV_FORMAT_MLA_LATENT,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (
        dense_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()

    device = torch.device("npu")
    dtype = torch.bfloat16
    block_size = 128
    host_chunk_size = _env_int("LMCACHE_AIV_CORE_HOST_CHUNK", 256)
    samples = _env_int("LMCACHE_AIV_CORE_SAMPLES", 7, minimum=3)
    if total_tokens % host_chunk_size:
        raise ValueError(
            f"total_tokens={total_tokens} must be divisible by "
            f"LMCACHE_AIV_CORE_HOST_CHUNK={host_chunk_size}"
        )

    torch.manual_seed(20260815)
    slots = _block_fragmented_slots(total_tokens, block_size)
    slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)
    physical_capacity = max(slots) + 1
    num_blocks = math.ceil(physical_capacity / block_size)
    num_chunks = total_tokens // host_chunk_size
    chunk_offsets = torch.arange(
        0,
        total_tokens,
        host_chunk_size,
        dtype=torch.int32,
        device=device,
    )
    chunk_sizes = torch.full(
        (num_chunks,),
        host_chunk_size,
        dtype=torch.int32,
        device=device,
    )

    latent_dims = (512, 64)
    index_dims = (128,)
    latent_cache = tuple(
        torch.randn(
            (num_blocks, block_size, 1, hidden),
            dtype=dtype,
            device=device,
        )
        for hidden in latent_dims
    )
    index_cache = (
        torch.randn(
            (num_blocks, block_size, 1, index_dims[0]),
            dtype=dtype,
            device=device,
        ),
    )

    allocator = PinMemoryAllocator(128 * 1024 * 1024)
    memory_objects = []

    def allocate_chunks(elements_per_token: int) -> list[torch.Tensor]:
        chunks = []
        for _ in range(num_chunks):
            memory_object = allocator.allocate(
                torch.Size([host_chunk_size * elements_per_token]),
                dtype,
            )
            assert memory_object is not None and memory_object.tensor is not None
            memory_object.tensor.fill_(-17.0)
            memory_objects.append(memory_object)
            chunks.append(memory_object.tensor)
        return chunks

    try:
        destinations = {}
        for mode in ("all", "eight"):
            latent_chunks = allocate_chunks(sum(latent_dims))
            index_chunks = allocate_chunks(sum(index_dims))
            destinations[mode] = {
                "latent": latent_chunks,
                "index": index_chunks,
                "latent_ptrs": build_chunk_ptrs_npu(latent_chunks, device),
                "index_ptrs": build_chunk_ptrs_npu(index_chunks, device),
            }

        stream = torch.npu.Stream()

        def launch(mode: str, limit: int | None) -> None:
            destination = destinations[mode]
            with _direct_aiv_core_limit(limit), torch.npu.stream(stream):
                dense_mla_dsa_batched_direct_kv_transfer(
                    destination["latent"],
                    latent_cache,
                    slot_mapping,
                    chunk_offsets,
                    chunk_sizes,
                    total_tokens,
                    KV_FORMAT_MLA_LATENT,
                    False,
                    False,
                    latent_dims[0],
                    latent_dims[1],
                    0,
                    False,
                    True,
                    destination["latent_ptrs"],
                    host_chunk_size,
                )
                dense_mla_dsa_batched_direct_kv_transfer(
                    destination["index"],
                    index_cache,
                    slot_mapping,
                    chunk_offsets,
                    chunk_sizes,
                    total_tokens,
                    KV_FORMAT_DSA_INDEX,
                    False,
                    False,
                    index_dims[0],
                    0,
                    index_dims[0],
                    False,
                    True,
                    destination["index_ptrs"],
                    host_chunk_size,
                )

        # Warm both launch configurations before collecting interleaved samples.
        launch("all", None)
        launch("eight", 8)
        torch.npu.synchronize()

        device_samples: dict[str, list[float]] = {"all": [], "eight": []}
        wall_samples: dict[str, list[float]] = {"all": [], "eight": []}
        for _ in range(samples):
            for mode, limit in (("all", None), ("eight", 8)):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                wall_start_ns = time.perf_counter_ns()
                with torch.npu.stream(stream):
                    start.record()
                launch(mode, limit)
                with torch.npu.stream(stream):
                    end.record()
                end.synchronize()
                wall_samples[mode].append(
                    (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
                )
                device_samples[mode].append(start.elapsed_time(end))

        for group in ("latent", "index"):
            for actual, expected in zip(
                destinations["eight"][group],
                destinations["all"][group],
                strict=True,
            ):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        all_device_ms = statistics.median(device_samples["all"])
        eight_device_ms = statistics.median(device_samples["eight"])
        all_wall_ms = statistics.median(wall_samples["all"])
        eight_wall_ms = statistics.median(wall_samples["eight"])
        print("\nDense two-group single-layer D2H core-count comparison")
        print("tokens host_chunk mapping launch_aivs device_median_ms wall_median_ms")
        print(
            f"{total_tokens} {host_chunk_size} block128-fragmented "
            f"all {all_device_ms:.3f} {all_wall_ms:.3f}"
        )
        print(
            f"{total_tokens} {host_chunk_size} block128-fragmented "
            f"8 {eight_device_ms:.3f} {eight_wall_ms:.3f}"
        )
        print(
            f"8-AIV/all-AIV device-time ratio: {eight_device_ms / all_device_ms:.3f}x"
        )
    finally:
        if _npu_available():
            torch.npu.synchronize()
        for memory_object in memory_objects:
            memory_object.ref_count_down()
        allocator.close()
