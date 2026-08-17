# SPDX-License-Identifier: Apache-2.0
"""Measure HCCL all-reduce with simultaneous layerwise load and save.

This is an Ascend hardware integration test.  It must be launched with at
least two HCCL ranks, for example::

    torchrun --nproc-per-node=8 -m pytest -q -s \
      tests/v1/test_hccl_layerwise_transfer_overlap.py::test_hccl_allreduce_with_layerwise_load_and_save

The pass criterion is deliberately not an overlap ratio reconstructed from
three independent event ranges.  All three streams wait on one release event,
and a control stream records one completion event only after it has waited for
all three operations.  The elapsed time between those two events is the actual
device makespan and must remain close to the 500 us all-reduce window.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable

# Third Party
import pytest
import torch
import torch.distributed as dist


_BF16_BYTES = 2
_MLA_LATENT_ELEMENTS_PER_TOKEN = 512 + 64
_DSA_INDEX_ELEMENTS_PER_TOKEN = 128
_TRANSFER_ELEMENTS_PER_TOKEN = (
    _MLA_LATENT_ELEMENTS_PER_TOKEN + _DSA_INDEX_ELEMENTS_PER_TOKEN
)
_TRANSFER_BYTES_PER_TOKEN = _TRANSFER_ELEMENTS_PER_TOKEN * _BF16_BYTES


@dataclass(frozen=True)
class _TransferShape:
    num_tokens: int
    payload_bytes: int
    theoretical_us: float


@dataclass(frozen=True)
class _HcclContext:
    local_rank: int
    rank: int
    world_size: int
    device: torch.device


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
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
    if value <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")
    return value


def _transfer_shape_for_window(
    *,
    num_tokens: int,
    bandwidth_gbps: float,
    block_size: int,
) -> _TransferShape:
    """Calculate the two-group transfer time for one prefill chunk."""
    if num_tokens <= 0 or num_tokens % block_size:
        raise ValueError("num_tokens must be a positive cache-block multiple")
    payload_bytes = num_tokens * _TRANSFER_BYTES_PER_TOKEN
    theoretical_us = payload_bytes / (bandwidth_gbps * 1e9) * 1e6
    return _TransferShape(num_tokens, payload_bytes, theoretical_us)


def test_transfer_payload_is_sub_500us_at_10gbps() -> None:
    """Lock down the payload arithmetic used by the hardware test."""
    shape = _transfer_shape_for_window(
        num_tokens=2048,
        bandwidth_gbps=10.0,
        block_size=128,
    )

    assert _TRANSFER_BYTES_PER_TOKEN == 1408
    assert shape.num_tokens == 2048
    assert shape.payload_bytes == 2_883_584
    assert shape.theoretical_us == pytest.approx(288.3584)
    assert shape.theoretical_us < 500.0


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture
def hccl_context() -> _HcclContext:
    """Bind LOCAL_RANK and own the lifetime of a torchrun HCCL group."""
    if not _npu_available():
        pytest.skip("requires an Ascend NPU")

    import torch_npu  # noqa: F401

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    owns_process_group = not dist.is_initialized()
    if owns_process_group and int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch this test with torchrun and at least two HCCL ranks")

    try:
        if owns_process_group:
            dist.init_process_group(backend="hccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        backend = str(dist.get_backend()).lower()
        if world_size < 2:
            pytest.fail(f"the initialized process group has only {world_size} rank")
        if "hccl" not in backend:
            pytest.fail(
                f"the initialized process group backend is {backend!r}, not HCCL"
            )
        yield _HcclContext(local_rank, rank, world_size, device)
    finally:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _add_benchmark_helpers_to_path() -> None:
    benchmark_dir = (
        Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
    )
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))


def _global_rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.cpu().item())


def _time_one_stream(
    *,
    stream,
    launch: Callable[[], None],
    device: torch.device,
    prepare: Callable[[], None] | None = None,
) -> float:
    """Return the slowest-rank device time for one operation group."""
    dist.barrier()
    torch.npu.synchronize()
    if prepare is not None:
        with torch.npu.stream(stream):
            prepare()
        stream.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(stream):
        start.record()
        launch()
        end.record()
    end.synchronize()
    return _global_rank_max(float(start.elapsed_time(end)), device)


def _time_common_release(
    *,
    control_stream,
    collective_stream,
    save_stream,
    load_stream,
    launch_gate: Callable[[], None],
    prepare_collective: Callable[[], None],
    launch_collective: Callable[[], None],
    launch_save: Callable[[], None],
    launch_load: Callable[[], None],
    validate_collective: Callable[[], None],
    device: torch.device,
) -> float:
    """Measure one common-release/common-fence device makespan."""
    dist.barrier()
    torch.npu.synchronize()

    release = torch.npu.Event(enable_timing=True)
    collective_done = torch.npu.Event(enable_timing=True)
    save_done = torch.npu.Event(enable_timing=True)
    load_done = torch.npu.Event(enable_timing=True)
    all_done = torch.npu.Event(enable_timing=True)

    # The gate keeps the release event in the future while Python queues all
    # three operation groups.  Without it, the first transfer could begin
    # before the HCCL call has even been submitted, so the streams would not
    # actually share a start point.
    with torch.npu.stream(control_stream):
        launch_gate()
        prepare_collective()
        release.record()

    # Queue both transfers before the collective.  This prevents a host-side
    # synchronous HCCL launch from delaying submission of the transfer work.
    with torch.npu.stream(save_stream):
        save_stream.wait_event(release)
        launch_save()
        save_done.record()
    with torch.npu.stream(load_stream):
        load_stream.wait_event(release)
        launch_load()
        load_done.record()
    with torch.npu.stream(collective_stream):
        collective_stream.wait_event(release)
        launch_collective()
        collective_done.record()

    with torch.npu.stream(control_stream):
        control_stream.wait_event(collective_done)
        control_stream.wait_event(save_done)
        control_stream.wait_event(load_done)
        all_done.record()

    all_done.synchronize()
    validate_collective()
    return _global_rank_max(float(release.elapsed_time(all_done)), device)


def _time_strict_serial(
    *,
    stream,
    launch_collective: Callable[[], None],
    launch_save: Callable[[], None],
    launch_load: Callable[[], None],
    prepare_collective: Callable[[], None],
    device: torch.device,
) -> float:
    """Measure a same-stream control that cannot claim false overlap."""
    dist.barrier()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        prepare_collective()
    stream.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(stream):
        start.record()
        launch_collective()
        launch_save()
        launch_load()
        end.record()
    end.synchronize()
    return _global_rank_max(float(start.elapsed_time(end)), device)


def _median_samples(
    samples: int,
    measure: Callable[[], float],
) -> float:
    return statistics.median(measure() for _ in range(samples))


@pytest.mark.skipif(
    not _npu_available(),
    reason="HCCL/layerwise-transfer overlap requires an Ascend NPU",
)
def test_hccl_allreduce_with_layerwise_load_and_save(
    hccl_context: _HcclContext,
) -> None:
    """A 500 us HCCL window must hide one two-group load and one save."""
    import torch_npu  # noqa: F401

    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    _add_benchmark_helpers_to_path()
    from kv_cache_fixtures import (
        check_paged_kv_cache_equal,
        generate_dsa_index_kv_cache,
        generate_mla_kv_cache,
    )
    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        build_load_benchmark_harness,
        ensure_ascend_host_memory_registered,
        run_dense_direct_fast_load_layer,
        run_dense_direct_fast_store_layer,
    )

    device = hccl_context.device
    rank = hccl_context.rank
    world_size = hccl_context.world_size

    target_us = _env_float("LMCACHE_HCCL_OVERLAP_TARGET_US", 500.0)
    bandwidth_gbps = _env_float("LMCACHE_HCCL_OVERLAP_BANDWIDTH_GBPS", 10.0)
    block_size = 128
    host_chunk_size = _env_int("LMCACHE_HCCL_OVERLAP_HOST_CHUNK", 128)
    if host_chunk_size % block_size:
        raise ValueError("LMCACHE_HCCL_OVERLAP_HOST_CHUNK must be divisible by 128")
    transfer_shape = _transfer_shape_for_window(
        num_tokens=_env_int("LMCACHE_HCCL_OVERLAP_TRANSFER_TOKENS", 2048),
        bandwidth_gbps=bandwidth_gbps,
        block_size=block_size,
    )
    num_tokens = transfer_shape.num_tokens
    if num_tokens % host_chunk_size:
        raise ValueError(
            "LMCACHE_HCCL_OVERLAP_TRANSFER_TOKENS must be divisible by "
            "LMCACHE_HCCL_OVERLAP_HOST_CHUNK"
        )
    if transfer_shape.theoretical_us >= target_us:
        raise ValueError(
            "the configured transfer does not fit the target window at the "
            f"configured bandwidth: transfer={transfer_shape.theoretical_us:.3f} us, "
            f"target={target_us:.3f} us"
        )
    num_blocks = math.ceil(num_tokens / block_size) * 2

    allreduce_rows = _env_int("LMCACHE_HCCL_OVERLAP_ALLREDUCE_ROWS", 2048)
    allreduce_hidden = _env_int("LMCACHE_HCCL_OVERLAP_ALLREDUCE_HIDDEN", 7168)
    warmups = _env_int("LMCACHE_HCCL_OVERLAP_WARMUPS", 3)
    samples = _env_int("LMCACHE_HCCL_OVERLAP_SAMPLES", 7, minimum=3)
    allreduce_tolerance_us = _env_float(
        "LMCACHE_HCCL_OVERLAP_ALLREDUCE_TOLERANCE_US", 50.0
    )
    makespan_slack_us = _env_float("LMCACHE_HCCL_OVERLAP_MAKESPAN_SLACK_US", 50.0)

    ensure_ascend_host_memory_registered()
    torch.manual_seed(20260817 + rank)
    dtype = torch.bfloat16
    common_cache = dict(
        num_blocks=num_blocks,
        device=str(device),
        num_layers=1,
        block_size=block_size,
        dtype=dtype,
    )

    def make_latent_cache():
        return generate_mla_kv_cache(
            num_kv_heads=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            **common_cache,
        )

    def make_index_cache():
        return generate_dsa_index_kv_cache(
            dsa_head_dim=128,
            **common_cache,
        )

    # Four non-overlapping registered-host slices model a true three-bank
    # transition: save N-1 and load N+1 never alias each other's host buffers.
    slab_owner = PinMemoryAllocator(64 * 1024 * 1024)
    slab_slice_bytes = 16 * 1024 * 1024
    harnesses = []
    save_latent = None
    save_index = None
    load_latent = None
    load_index = None

    def build_harness(fmt: str, src_kv_cache, slab_index: int):
        slab_begin = slab_index * slab_slice_bytes
        slab_end = slab_begin + slab_slice_bytes
        harness = build_load_benchmark_harness(
            fmt=fmt,
            src_kv_cache=src_kv_cache,
            num_tokens=num_tokens,
            chunk_size=host_chunk_size,
            num_layers=1,
            num_selected=num_tokens,
            seed=20260817,
            shared_cpu_slab=slab_owner.buffer[slab_begin:slab_end],
            populate_cpu_source=True,
        )
        harnesses.append(harness)
        return harness

    try:
        save_latent = build_harness("mla_latent", make_latent_cache(), 0)
        save_index = build_harness("dsa_index", make_index_cache(), 1)
        load_latent = build_harness("mla_latent", make_latent_cache(), 2)
        load_index = build_harness("dsa_index", make_index_cache(), 3)

        # Ensure correctness checks for the save path cannot pass merely
        # because build_load_benchmark_harness pre-populated the destinations.
        for harness in (save_latent, save_index):
            for chunk in harness.stacked_by_layer[0]:
                chunk.fill_(-17.0)

        def launch_harness(harness, *, direction: bool) -> None:
            common = dict(
                layer_state=(
                    harness.store_layer_states[0]
                    if direction
                    else harness.layer_states[0]
                ),
                slot_mapping_full=harness.slot_mapping_full,
                chunk_ptrs_npu=harness.chunk_ptrs_by_layer[0],
                chunk_offsets_npu=harness.chunk_offsets_npu,
                chunk_sizes_npu=harness.chunk_sizes_npu,
                total_tokens=harness.num_tokens,
                kv_format=harness.kv_format,
                fixed_chunk_size=harness.fixed_chunk_size,
            )
            if direction:
                run_dense_direct_fast_store_layer(**common)
            else:
                run_dense_direct_fast_load_layer(**common)

        def launch_save() -> None:
            launch_harness(save_latent, direction=True)
            launch_harness(save_index, direction=True)

        def launch_load() -> None:
            launch_harness(load_latent, direction=False)
            launch_harness(load_index, direction=False)

        allreduce_tensor = torch.zeros(
            (allreduce_rows, allreduce_hidden),
            dtype=dtype,
            device=device,
        )
        expected_collective_value = world_size * (world_size + 1) // 2

        def prepare_collective() -> None:
            allreduce_tensor.fill_(rank + 1)

        def validate_collective() -> None:
            first = float(allreduce_tensor[0, 0].item())
            last = float(allreduce_tensor[-1, -1].item())
            assert first == expected_collective_value
            assert last == expected_collective_value

        def launch_collective() -> None:
            work = dist.all_reduce(
                allreduce_tensor,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            # Make the caller's current stream depend on HCCL completion before
            # its done event is recorded.  Otherwise an internal HCCL stream
            # could make both the common fence and serial control look shorter
            # than the collective itself.
            work.wait()

        control_stream = torch.npu.Stream()
        collective_stream = torch.npu.Stream()
        save_stream = torch.npu.Stream()
        load_stream = torch.npu.Stream()
        serial_stream = torch.npu.Stream()

        gate_size = _env_int("LMCACHE_HCCL_OVERLAP_GATE_SIZE", 4096)
        gate_target_us = _env_float("LMCACHE_HCCL_OVERLAP_GATE_US", 10_000.0)
        gate_left = torch.randn((gate_size, gate_size), dtype=dtype, device=device)
        gate_right = torch.randn_like(gate_left)
        gate_output = torch.empty_like(gate_left)

        def launch_one_gate_matmul() -> None:
            torch.mm(gate_left, gate_right, out=gate_output)

        # Compile/validate every transfer kernel and warm HCCL before timing.
        for _ in range(warmups):
            with torch.npu.stream(save_stream):
                launch_save()
            with torch.npu.stream(load_stream):
                launch_load()
            with torch.npu.stream(collective_stream):
                prepare_collective()
                launch_collective()
            with torch.npu.stream(control_stream):
                launch_one_gate_matmul()
            torch.npu.synchronize()

        one_gate_ms = _median_samples(
            3,
            lambda: _time_one_stream(
                stream=control_stream,
                launch=launch_one_gate_matmul,
                device=device,
            ),
        )
        gate_iterations = max(
            1, math.ceil(gate_target_us / max(one_gate_ms * 1000.0, 1.0))
        )

        def launch_gate() -> None:
            for _ in range(gate_iterations):
                launch_one_gate_matmul()

        gate_samples = [
            _time_one_stream(
                stream=control_stream,
                launch=launch_gate,
                device=device,
            )
            for _ in range(3)
        ]
        if min(gate_samples) * 1000.0 < gate_target_us:
            gate_iterations = math.ceil(
                gate_iterations * gate_target_us / max(min(gate_samples) * 1000.0, 1.0)
            )
            gate_samples = [
                _time_one_stream(
                    stream=control_stream,
                    launch=launch_gate,
                    device=device,
                )
                for _ in range(3)
            ]
        gate_ms = statistics.median(gate_samples)
        assert min(gate_samples) * 1000.0 >= gate_target_us * 0.95, (
            "the common-start gate is too short to queue all three streams: "
            f"target={gate_target_us:.3f} us, "
            f"minimum={min(gate_samples) * 1000.0:.3f} us"
        )

        collective_samples = [
            _time_one_stream(
                stream=collective_stream,
                launch=launch_collective,
                device=device,
                prepare=prepare_collective,
            )
            for _ in range(samples)
        ]
        save_samples = [
            _time_one_stream(
                stream=save_stream,
                launch=launch_save,
                device=device,
            )
            for _ in range(samples)
        ]
        load_samples = [
            _time_one_stream(
                stream=load_stream,
                launch=launch_load,
                device=device,
            )
            for _ in range(samples)
        ]
        collective_ms = statistics.median(collective_samples)
        save_ms = statistics.median(save_samples)
        load_ms = statistics.median(load_samples)

        concurrent_samples = []
        serial_samples = []
        for sample_index in range(samples):
            measure_concurrent = lambda: _time_common_release(
                control_stream=control_stream,
                collective_stream=collective_stream,
                save_stream=save_stream,
                load_stream=load_stream,
                launch_gate=launch_gate,
                prepare_collective=prepare_collective,
                launch_collective=launch_collective,
                launch_save=launch_save,
                launch_load=launch_load,
                validate_collective=validate_collective,
                device=device,
            )
            measure_serial = lambda: _time_strict_serial(
                stream=serial_stream,
                launch_collective=launch_collective,
                launch_save=launch_save,
                launch_load=launch_load,
                prepare_collective=prepare_collective,
                device=device,
            )
            # Alternate order to reduce frequency/thermal drift bias.
            if sample_index % 2 == 0:
                concurrent_samples.append(measure_concurrent())
                serial_samples.append(measure_serial())
            else:
                serial_samples.append(measure_serial())
                concurrent_samples.append(measure_concurrent())

        concurrent_ms = statistics.median(concurrent_samples)
        serial_ms = statistics.median(serial_samples)

        # Check both transfer directions after their final timed launch.
        torch.npu.synchronize()
        for actual, expected in zip(
            save_latent.stacked_by_layer[0],
            save_latent.expected_stacked_by_layer[0],
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(
            save_index.stacked_by_layer[0],
            save_index.expected_stacked_by_layer[0],
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        check_paged_kv_cache_equal(
            load_latent.src_kv_cache,
            load_latent.dst_direct_fast,
            load_latent.slot_mapping_full,
            num_heads=1,
            kv_format=5,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            label="concurrent MLA latent load",
        )
        check_paged_kv_cache_equal(
            load_index.src_kv_cache,
            load_index.dst_direct_fast,
            load_index.slot_mapping_full,
            kv_format=6,
            dsa_head_dim=128,
            label="concurrent DSA index load",
        )

        collective_us = collective_ms * 1000.0
        save_us = save_ms * 1000.0
        load_us = load_ms * 1000.0
        concurrent_us = concurrent_ms * 1000.0
        serial_us = serial_ms * 1000.0
        collective_samples_us = [sample * 1000.0 for sample in collective_samples]
        save_samples_us = [sample * 1000.0 for sample in save_samples]
        load_samples_us = [sample * 1000.0 for sample in load_samples]
        concurrent_samples_us = [sample * 1000.0 for sample in concurrent_samples]
        serial_samples_us = [sample * 1000.0 for sample in serial_samples]
        max_makespan_us = target_us + makespan_slack_us
        dominant_max_us = max(
            max(collective_samples_us),
            max(save_samples_us),
            max(load_samples_us),
        )
        isolated_sum_us = collective_us + save_us + load_us

        def format_samples(values: list[float]) -> str:
            return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"

        if rank == 0:
            print("\nHCCL + layerwise load/save device makespan")
            print(
                f"ranks={world_size} allreduce={allreduce_rows}x"
                f"{allreduce_hidden} bf16 payload_per_direction="
                f"{transfer_shape.payload_bytes} B"
            )
            print(f"10GB/s-model transfer time: {transfer_shape.theoretical_us:.3f} us")
            print(
                f"common-start gate:         {gate_ms * 1000.0:.3f} us "
                f"({gate_iterations} matmuls; excluded from makespan)"
            )
            print(f"HCCL-only median:          {collective_us:.3f} us")
            print(f"save-only median:          {save_us:.3f} us")
            print(f"load-only median:          {load_us:.3f} us")
            print(f"HCCL samples:              {format_samples(collective_samples_us)}")
            print(f"save samples:              {format_samples(save_samples_us)}")
            print(f"load samples:              {format_samples(load_samples_us)}")
            print(f"strict-serial median:      {serial_us:.3f} us")
            print(f"serial samples:            {format_samples(serial_samples_us)}")
            print(f"prequeued makespan median: {concurrent_us:.3f} us")
            print(f"prequeued samples:         {format_samples(concurrent_samples_us)}")
            print(f"hard makespan ceiling:     {max_makespan_us:.3f} us")

        assert abs(collective_us - target_us) <= allreduce_tolerance_us, (
            "the real HCCL all-reduce is not the requested 500 us window: "
            f"target={target_us:.3f} us, measured={collective_us:.3f} us; "
            "adjust LMCACHE_HCCL_OVERLAP_ALLREDUCE_ROWS/HIDDEN if needed"
        )
        assert max(save_samples_us) < target_us, (
            f"a layerwise save exceeds {target_us:.3f} us: "
            f"samples={format_samples(save_samples_us)} us"
        )
        assert max(load_samples_us) < target_us, (
            f"a layerwise load exceeds {target_us:.3f} us: "
            f"samples={format_samples(load_samples_us)} us"
        )
        assert max(concurrent_samples_us) <= max_makespan_us, (
            "HCCL + save + load did not fit in the requested device window: "
            f"ceiling={max_makespan_us:.3f} us, "
            f"samples={format_samples(concurrent_samples_us)} us, "
            f"strict_serial_median={serial_us:.3f} us"
        )
        assert max(concurrent_samples_us) <= dominant_max_us + makespan_slack_us, (
            "the common-release makespan is not close to the slowest isolated "
            f"operation: dominant_max={dominant_max_us:.3f} us, "
            f"concurrent_max={max(concurrent_samples_us):.3f} us"
        )
        assert (
            isolated_sum_us * 0.75
            <= serial_us
            <= (isolated_sum_us * 1.35 + makespan_slack_us)
        ), (
            "the same-stream control is inconsistent with the isolated sum; "
            "HCCL may not be fenced onto the measured stream: "
            f"isolated_sum={isolated_sum_us:.3f} us, serial={serial_us:.3f} us"
        )
        assert serial_us > concurrent_us + makespan_slack_us, (
            "the same-stream control is not meaningfully slower than the "
            f"concurrent launch: serial={serial_us:.3f} us, "
            f"concurrent={concurrent_us:.3f} us"
        )
    finally:
        try:
            if _npu_available():
                torch.npu.synchronize()
        finally:
            try:
                first_close_error = None
                for harness in reversed(harnesses):
                    try:
                        harness.close()
                    except BaseException as exc:  # preserve cleanup of the rest
                        if first_close_error is None:
                            first_close_error = exc
                if first_close_error is not None:
                    raise first_close_error
            finally:
                harnesses.clear()
                save_latent = None
                save_index = None
                load_latent = None
                load_index = None
                slab_owner.close()
