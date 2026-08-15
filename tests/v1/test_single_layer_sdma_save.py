# SPDX-License-Identifier: Apache-2.0
"""Feasibility test for an SDMA-only layerwise MLA/DSA save path.

This file is intentionally test-only.  It does not change the LMCache
connector or the production custom operator.  The experimental path calls
``aclrtMemcpyAsync`` directly for contiguous runs in ``slot_mapping`` so it
does not launch an AICore/AIV copy kernel.

Run on an Ascend host with::

    python -m pytest -q -s tests/v1/test_single_layer_sdma_save.py

The NPU test compares the experimental result with the current
``single_layer_paged_kv_copy`` implementation for both GLM two-group layouts,
then reports current-AIV versus SDMA save latency and checks whether a 2048
token save is hidden by approximately 1 ms and 2 ms of independent compute.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable, Sequence

# Third Party
import pytest
import torch


_ACL_MEMCPY_DEVICE_TO_HOST = 2


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


@dataclass(frozen=True)
class _DenseRun:
    """One run contiguous in both logical destination and physical slots."""

    chunk_id: int
    local_token: int
    slot: int
    length: int


def _coalesce_dense_runs(
    slot_mapping: Sequence[int],
    *,
    chunk_size: int,
    total_tokens: int,
) -> list[_DenseRun]:
    """Split an arbitrary valid slot mapping into exact contiguous DMA runs.

    A run is split at an LMCache chunk boundary, at ``-1`` padding, or when
    adjacent logical tokens do not map to adjacent physical slots.  Thus the
    optimization preserves the current operator's token-by-token semantics;
    a fully fragmented mapping simply degenerates to one DMA per token/plane.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if len(slot_mapping) > total_tokens:
        raise ValueError("slot_mapping cannot be longer than total_tokens")

    runs: list[_DenseRun] = []
    for logical_token, raw_slot in enumerate(slot_mapping):
        slot = int(raw_slot)
        if slot == -1:
            continue
        if slot < 0:
            raise ValueError(f"invalid slot_mapping[{logical_token}]={slot}")

        chunk_id, local_token = divmod(logical_token, chunk_size)
        if runs:
            previous = runs[-1]
            if (
                previous.chunk_id == chunk_id
                and previous.local_token + previous.length == local_token
                and previous.slot + previous.length == slot
            ):
                runs[-1] = _DenseRun(
                    chunk_id=previous.chunk_id,
                    local_token=previous.local_token,
                    slot=previous.slot,
                    length=previous.length + 1,
                )
                continue
        runs.append(
            _DenseRun(
                chunk_id=chunk_id,
                local_token=local_token,
                slot=slot,
                length=1,
            )
        )
    return runs


def _build_block_fragmented_mapping(
    total_tokens: int,
    *,
    block_size: int,
) -> tuple[list[int], int]:
    """Place adjacent logical blocks in nonadjacent physical cache blocks."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_logical_blocks = math.ceil(total_tokens / block_size)
    physical_block_ids = [
        logical_block * 2 for logical_block in range(num_logical_blocks)
    ]
    mapping = [
        physical_block_ids[logical_token // block_size] * block_size
        + logical_token % block_size
        for logical_token in range(total_tokens)
    ]
    capacity = (physical_block_ids[-1] + 1) * block_size
    return mapping, capacity


def test_coalesce_dense_runs_preserves_arbitrary_slot_mapping() -> None:
    mapping = [5, 6, 7, -1, 11, 12, 20, 21, 22, 3, 4, 9]

    runs = _coalesce_dense_runs(mapping, chunk_size=4, total_tokens=len(mapping))

    reconstructed: dict[int, int] = {}
    for run in runs:
        for offset in range(run.length):
            logical = run.chunk_id * 4 + run.local_token + offset
            reconstructed[logical] = run.slot + offset
    expected = {index: slot for index, slot in enumerate(mapping) if slot != -1}
    assert reconstructed == expected
    assert runs == [
        _DenseRun(0, 0, 5, 3),
        _DenseRun(1, 0, 11, 2),
        _DenseRun(1, 2, 20, 2),
        _DenseRun(2, 0, 22, 1),
        _DenseRun(2, 1, 3, 2),
        _DenseRun(2, 3, 9, 1),
    ]


class _AclRuntime:
    """Minimal test-only binding for asynchronous D2H runtime copies."""

    def __init__(self) -> None:
        try:
            library = ctypes.CDLL("libascendcl.so")
        except OSError as exc:
            raise RuntimeError("could not load libascendcl.so") from exc
        memcpy_async = library.aclrtMemcpyAsync
        memcpy_async.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        memcpy_async.restype = ctypes.c_int
        self._library = library
        self._memcpy_async = memcpy_async

    def d2h_async(
        self,
        *,
        dst: int,
        src: int,
        size: int,
        stream: int,
    ) -> None:
        result = self._memcpy_async(
            ctypes.c_void_p(dst),
            ctypes.c_size_t(size),
            ctypes.c_void_p(src),
            ctypes.c_size_t(size),
            _ACL_MEMCPY_DEVICE_TO_HOST,
            ctypes.c_void_p(stream),
        )
        if result != 0:
            raise RuntimeError(
                f"aclrtMemcpyAsync D2H failed with aclError={result}, size={size}"
            )


def _launch_sdma_group_save(
    runtime: _AclRuntime,
    *,
    host_chunks: Sequence[torch.Tensor],
    paged_planes: Sequence[torch.Tensor],
    runs: Sequence[_DenseRun],
    total_tokens: int,
    chunk_size: int,
    stream: object,
) -> int:
    """Gather paged planes into planar LMCache chunks using only runtime DMA."""
    if not paged_planes:
        raise ValueError("paged_planes must not be empty")
    element_size = paged_planes[0].element_size()
    plane_dims = [int(plane.shape[-2] * plane.shape[-1]) for plane in paged_planes]
    plane_elems = sum(plane_dims)
    if any(not plane.is_contiguous() for plane in paged_planes):
        raise ValueError("the real GLM paged cache planes must be contiguous")
    if any(
        plane.dtype != paged_planes[0].dtype or plane.element_size() != element_size
        for plane in paged_planes
    ):
        raise ValueError("all paged planes must have the same dtype")

    stream_handle = int(stream.npu_stream)
    copy_count = 0
    for run in runs:
        if run.chunk_id >= len(host_chunks):
            raise ValueError(f"run references missing chunk {run.chunk_id}")
        chunk_tokens = min(
            chunk_size,
            total_tokens - run.chunk_id * chunk_size,
        )
        chunk = host_chunks[run.chunk_id]
        if chunk.device.type != "cpu" or not chunk.is_contiguous():
            raise ValueError(
                "LMCache destination chunks must be contiguous CPU tensors"
            )
        if chunk.numel() != chunk_tokens * plane_elems:
            raise ValueError(
                f"chunk {run.chunk_id} has {chunk.numel()} elements, expected "
                f"{chunk_tokens * plane_elems}"
            )
        if run.local_token + run.length > chunk_tokens:
            raise ValueError("DMA run crosses an LMCache chunk boundary")

        plane_base = 0
        for plane, hidden_dims in zip(paged_planes, plane_dims, strict=True):
            max_slots = plane.numel() // hidden_dims
            if run.slot + run.length > max_slots:
                raise ValueError("slot_mapping exceeds the paged cache capacity")
            copy_elements = run.length * hidden_dims
            dst_element = plane_base + run.local_token * hidden_dims
            src_element = run.slot * hidden_dims
            runtime.d2h_async(
                dst=int(chunk.data_ptr()) + dst_element * element_size,
                src=int(plane.data_ptr()) + src_element * element_size,
                size=copy_elements * element_size,
                stream=stream_handle,
            )
            copy_count += 1
            plane_base += chunk_tokens * hidden_dims
    return copy_count


def _launch_sdma_two_group_save(
    runtime: _AclRuntime,
    *,
    latent_chunks: Sequence[torch.Tensor],
    index_chunks: Sequence[torch.Tensor],
    latent_cache: Sequence[torch.Tensor],
    index_cache: Sequence[torch.Tensor],
    runs: Sequence[_DenseRun],
    total_tokens: int,
    chunk_size: int,
    stream: object,
) -> int:
    latent_copies = _launch_sdma_group_save(
        runtime,
        host_chunks=latent_chunks,
        paged_planes=latent_cache,
        runs=runs,
        total_tokens=total_tokens,
        chunk_size=chunk_size,
        stream=stream,
    )
    index_copies = _launch_sdma_group_save(
        runtime,
        host_chunks=index_chunks,
        paged_planes=index_cache,
        runs=runs,
        total_tokens=total_tokens,
        chunk_size=chunk_size,
        stream=stream,
    )
    return latent_copies + index_copies


def test_sdma_group_save_uses_exact_planar_offsets_for_every_run() -> None:
    class RecordingRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def d2h_async(
            self,
            *,
            dst: int,
            src: int,
            size: int,
            stream: int,
        ) -> None:
            self.calls.append((dst, src, size, stream))

    class FakeStream:
        npu_stream = 73

    mapping = [2, 3, -1, 7, 8, 20]
    runs = _coalesce_dense_runs(
        mapping,
        chunk_size=4,
        total_tokens=len(mapping),
    )
    first_plane = torch.empty((32, 1, 4), dtype=torch.bfloat16)
    second_plane = torch.empty((32, 1, 2), dtype=torch.bfloat16)
    chunks = [
        torch.empty(4 * 6, dtype=torch.bfloat16),
        torch.empty(2 * 6, dtype=torch.bfloat16),
    ]
    runtime = RecordingRuntime()

    copy_count = _launch_sdma_group_save(
        runtime,  # type: ignore[arg-type]
        host_chunks=chunks,
        paged_planes=(first_plane, second_plane),
        runs=runs,
        total_tokens=len(mapping),
        chunk_size=4,
        stream=FakeStream(),
    )

    element_size = 2
    expected = []
    for run in runs:
        chunk_tokens = 4 if run.chunk_id == 0 else 2
        plane_base = 0
        for plane, hidden_dims in ((first_plane, 4), (second_plane, 2)):
            expected.append(
                (
                    chunks[run.chunk_id].data_ptr()
                    + (plane_base + run.local_token * hidden_dims) * element_size,
                    plane.data_ptr() + run.slot * hidden_dims * element_size,
                    run.length * hidden_dims * element_size,
                    73,
                )
            )
            plane_base += chunk_tokens * hidden_dims
    assert copy_count == len(expected)
    assert runtime.calls == expected


def _allocate_chunks(
    allocator: object,
    *,
    total_tokens: int,
    chunk_size: int,
    plane_dims: Sequence[int],
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[object]]:
    chunks = []
    memory_objects = []
    for start in range(0, total_tokens, chunk_size):
        tokens = min(chunk_size, total_tokens - start)
        memory_object = allocator.allocate(
            torch.Size([tokens * sum(plane_dims)]),
            dtype,
        )
        if memory_object is None or memory_object.tensor is None:
            raise RuntimeError("PinMemoryAllocator could not allocate a test chunk")
        chunks.append(memory_object.tensor)
        memory_objects.append(memory_object)
    return chunks, memory_objects


def _median_wall_ms(fn: Callable[[], None], samples: int) -> float:
    values = []
    for _ in range(samples):
        torch.npu.synchronize()
        started = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return statistics.median(values)


def _launch_compute_unit(
    *,
    stream: object,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    iterations: int,
) -> None:
    with torch.npu.stream(stream):
        for _ in range(iterations):
            # MM occupies Cube/AIC and add occupies vector/AIV.  The SDMA
            # experiment therefore cannot pass merely by using the idle half
            # of the compute cores.
            torch.mm(left, right, out=output)
            torch.add(output, 0.001, out=output)


def _event_duration_ms(fn: Callable[[], None], stream: object) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(stream):
        start.record()
        fn()
        end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _calibrate_compute_iterations(
    *,
    target_ms: float,
    stream: object,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
) -> tuple[int, float]:
    def measure(iterations: int) -> float:
        return statistics.median(
            _event_duration_ms(
                lambda: _launch_compute_unit(
                    stream=stream,
                    left=left,
                    right=right,
                    output=output,
                    iterations=iterations,
                ),
                stream,
            )
            for _ in range(3)
        )

    one_iteration_ms = max(measure(1), 0.001)
    iterations = max(1, min(512, round(target_ms / one_iteration_ms)))
    for _ in range(2):
        observed_ms = max(measure(iterations), 0.001)
        adjusted = max(1, min(512, round(iterations * target_ms / observed_ms)))
        if adjusted == iterations:
            break
        iterations = adjusted
    return iterations, measure(iterations)


@dataclass(frozen=True)
class _OverlapSample:
    compute_ms: float
    dma_ms: float
    makespan_ms: float
    overlap_ms: float
    dma_tail_ms: float


def _measure_overlap_sample(
    *,
    launch_compute: Callable[[], None],
    launch_dma: Callable[[], None],
    compute_stream: object,
    dma_stream: object,
) -> _OverlapSample:
    torch.npu.synchronize()
    origin = torch.npu.Event(enable_timing=True)
    compute_start = torch.npu.Event(enable_timing=True)
    compute_end = torch.npu.Event(enable_timing=True)
    dma_start = torch.npu.Event(enable_timing=True)
    dma_end = torch.npu.Event(enable_timing=True)

    # Record a completed common origin so timestamps from the two independent
    # streams share the same reference.  Production submits the previous
    # layer's save before launching the next layer's compute, so preserve that
    # order here.  Submitting compute first is misleading for a short compute
    # window: Python may still be enqueueing its operators after the device has
    # already completed them, leaving no opportunity for the later DMA launch
    # to overlap.
    origin.record()
    origin.synchronize()
    with torch.npu.stream(dma_stream):
        dma_start.record()
        launch_dma()
        dma_end.record()
    with torch.npu.stream(compute_stream):
        compute_start.record()
        launch_compute()
        compute_end.record()
    compute_end.synchronize()
    dma_end.synchronize()

    compute_start_ms = float(origin.elapsed_time(compute_start))
    compute_end_ms = float(origin.elapsed_time(compute_end))
    dma_start_ms = float(origin.elapsed_time(dma_start))
    dma_end_ms = float(origin.elapsed_time(dma_end))
    compute_ms = compute_end_ms - compute_start_ms
    dma_ms = dma_end_ms - dma_start_ms
    overlap_ms = max(
        0.0,
        min(compute_end_ms, dma_end_ms) - max(compute_start_ms, dma_start_ms),
    )
    makespan_ms = max(compute_end_ms, dma_end_ms) - min(
        compute_start_ms,
        dma_start_ms,
    )
    return _OverlapSample(
        compute_ms=compute_ms,
        dma_ms=dma_ms,
        makespan_ms=makespan_ms,
        overlap_ms=overlap_ms,
        dma_tail_ms=dma_end_ms - compute_end_ms,
    )


def _profile_sdma_only(
    *,
    launch_dma: Callable[[], None],
    dma_stream: object,
) -> Path:
    import torch_npu

    profile_root = Path(
        os.environ.get(
            "LMCACHE_SDMA_PROFILE_DIR",
            "/tmp/lmcache-single-layer-sdma-profile",
        )
    )
    profile_root.mkdir(parents=True, exist_ok=True)
    trace_path = profile_root / f"trace_view_{time.time_ns()}.json"
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    with profiler:
        with torch.profiler.record_function("lmcache_sdma_save"):
            launch_dma()
        dma_stream.synchronize()
    profiler.export_chrome_trace(str(trace_path))

    raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = (
        raw_trace.get("traceEvents", []) if isinstance(raw_trace, dict) else raw_trace
    )
    hardware_pids = {
        event.get("pid")
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    assert hardware_pids, (
        f"the profiler trace has no Ascend Hardware process: {trace_path}"
    )
    hardware_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "X"
        and event.get("pid") in hardware_pids
        and float(event.get("dur", 0.0)) > 0.0
    ]

    def is_dma_event(event: dict[str, object]) -> bool:
        searchable = json.dumps(
            {"name": event.get("name"), "args": event.get("args")},
            ensure_ascii=True,
        ).lower()
        return "memcpy" in searchable or "sdma" in searchable

    def is_compute_core_event(event: dict[str, object]) -> bool:
        searchable = json.dumps(
            {"name": event.get("name"), "args": event.get("args")},
            ensure_ascii=True,
        ).lower()
        compute_markers = (
            "ai_core",
            "aicore",
            "ai core",
            "ai_vector_core",
            "aivector",
            "ai vector core",
            "vector_core",
            "single_layer_paged_kv_copy",
        )
        return any(marker in searchable for marker in compute_markers)

    dma_events = [event for event in hardware_events if is_dma_event(event)]
    compute_events = [
        event for event in hardware_events if is_compute_core_event(event)
    ]
    assert dma_events, (
        "the SDMA-only trace contains no Memcpy/SDMA hardware task; inspect "
        f"{trace_path}"
    )
    assert not compute_events, (
        "the SDMA-only trace unexpectedly contains AICore/AIV tasks: "
        f"{sorted({str(event.get('name')) for event in compute_events})}; "
        f"inspect {trace_path}"
    )
    assert not any(
        "single_layer_paged_kv_copy" in str(event.get("name", ""))
        for event in hardware_events
    )
    print(
        f"SDMA-only profiler: {len(dma_events)} DMA tasks, "
        f"no AICore/AIV task; trace={trace_path}"
    )
    return trace_path


@pytest.mark.skipif(
    not _npu_available(),
    reason="exact SDMA layer-save experiment requires an Ascend NPU",
)
def test_exact_two_group_sdma_save_and_compute_overlap() -> None:
    """Compare exact semantics, then test 1/2 ms compute-window coverage."""
    import torch_npu  # noqa: F401

    from lmcache.v1.memory_management import PinMemoryAllocator

    benchmark_dir = (
        Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
    )
    import sys

    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
    from load_benchmark_utils import (  # noqa: PLC0415
        KV_FORMAT_DSA_INDEX,
        KV_FORMAT_MLA_LATENT,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (  # noqa: PLC0415
        dense_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()
    torch.manual_seed(20260815)
    device = torch.device("npu")
    dtype = torch.bfloat16
    block_size = 128
    latent_dims = (512, 64)
    index_dims = (128,)
    runtime = _AclRuntime()
    memory_objects: list[object] = []

    # 64 MiB is deliberately larger than both correctness and performance
    # destinations, including allocator alignment.
    allocator = PinMemoryAllocator(64 * 1024 * 1024)
    try:
        # First compare against the current AIV operator using a deliberately
        # fragmented mapping, padding, repeated slots, and chunk crossings.
        correctness_tokens = 37
        correctness_chunk = 16
        capacity = 64
        correctness_mapping = [
            5,
            6,
            7,
            -1,
            31,
            4,
            4,
            20,
            21,
            22,
            9,
            3,
            40,
            41,
            42,
            43,
            44,
            2,
            -1,
            19,
            18,
            17,
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            8,
            1,
            0,
            50,
            51,
            30,
            29,
            28,
        ]
        correctness_runs = _coalesce_dense_runs(
            correctness_mapping,
            chunk_size=correctness_chunk,
            total_tokens=correctness_tokens,
        )
        num_blocks = math.ceil(capacity / block_size)
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
        aiv_latent, objects = _allocate_chunks(
            allocator,
            total_tokens=correctness_tokens,
            chunk_size=correctness_chunk,
            plane_dims=latent_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        aiv_index, objects = _allocate_chunks(
            allocator,
            total_tokens=correctness_tokens,
            chunk_size=correctness_chunk,
            plane_dims=index_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        sdma_latent, objects = _allocate_chunks(
            allocator,
            total_tokens=correctness_tokens,
            chunk_size=correctness_chunk,
            plane_dims=latent_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        sdma_index, objects = _allocate_chunks(
            allocator,
            total_tokens=correctness_tokens,
            chunk_size=correctness_chunk,
            plane_dims=index_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        for chunk in aiv_latent + aiv_index + sdma_latent + sdma_index:
            chunk.fill_(-17.0)

        offsets = list(range(0, correctness_tokens, correctness_chunk))
        sizes = [
            min(correctness_chunk, correctness_tokens - offset) for offset in offsets
        ]
        offsets_npu = torch.tensor(offsets, dtype=torch.int32, device=device)
        sizes_npu = torch.tensor(sizes, dtype=torch.int32, device=device)
        mapping_npu = torch.tensor(
            correctness_mapping,
            dtype=torch.long,
            device=device,
        )
        dense_mla_dsa_batched_direct_kv_transfer(
            aiv_latent,
            latent_cache,
            mapping_npu,
            offsets_npu,
            sizes_npu,
            correctness_tokens,
            KV_FORMAT_MLA_LATENT,
            False,
            False,
            latent_dims[0],
            latent_dims[1],
            0,
            False,
            True,
            build_chunk_ptrs_npu(aiv_latent, device),
            correctness_chunk,
        )
        dense_mla_dsa_batched_direct_kv_transfer(
            aiv_index,
            index_cache,
            mapping_npu,
            offsets_npu,
            sizes_npu,
            correctness_tokens,
            KV_FORMAT_DSA_INDEX,
            False,
            False,
            index_dims[0],
            0,
            index_dims[0],
            False,
            True,
            build_chunk_ptrs_npu(aiv_index, device),
            correctness_chunk,
        )
        # The ctypes runtime call is intentionally outside PyTorch's tensor
        # dependency tracker, so explicitly finish source initialization and
        # the reference gather before the independent SDMA stream reads it.
        torch.npu.synchronize()
        correctness_stream = torch.npu.Stream()
        _launch_sdma_two_group_save(
            runtime,
            latent_chunks=sdma_latent,
            index_chunks=sdma_index,
            latent_cache=latent_cache,
            index_cache=index_cache,
            runs=correctness_runs,
            total_tokens=correctness_tokens,
            chunk_size=correctness_chunk,
            stream=correctness_stream,
        )
        torch.npu.synchronize()
        for actual, expected in zip(sdma_latent, aiv_latent, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for actual, expected in zip(sdma_index, aiv_index, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        # Now measure the real GLM per-rank payload for one 2048-token layer.
        total_tokens = _env_int("LMCACHE_SDMA_TOKENS", 2048)
        chunk_size = _env_int("LMCACHE_SDMA_HOST_CHUNK", 256)
        if total_tokens % chunk_size:
            raise ValueError(
                "LMCACHE_SDMA_TOKENS must be divisible by "
                "LMCACHE_SDMA_HOST_CHUNK for this diagnostic"
            )
        mapping, capacity = _build_block_fragmented_mapping(
            total_tokens,
            block_size=block_size,
        )
        num_blocks = math.ceil(capacity / block_size)
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
        host_latent, objects = _allocate_chunks(
            allocator,
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            plane_dims=latent_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        host_index, objects = _allocate_chunks(
            allocator,
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            plane_dims=index_dims,
            dtype=dtype,
        )
        memory_objects.extend(objects)
        runs = _coalesce_dense_runs(
            mapping,
            chunk_size=chunk_size,
            total_tokens=total_tokens,
        )
        expected_dma_calls = len(runs) * (len(latent_dims) + len(index_dims))
        dma_stream = torch.npu.Stream()
        aiv_stream = torch.npu.Stream()

        def launch_sdma() -> None:
            copies = _launch_sdma_two_group_save(
                runtime,
                latent_chunks=host_latent,
                index_chunks=host_index,
                latent_cache=latent_cache,
                index_cache=index_cache,
                runs=runs,
                total_tokens=total_tokens,
                chunk_size=chunk_size,
                stream=dma_stream,
            )
            assert copies == expected_dma_calls

        offsets = list(range(0, total_tokens, chunk_size))
        offsets_npu = torch.tensor(offsets, dtype=torch.int32, device=device)
        sizes_npu = torch.full(
            (len(offsets),),
            chunk_size,
            dtype=torch.int32,
            device=device,
        )
        mapping_npu = torch.tensor(mapping, dtype=torch.long, device=device)
        latent_ptrs = build_chunk_ptrs_npu(host_latent, device)
        index_ptrs = build_chunk_ptrs_npu(host_index, device)

        def launch_aiv() -> None:
            with torch.npu.stream(aiv_stream):
                dense_mla_dsa_batched_direct_kv_transfer(
                    host_latent,
                    latent_cache,
                    mapping_npu,
                    offsets_npu,
                    sizes_npu,
                    total_tokens,
                    KV_FORMAT_MLA_LATENT,
                    False,
                    False,
                    latent_dims[0],
                    latent_dims[1],
                    0,
                    False,
                    True,
                    latent_ptrs,
                    chunk_size,
                )
                dense_mla_dsa_batched_direct_kv_transfer(
                    host_index,
                    index_cache,
                    mapping_npu,
                    offsets_npu,
                    sizes_npu,
                    total_tokens,
                    KV_FORMAT_DSA_INDEX,
                    False,
                    False,
                    index_dims[0],
                    0,
                    index_dims[0],
                    False,
                    True,
                    index_ptrs,
                    chunk_size,
                )

        matmul_size = _env_int("LMCACHE_SDMA_MATMUL_SIZE", 1024, minimum=256)
        samples = _env_int("LMCACHE_SDMA_SAMPLES", 7, minimum=3)
        left = torch.randn((matmul_size, matmul_size), dtype=dtype, device=device)
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        compute_stream = torch.npu.Stream()

        # Compile/warm all operators and runtime paths before timing.
        torch.npu.synchronize()
        _launch_compute_unit(
            stream=compute_stream,
            left=left,
            right=right,
            output=output,
            iterations=1,
        )
        launch_aiv()
        launch_sdma()
        torch.npu.synchronize()

        aiv_ms = _median_wall_ms(launch_aiv, samples)
        sdma_ms = _median_wall_ms(launch_sdma, samples)
        payload_mib = (total_tokens * (sum(latent_dims) + sum(index_dims)) * 2) / (
            1024 * 1024
        )
        print("\nExact two-group layer save")
        print(
            f"tokens={total_tokens} cache_block={block_size} "
            f"physical_runs={len(runs)} host_chunk={chunk_size} "
            f"payload={payload_mib:.3f} MiB SDMA_calls={expected_dma_calls}"
        )
        print(f"current AIV save median: {aiv_ms:.3f} ms")
        print(f"SDMA save median:        {sdma_ms:.3f} ms")

        # A D2H-only trace must contain SDMA/Memcpy tasks and no AICore/AIV
        # task.  This is stronger evidence than inferring the backend from a
        # Python API name.
        trace_path = _profile_sdma_only(
            launch_dma=launch_sdma,
            dma_stream=dma_stream,
        )
        assert trace_path.is_file()

        overlap_failures = []
        for target_ms in (1.0, 2.0):
            iterations, calibrated_ms = _calibrate_compute_iterations(
                target_ms=target_ms,
                stream=compute_stream,
                left=left,
                right=right,
                output=output,
            )

            def launch_compute(iterations: int = iterations) -> None:
                _launch_compute_unit(
                    stream=compute_stream,
                    left=left,
                    right=right,
                    output=output,
                    iterations=iterations,
                )

            # Warm the calibrated iteration count before collecting samples.
            launch_compute()
            launch_sdma()
            torch.npu.synchronize()
            overlap_samples = [
                _measure_overlap_sample(
                    launch_compute=launch_compute,
                    launch_dma=launch_sdma,
                    compute_stream=compute_stream,
                    dma_stream=dma_stream,
                )
                for _ in range(samples)
            ]
            compute_ms = statistics.median(
                sample.compute_ms for sample in overlap_samples
            )
            dma_overlap_ms = statistics.median(
                sample.dma_ms for sample in overlap_samples
            )
            makespan_ms = statistics.median(
                sample.makespan_ms for sample in overlap_samples
            )
            overlap_ms = statistics.median(
                sample.overlap_ms for sample in overlap_samples
            )
            dma_tail_ms = statistics.median(
                sample.dma_tail_ms for sample in overlap_samples
            )
            overlap_ratio = overlap_ms / max(dma_overlap_ms, 1e-6)
            serial_estimate_ms = compute_ms + dma_overlap_ms
            print(
                f"target_compute={target_ms:.1f} ms iterations={iterations} "
                f"calibrated={calibrated_ms:.3f} ms compute={compute_ms:.3f} ms "
                f"DMA={dma_overlap_ms:.3f} ms serial_estimate="
                f"{serial_estimate_ms:.3f} ms parallel={makespan_ms:.3f} ms "
                f"overlap={overlap_ms:.3f} ms ({overlap_ratio:.1%}) "
                f"DMA_tail={dma_tail_ms:.3f} ms"
            )

            if overlap_ratio < 0.50:
                overlap_failures.append(
                    "SDMA did not execute concurrently with AIC+AIV compute: "
                    f"target={target_ms} ms, overlap_ratio={overlap_ratio:.1%}"
                )
            if dma_tail_ms > 0.05:
                overlap_failures.append(
                    "the compute window did not cover the SDMA layer save: "
                    f"target={target_ms} ms, median DMA tail={dma_tail_ms:.3f} ms"
                )
        assert not overlap_failures, "; ".join(overlap_failures)
    finally:
        if _npu_available():
            torch.npu.synchronize()
        for memory_object in memory_objects:
            memory_object.ref_count_down()
        allocator.close()
