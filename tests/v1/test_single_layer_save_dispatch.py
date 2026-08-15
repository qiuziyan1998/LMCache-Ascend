# SPDX-License-Identifier: Apache-2.0
"""Measure host submission and device scheduling of layerwise KV saves.

This is an Ascend hardware diagnostic.  It launches the same dense MLA D2H
kernel used by layerwise prefill, captures a torch_npu trace for actual kernel
start times, and surrounds each launch with timing events on the store stream.
A deliberately late-submission run is included as a control so the output
distinguishes these two cases:

* Python submits the save while the layer compute is still running, then the
  save waits in the device stream for its dependency/resources.
* Python does not submit the save until after the layer compute has finished.

Run with ``pytest -q -s tests/v1/test_single_layer_save_dispatch.py`` so the
per-layer timing table is visible.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

# Third Party
import pytest
import torch


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


@dataclass
class _PendingTiming:
    layer: int
    bank: int
    compute_done_at_submit: bool
    compute_done_at_kernel_call: bool
    save_done_at_return: bool
    host_prepare_ms: float
    host_enqueue_ms: float
    compute_start: Any
    compute_end: Any
    save_start: Any
    save_end: Any


@dataclass
class _LayerTiming:
    layer: int
    bank: int
    compute_done_at_submit: bool
    compute_done_at_kernel_call: bool
    save_done_at_return: bool
    host_prepare_ms: float
    host_enqueue_ms: float
    compute_start_ms: float
    compute_end_ms: float
    save_start_ms: float
    save_end_ms: float
    ready_to_save_start_ms: float
    profiler_launch_to_kernel_ms: float | None = None
    profiler_kernel_ms: float | None = None
    profiler_stream_id: str = "?"

    @property
    def compute_ms(self) -> float:
        return self.compute_end_ms - self.compute_start_ms

    @property
    def save_ms(self) -> float:
        return self.save_end_ms - self.save_start_ms


@dataclass(frozen=True)
class _ProfiledKernel:
    name: str
    launch_marker: str
    stream_id: str
    start_us: float
    duration_us: float
    launch_to_start_us: float


def _event_timestamp_us(event: dict[str, Any]) -> float:
    return float(event["ts"])


def _load_profiled_save_kernels(trace_path: Path) -> list[_ProfiledKernel]:
    """Pair explicit CPU launch markers with hardware kernels in submit order."""
    raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = (
        raw_trace.get("traceEvents", []) if isinstance(raw_trace, dict) else raw_trace
    )
    if not isinstance(events, list):
        raise AssertionError(f"invalid trace event container in {trace_path}")

    hardware_pids = {
        event.get("pid")
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    launch_events = []
    kernel_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        phase = event.get("ph")
        if (
            phase == "X"
            and str(event.get("name", "")).startswith("lmcache_save_submit::")
            and event.get("pid") not in hardware_pids
        ):
            launch_events.append(event)
        elif (
            phase == "X"
            and "single_layer_paged_kv_copy" in str(event.get("name", ""))
            and (not hardware_pids or event.get("pid") in hardware_pids)
        ):
            kernel_events.append(event)

    launch_events.sort(key=_event_timestamp_us)
    kernel_events.sort(key=_event_timestamp_us)
    if len(launch_events) != len(kernel_events):
        raise AssertionError(
            f"found {len(launch_events)} CPU save launch markers but "
            f"{len(kernel_events)} hardware save kernels in {trace_path}"
        )
    profiled = []
    for launch, kernel in zip(launch_events, kernel_events, strict=True):
        launch_us = _event_timestamp_us(launch)
        kernel_start_us = _event_timestamp_us(kernel)
        kernel_tid = str(kernel.get("tid", ""))
        if kernel_start_us < launch_us:
            raise AssertionError(
                f"hardware kernel {kernel.get('name')!r} started before its "
                f"paired CPU launch marker {launch.get('name')!r}; inspect "
                f"{trace_path}"
            )
        profiled.append(
            _ProfiledKernel(
                name=str(kernel.get("name", "")),
                launch_marker=str(launch.get("name", "")),
                stream_id=kernel_tid,
                start_us=kernel_start_us,
                duration_us=float(kernel.get("dur", 0.0)),
                launch_to_start_us=kernel_start_us - launch_us,
            )
        )
    return profiled


def test_profile_launch_marker_parser_uses_actual_hardware_kernel_start(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace_view.json"
    trace_path.write_text(
        json.dumps(
            [
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": 4,
                    "tid": 0,
                    "args": {"name": "Ascend Hardware"},
                },
                {
                    "ph": "X",
                    "name": "lmcache_save_submit::immediate::layer_0",
                    "pid": 1,
                    "tid": 9,
                    "ts": 100.0,
                    "dur": 5.0,
                },
                {
                    "ph": "X",
                    "name": (
                        "single_layer_paged_kv_copy_v2_mla_"
                        "dense_multi_chunk_bfloat16_t_int64_t"
                    ),
                    "pid": 4,
                    "tid": 38,
                    "ts": 250.0,
                    "dur": 25.0,
                },
                {
                    "ph": "X",
                    "name": "single_layer_paged_kv_copy_cpu_wrapper",
                    "pid": 1,
                    "tid": 9,
                    "ts": 90.0,
                    "dur": 5.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    kernels = _load_profiled_save_kernels(trace_path)

    assert len(kernels) == 1
    assert kernels[0].launch_marker.endswith("immediate::layer_0")
    assert kernels[0].stream_id == "38"
    assert kernels[0].start_us == pytest.approx(250.0)
    assert kernels[0].duration_us == pytest.approx(25.0)
    assert kernels[0].launch_to_start_us == pytest.approx(150.0)


def _attach_profiled_kernels(
    timings: list[_LayerTiming],
    kernels: list[_ProfiledKernel],
) -> None:
    if len(timings) != len(kernels):
        raise AssertionError(
            f"expected {len(timings)} profiled save kernels, found {len(kernels)}"
        )
    for timing, kernel in zip(timings, kernels, strict=True):
        timing.profiler_launch_to_kernel_ms = kernel.launch_to_start_us / 1000.0
        timing.profiler_kernel_ms = kernel.duration_us / 1000.0
        timing.profiler_stream_id = kernel.stream_id


def _print_pipeline(label: str, timings: list[_LayerTiming]) -> None:
    print(f"\n{label}: single_layer_paged_kv_copy dispatch timing")
    print(
        "layer bank stream done@submit done@kernel_call save_done@return "
        "host_prepare_ms host_enqueue_ms compute_ms "
        "ready_to_stream_boundary_ms launch_to_actual_kernel_ms "
        "actual_kernel_ms store_stream_span_ms"
    )
    for item in timings:
        launch_to_kernel = item.profiler_launch_to_kernel_ms
        kernel_ms = item.profiler_kernel_ms
        print(
            f"{item.layer:>5} {item.bank:>4} {item.profiler_stream_id:>6} "
            f"{str(item.compute_done_at_submit):>11} "
            f"{str(item.compute_done_at_kernel_call):>16} "
            f"{str(item.save_done_at_return):>16} "
            f"{item.host_prepare_ms:>15.3f} "
            f"{item.host_enqueue_ms:>15.3f} "
            f"{item.compute_ms:>10.3f} "
            f"{item.ready_to_save_start_ms:>27.3f} "
            f"{launch_to_kernel if launch_to_kernel is not None else math.nan:>26.3f} "
            f"{kernel_ms if kernel_ms is not None else math.nan:>16.3f} "
            f"{item.save_ms:>20.3f}"
        )


def _run_pipeline(
    *,
    scenario: str,
    layer_count: int,
    bank_count: int,
    compute_iters: int,
    compute_stream: Any,
    store_stream: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    launch_save: Callable[[int, int], None],
    synchronize_before_submit: bool,
) -> list[_LayerTiming]:
    torch.npu.synchronize()

    origin_event = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(compute_stream):
        origin_event.record()
    origin_event.synchronize()

    bank_done_events: list[Any | None] = [None] * bank_count
    pending: list[_PendingTiming] = []

    for layer in range(layer_count):
        bank = layer % bank_count
        compute_start = torch.npu.Event(enable_timing=True)
        compute_end = torch.npu.Event(enable_timing=True)
        save_start = torch.npu.Event(enable_timing=True)
        save_end = torch.npu.Event(enable_timing=True)

        with torch.npu.stream(compute_stream):
            compute_start.record()
            for _ in range(compute_iters):
                torch.mm(left, right, out=output)
            compute_end.record()

        # This is the intentional difference between the real-style launch and
        # the control.  query() itself does not synchronize the host.
        if synchronize_before_submit:
            compute_end.synchronize()
        compute_done_at_submit = bool(compute_end.query())

        host_prepare_ns = time.perf_counter_ns()
        with torch.npu.stream(store_stream):
            # Match _run_dense_direct_kv_transfer_layer(): the store stream
            # waits for all work currently queued on the compute stream.
            store_stream.wait_stream(compute_stream)
            compute_done_at_kernel_call = bool(compute_end.query())
            host_kernel_call_ns = time.perf_counter_ns()
            save_start.record()
            with torch.profiler.record_function(
                f"lmcache_save_submit::{scenario}::layer_{layer}"
            ):
                launch_save(layer, bank)
            host_return_ns = time.perf_counter_ns()
            save_end.record()
        save_done_at_return = bool(save_end.query())
        bank_done_events[bank] = save_end

        # Match the three-bank lifetime rule in batched_from_gpu(): after a
        # layer save is queued, protect the bank that the next layer will use.
        next_bank = (bank + 1) % bank_count
        next_bank_done = bank_done_events[next_bank]
        if next_bank_done is not None:
            with torch.npu.stream(compute_stream):
                compute_stream.wait_event(next_bank_done)

        pending.append(
            _PendingTiming(
                layer=layer,
                bank=bank,
                compute_done_at_submit=compute_done_at_submit,
                compute_done_at_kernel_call=compute_done_at_kernel_call,
                save_done_at_return=save_done_at_return,
                host_prepare_ms=(host_kernel_call_ns - host_prepare_ns) / 1_000_000.0,
                host_enqueue_ms=(host_return_ns - host_kernel_call_ns) / 1_000_000.0,
                compute_start=compute_start,
                compute_end=compute_end,
                save_start=save_start,
                save_end=save_end,
            )
        )

    torch.npu.synchronize()

    timings = []
    for item in pending:
        save_start_ms = origin_event.elapsed_time(item.save_start)
        timings.append(
            _LayerTiming(
                layer=item.layer,
                bank=item.bank,
                compute_done_at_submit=item.compute_done_at_submit,
                compute_done_at_kernel_call=item.compute_done_at_kernel_call,
                save_done_at_return=item.save_done_at_return,
                host_prepare_ms=item.host_prepare_ms,
                host_enqueue_ms=item.host_enqueue_ms,
                compute_start_ms=origin_event.elapsed_time(item.compute_start),
                compute_end_ms=origin_event.elapsed_time(item.compute_end),
                save_start_ms=save_start_ms,
                save_end_ms=origin_event.elapsed_time(item.save_end),
                ready_to_save_start_ms=item.compute_end.elapsed_time(item.save_start),
            )
        )
    return timings


@pytest.mark.skipif(
    not _npu_available(),
    reason="single-layer save dispatch timing requires an Ascend NPU",
)
def test_single_layer_paged_kv_copy_submission_to_execution_delay() -> None:
    """Time immediate submission and a deliberately late-submit control."""
    _add_benchmark_helpers_to_path()
    import torch_npu

    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

    from lmcache.v1.memory_management import PinMemoryAllocator
    from load_benchmark_utils import (
        KV_FORMAT_MLA,
        build_chunk_ptrs_npu,
        ensure_ascend_host_memory_registered,
    )
    from lmcache_ascend.v1.npu_connector.utils import (
        dense_mla_dsa_batched_direct_kv_transfer,
    )

    ensure_ascend_host_memory_registered()

    layer_count = _env_int("LMCACHE_SAVE_DISPATCH_LAYERS", 6, minimum=4)
    bank_count = 3
    total_tokens = _env_int("LMCACHE_SAVE_DISPATCH_TOKENS", 2048)
    host_chunk_size = _env_int("LMCACHE_SAVE_DISPATCH_HOST_CHUNK", 256)
    matmul_size = _env_int("LMCACHE_SAVE_DISPATCH_MATMUL_SIZE", 4096, minimum=256)
    compute_iters = _env_int("LMCACHE_SAVE_DISPATCH_COMPUTE_ITERS", 16)
    if total_tokens % host_chunk_size:
        raise ValueError(
            "LMCACHE_SAVE_DISPATCH_TOKENS must be divisible by "
            "LMCACHE_SAVE_DISPATCH_HOST_CHUNK"
        )

    torch.manual_seed(20260814)
    device = torch.device("npu")
    dtype = torch.bfloat16
    k_hidden_dims = 512
    v_hidden_dims = 128
    block_size = 16
    num_blocks = math.ceil(total_tokens / block_size)
    num_host_chunks = total_tokens // host_chunk_size
    elements_per_chunk = host_chunk_size * (k_hidden_dims + v_hidden_dims)
    required_host_bytes = (
        layer_count * total_tokens * (k_hidden_dims + v_hidden_dims) * 2
    )
    allocator = PinMemoryAllocator(max(64 * 1024 * 1024, required_host_bytes * 2))
    memory_objs = []

    try:
        host_layers = []
        chunk_ptrs_by_layer = []
        for _ in range(layer_count):
            host_chunks = []
            for _ in range(num_host_chunks):
                memory_obj = allocator.allocate(
                    torch.Size([elements_per_chunk]),
                    dtype,
                )
                assert memory_obj is not None and memory_obj.tensor is not None
                memory_obj.tensor.fill_(-17.0)
                memory_objs.append(memory_obj)
                host_chunks.append(memory_obj.tensor)
            host_layers.append(host_chunks)
            chunk_ptrs_by_layer.append(build_chunk_ptrs_npu(host_chunks, device))

        kv_banks = []
        for _ in range(bank_count):
            k_cache = torch.randn(
                (num_blocks, block_size, 1, k_hidden_dims),
                dtype=dtype,
                device=device,
            )
            v_cache = torch.randn(
                (num_blocks, block_size, 1, v_hidden_dims),
                dtype=dtype,
                device=device,
            )
            kv_banks.append((k_cache, v_cache))

        slot_mapping = torch.arange(total_tokens, dtype=torch.long, device=device)
        chunk_offsets = torch.arange(
            0,
            total_tokens,
            host_chunk_size,
            dtype=torch.int32,
            device=device,
        )
        chunk_sizes = torch.full(
            (num_host_chunks,),
            host_chunk_size,
            dtype=torch.int32,
            device=device,
        )

        left = torch.randn((matmul_size, matmul_size), dtype=dtype, device=device)
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        compute_stream = torch.npu.Stream()
        store_stream = torch.npu.Stream()

        def launch_save(layer: int, bank: int) -> None:
            dense_mla_dsa_batched_direct_kv_transfer(
                host_layers[layer],
                kv_banks[bank],
                slot_mapping,
                chunk_offsets,
                chunk_sizes,
                total_tokens,
                KV_FORMAT_MLA,
                False,
                False,
                k_hidden_dims,
                v_hidden_dims,
                0,
                False,
                True,
                chunk_ptrs_by_layer[layer],
                host_chunk_size,
            )

        # Compile/warm all paths before measuring host enqueue time.
        with torch.npu.stream(compute_stream):
            torch.mm(left, right, out=output)
        with torch.npu.stream(store_stream):
            launch_save(0, 0)
        torch.npu.synchronize()

        profile_root = Path(
            os.environ.get(
                "LMCACHE_SAVE_DISPATCH_PROFILE_DIR",
                "/tmp/lmcache-save-dispatch-profile",
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
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1,
            ),
            with_stack=False,
            profile_memory=False,
            with_modules=False,
            experimental_config=experimental_config,
        )
        with profiler:
            immediate = _run_pipeline(
                scenario="immediate",
                layer_count=layer_count,
                bank_count=bank_count,
                compute_iters=compute_iters,
                compute_stream=compute_stream,
                store_stream=store_stream,
                left=left,
                right=right,
                output=output,
                launch_save=launch_save,
                synchronize_before_submit=False,
            )
            late_control = _run_pipeline(
                scenario="late",
                layer_count=layer_count,
                bank_count=bank_count,
                compute_iters=compute_iters,
                compute_stream=compute_stream,
                store_stream=store_stream,
                left=left,
                right=right,
                output=output,
                launch_save=launch_save,
                synchronize_before_submit=True,
            )
        profiler.export_chrome_trace(str(trace_path))
        assert trace_path.is_file(), f"torch_npu profiler did not create {trace_path}"
        profiled_kernels = _load_profiled_save_kernels(trace_path)
        expected_kernel_count = layer_count * 2
        assert len(profiled_kernels) == expected_kernel_count, (
            f"expected {expected_kernel_count} single-layer save kernels in "
            f"{trace_path}, found {len(profiled_kernels)}"
        )
        _attach_profiled_kernels(immediate, profiled_kernels[:layer_count])
        _attach_profiled_kernels(late_control, profiled_kernels[layer_count:])
        _print_pipeline("immediate-submit (matches layerwise prefill)", immediate)
        _print_pipeline(
            "late-submit control (compute_end.synchronize first)",
            late_control,
        )
        print(f"\nMindStudio trace: {trace_path}")

        immediate_early_count = sum(
            not item.compute_done_at_kernel_call for item in immediate
        )
        print("\nInterpretation")
        print(
            f"immediate kernel calls made before compute completed: "
            f"{immediate_early_count}/{layer_count}"
        )
        print(
            "launch_to_actual_kernel_ms is the trace timestamp difference "
            "between the explicit CPU launch marker and the hardware kernel "
            "start, not a Python/device clock estimate. If "
            "done@kernel_call is False, Python called the op before compute "
            "finished. If it is True only after host_prepare_ms, the delay "
            "occurred on the host before the op call."
        )

        assert immediate_early_count > 0, (
            "The immediate path did not submit any save before compute "
            "completed. Increase LMCACHE_SAVE_DISPATCH_COMPUTE_ITERS if the "
            "compute probe is too short; otherwise this indicates a host-side "
            "synchronization before the save launch."
        )
        assert all(item.compute_done_at_kernel_call for item in late_control)
        assert all(
            item.profiler_launch_to_kernel_ms is not None
            and item.profiler_launch_to_kernel_ms >= 0.0
            for item in immediate + late_control
        )
        assert any(
            torch.any(chunk != -17.0).item()
            for chunks in host_layers
            for chunk in chunks
        )
    finally:
        torch.npu.synchronize()
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()
        allocator.close()
