# SPDX-License-Identifier: Apache-2.0
"""Validate the hardware probe's data checks with adversarial CPU queues.

This verifies the probe, not torch_npu or NPU hardware. The connector handoff
methods are read from production; buffers change only when a queued task runs.
"""

import ast
from contextlib import contextmanager
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task:
    def __init__(self, fn, dependencies):
        self.fn = fn
        self.dependencies = dependencies
        self.done = False

    def finish(self):
        if not self.done:
            for dependency in self.dependencies:
                dependency.finish()
            self.fn()
            self.done = True


class Stream:
    def __init__(self, runtime):
        self.runtime = runtime
        self.last = None
        runtime.streams.append(self)

    def enqueue(self, fn, *dependencies):
        deps = tuple(dep for dep in (self.last, *dependencies) if dep is not None)
        self.last = Task(fn, deps)

    def wait_event(self, event):
        if self.runtime.drop_bank_wait and self is not self.runtime.compute:
            return
        self.enqueue(lambda: None, event.task)


class Event:
    def __init__(self):
        self.task = None

    def record(self, stream):
        self.task = stream.last

    def synchronize(self):
        if self.task is not None:
            self.task.finish()


class DeviceTensor:
    def __init__(self, tensor, runtime):
        self.tensor = tensor
        self.runtime = runtime

    def __getitem__(self, index):
        return DeviceTensor(self.tensor[index], self.runtime)

    def copy_(self, source):
        self.runtime.current.enqueue(lambda: self.tensor.copy_(source.tensor))

    def add_(self, value):
        self.runtime.current.enqueue(lambda: self.tensor.add_(value))

    def data_ptr(self):
        return self.tensor.data_ptr()

    def cpu(self):
        return self.tensor


class Runtime:
    Event = Event
    float32 = torch.float32
    arange = staticmethod(torch.arange)

    def __init__(self, drop_bank_wait=False):
        self.streams = []
        self.drop_bank_wait = drop_bank_wait
        self.compute = self.Stream()
        self.current = self.compute
        self.npu = self

    def Stream(self):
        return Stream(self)

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, value):
        old, self.current = self.current, value
        try:
            yield
        finally:
            self.current = old

    def synchronize(self):
        # DMA gets priority, unless a real dependency orders compute first.
        for stream in reversed(self.streams):
            if stream.last is not None:
                stream.last.finish()

    def empty(self, shape, *, device=None, pin_memory=False, dtype=None):
        value = torch.full(shape, -777, dtype=dtype or torch.float32)
        return DeviceTensor(value, self) if device == "npu" else value

    def empty_like(self, value, **kwargs):
        if isinstance(value, DeviceTensor):
            return self.empty(
                value.tensor.shape, device="npu", dtype=value.tensor.dtype
            )
        return torch.empty_like(value)

    def ones(self, shape, **kwargs):
        return self.empty(shape, **kwargs)

    def ones_like(self, value):
        return self.empty_like(value)

    def mm(self, left, right, out):
        self.current.enqueue(lambda: out.tensor.fill_(1))

    def layerwise_prefill_dma_copy(self, copies, from_npu):
        for dst, src, size in copies:
            self.current.enqueue(lambda d=dst, s=src, n=size: ctypes.memmove(d, s, n))


def production_connector(runtime):
    path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    names = {
        "_layerwise_prefill_transfer_state",
        "_layerwise_prefill_transfer_generation",
        "_layerwise_prefill_bank",
        "_layerwise_prefill_dma_stream",
        "record_layerwise_prefill_bank_use",
        "wait_for_layerwise_prefill_load",
    }
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = dict(
        torch=runtime,
        Any=Any,
        Optional=Optional,
        prefill_start_timing_enabled=lambda: False,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[cls.name]


@pytest.mark.parametrize("banks", [2, 4, 8])
@pytest.mark.parametrize("mode", ["event", "p_use", "same_stream"])
def test_probe_uses_completed_payloads(banks, mode):
    probe = load_file("bank_probe", "tools/prefill_bank_handoff_check.py")
    fence = load_file("fence_probe", "lmcache_ascend/v1/kv_fence_debug.py")
    runtime = Runtime()
    args = SimpleNamespace(
        layers=19,
        requests=4,
        tokens=17,
        width=8,
        matmul_size=4,
        compute_repeats=1,
        banks=banks,
    )
    result = probe.run_case(
        runtime, runtime, production_connector(runtime), fence.KVFenceDebug, args, mode
    )
    assert result["passed"], result


@pytest.mark.parametrize("mode", ["event", "p_use"])
@pytest.mark.parametrize("drop_host_wait", [False, True])
def test_default_host_fence_protects_missing_bank_wait(mode, drop_host_wait):
    probe = load_file("broken_probe", "tools/prefill_bank_handoff_check.py")
    fence = load_file("broken_fence", "lmcache_ascend/v1/kv_fence_debug.py")
    runtime = Runtime(drop_bank_wait=True)
    if drop_host_wait:
        class NoHostWaitEvent(Event):
            def synchronize(self):
                pass

        runtime.Event = NoHostWaitEvent
    args = SimpleNamespace(
        layers=11,
        requests=3,
        tokens=13,
        width=8,
        matmul_size=4,
        compute_repeats=1,
        banks=4,
    )
    result = probe.run_case(
        runtime, runtime, production_connector(runtime), fence.KVFenceDebug, args, mode
    )
    # Dropping both dependencies must still expose corrupted payloads.
    assert result["passed"] is not drop_host_wait, result
