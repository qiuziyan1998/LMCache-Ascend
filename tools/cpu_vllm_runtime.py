# SPDX-License-Identifier: Apache-2.0
"""Hardware substitutions for executing the imported vLLM/LMCache code on CPU."""

from __future__ import annotations

import ctypes
import importlib.machinery
import mmap
import os
from pathlib import Path
import sys
import threading
import types
from functools import wraps
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from prefill_cpu_sim import Device, Event, Stream


class CPUHardware:
    """Preserve asynchronous stream order while replacing native device calls."""

    def __init__(self, root: Path, seed: int = 0):
        self.root = root
        self.lock = threading.RLock()

        class LockedDevice(Device):
            def __init__(self, lock):
                self.lock, self.local = lock, threading.local()
                self.default_stream = None
                super().__init__(seed, schedule="dma-lag")

            @property
            def current(self):
                return getattr(self.local, "current", self.default_stream)

            @current.setter
            def current(self, value):
                self.local.current = value

            def enqueue(self, *args, **kwargs):
                with self.lock:
                    return super().enqueue(*args, **kwargs)

            def step(self):
                with self.lock:
                    command = self.ready[0][2] if self.ready else None
                    result = super().step()
                    # Completed commands must not retain full 500k-token inputs.
                    if command is not None:
                        command.action = lambda: None
                    return result

            def until(self, *args):
                with self.lock:
                    return super().until(*args)

            def drain(self):
                with self.lock:
                    return super().drain()

        self.queue = LockedDevice(self.lock)
        self.queue.current = Stream(self.queue, "default/compute")
        self.queue.default_stream = self.queue.current
        self.pinned: dict[int, torch.Tensor] = {}
        self.named_memory: dict[str, torch.Tensor] = {}
        self.old = {}
        self.virtual_allocations = []
        self.pending_writers = {}
        self.transfer_stats = dict(
            poisoned_operations=0,
            completed_operations=0,
            guarded_reads=0,
            early_reads=0,
        )
        hardware = self

        class TorchStream(Stream):
            def __init__(self, *args, **kwargs):
                Stream.__init__(self, hardware.queue, f"stream-{id(self)}")
                self.cuda_stream = self.npu_stream = id(self)

            def query(self):
                return self.tail is None or self.tail.done

            def wait_stream(self, stream):
                with hardware.lock:
                    return super().wait_stream(stream)

            def wait_event(self, event):
                with hardware.lock:
                    if (
                        hardware.queue.fault == "skip-save-wait"
                        and not self.name.endswith("compute")
                        and event.stream is not None
                        and event.stream.name.endswith("compute")
                    ):
                        return
                    return super().wait_event(event)

            def synchronize(self):
                with hardware.lock:
                    return super().synchronize()

        class TorchEvent(Event):
            def record(self, stream=None):
                with hardware.lock:
                    super().record(stream or hardware.queue.current)
                    if hardware.queue.fault == "early-ready-event":
                        self.tail = getattr(self.tail, "write_begin", self.tail)

            def synchronize(self):
                with hardware.lock:
                    if self.stream is not None:
                        super().synchronize()

            def wait(self, stream=None):
                (stream or hardware.queue.current).wait_event(self)

        self.Stream, self.Event = TorchStream, TorchEvent

    def _set(self, owner, name, value):
        self.old[owner, name] = getattr(owner, name, None)
        setattr(owner, name, value)

    @contextmanager
    def stream(self, stream):
        previous = self.queue.current
        self.queue.current = stream
        try:
            yield
        finally:
            self.queue.current = previous

    @contextmanager
    def device(self, *args, **kwargs):
        yield

    def alloc(self, size, *args):
        if size >= 16 * 1024**2:
            memory = mmap.mmap(-1, size)
            self.virtual_allocations.append(memory)
            tensor = torch.frombuffer(memory, dtype=torch.uint8)
        else:
            tensor = torch.full((size,), 255, dtype=torch.uint8)
        self.pinned[tensor.data_ptr()] = tensor
        self.queue.memory(tensor, tensor)
        return tensor.data_ptr()

    def alloc_shm(self, size, name, *args):
        if name not in self.named_memory:
            pointer = self.alloc(size)
            self.named_memory[name] = self.pinned[pointer]
        return self.named_memory[name].data_ptr()

    def attach_shm(self, size, name, *args):
        if name not in self.named_memory:
            raise KeyError(
                f"Shared mapping {name!r} not found; "
                f"allocated={list(self.named_memory)}"
            )
        return self.named_memory[name].data_ptr()

    def free(self, pointer, *args):
        self.pinned.pop(pointer, None)

    @staticmethod
    def merge_ranges(ranges):
        """Coalesce exact byte ranges; never enlarge the accessed region."""
        merged = []
        for start, size in sorted(ranges):
            if size <= 0:
                continue
            end = start + size
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return [(start, end - start) for start, end in merged]

    def _writers(self, ranges):
        commands = {}
        for start, size in ranges:
            end = start + size
            for tile in range(start // 65536, (end - 1) // 65536 + 1):
                entries = self.pending_writers.get(tile, ())
                remaining = [(a, b, c) for a, b, c in entries if not c.done]
                if remaining:
                    self.pending_writers[tile] = remaining
                else:
                    self.pending_writers.pop(tile, None)
                for a, b, command in remaining:
                    if a < end and start < b:
                        commands[command.number] = command
        return tuple(commands.values())

    def submit_access(self, label, action, *, reads=(), writes=()):
        """Poison at device start, commit at completion, check read readiness.

        Poisoning at host submission would overwrite a bank still being read
        by an earlier layer. The begin command obeys the actual stream/event
        dependencies instead. A captured writer-version check also catches
        reads before begin, even when old bytes happen to equal the new bytes.
        """
        with self.lock:
            reads = self.merge_ranges(reads)
            writes = self.merge_ranges(writes)
            required = self._writers(reads)

            def begin():
                for pointer, size in writes:
                    self.queue.check_address(pointer, size)
                    ctypes.memset(pointer, 255, size)
                self.transfer_stats["poisoned_operations"] += 1

            def complete():
                for command in required:
                    self.transfer_stats["guarded_reads"] += 1
                    if not command.done:
                        self.transfer_stats["early_reads"] += 1
                        raise AssertionError(
                            f"read before completion: {label} needs "
                            f"{command.number}:{command.label}"
                        )
                action()
                self.transfer_stats["completed_operations"] += 1

            started = self.queue.current.submit(label + " begin/poison", begin)
            command = self.queue.current.submit(label + " complete", complete)
            command.write_begin = started
            for start, size in writes:
                end = start + size
                for tile in range(start // 65536, (end - 1) // 65536 + 1):
                    entries = self.pending_writers.setdefault(tile, [])
                    entries[:] = [(a, b, c) for a, b, c in entries if not c.done]
                    entries.append((start, end, command))
            return command

    @staticmethod
    def tensor_ranges(tensor, rows):
        """Return exact contiguous physical row runs without per-token loops."""
        values = np.unique(rows.cpu().numpy().astype(np.int64, copy=False))
        if not len(values):
            return []
        breaks = np.flatnonzero(np.diff(values) != 1) + 1
        starts = values[np.r_[0, breaks]]
        ends = values[np.r_[breaks - 1, len(values) - 1]] + 1
        stride = tensor.shape[-1] * tensor.element_size()
        return [
            (tensor.data_ptr() + int(a) * stride, int(b - a) * stride)
            for a, b in zip(starts, ends, strict=True)
        ]

    def dma(self, copies, to_cpu):
        # Keep raw descriptors immutable, just as pybind converts them before
        # returning to Python. Device effects happen only when the queue runs.
        rows = tuple(tuple(int(x) for x in row) for row in copies)
        with self.lock:

            def copy():
                for dst, src, size in rows:
                    self.queue.check_address(src, size)
                    self.queue.check_address(dst, size)
                    ctypes.memmove(dst, src, size)

            self.submit_access(
                "save native D2H" if to_cpu else "load native H2D",
                copy,
                reads=[(src, size) for _, src, size in rows],
                writes=[(dst, size) for dst, _, size in rows],
            )
            self.queue.counts["save" if to_cpu else "load"] += 1

    def layer_state(self, sample, caches, mapping, *args):
        return SimpleNamespace(
            caches=tuple(caches) if isinstance(caches, (list, tuple)) else (caches,),
            sample=sample,
            capacity=args[-1] if args else None,
        )

    def destination_state(self, caches, mapping, *args):
        return self.layer_state(None, caches, mapping, *args)

    def sparse_copy(
        self, state, slots, selected, pointers, chunk_size, total_tokens, *args
    ):
        caches = state.caches
        slots, selected, pointers = slots.clone(), selected.clone(), pointers.clone()
        widths = [int(x.shape[-1]) for x in caches]
        element = caches[0].element_size()
        destination = slots.flatten().long()
        token_ids = selected.flatten().long()
        valid = (token_ids >= 0) & (token_ids < total_tokens) & (destination >= 0)
        destination, token_ids = destination[valid], token_ids[valid]
        chunks = token_ids // chunk_size
        if len(chunks) > 1 and bool(torch.any(chunks[1:] < chunks[:-1])):
            order = torch.argsort(chunks, stable=True)
            chunks, destination, token_ids = (
                chunks[order],
                destination[order],
                token_ids[order],
            )
        chunk_ids, counts = torch.unique_consecutive(chunks, return_counts=True)
        plans, reads, writes = [], [], []
        begin = 0
        for chunk, count in zip(chunk_ids.tolist(), counts.tolist(), strict=True):
            targets = destination[begin : begin + count]
            offsets_in_chunk = token_ids[begin : begin + count] % chunk_size
            begin += count
            tokens = min(chunk_size, total_tokens - chunk * chunk_size)
            source = int(pointers[chunk])
            prefix = 0
            for cache, width in zip(caches, widths, strict=True):
                pointer = source + prefix * tokens * element
                cpu = self.pointer_tensor(pointer, tokens * width, cache.dtype)
                plans.append(
                    (
                        cache.reshape(-1, width),
                        targets,
                        cpu.reshape(tokens, width),
                        offsets_in_chunk,
                    )
                )
                reads.extend(
                    self.tensor_ranges(cpu.reshape(tokens, width), offsets_in_chunk)
                )
                writes.extend(self.tensor_ranges(cache, targets))
                prefix += width

        def copy():
            for view, targets, cpu, source_rows in plans:
                view[targets] = cpu[source_rows]

        with self.lock:
            self.submit_access(
                "load native sparse H2D", copy, reads=reads, writes=writes
            )
            self.queue.counts["load"] += 1

    def pointer_tensor(self, pointer, count, dtype):
        element = torch.empty((), dtype=dtype).element_size()
        self.queue.check_address(pointer, count * element)
        return torch.frombuffer(
            (ctypes.c_ubyte * (count * element)).from_address(pointer),
            dtype=dtype,
        )

    def dense_copy(
        self,
        state,
        slots,
        pointers,
        offsets,
        sizes,
        total_tokens,
        interleaved,
        validate=False,
        fixed_chunk_size=0,
        *,
        direction=False,
    ):
        caches = state.caches
        widths = [int(cache.shape[-1]) for cache in caches]
        element = caches[0].element_size()
        plans, reads, writes = [], [], []
        for chunk, pointer in enumerate(pointers.tolist()):
            start = (
                chunk * fixed_chunk_size if fixed_chunk_size else int(offsets[chunk])
            )
            count = (
                min(fixed_chunk_size, total_tokens - start)
                if fixed_chunk_size
                else int(sizes[chunk])
            )
            selected = slots[start : start + count].long().clone()
            valid = selected >= 0
            prefix = 0
            raw = self.pointer_tensor(
                int(pointer), count * sum(widths), caches[0].dtype
            )
            for cache, width in zip(caches, widths, strict=True):
                cpu = (
                    raw.reshape(count, sum(widths))[:, prefix : prefix + width]
                    if interleaved
                    else raw[prefix * count : (prefix + width) * count].reshape(
                        count, width
                    )
                )
                view = cache.reshape(-1, width)
                plans.append((view, selected, valid, cpu))
                cpu_ranges = (
                    [(int(pointer), count * sum(widths) * element)]
                    if interleaved
                    else [(cpu.data_ptr(), cpu.numel() * element)]
                )
                device_ranges = self.tensor_ranges(cache, selected[valid])
                reads.extend(device_ranges if direction else cpu_ranges)
                writes.extend(cpu_ranges if direction else device_ranges)
                prefix += width

        def copy():
            for view, selected, valid, cpu in plans:
                if direction:
                    cpu[valid] = view[selected[valid]]
                else:
                    view[selected[valid]] = cpu[valid]

        with self.lock:
            self.submit_access(
                "save dense" if direction else "load dense",
                copy,
                reads=reads,
                writes=writes,
            )
            self.queue.counts["save" if direction else "load"] += 1

    def dense_fast(
        self,
        state,
        slots,
        pointers,
        offsets,
        sizes,
        total_tokens,
        interleaved,
        direction,
        validate=False,
        fixed_chunk_size=0,
    ):
        return self.dense_copy(
            state,
            slots,
            pointers,
            offsets,
            sizes,
            total_tokens,
            interleaved,
            validate,
            fixed_chunk_size,
            direction=direction,
        )

    def unknown(self, name):
        def fail(*args, **kwargs):
            raise NotImplementedError(f"CPU substitute missing native operation {name}")

        return fail

    def native_module(self, name):
        module = types.ModuleType(name)
        operations = dict(
            alloc_pinned_ptr=self.alloc,
            alloc_pinned_numa_ptr=self.alloc,
            alloc_shm_pinned_ptr=self.alloc_shm,
            free_pinned_ptr=self.free,
            free_pinned_numa_ptr=self.free,
            free_shm_pinned_ptr=self.free,
            get_device_ptr=lambda pointer, size=0: int(pointer),
            attach_shm_pinned_ptr=self.attach_shm,
            detach_shm_pinned_ptr=lambda *args: None,
            unlink_shm=lambda *args: None,
            layerwise_prefill_dma_copy=self.dma,
            prepare_sparse_direct_layer_state=self.layer_state,
            prepare_sparse_direct_destination_state=self.destination_state,
            sparse_mla_dsa_batched_direct_kv_transfer_fast=self.sparse_copy,
            sparse_mla_dsa_batched_direct_kv_transfer_prepared=self.sparse_copy,
            dense_mla_dsa_batched_direct_kv_transfer_prepared=self.dense_copy,
            dense_mla_dsa_batched_direct_kv_transfer_fast=self.dense_fast,
        )
        module.__dict__.update(operations)
        module.__getattr__ = lambda attr: (
            self.unknown(attr) if not attr.startswith("__") else None
        )
        return module

    def install_mooncake(self):
        """Replace the native store SDK, retaining the real Python connector."""
        hardware = self
        self.remote = {}
        self.remote_calls = []

        class Store:
            def setup(self, *args):
                self.hostname = args[0]
                self.network_stream = hardware.Stream()
                self.network_stream.name = "native-remote-transfer"
                return 0

            def get_hostname(self):
                return self.hostname

            def register_buffer(self, *args):
                return 0

            def unregister_buffer(self, *args):
                return 0

            def is_exist(self, key):
                return int(key in hardware.remote)

            def batch_is_exist(self, keys):
                result = [self.is_exist(key) for key in keys]
                return result

            def batch_put_from_multi_buffers(self, keys, pointers, sizes, *args):
                with hardware.lock:
                    rows = [
                        (key, tuple(row), tuple(lengths))
                        for key, row, lengths in zip(keys, pointers, sizes, strict=True)
                    ]

                    def publish():
                        for key, row, lengths in rows:
                            hardware.remote[key] = b"".join(
                                ctypes.string_at(p, n)
                                for p, n in zip(row, lengths, strict=True)
                            )
                            if hardware.queue.fault == "corrupt-remote":
                                data = bytearray(hardware.remote[key])
                                data[0] ^= 1
                                hardware.remote[key] = bytes(data)
                            hardware.remote_calls.append(("put", key, sum(lengths)))
                            hardware.queue.counts["publish"] += 1

                    with hardware.stream(self.network_stream):
                        finished = hardware.submit_access(
                            "publish native remote",
                            publish,
                            reads=[
                                (p, n)
                                for _, row, lengths in rows
                                for p, n in zip(row, lengths, strict=True)
                            ],
                        )
                    hardware.queue.until(finished)
                return [0] * len(keys)

            def batch_get_into_multi_buffers(self, keys, pointers, sizes):
                statuses = []
                with hardware.lock:
                    plans = []
                    for key, row, lengths in zip(keys, pointers, sizes, strict=True):
                        data = hardware.remote.get(key)
                        if data is None:
                            statuses.append(-1)
                            continue
                        if len(data) != sum(lengths):
                            raise AssertionError(
                                f"Remote payload size mismatch: {len(data)} "
                                f"!= {sum(lengths)}"
                            )
                        offset = 0
                        for pointer, size in zip(row, lengths, strict=True):
                            hardware.queue.check_address(pointer, size)
                            plans.append((pointer, size, data[offset : offset + size]))
                            offset += size
                        hardware.remote_calls.append(("get", key, len(data)))
                        statuses.append(len(data))
                    if plans:

                        def receive():
                            for pointer, size, payload in plans:
                                ctypes.memmove(pointer, payload, size)

                        with hardware.stream(self.network_stream):
                            finished = hardware.submit_access(
                                "load native remote",
                                receive,
                                writes=[(pointer, size) for pointer, size, _ in plans],
                            )
                        hardware.queue.until(finished)
                return statuses

            def batch_put_from(self, keys, pointers, sizes, *args):
                return self.batch_put_from_multi_buffers(
                    keys, [[p] for p in pointers], [[s] for s in sizes]
                )

            def batch_get_into(self, keys, pointers, sizes):
                return self.batch_get_into_multi_buffers(
                    keys, [[p] for p in pointers], [[s] for s in sizes]
                )

            def close(self):
                pass

        package = types.ModuleType("mooncake")
        module = types.ModuleType("mooncake.store")
        module.MooncakeDistributedStore = Store
        module.ReplicateConfig = SimpleNamespace
        package.store = module
        sys.modules["mooncake"], sys.modules["mooncake.store"] = package, module

    def install(self):
        """Install this process's CPU-only hardware backend before NPU imports."""
        # Import vLLM/transformers first, before registering a fake torch_npu.
        # This avoids selecting transformers' optional native NPU kernels.
        from vllm.v1.engine.core import EngineCore  # noqa: F401
        import torch._dynamo  # noqa: F401

        torch._dynamo.config.disable = True

        for name, directory in (
            ("lmcache_ascend", self.root / "lmcache_ascend"),
            ("vllm_ascend", self.root.parent / "vllm-ascend/vllm_ascend"),
            (
                "vllm_ascend.patch.platform",
                self.root.parent / "vllm-ascend/vllm_ascend/patch/platform",
            ),
            (
                "vllm_ascend.patch.worker",
                self.root.parent / "vllm-ascend/vllm_ascend/patch/worker",
            ),
            (
                "lmcache_ascend.v1.npu_connector",
                self.root / "lmcache_ascend/v1/npu_connector",
            ),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            sys.modules[name] = package
        build = types.ModuleType("lmcache_ascend._build_info")
        build.__framework_name__ = "pytorch"
        sys.modules[build.__name__] = build
        ascend_build = types.ModuleType("vllm_ascend._build_info")
        ascend_build.__device_type__ = "A3"
        sys.modules[ascend_build.__name__] = ascend_build
        sys.modules["lmcache.c_ops"] = self.native_module("lmcache.c_ops")
        sys.modules["lmcache_ascend.c_ops"] = self.native_module("lmcache_ascend.c_ops")

        # These imports have no device side effects and remain genuine modules.
        import lmcache.v1.cache_engine  # noqa: F401
        import lmcache_ascend.v1.cache_engine  # noqa: F401
        import lmcache_ascend.integration.vllm.vllm_v1_adapter  # noqa: F401
        import lmcache.v1.memory_management as memory_management

        self._set(memory_management, "lmc_ops", sys.modules["lmcache.c_ops"])
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
        from lmcache_ascend.v1.kv_layer_groups import (
            build_kv_layer_groups,
            patched_hidden_dim_size,
        )

        # Same data-layout installation as the Ascend package entry point.
        self._set(KVLayerGroupsManager, "build_kv_layer_groups", build_kv_layer_groups)
        self._set(
            KVLayerGroupInfo, "hidden_dim_size", property(patched_hidden_dim_size)
        )
        from lmcache.v1.token_database import TokenDatabase
        from lmcache_ascend.v1.tokens_hash import _hash_tokens

        self._set(TokenDatabase, "_hash_tokens", _hash_tokens)

        npu = types.ModuleType("torch_npu")
        npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
        npu.npu = SimpleNamespace()
        sys.modules["torch_npu"] = npu
        api = SimpleNamespace(
            Stream=self.Stream,
            Event=self.Event,
            current_stream=lambda *args: self.queue.current,
            stream=self.stream,
            current_device=lambda: 0,
            device_count=lambda: 1,
            set_device=lambda *_: None,
            synchronize=lambda: self.queue.drain(),
            is_available=lambda: False,
            is_current_stream_capturing=lambda: False,
            device=self.device,
        )
        self._set(torch, "npu", api)
        npu.npu = api

        class UnsupportedGraph:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("ACL graph execution is not emulated")

        api.NPUGraph = UnsupportedGraph
        api.ExternalEvent = self.Event
        api.get_soc_version = lambda: 250
        api.config = SimpleNamespace(allow_internal_format=False)
        api.default_stream = lambda *args: self.queue.default_stream
        api.mem_get_info = lambda *args: (8 * 1024**3, 8 * 1024**3)
        npu._C = SimpleNamespace(_NPUTaskGroupHandle=UnsupportedGraph)
        # Run the original Python policies without optional CPU JIT compilation.
        if "numba" not in sys.modules:
            numba = types.ModuleType("numba")
            numba.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)
            numba.njit = lambda function=None, **kwargs: (
                function
                if callable(function)
                else lambda implementation: implementation
            )
            numba.jit = numba.njit
            numba.prange = range
            numba.get_num_threads = lambda: 1
            numba.set_num_threads = lambda count: None
            sys.modules["numba"] = numba
        for name in (
            "Stream",
            "Event",
            "current_stream",
            "stream",
            "synchronize",
            "current_device",
        ):
            self._set(torch.cuda, name, getattr(api, name))
        self._set(torch.Tensor, "record_stream", lambda *_: None)
        self._set(torch.Tensor, "pin_memory", lambda value, *_args, **_kwargs: value)
        original_zeros = torch.zeros
        for name in (
            "tensor",
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "zeros_like",
            "ones_like",
            "full_like",
        ):
            original = getattr(torch, name)

            def unpinned(function):
                @wraps(function)
                def allocate(*args, **kwargs):
                    pinned = kwargs.pop("pin_memory", False)
                    shape = args[0] if args else kwargs.get("size")
                    if (
                        function is original_zeros
                        and isinstance(shape, int)
                        and shape >= 16 * 1024**2
                        and kwargs.get("dtype") in (torch.uint8, torch.int8)
                        and str(kwargs.get("device", "cpu")) == "cpu"
                        and "out" not in kwargs
                    ):
                        # Fresh anonymous mappings are zero-filled, like
                        # torch.zeros, but unused KV capacity stays unresident.
                        memory = mmap.mmap(-1, shape)
                        self.virtual_allocations.append(memory)
                        return torch.frombuffer(memory, dtype=kwargs["dtype"])
                    tensor = function(*args, **kwargs)
                    if pinned and tensor.device.type == "cpu":
                        self.queue.memory(tensor, tensor)
                    return tensor

                return allocate

            self._set(torch, name, unpinned(original))
        return self


def prepare_paths(root: Path) -> None:
    paths = [
        root,
        root.parent / "vllm",
        root.parent / "LMCache",
        root.parent / "vllm-ascend",
    ]
    sys.path[:0] = [str(path) for path in paths]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_VERSION", "0.18.0")
    if sys.platform == "win32":
        # Routed-expert capture is disabled. Fail if it is accidentally used.
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_UN = 2, 8

        def unsupported(*args):
            raise RuntimeError("Routed-expert file locking is outside this CPU test")

        fcntl.flock = unsupported
        sys.modules["fcntl"] = fcntl
