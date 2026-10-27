# SPDX-License-Identifier: Apache-2.0
"""Hardware substitutions for executing the imported vLLM/LMCache code on CPU."""

from __future__ import annotations

import ctypes
import importlib.machinery
import os
from pathlib import Path
import sys
import threading
import types
from contextlib import contextmanager
from types import SimpleNamespace

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
                    return super().step()

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
                    return super().wait_event(event)

            def synchronize(self):
                with hardware.lock:
                    return super().synchronize()

        class TorchEvent(Event):
            def record(self, stream=None):
                with hardware.lock:
                    super().record(stream or hardware.queue.current)

            def synchronize(self):
                with hardware.lock:
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
        tensor = torch.empty(size, dtype=torch.uint8)
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

    def dma(self, copies, to_cpu):
        # Keep raw descriptors immutable, just as pybind converts them before
        # returning to Python. Device effects happen only when the queue runs.
        rows = tuple(tuple(int(x) for x in row) for row in copies)
        stream = self.queue.current
        with self.lock:

            def copy():
                for dst, src, size in rows:
                    self.queue.check_address(src, size)
                    self.queue.check_address(dst, size)
                    ctypes.memmove(dst, src, size)

            stream.submit("save native D2H" if to_cpu else "load native H2D", copy)
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

        def copy():
            for target, token in zip(
                slots.flatten().tolist(), selected.flatten().tolist(), strict=True
            ):
                if token < 0 or token >= total_tokens or target < 0:
                    continue
                chunk, offset = divmod(token, chunk_size)
                tokens = min(chunk_size, total_tokens - chunk * chunk_size)
                source = int(pointers[chunk])
                prefix = 0
                for cache, width in zip(caches, widths, strict=True):
                    dst = cache.data_ptr() + target * width * element
                    src = source + (prefix * tokens + offset * width) * element
                    self.queue.check_address(src, width * element)
                    self.queue.check_address(dst, width * element)
                    ctypes.memmove(dst, src, width * element)
                    prefix += width

        with self.lock:
            self.queue.current.submit("native sparse H2D", copy)
            self.queue.counts["load"] += 1

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

        def copy():
            for chunk, pointer in enumerate(pointers.tolist()):
                start = (
                    chunk * fixed_chunk_size
                    if fixed_chunk_size
                    else int(offsets[chunk])
                )
                count = (
                    min(fixed_chunk_size, total_tokens - start)
                    if fixed_chunk_size
                    else int(sizes[chunk])
                )
                prefix = 0
                for cache, width in zip(caches, widths, strict=True):
                    for token in range(count):
                        slot = int(slots[start + token])
                        if slot < 0:
                            continue
                        offset = (
                            token * sum(widths) + prefix
                            if interleaved
                            else prefix * count + token * width
                        )
                        cpu = int(pointer) + offset * element
                        npu = cache.data_ptr() + slot * width * element
                        self.queue.check_address(cpu, width * element)
                        self.queue.check_address(npu, width * element)
                        ctypes.memmove(
                            cpu if direction else npu,
                            npu if direction else cpu,
                            width * element,
                        )
                    prefix += width

        with self.lock:
            self.queue.current.submit("save dense" if direction else "load dense", copy)
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
                    for key, row, lengths in zip(keys, pointers, sizes, strict=True):
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
                return [0] * len(keys)

            def batch_get_into_multi_buffers(self, keys, pointers, sizes):
                statuses = []
                with hardware.lock:
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
                            ctypes.memmove(pointer, data[offset : offset + size], size)
                            offset += size
                        hardware.remote_calls.append(("get", key, len(data)))
                        statuses.append(len(data))
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

        for name, directory in (
            ("lmcache_ascend", self.root / "lmcache_ascend"),
            ("vllm_ascend", self.root.parent / "vllm-ascend/vllm_ascend"),
            (
                "vllm_ascend.patch.platform",
                self.root.parent / "vllm-ascend/vllm_ascend/patch/platform",
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
    if sys.platform == "win32":
        # Routed-expert capture is disabled. Fail if it is accidentally used.
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_UN = 2, 8

        def unsupported(*args):
            raise RuntimeError("Routed-expert file locking is outside this CPU test")

        fcntl.flock = unsupported
        sys.modules["fcntl"] = fcntl
