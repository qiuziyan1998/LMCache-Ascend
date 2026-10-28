# SPDX-License-Identifier: Apache-2.0
"""Run production prefill DMA generators on a deterministic CPU device model.

No vLLM model, NPU runtime, service, or diagnostic recorder is required. The
production Python methods are compiled unchanged from the adjacent checkouts;
only device operators and framework construction are replaced. Distinct values
identify every request, layer, token and KV component. This checks ordering and
addressing, not numerical equivalence of real attention kernels.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import ctypes
import heapq
import importlib.util
import json
import math
import random
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import patch

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SENTINEL = -12345


class OrderingError(AssertionError):
    """A real consumer observed incorrect or unfinished simulated device data."""


@dataclass(eq=False)
class Command:
    number: int
    label: str
    action: Callable[[], None]
    pending: int = 0
    successors: list[Command] = field(default_factory=list)
    done: bool = False
    ready_at: float = 0.0


class Device:
    """Execute a legal, reproducible interleaving of independent stream FIFOs."""

    def __init__(
        self,
        seed: int,
        fault: str = "none",
        debug_sync: bool = False,
        schedule: str = "random",
    ):
        self.rng = random.Random(seed)
        self.seed, self.fault, self.debug_sync = seed, fault, debug_sync
        self.schedule = schedule
        self.clock = 0.0
        self.ready: list[tuple[float, int, Command]] = []
        self.commands: list[Command] = []
        self.history: list[str] = []
        self.current: Stream | None = None
        self.active = "setup"
        self.counts = dict(load=0, save=0, compute=0, publish=0, collectives=0)
        self.ranges: list[tuple[int, int, object]] = []
        self.range_starts: list[int] = []

    def memory(self, tensor: torch.Tensor, owner: object) -> None:
        """Register legal allocations so bad raw DMA addresses fail safely."""
        start = tensor.data_ptr()
        index = bisect.bisect_left(self.range_starts, start)
        self.range_starts.insert(index, start)
        self.ranges.insert(
            index, (start, start + tensor.numel() * tensor.element_size(), owner)
        )

    def check_address(self, ptr: int, size: int) -> None:
        index = bisect.bisect_right(self.range_starts, ptr) - 1
        if index >= 0:
            start, end, owner = self.ranges[index]
            if start <= ptr and ptr + size <= end:
                if hasattr(owner, "is_valid") and not owner.is_valid():
                    raise OrderingError(f"DMA uses released CPU page: {self.active}")
                return
        raise OrderingError(
            f"DMA pointer is outside a registered allocation: {ptr}, {size}"
        )

    def enqueue(self, label: str, action: Callable[[], None], deps=()) -> Command:
        command = Command(len(self.commands), label, action)
        self.commands.append(command)
        for dep in set(dep for dep in deps if dep is not None):
            if not dep.done:
                command.pending += 1
                dep.successors.append(command)
        if not command.pending:
            self.make_ready(command)
        return command

    def make_ready(self, command: Command) -> None:
        # Transfers, compute and transport progress independently. The wide
        # latency range prevents submission order from becoming completion order.
        delay = self.rng.uniform(0.05, 15.0)
        if command.label.startswith("publish"):
            delay *= 0.1
        if self.schedule == "dma-lag" and command.label.startswith(("save", "load")):
            delay *= 50
        if self.schedule == "compute-lag" and command.label.startswith("compute"):
            delay *= 50
        command.ready_at = self.clock + delay
        heapq.heappush(self.ready, (command.ready_at, command.number, command))

    def step(self) -> None:
        if not self.ready:
            unfinished = [c.label for c in self.commands if not c.done][:8]
            raise OrderingError(f"Dependency deadlock: {unfinished}")
        self.clock, _, command = heapq.heappop(self.ready)
        self.active = command.label
        try:
            command.action()
        except Exception as error:
            raise OrderingError(
                f"seed={self.seed} command={command.number} {command.label}: {error}\n"
                + "\n".join(self.history[-12:])
            ) from error
        command.done = True
        self.history.append(f"{command.number}: {command.label}")
        if len(self.history) > 32:
            del self.history[:16]
        for successor in command.successors:
            successor.pending -= 1
            if not successor.pending:
                self.make_ready(successor)
        command.successors.clear()

    def until(self, command: Command | None) -> None:
        while command is not None and not command.done:
            self.step()

    def drain(self) -> None:
        while self.ready:
            self.step()
        if any(not c.done for c in self.commands):
            raise OrderingError("Unfinished device commands")

    def dma(self, rows, to_cpu: bool) -> None:
        stream = self.current
        assert stream is not None
        # The C++ binding copies the Python rows before asynchronously using
        # addresses. Do the same here; do not capture a mutable planner list.
        rows = tuple(tuple(int(value) for value in row) for row in rows)
        label = f"{'save' if to_cpu else 'load'} {self.active} {stream.name}"

        def copy() -> None:
            for dst, src, size in rows:
                self.check_address(src, size)
                self.check_address(dst, size)
                ctypes.memmove(dst, src, size)

        stream.submit(label, copy)
        self.counts["save" if to_cpu else "load"] += 1
        if self.debug_sync:
            stream.synchronize()


class Stream:
    def __init__(self, device: Device, name: str):
        self.device, self.name, self.tail = device, name, None

    def submit(
        self, label: str, action: Callable[[], None] = lambda: None, deps=()
    ) -> Command:
        self.tail = self.device.enqueue(label, action, (self.tail, *deps))
        return self.tail

    def wait_event(self, event: Event) -> None:
        if self.device.fault == "skip-load-wait" and self.name.endswith("compute"):
            return
        if (
            self.device.fault == "skip-bank-handoff"
            and "bank" in self.name
            and event.stream.name.endswith("compute")
        ):
            return
        self.submit(f"wait {self.name}", deps=(event.tail,))

    def wait_stream(self, stream: Stream) -> None:
        self.submit(f"wait-stream {self.name} <- {stream.name}", deps=(stream.tail,))

    def synchronize(self) -> None:
        self.device.until(self.tail)


class Event:
    def __init__(self, **_kwargs):
        self.tail: Command | None = None
        self.stream: Stream | None = None

    def record(self, stream: Stream | None = None) -> None:
        assert stream is not None
        self.stream, self.tail = stream, stream.tail

    def query(self) -> bool:
        return self.tail is None or self.tail.done

    def synchronize(self) -> None:
        assert self.stream is not None
        self.stream.device.until(self.tail)


class NPU:
    Event = Event

    def __init__(self, device: Device):
        self.device = device

    def current_stream(self, *_args) -> Stream:
        assert self.device.current is not None
        return self.device.current

    @contextmanager
    def stream(self, stream: Stream) -> Iterator[None]:
        previous = self.device.current
        self.device.current = stream
        try:
            yield
        finally:
            self.device.current = previous


class TorchCPU:
    def __init__(self, device: Device):
        self.npu = self.cuda = NPU(device)

    def __getattr__(self, name):
        return getattr(torch, name)


class Page:
    """CPU allocator lease, with the same pointers used by the real DMA planner."""

    def __init__(self, tensor: torch.Tensor, device: Device):
        self.tensor, self.refs = tensor, 1
        self.metadata = SimpleNamespace(fmt="sim")
        device.memory(tensor, self)

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    def is_valid(self) -> bool:
        return self.refs > 0

    def ref_count_up(self) -> None:
        if not self.is_valid():
            raise OrderingError("Retaining a released page")
        self.refs += 1

    def ref_count_down(self) -> None:
        self.refs -= 1
        if self.refs < 0:
            raise OrderingError("CPU page released twice")


class IndexerState:
    """CPU equivalent of the native immutable IndexerC8State descriptor."""

    def __init__(self, keys, scales=None, slot_factor=1):
        self.keys, self.scales, self.slot_factor = keys, scales, slot_factor


class MergedPage(Page):
    """One CPU allocation containing every layer of a token chunk."""

    def __init__(self, tokens, group, layers, device, is_c8):
        bytes_per_layer = tokens * (18 if group == 0 else 4)
        raw = torch.full((bytes_per_layer * layers,), 0xA5, dtype=torch.uint8)
        super().__init__(raw, device)
        self.valid_tokens, self.num_layers = tokens, layers
        self.group_prefix_sum = tuple(i * bytes_per_layer for i in range(layers + 1))
        self._base_data_ptr = raw.data_ptr()
        self.views = tuple(
            raw[i * bytes_per_layer : (i + 1) * bytes_per_layer].view(
                torch.uint8 if group == 1 and is_c8(i) else torch.int16
            )
            for i in range(layers)
        )

    @property
    def valid(self):
        return self.is_valid()

    def get_tensor(self, layer):
        return self.views[layer]


def layer_tensor(obj, layer):
    return obj.layer_tensor(layer) if isinstance(obj, MergedPage) else obj.tensor


def revision(path: Path) -> str:
    """Record which checkout supplied production code for this run."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def production_methods(
    path: Path, class_name: str, names: tuple[str, ...], scope: dict
) -> dict:
    """Compile complete methods unchanged, avoiding hardware-only imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - methods.keys()
    if missing:
        raise ValueError(
            f"Production methods disappeared from {path}: {sorted(missing)}"
        )
    # Decorators measure hardware/profiler state; their wrapped bodies are the
    # unchanged production implementation executed by the CPU backend.
    static = set()
    for method in methods.values():
        if any(
            isinstance(item, ast.Name) and item.id == "staticmethod"
            for item in method.decorator_list
        ):
            static.add(method.name)
        method.decorator_list = []
    source = ast.parse("from __future__ import annotations")
    source.body.extend(methods.values())
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), scope)
    for name in static:
        scope[name].sim_static = True
    return {name: scope[name] for name in names}


def bind(obj, methods: dict) -> None:
    for name, method in methods.items():
        setattr(
            obj,
            name,
            method if getattr(method, "sim_static", False) else MethodType(method, obj),
        )


@dataclass
class Metadata:
    requests: list


def load_store_protocol(path: Path):
    """Compile the real cache engine's two-yield per-layer store protocol."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendLMCacheEngine"
    )
    store = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "store_layer"
    )
    block = next(
        n
        for n in ast.walk(store)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "deferred_layerwise_put"
        and n.body
        and isinstance(n.body[0], ast.AnnAssign)
        and ast.unparse(n.body[0].target) == "persisted_layers"
    )
    wrapper = ast.parse(
        "def store_protocol(mem_obj_generator, num_layers, "
        "publish_completed_layer):\n    pass"
    ).body[0]
    wrapper.body = block.body
    module = ast.parse("from __future__ import annotations")
    module.body.append(wrapper)
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["store_protocol"]


def load_metadata_getter(path: Path):
    """Keep the runner's actual fixed bank selection and view cache."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    getter = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_get_block_table_and_slot_mapping"
    )
    factory = ast.parse(
        "def make_getter(self, num_reqs, num_tokens, kv_cache_groups):\n"
        "    num_reqs_padded, num_tokens_padded = num_reqs, num_tokens\n"
        "    prefill_bank_views = {}\n"
    ).body[0]
    factory.body.extend(
        [getter, ast.parse("return _get_block_table_and_slot_mapping").body[0]]
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(factory)
    scope = dict(
        torch=torch, np=np, EncoderOnlyAttentionSpec=type("UnusedEncoderSpec", (), {})
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["make_getter"]


class BlockTable:
    def __init__(self, table: torch.Tensor, slots: torch.Tensor):
        self.table = table
        self.slot_mapping = SimpleNamespace(gpu=slots)

    def get_device_tensor(self) -> torch.Tensor:
        return self.table


def source_protocol(generator, sources, layers):
    """Supply CPU objects at the engine-to-NPU boundary, including layer zero."""
    generator.send(sources(0))
    yield None
    for layer in range(1, layers):
        generator.send(sources(layer))
        yield None
    next(generator)
    yield torch.ones(1, dtype=torch.bool)


def require_equal(actual: torch.Tensor, expected: torch.Tensor, where: str) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).reshape(-1).nonzero()[0].item()
        raise OrderingError(
            f"{where}: element={mismatch}, "
            f"actual={actual.reshape(-1)[mismatch].item()}, "
            f"expected={expected.reshape(-1)[mismatch].item()}"
        )


@dataclass
class Request:
    number: int
    total: int
    completed: int = 0
    blocks: dict[int, list[list[int]]] = field(default_factory=dict)
    bundles: dict[int, list[list[int]]] = field(default_factory=dict)
    pages: dict[tuple[int, int], list[Page]] = field(default_factory=dict)
    retired_pages: list[Page] = field(default_factory=list)
    remote: dict[tuple[int, int], list[torch.Tensor]] = field(default_factory=dict)
    publications: list[Command] = field(default_factory=list)
    req_id: str = ""
    is_sparse_decode: bool = False
    block_allocation_mode: str = "prefill_child"
    save_spec: Any = field(
        default_factory=lambda: SimpleNamespace(
            can_save=True, can_save_latent=True, can_save_indexer=True
        )
    )


class Simulation:
    """Wire the actual allocator, address plans and DMA generators together."""

    def __init__(self, args, seed: int):
        self.args, self.device = (
            args,
            Device(seed, args.fault, args.debug_sync, args.schedule),
        )
        self.rng = random.Random(seed)
        self.layer_counts = (args.layers, args.indexer_layers)
        self.index_at = sorted(
            {
                round(i * (args.layers - 2) / max(1, args.indexer_layers - 1))
                for i in range(args.indexer_layers)
            }
        )
        if len(self.index_at) != args.indexer_layers:
            raise ValueError("Indexer layer count exceeds the main layers")
        self.methods_used: list[str] = []
        page_methods = production_methods(
            args.lmcache / "lmcache/v1/memory_management.py",
            "LayerPageMemoryObj",
            ("layer_tensor", "layer_data_ptr", "layer_size_bytes"),
            {},
        )
        for name, method in page_methods.items():
            setattr(MergedPage, name, method)
        self.dma_module = module_from_file(
            "prefill_sim_dma", ROOT / "lmcache_ascend/v1/npu_connector/layerwise_dma.py"
        )
        self.pool_module = module_from_file(
            "prefill_sim_pool", args.vllm / "vllm/v1/core/dsa_shared_pool.py"
        )
        self.worker_module = module_from_file(
            "prefill_sim_views", args.ascend / "vllm_ascend/worker/dsa_shared_pool.py"
        )
        self.requests = [
            Request(
                i,
                args.tokens
                + self.rng.randrange(-args.cache_chunk + 1, args.cache_chunk),
                req_id=f"request-{i}",
            )
            for i in range(args.requests)
        ]
        self.raw_owners: list[torch.Tensor] = []
        self.workers = []
        self.streams: dict[tuple[int, int, int], Stream] = {}
        self.build_allocator()
        self.build_production()
        self.build_callbacks()
        self.build_runner()
        self.checked_tokens = 0
        self.batch_count = 0

    def is_c8(self, layer: int) -> bool:
        return self.args.indexer_format == "mixed-c8" and layer % 3 != 1

    def build_allocator(self) -> None:
        p = self.pool_module
        # 8 NOPE + 1 ROPE versus 2 indexer components preserves the production
        # 9-page shared layout, using int16 instead of BF16 for exact comparison.
        bs = self.args.block_size
        parent = p.DSASharedBlockLayout(
            bs * 18,
            bs * 4,
            self.args.requests * 4 + 8,
            bundle_multiplier=2,
            k_nope_dim=8,
            k_pe_dim=1,
            indexer_dim=2,
        )
        self.pool = p.PrefillLayerBundlePool(
            p.DSASharedBundleAllocator(parent),
            self.args.layers,
            bank_count=self.args.banks,
        )
        self.layout = self.pool.layout
        raw_bytes = self.layout.slot_count * self.layout.bundle_page_size_bytes
        for rank in range(self.args.tp):
            raw = torch.full((raw_bytes,), 0xA5, dtype=torch.uint8)
            self.device.memory(raw, raw)
            self.raw_owners.append(raw)
            views = self.worker_module.reshape_dsa_shared_pool_raw
            latent = views(raw, torch.int16, bs, 1, 8, 1, 2, is_indexer=False)
            indexer = views(raw, torch.int16, bs, 1, 8, 1, 2, is_indexer=True)
            if self.args.indexer_format == "mixed-c8":
                scale = torch.zeros((indexer[0].numel() * 2,), dtype=torch.uint8)
                self.device.memory(scale, scale)
                self.raw_owners.append(scale)
                c8 = views(
                    raw,
                    torch.int16,
                    bs,
                    1,
                    8,
                    1,
                    2,
                    is_indexer=True,
                    indexer_dtype=torch.int8,
                    indexer_scale=scale,
                    shared_indexer_dtype=torch.int16,
                )
                indexers = [
                    c8 if self.is_c8(layer) else indexer
                    for layer in range(self.args.indexer_layers)
                ]
            else:
                indexers = [indexer] * self.args.indexer_layers
            self.workers.append(
                SimpleNamespace(
                    rank=rank,
                    compute=Stream(self.device, f"r{rank}/compute"),
                    caches=([latent] * self.args.layers, indexers),
                )
            )

    def allocate(self, request: Request, end: int) -> None:
        for group, owner in enumerate(
            (
                self.pool_module.DSASharedBlockOwner.LATENT,
                self.pool_module.DSASharedBlockOwner.INDEXER,
            )
        ):
            maps = request.blocks.setdefault(
                group, [[] for _ in range(self.args.banks)]
            )
            bundles = request.bundles.setdefault(
                group, [[] for _ in range(self.args.banks)]
            )
            needed = math.ceil(end / self.args.block_size)
            count = self.pool.logical_bundle_count_for_blocks(
                owner, max(0, needed - len(maps[0]))
            )
            for bank, added in enumerate(self.pool.allocate_banks(owner, count)):
                bundles[bank].extend(added)
                for bundle in added:
                    maps[bank].extend(self.layout.block_ids_for_bundle(owner, bundle))

    def build_production(self) -> None:
        device, dma = self.device, self.dma_module
        fmt = SimpleNamespace(
            MLA_LATENT=SimpleNamespace(value=1), DSA_INDEX=SimpleNamespace(value=2)
        )
        connector_path = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
        names = (
            "batched_to_gpu",
            "batched_from_gpu",
            "_layerwise_prefill_transfer_state",
            "_layerwise_prefill_transfer_generation",
            "_check_layerwise_prefill_transfer_generation",
            "_set_layerwise_prefill_bank_count",
            "_layerwise_prefill_bank",
            "record_layerwise_prefill_bank_use",
            "_require_layerwise_prefill_bank_use",
            "wait_for_layerwise_prefill_load",
            "layerwise_prefill_store_fences",
            "wait_for_layerwise_prefill_source_publication",
            "release_layerwise_prefill_dma_cache",
            "_retain_unfenced_layerwise_prefill_load",
            "_run_sparse_direct_kv_transfer_layer",
            "_run_c8_indexer_bootstrap",
            "_prepare_prefill_c8_packets",
            "_submit_prefill_c8_packet",
        )
        scope = dict(
            torch=TorchCPU(device),
            KVCacheFormat=fmt,
            _DENSE_DIRECT_LOAD_DISABLE=False,
            _DENSE_DIRECT_STORE_DISABLE=False,
            LayerPageMemoryObj=MergedPage,
            LayerPageSource=type("UnusedLayerSource", (), {}),
            _layer_source_memory_objs=lambda objects, layer: objects,
            _layer_memory_tensor=layer_tensor,
            _layer_memory_host_ptr=lambda obj, layer: (
                obj.layer_data_ptr(layer)
                if isinstance(obj, MergedPage)
                else obj.data_ptr
            ),
            prefill_start_timing_enabled=lambda: False,
            prefill_reuse_debug_enabled=lambda rank: False,
            _mtp_dw_deep_diag_enabled=lambda: False,
            _sparse_h2d_python_stage=lambda *_: None,
            sparse_mla_dsa_batched_direct_kv_transfer=self.sparse_copy,
            logger=SimpleNamespace(
                isEnabledFor=lambda *_: False, debug=lambda *_: None
            ),
            lmc_ops=SimpleNamespace(
                layerwise_prefill_dma_copy=device.dma,
                IndexerC8State=IndexerState,
                indexer_c8_transfer_prepared=self.c8_copy,
            ),
            bind_copy_addresses=dma.bind_copy_addresses,
            bind_incremental_copy_addresses=dma.bind_incremental_copy_addresses,
            prepare_source_addresses=dma.prepare_source_addresses,
            plan_block_id_ranges_incremental=dma.plan_block_id_ranges_incremental,
        )
        tree = ast.parse(connector_path.read_text(encoding="utf-8"))
        planner = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_prefill_dma_plans"
        )
        module = ast.parse("from __future__ import annotations")
        module.body.append(planner)
        exec(
            compile(ast.fix_missing_locations(module), str(connector_path), "exec"),
            scope,
        )
        methods = production_methods(
            connector_path, "VLLMPagedMemLayerwiseNPUConnector", names, scope
        )
        self.methods_used.extend(f"connector.{name}" for name in names)
        for worker in self.workers:
            obj = SimpleNamespace(
                _prefill_worker_id=worker.rank,
                _layerwise_prefill_dma=True,
                kvcaches=worker.caches[0],
                _prefill_dma_bound_loads={},
                _layerwise_prefill_load_owners={},
                _layerwise_prefill_load_request_events={},
                use_gpu=False,
                load_stream=Stream(device, f"r{worker.rank}/metadata"),
                store_stream=Stream(device, f"r{worker.rank}/store"),
                initialize_kvcaches_ptr=lambda **_: None,
                _is_mla_dsa_format=lambda _: True,
                _layerwise_token_major=lambda _: True,
                _expected_memory_format=lambda _: "sim",
                _sparse_lmc_host_interleaved=lambda _: True,
                _check_layerwise_transfer_invariants=lambda **_: None,
                _stream_context_or_null=lambda stream: NPU(device).stream(stream),
            )
            obj._expected_group_layers = lambda group: self.layer_counts[group]
            obj.checkpoint_plane_widths = lambda group: (8, 1) if group == 0 else (2,)
            obj._lmc_plane_num_tokens = lambda tensor, group: (
                tensor.numel()
                // (9 if group == 0 else 4 if tensor.dtype == torch.uint8 else 2)
            )
            obj._lazy_initialize_buffer_with_staging = lambda caches, kv_group, **_: (
                SimpleNamespace(
                    indexer_c8=kv_group == 1 and self.args.indexer_format == "mixed-c8",
                    layer_slot_factors=tuple(
                        2 if self.is_c8(layer) else 1
                        for layer in range(self.args.indexer_layers)
                    ),
                    layer_token_bytes=(4,) * self.args.indexer_layers,
                    kv_device="cpu",
                    kv_format=fmt.MLA_LATENT if kv_group == 0 else fmt.DSA_INDEX,
                    vllm_two_major=False,
                    k_hidden_dims=8 if kv_group == 0 else 0,
                    v_hidden_dims=1 if kv_group == 0 else 0,
                    dsa_hidden_dims=2 if kv_group else 0,
                )
            )
            obj.prefill_dma_cycles = dict(
                enumerate(
                    dma.build_group_cycles(
                        worker.caches[0][0],
                        worker.caches[1][0],
                        self.args.cache_chunk,
                        2,
                        (8, 1),
                    )
                )
            )
            # Logical indexer IDs refer to BF16 capacity even on C8 layers.
            bf16_indexer = next(
                (row for row in worker.caches[1] if row[0].dtype == torch.int16),
                worker.caches[1][0],
            )
            obj.prefill_dma_cycles = dict(
                enumerate(
                    dma.build_group_cycles(
                        worker.caches[0][0],
                        bf16_indexer,
                        self.args.cache_chunk,
                        2,
                        (8, 1),
                    )
                )
            )
            obj.indexer_c8_layout = SimpleNamespace(mixed=True, is_c8=self.is_c8)
            obj._resolve_registered_cpu_source_device_ptr = lambda obj, layer_id, **_: (
                obj.layer_data_ptr(layer_id)
                if isinstance(obj, MergedPage)
                else obj.data_ptr
            )
            obj._layerwise_prefill_dma_stream = lambda group, bank, rank=worker.rank: (
                self.streams.setdefault(
                    (rank, group, bank), Stream(device, f"r{rank}/g{group}/bank{bank}")
                )
            )
            bind(obj, methods)
            if device.fault == "stale-publication":
                obj.layerwise_prefill_store_fences = MethodType(stale_publication, obj)
            worker.connector = obj

    def c8_copy(
        self, state, pointers, offsets, counts, slots, capacity, to_cpu, **_kwargs
    ) -> None:
        """Queued CPU implementation of the native mixed C8 packet layout."""
        stream = self.device.current
        assert stream is not None

        def copy() -> None:
            key_bytes = state.keys.shape[-1] * state.keys.element_size()
            starts, lengths = offsets.tolist(), counts.tolist()
            if _kwargs.get("fixed_chunks"):
                starts = [i * capacity for i in range(pointers.numel())]
                lengths = [min(capacity, slots.numel() - start) for start in starts]
            for pointer, offset, count in zip(
                pointers.tolist(), starts, lengths, strict=True
            ):
                for token in range(count):
                    slot = int(slots[offset + token])
                    if slot < 0:
                        continue
                    physical = slot + slot // 128 * (state.slot_factor - 1) * 128
                    pairs = [
                        (
                            pointer + token * key_bytes,
                            state.keys.data_ptr() + physical * key_bytes,
                            key_bytes,
                        )
                    ]
                    if state.scales is not None:
                        pairs.append(
                            (
                                pointer + count * key_bytes + token * 2,
                                state.scales.data_ptr() + physical * 2,
                                2,
                            )
                        )
                    for cpu, npu, size in pairs:
                        self.device.check_address(cpu, size)
                        self.device.check_address(npu, size)
                        ctypes.memmove(
                            cpu if to_cpu else npu, npu if to_cpu else cpu, size
                        )

        stream.submit(
            f"{'save' if to_cpu else 'load'} C8 {self.device.active} {stream.name}",
            copy,
        )
        self.device.counts["save" if to_cpu else "load"] += 1
        if self.device.debug_sync:
            stream.synchronize()

    def sparse_copy(
        self, cpu_tensors, planes, slots, selected, chunk_size, total_tokens, *_args
    ) -> None:
        """CPU implementation of the native selected-KV copy, still queued."""
        stream = self.device.current
        assert stream is not None
        widths = tuple(plane.shape[-1] for plane in planes)

        def copy() -> None:
            for row, position in enumerate(selected.tolist()):
                if position < 0 or position >= total_tokens:
                    continue
                chunk, token = divmod(position, chunk_size)
                source = cpu_tensors[chunk]
                capacity = source.numel() // sum(widths)
                cursor = 0
                for plane, width in zip(planes, widths, strict=True):
                    values = source[
                        cursor + token * width : cursor + (token + 1) * width
                    ]
                    plane.reshape(-1, width)[slots[row]].copy_(values)
                    cursor += capacity * width

        stream.submit("D selected KV load", copy)
        self.device.counts["load"] += 1

    def decode_first_token(self, request: Request) -> None:
        """Exercise the production D sparse consumer wait with selected CPU KV."""
        device = self.device
        compute, load = Stream(device, "D/compute"), Stream(device, "D/load")
        device.current = compute
        connector = self.workers[0].connector
        connector._sparse_direct_pointer_cache_signature = lambda **_: ()
        connector._get_or_create_sparse_direct_layer_state = lambda **_: (None, None)
        connector._validate_sparse_fixed_chunk_coverage = lambda chunks, size, tokens: (
            None
        )
        connector._dense_direct_dummy_metadata_tensor = lambda: torch.empty(
            0, dtype=torch.long
        )
        # CPU tensors need no caching-allocator stream registration. Preserve
        # all their lifetimes explicitly while the fake NPU executes callbacks.
        with patch.object(torch.Tensor, "record_stream", lambda *_: None):
            for group, layers in enumerate(self.layer_counts):
                if group == 1 and self.args.indexer_format == "mixed-c8":
                    self.decode_indexer(request, connector, compute)
                    continue
                widths = (8, 1) if group == 0 else (2,)
                for layer in range(layers):
                    cpu_tensors = request.remote[group, layer]
                    selected = torch.tensor(
                        sorted(
                            {
                                0,
                                request.total - 1,
                                request.total // 2,
                                *range(0, request.total, max(1, request.total // 17)),
                            }
                        ),
                        dtype=torch.long,
                    )
                    # Two speculative rows intentionally select common tokens.
                    selected = torch.cat((selected, selected.flip(0)))
                    slots = torch.arange(selected.numel())
                    planes = tuple(
                        torch.full(
                            (selected.numel(), width), SENTINEL, dtype=torch.int16
                        )
                        for width in widths
                    )
                    ptrs = torch.tensor([tensor.data_ptr() for tensor in cpu_tensors])
                    connector._run_sparse_direct_kv_transfer_layer(
                        kvcaches_ref=[planes] * layers,
                        kv_group=group,
                        layer_id=layer,
                        load_stream=load,
                        load_stream_idx=0,
                        current_stream=compute,
                        slot_mapping_packed=slots,
                        selected_token_idx=selected,
                        chunk_size=self.args.cache_chunk,
                        total_tokens=request.total,
                        chunk_ptrs_npu=ptrs,
                        sparse_kv_format=1 + group,
                        sparse_token_major=True,
                        sparse_vllm_two_major=False,
                        sparse_k_hidden_dims=8 if group == 0 else 0,
                        sparse_v_hidden_dims=1 if group == 0 else 0,
                        sparse_dsa_hidden_dims=2 if group == 1 else 0,
                        sparse_host_interleaved=True,
                        layer_tensors=cpu_tensors,
                        cpu_tensors=cpu_tensors,
                        req_id=request.req_id,
                    )
                    expected = self.expected(request, group, layer, 0, request.total)[
                        selected
                    ]

                    def verify(
                        planes=planes, expected=expected, group=group, layer=layer
                    ):
                        require_equal(
                            torch.cat(planes, dim=1),
                            expected,
                            f"D first decode {request.req_id} g{group} l{layer}",
                        )

                    compute.submit(
                        f"D compute {request.req_id} g{group} l{layer}", verify
                    )
        device.until(compute.tail)

    def decode_indexer(self, request, connector, compute) -> None:
        """Run the real D C8 full-prefix bootstrap for mixed indexer layers."""
        states = []
        for layer in range(self.args.indexer_layers):
            keys = torch.empty(
                (request.total, 2),
                dtype=torch.int8 if self.is_c8(layer) else torch.int16,
            )
            scales = (
                torch.empty((request.total, 1), dtype=torch.float16)
                if self.is_c8(layer)
                else None
            )
            for tensor in (keys, scales):
                if tensor is not None:
                    self.device.memory(tensor, tensor)
            states.append(IndexerState(keys, scales, 1))
        plan = SimpleNamespace(states=states)
        slots = torch.arange(request.total)
        for layer, state in enumerate(states):
            sources = request.remote[1, layer]
            for source in sources:
                self.device.memory(source, source)
            connector._run_c8_indexer_bootstrap(
                plan,
                torch.tensor([source.data_ptr() for source in sources]),
                layer,
                slots,
                request.total,
                self.args.cache_chunk,
                False,
            )
            expected = self.expected(request, 1, layer, 0, request.total)

            def verify(state=state, expected=expected, layer=layer):
                values = (
                    state.keys
                    if state.scales is None
                    else torch.cat((state.keys, state.scales), dim=1)
                )
                require_equal(
                    values, expected, f"D indexer bootstrap {request.req_id} l{layer}"
                )

            compute.submit(f"D indexer consume {request.req_id} l{layer}", verify)

    def build_callbacks(self) -> None:
        path = self.args.lmcache / "lmcache/integration/vllm/vllm_v1_adapter.py"
        names = (
            "_layerwise_layer_id_from_name",
            "_layerwise_has_indexer_model_layer",
            "_layerwise_required_wait_groups",
            "_layerwise_wait_should_advance",
            "_layerwise_prefill_transfer_layer_id",
            "_layerwise_prefill_load_position",
            "record_layerwise_prefill_bank_use",
            "_wait_for_layerwise_prefill_bank",
            "_advance_dense_layerwise_retriever",
            "_complete_layerwise_retrieve_group",
            "_advance_deferred_layerwise_prefill_load",
            "_submit_initial_layerwise_prefill_load",
            "submit_layerwise_prefill_load",
            "save_kv_layer",
            "finish_layerwise_prefill_save",
            "_store_result_from_yield",
            "_drain_layerwise_retrievers",
        )
        scope = dict(
            torch=TorchCPU(self.device),
            LMCacheConnectorMetadata=Metadata,
            LayerwiseStoreResult=type("UnusedStoreResult", (), {}),
            prefill_start_timing_enabled=lambda: False,
            serving_perf_enabled=lambda: False,
            logger=SimpleNamespace(debug=lambda *_: None, warning=lambda *_: None),
        )
        methods = production_methods(path, "LMCacheConnectorV1Impl", names, scope)
        self.methods_used.extend(f"adapter.{name}" for name in names)
        self.store_protocol = load_store_protocol(
            ROOT / "lmcache_ascend/v1/cache_engine.py"
        )
        self.methods_used.append("cache_engine.store_layer.deferred_layerwise_put")
        for worker in self.workers:
            bridge = SimpleNamespace(
                _layerwise_prefill_p_node=True,
                _layerwise_prefill_dma=True,
                _layerwise_prefill_bank_count=self.args.banks,
                num_layers=self.args.layers,
                _latent_layer_names=[
                    f"model.layers.{i}.attn" for i in range(self.args.layers)
                ],
                _indexer_layer_names=[
                    f"model.layers.{i}.indexer.k_cache" for i in self.index_at
                ],
                _indexer_model_layers=set(self.index_at),
                _is_dsa_two_groups=lambda: True,
                _layerwise_wait_group=lambda name: int("indexer" in name),
                _sparse_retrieve_state_guard=lambda *_: nullcontext(),
                _layerwise_prefill_slot_mapping=lambda *_: None,
                _validate_dense_retrieve_result=lambda *_args, **_kwargs: None,
                _finalize_worker_retrieve_state_from_metadata=lambda *_: None,
                _close_layerwise_retriever=lambda generator: generator.close(),
                _should_defer_latent_save_under_tp=lambda: False,
                _note_decode_window_save_seen=lambda *_: None,
                _layerwise_save_storer_key=lambda request, group: (
                    request.req_id,
                    "normal",
                    group,
                ),
                use_layerwise=True,
                kv_role="kv_producer",
                config=SimpleNamespace(dsa_two_groups=True),
                kv_caches={"sim": True},
                _kvcaches_list=[True],
                _kvcaches_for_group=lambda group, worker=worker: worker.caches[group],
                lmcache_engine=SimpleNamespace(gpu_connector=worker.connector),
            )
            bridge._layerwise_group_ordinals = tuple(
                {name: i for i, name in enumerate(row)}
                for row in (bridge._latent_layer_names, bridge._indexer_layer_names)
            )
            bridge._parent = SimpleNamespace(_connector_metadata=None)
            bridge._parent._get_connector_metadata = lambda bridge=bridge: (
                bridge._parent._connector_metadata
            )
            bind(bridge, methods)
            worker.bridge = bridge
        self.active_bridge = None
        sfa_scope = dict(
            torch=TorchCPU(self.device),
            maybe_save_kv_layer_in_layerwise_prefill_transfer_window=(
                self.save_callback
            ),
            maybe_save_kv_layer_outside_layerwise_prefill_transfer_window=lambda *_: (
                None
            ),
            maybe_submit_layerwise_prefill_load=lambda name: (
                self.active_bridge.submit_layerwise_prefill_load(name) or True
            ),
            maybe_finish_layerwise_prefill_save=lambda name: (
                self.active_bridge.finish_layerwise_prefill_save(name) or True
            ),
        )
        sfa_names = (
            "_submit_sfa_transfer_window_save_operations",
            "_submit_sfa_post_transfer_window_save_operations",
            "_submit_sfa_layerwise_transfer_window",
            "_finish_sfa_layerwise_transfer_window",
        )
        sfa_methods = production_methods(
            self.args.ascend / "vllm_ascend/attention/sfa_v1.py",
            "AscendSFAImpl",
            sfa_names,
            sfa_scope,
        )
        self.sfa = SimpleNamespace()
        bind(self.sfa, sfa_methods)
        self.methods_used.extend(f"sfa.{name}" for name in sfa_names)

    def save_callback(self, name, caches):
        self.active_bridge.save_kv_layer(name, caches, None)

    def build_runner(self) -> None:
        path = self.args.ascend / "vllm_ascend/worker/model_runner_v1.py"
        self.metadata_factory = load_metadata_getter(path)
        self.methods_used.append("runner._get_block_table_and_slot_mapping")

    def prepare_runner(self, worker, active, ranges) -> None:
        runner = SimpleNamespace(
            layerwise_prefill_p_node=True,
            layerwise_prefill_bank_count=self.args.banks,
            pcp_size=1,
            use_cp=False,
            device="cpu",
            model_config=SimpleNamespace(enable_return_routed_experts=False),
        )
        num_tokens = sum(ranges[r.number][1] - ranges[r.number][0] for r in active)
        tables = []
        for bank in range(self.args.banks):
            groups = []
            for group in range(2):
                width = max(len(r.blocks[group][bank]) for r in active)
                table = torch.zeros((len(active), width), dtype=torch.long)
                slot_rows = []
                for i, request in enumerate(active):
                    blocks = torch.tensor(request.blocks[group][bank])
                    table[i, : len(blocks)] = blocks
                    start, end = ranges[request.number]
                    pos = torch.arange(start, end)
                    slot_rows.append(
                        blocks[pos // self.args.block_size] * self.args.block_size
                        + pos % self.args.block_size
                    )
                groups.append(BlockTable(table, torch.cat(slot_rows)))
            tables.append(groups)
        runner.input_batch = SimpleNamespace(
            req_ids=[r.req_id for r in active],
            layerwise_prefill_block_tables=tables,
            block_table=tables[0],
        )
        worker.runner = runner
        worker.metadata_getter = self.metadata_factory(
            runner,
            len(active),
            num_tokens,
            [SimpleNamespace(kv_cache_spec=object())] * 2,
        )
        worker.request_row = {r.req_id: i for i, r in enumerate(active)}

    def expected(
        self, req: Request, group: int, layer: int, start: int, end: int
    ) -> torch.Tensor:
        width = 9 if group == 0 else 2
        positions = torch.arange(start, end, dtype=torch.int64)
        values = (
            positions[:, None] * 71
            + torch.arange(width) * 23
            + req.number * 997
            + layer * 313
            + group * 7901
        ) % 30000
        if group == 1 and self.is_c8(layer):
            keys = (values % 127).to(torch.float16)
            scales = ((positions + req.number * 3 + layer) % 31).to(torch.float16)[
                :, None
            ] / 16
            return torch.cat((keys, scales), dim=1)
        return values.to(torch.int16)

    def new_pages(
        self, request: Request, start: int, end: int
    ) -> tuple[list[int], list[int], dict[int, list[list[Page]]]]:
        # LMCache republishes a partial chunk as a new, longer object. It does
        # not append a second short object at an unaligned logical position.
        store_start = start // self.args.cache_chunk * self.args.cache_chunk
        starts = list(range(store_start, end, self.args.cache_chunk))
        ends = [min(s + self.args.cache_chunk, end) for s in starts]
        pages = {}
        for group, count in enumerate(self.layer_counts):
            rows = []
            merged = (
                [
                    MergedPage(e - s, group, count, self.device, self.is_c8)
                    for s, e in zip(starts, ends, strict=True)
                ]
                if self.args.cpu_objects == "merged"
                else None
            )
            for layer in range(count):
                c8 = group == 1 and self.is_c8(layer)
                row = (
                    merged
                    if merged is not None
                    else [
                        Page(
                            torch.full(
                                ((e - s) * (9 if group == 0 else 4 if c8 else 2),),
                                0xA5 if c8 else SENTINEL,
                                dtype=torch.uint8 if c8 else torch.int16,
                            ),
                            self.device,
                        )
                        for s, e in zip(starts, ends, strict=True)
                    ]
                )
                rows.append(row)
                previous = request.pages.setdefault((group, layer), [])
                keep = store_start // self.args.cache_chunk
                request.retired_pages.extend(previous[keep:])
                request.pages[group, layer] = previous[:keep] + row
            pages[group] = rows
        return starts, ends, pages

    def slots(
        self, request: Request, group: int, layer: int, start: int, end: int
    ) -> torch.Tensor:
        positions = torch.arange(start, end)
        bank = layer % self.args.banks
        blocks = torch.tensor(request.blocks[group][bank])
        slots = (
            blocks[positions // self.args.block_size] * self.args.block_size
            + positions % self.args.block_size
        )
        if group == 1 and self.is_c8(layer):
            slots += slots // 128 * 128
        return slots

    def compute(
        self, worker, request: Request, group: int, layer: int, start: int, end: int
    ) -> None:
        planes = worker.caches[group][layer]
        table, _ = worker.metadata_getter(group, layer % self.args.banks)
        positions = torch.arange(end)
        slots = (
            table[worker.request_row[request.req_id], positions // self.args.block_size]
            * self.args.block_size
            + positions % self.args.block_size
        )
        if group == 1 and self.is_c8(layer):
            slots += slots // 128 * 128
        require_equal(
            slots,
            self.slots(request, group, layer, 0, end),
            "runner physical bank mapping",
        )
        expected = self.expected(request, group, layer, 0, end)

        def run() -> None:
            actual = torch.cat(
                [plane.reshape(-1, plane.shape[-1])[slots[:start]] for plane in planes],
                dim=1,
            )
            require_equal(
                actual,
                expected[:start],
                f"r{worker.rank} {request.req_id} g{group} l{layer} "
                f"history [0,{start})",
            )
            cursor = 0
            for plane in planes:
                width = plane.shape[-1]
                plane.reshape(-1, width)[slots[start:end]] = expected[
                    start:end, cursor : cursor + width
                ].to(plane.dtype)
                cursor += width
            self.checked_tokens += end

        worker.compute.submit(
            f"compute r{worker.rank} {request.req_id} g{group} l{layer} "
            f"[{start},{end})",
            run,
        )
        self.device.counts["compute"] += 1

    def page_expected(
        self, req: Request, group: int, layer: int, start: int, end: int
    ) -> torch.Tensor:
        rows = self.expected(req, group, layer, start, end)
        if group == 1 and self.is_c8(layer):
            return torch.cat(
                (
                    rows[:, :2].to(torch.int8).reshape(-1).view(torch.uint8),
                    rows[:, 2:].contiguous().reshape(-1).view(torch.uint8),
                )
            )
        return (
            torch.cat((rows[:, :8].reshape(-1), rows[:, 8:].reshape(-1)))
            if group == 0
            else rows.reshape(-1)
        )

    def publish(
        self, request: Request, starts: list[int], ends: list[int], pages: dict
    ) -> Command:
        connector = self.workers[0].connector
        events = tuple(
            event
            for group in range(2)
            for event in connector.layerwise_prefill_store_fences(group)
        )

        def transfer() -> None:
            for group, rows in pages.items():
                for layer, row in enumerate(rows):
                    target = request.remote.setdefault((group, layer), [])
                    for start, end, page in zip(starts, ends, row, strict=True):
                        require_equal(
                            layer_tensor(page, layer),
                            self.page_expected(request, group, layer, start, end),
                            f"publish {request.req_id} g{group} l{layer} "
                            f"[{start},{end})",
                        )
                        index = start // self.args.cache_chunk
                        target.extend([None] * max(0, index + 1 - len(target)))
                        # Different chunk keys can complete out of order;
                        # final lookup selects the longest valid successor.
                        data = layer_tensor(page, layer)
                        if (
                            target[index] is None
                            or data.numel() >= target[index].numel()
                        ):
                            target[index] = data.clone()

        self.device.counts["publish"] += 1
        return self.device.enqueue(
            f"publish {request.req_id} [{starts[0]},{ends[-1]})",
            transfer,
            [e.tail for e in events],
        )

    def run(self) -> dict:
        with patch.object(torch.Tensor, "record_stream", lambda *_: None):
            return self.run_with_cpu_streams()

    def run_with_cpu_streams(self) -> dict:
        waiting = list(self.requests)
        active: list[Request] = []
        while waiting or active:
            while waiting and len(active) < self.args.batch_size:
                active.append(waiting.pop(0))
            self.run_batch(active)
            finished = [
                request for request in active if request.completed == request.total
            ]
            for request in finished:
                self.finish(request)
                active.remove(request)
        self.device.drain()
        return dict(
            seed=self.device.seed,
            status="passed",
            requests=len(self.requests),
            tp=self.args.tp,
            banks=self.args.banks,
            batches=self.batch_count,
            checked_tokens=self.checked_tokens,
            commands=len(self.device.commands),
            operations=self.device.counts,
            production_methods=self.methods_used,
            indexer_format=self.args.indexer_format,
            cpu_objects=self.args.cpu_objects,
            schedule=self.args.schedule,
            code_revisions={
                "LMCache-Ascend": revision(ROOT),
                "LMCache": revision(self.args.lmcache),
                "vllm-ascend": revision(self.args.ascend),
                "vllm": revision(self.args.vllm),
            },
            scope=(
                "CPU ordering/address/lifetime simulation; "
                "NPU kernels and ACL graph replay are not executed"
            ),
        )

    def run_batch(self, active: list[Request]) -> None:
        self.batch_count += 1
        self.rng.shuffle(active)
        ranges, new, previous_pages = {}, {}, {}
        remaining = self.args.prefill_chunk
        for request in active:
            size = min(
                request.total - request.completed,
                max(self.args.cache_chunk, remaining // (len(active) or 1)),
            )
            end = request.completed + size
            ranges[request.number] = (request.completed, end)
            self.allocate(request, end)
            previous_pages[request.number] = dict(request.pages)
            new[request.number] = self.new_pages(request, request.completed, end)
            remaining = max(0, remaining - size)
        loads, stores = {}, {}
        for worker in self.workers:
            self.device.current = worker.compute
            self.prepare_runner(worker, active, ranges)
            obj = worker.connector
            bridge = worker.bridge
            bridge.current_layer = 0
            bridge._layerwise_waited_groups = set()
            bridge._deferred_layerwise_prefill_drained_groups = set()
            bridge._deferred_layerwise_prefill_load_active = True
            bridge._initial_layerwise_prefill_load_submitted = False
            bridge._layerwise_save_storers = {}
            bridge._layerwise_prefill_prepared_storer_keys = set()
            bridge._layerwise_prefill_pending_store_finishes = {}
            bridge.layerwise_retrievers = []
            bridge._layerwise_requests = []
            bridge._layerwise_retriever_is_sparse = []
            bridge._parent._connector_metadata = Metadata(active)
            for request in active:
                start, end = ranges[request.number]
                starts, ends, pages = new[request.number]
                for group in range(2):
                    common = dict(
                        kvcaches=worker.caches[group],
                        slot_mapping=torch.empty(0, dtype=torch.long),
                        sync=True,
                        kv_group=group,
                        req_id=request.req_id,
                        layerwise_prefill_bank_count=self.args.banks,
                        prefill_dma_block_ids_by_bank=tuple(
                            tuple(row) for row in request.blocks[group]
                        ),
                        prefill_dma_block_size=self.args.block_size,
                    )
                    if worker.rank == 0:
                        storer = obj.batched_from_gpu(
                            pages[group],
                            starts,
                            ends,
                            deferred_layerwise_put=True,
                            layerwise_prefill_async_store=True,
                            **common,
                        )
                        next(storer)
                        storer = self.store_protocol(
                            storer, self.layer_counts[group], lambda *_: None
                        )
                        next(storer)
                        stores[request.number, group] = storer
                        key = bridge._layerwise_save_storer_key(request, group)
                        bridge._layerwise_save_storers[key] = storer
                        bridge._layerwise_prefill_prepared_storer_keys.add(key)
                    if start:
                        historical = previous_pages[request.number][group, 0]
                        load_starts, load_ends, cursor = [], [], 0
                        for page in historical:
                            load_starts.append(cursor)
                            cursor += obj._lmc_plane_num_tokens(
                                layer_tensor(page, 0), group
                            )
                            load_ends.append(cursor)
                        if cursor != start:
                            raise OrderingError(
                                "Historical CPU page coverage is incomplete"
                            )
                        if worker.rank == 0:
                            obj.wait_for_layerwise_prefill_source_publication(group)
                        sources = [
                            previous_pages[request.number][group, layer]
                            for layer in range(self.layer_counts[group])
                        ]
                        loader = obj.batched_to_gpu(
                            load_starts,
                            load_ends,
                            deferred_layerwise_get=True,
                            prefill_c8_memory_objs=sources,
                            **common,
                        )
                        next(loader)
                        loader = source_protocol(
                            loader,
                            lambda layer, sources=sources: sources[layer],
                            self.layer_counts[group],
                        )
                        next(loader)
                        loads[worker.rank, request.number, group] = loader
                if start:
                    bridge.layerwise_retrievers.append(
                        tuple(
                            loads[worker.rank, request.number, group]
                            for group in range(2)
                        )
                    )
                    bridge._layerwise_requests.append(request)
                    bridge._layerwise_retriever_is_sparse.append(False)
            bridge.submit_layerwise_prefill_load(-1)
        self.execute_layers(active, ranges, new, loads, stores)
        for loader in loads.values():
            loader.close()
        for storer in stores.values():
            for _ in storer:
                pass
        publications = []
        for request in active:
            starts, ends, pages = new[request.number]
            publications.append(self.publish(request, starts, ends, pages))
        # The engine must receive model output before scheduling its next step;
        # unrelated async CPU-page stores are deliberately left in flight.
        for worker in self.workers:
            self.device.until(worker.compute.tail)
        for request, publication in zip(active, publications, strict=True):
            start, end = ranges[request.number]
            request.completed = end
            request.publication = publication
            request.publications.append(publication)

    def execute_layers(self, active, ranges, new, loads, stores) -> None:
        pending = []
        for model_layer in range(self.args.layers):
            previous = self.transfer_window(pending) if pending else None
            groups = [(0, model_layer)]
            if model_layer in self.index_at:
                groups.insert(0, (1, self.index_at.index(model_layer)))
            for group, layer in groups:
                for worker in self.workers:
                    self.device.current = worker.compute
                    name = (
                        worker.bridge._latent_layer_names
                        if group == 0
                        else worker.bridge._indexer_layer_names
                    )[layer]
                    worker.bridge._wait_for_layerwise_prefill_bank(name, group)
                    for request in active:
                        start, end = ranges[request.number]
                        self.compute(worker, request, group, layer, start, end)
                    event = Event()
                    event.record(worker.compute)
                    worker.bridge.record_layerwise_prefill_bank_use(name, event)
            terminal = model_layer == self.args.layers - 1 or (
                self.args.mtp_layers and model_layer == self.args.layers - 2
            )
            final = None
            if terminal:
                # The last backbone attention and the one-layer MTP forward
                # submit their own save before projection; neither has a later
                # SFA callback in the same forward context.
                self.finish_window(previous)
                previous = None
                final = self.transfer_window(groups)
            # A simple collective preserves TP lockstep while device/DMA work
            # remains delayed. The operation computes identity, not a real GEMM.
            collective = self.device.enqueue(
                f"collective layer={model_layer}",
                lambda: None,
                [worker.compute.tail for worker in self.workers],
            )
            self.device.counts["collectives"] += 1
            for worker in self.workers:
                worker.compute.submit(
                    f"projection r{worker.rank} l{model_layer}", deps=(collective,)
                )
            self.finish_window(previous)
            self.finish_window(final)
            pending = [] if terminal else groups

    def finish_window(self, pending) -> None:
        if pending is not None:
            self.active_bridge = self.workers[0].bridge
            self.device.current = self.workers[0].compute
            operations, names = pending
            self.sfa._finish_sfa_layerwise_transfer_window(operations, names)

    def transfer_window(self, groups):
        pending = None
        for worker in self.workers:
            self.device.current = worker.compute
            bridge = worker.bridge
            operations = [
                (
                    bridge._latent_layer_names[layer]
                    if group == 0
                    else bridge._indexer_layer_names[layer],
                    worker.caches[group][layer],
                )
                for group, layer in sorted(groups)
            ]
            for group, layer in groups:
                if layer + 1 == self.layer_counts[group]:
                    name = (
                        bridge._latent_layer_names
                        if group == 0
                        else bridge._indexer_layer_names
                    )[layer]
                    bridge._advance_deferred_layerwise_prefill_load(
                        name, finish_current=True
                    )
            if worker.rank == 0:
                self.active_bridge = bridge
                names = self.sfa._submit_sfa_layerwise_transfer_window(operations)
                pending = operations, names
            else:
                for name, _ in operations:
                    bridge.submit_layerwise_prefill_load(name)
        return pending

    def finish(self, request: Request) -> None:
        for publication in request.publications:
            self.device.until(publication)
        # Check the persisted representation that a fresh D instance receives,
        # including every layer and the final non-full chunk.
        for (group, layer), tensors in request.remote.items():
            cursor = 0
            for tensor in tensors:
                count = tensor.numel() // (
                    9 if group == 0 else 4 if tensor.dtype == torch.uint8 else 2
                )
                require_equal(
                    tensor,
                    self.page_expected(request, group, layer, cursor, cursor + count),
                    f"D input {request.req_id} g{group} l{layer}",
                )
                cursor += count
            if cursor != request.total:
                raise OrderingError("D input does not cover the prompt")
        self.decode_first_token(request)
        for worker in self.workers:
            worker.connector.release_layerwise_prefill_dma_cache(request.req_id)
        owners = {id(page): page for pages in request.pages.values() for page in pages}
        owners.update((id(page), page) for page in request.retired_pages)
        for page in owners.values():
            page.ref_count_down()
        for group, owner in enumerate(
            (
                self.pool_module.DSASharedBlockOwner.LATENT,
                self.pool_module.DSASharedBlockOwner.INDEXER,
            )
        ):
            self.pool.free(
                owner, [bundle for bank in request.bundles[group] for bundle in bank]
            )


def stale_publication(self, group):
    """Inject an early-layer event in place of the current bank's final save.

    With fixed bank assignment, max(layer) after a fully drained group now
    selects its actual final save. Use min(layer) to keep this negative test
    exercising publication before D2H completion.
    """
    chosen = {}
    for (kv_group, bank, layer), (
        generation,
        event,
    ) in self._layerwise_prefill_save_done_events.items():
        if (
            kv_group == group
            and generation == self._layerwise_prefill_transfer_generation(group)
        ):
            if bank not in chosen or layer < chosen[bank][0]:
                chosen[bank] = (layer, event)
    return tuple(chosen[bank][1] for bank in sorted(chosen))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The default runs 16 requests on 4 simulated TP ranks with 79 cache "
            "layers (including one MTP layer), 22 indexer layers, 4 banks and mixed "
            "C8. Use --schedule dma-lag/compute-lag and --seeds N to vary device "
            "interleavings. Fault modes intentionally fail; adding --debug-sync "
            "to stale-publication demonstrates how synchronous dumps mask that race."
        ),
    )
    result.add_argument("--vllm", type=Path, default=ROOT.parent / "vllm")
    result.add_argument("--ascend", type=Path, default=ROOT.parent / "vllm-ascend")
    result.add_argument("--lmcache", type=Path, default=ROOT.parent / "LMCache")
    result.add_argument("--requests", type=int, default=16)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--tp", type=int, default=4)
    result.add_argument("--banks", type=int, default=4)
    result.add_argument("--layers", type=int, default=79)
    result.add_argument("--mtp-layers", type=int, choices=(0, 1), default=1)
    result.add_argument("--indexer-layers", type=int, default=22)
    result.add_argument("--tokens", type=int, default=640)
    result.add_argument("--prefill-chunk", type=int, default=512)
    result.add_argument("--cache-chunk", type=int, default=128)
    result.add_argument("--block-size", type=int, default=128)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--seeds", type=int, default=1)
    result.add_argument(
        "--fault",
        choices=("none", "skip-load-wait", "skip-bank-handoff", "stale-publication"),
        default="none",
    )
    result.add_argument("--debug-sync", action="store_true")
    result.add_argument(
        "--schedule", choices=("random", "dma-lag", "compute-lag"), default="random"
    )
    result.add_argument(
        "--indexer-format", choices=("bf16", "mixed-c8"), default="mixed-c8"
    )
    result.add_argument("--cpu-objects", choices=("merged", "flat"), default="merged")
    result.add_argument("--output", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if (
        min(
            args.requests,
            args.batch_size,
            args.tp,
            args.banks,
            args.layers,
            args.indexer_layers,
            args.tokens,
            args.prefill_chunk,
            args.cache_chunk,
            args.block_size,
            args.seeds,
        )
        <= 0
    ):
        raise SystemExit("Sizes must be positive")
    if args.tokens < args.cache_chunk or args.prefill_chunk < args.cache_chunk:
        raise SystemExit("tokens and prefill-chunk must be at least cache-chunk")
    if args.indexer_layers >= args.layers:
        raise SystemExit("indexer-layers must be smaller than total cache layers")
    if args.indexer_format == "mixed-c8" and (
        args.block_size != 128 or args.indexer_layers < 2
    ):
        raise SystemExit(
            "mixed-c8 requires block-size=128 and at least two indexer layers"
        )
    results = []
    try:
        for seed in range(args.seed, args.seed + args.seeds):
            report = Simulation(args, seed).run()
            results.append(report)
            print(
                "[PREFILL_CPU_SIM] "
                + json.dumps(
                    {k: v for k, v in report.items() if k != "production_methods"}
                ),
                flush=True,
            )
    except OrderingError as error:
        report = dict(status="failed", seed=seed, error=str(error), fault=args.fault)
        results.append(report)
        print("[PREFILL_CPU_SIM] " + json.dumps(report), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
