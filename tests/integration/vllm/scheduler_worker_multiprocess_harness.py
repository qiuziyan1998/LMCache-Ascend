# SPDX-License-Identifier: Apache-2.0
"""
Harness for scheduler/worker split-process integration tests.

Worker process: LMCacheEngine + LMCacheLookupServer + connector lifecycle hooks.
Scheduler process: LMCacheLookupClient (ZMQ IPC, same as production).
"""

# Standard
import copy
import multiprocessing as mp
import queue
import sys
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    SaveSpec,
    WorkerRetrieveState,
)
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)

PROMPT_LEN = 256
CHUNK_SIZE = 256
NUM_LAYERS = 4
HIDDEN_DIM = 1024
NUM_BLOCKS = 32
BLOCK_SIZE = 16


def npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _patch_loaded_lmc_ops_bindings() -> None:
    """Re-bind lmc_ops on modules imported before Ascend c_ops patch."""
    import lmcache_ascend.c_ops as ascend_c_ops

    for mod_name in (
        "lmcache.v1.memory_management",
        "lmcache.v1.lazy_memory_allocator",
        "lmcache.v1.gpu_connector.utils",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "lmc_ops"):
            mod.lmc_ops = ascend_c_ops


def _init_spawn_env() -> None:
    """Bootstrap LMCache path and PYTHONHASHSEED in spawned child processes."""
    import os
    import sys

    tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ascend_root = os.path.abspath(os.path.join(tests_dir, ".."))
    for path in (ascend_root, tests_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if npu_available():
        # Must run before any lmcache import so memory_management binds ascend c_ops.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    # Local
    from bootstrap import prepare_environment

    prepare_environment()
    if npu_available():
        import lmcache_ascend  # noqa: F401  # patches Ascend engine + connectors
        _patch_loaded_lmc_ops_bindings()


def _make_sparse_req_meta(
    req_id: str,
    *,
    can_load: bool = True,
    can_save: bool = False,
    is_sparse_decode: bool = True,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_ids=list(range(PROMPT_LEN)),
        slot_mapping=[torch.arange(PROMPT_LEN, dtype=torch.long)],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=PROMPT_LEN,
            can_load=can_load,
        ),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_sparse_decode=is_sparse_decode,
    )


def _make_normal_decode_meta(
    req_id: str,
    *,
    can_load: bool = False,
    can_save: bool = True,
    token_count: int = PROMPT_LEN,
    skip_leading_tokens: int = 0,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_ids=list(range(token_count)),
        slot_mapping=[torch.arange(token_count, dtype=torch.long)],
        load_spec=(
            LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=token_count,
                can_load=can_load,
            )
            if can_load
            else None
        ),
        save_spec=SaveSpec(
            skip_leading_tokens=skip_leading_tokens,
            can_save=can_save,
        ),
        is_sparse_decode=False,
    )


def _prepare_requests_for_npu_wait_for_save(
    requests: list[ReqMeta],
) -> list[ReqMeta]:
    """Ascend wait_for_save expects a CPU tensor slot_mapping when saving."""
    prepared: list[ReqMeta] = []
    for request in requests:
        req = copy.copy(request)
        if req.save_spec is not None and req.save_spec.can_save:
            slot_mapping = req.slot_mapping
            if isinstance(slot_mapping, list):
                slot_mapping = slot_mapping[0]
            req.slot_mapping = slot_mapping.detach().cpu().long()
        prepared.append(req)
    return prepared


def _create_worker_adapter(
    engine: Any,
    *,
    device: str,
    kvcaches: list[torch.Tensor],
) -> Any:
    adapter: LMCacheConnectorV1Impl
    try:
        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            LMCacheAscendConnectorV1Impl,
        )

        adapter = object.__new__(LMCacheAscendConnectorV1Impl)
        adapter._finished_req_ids_waiting_for_save = set()
        adapter._late_finished_sending = set()
        adapter._wait_for_save_done = True
    except ImportError:
        adapter = object.__new__(LMCacheConnectorV1Impl)

    adapter.kv_role = "kv_both"
    adapter.use_layerwise = False
    adapter.store_async = False
    adapter.enable_blending = False
    adapter.device = device
    adapter._lmcache_chunk_size = CHUNK_SIZE
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._worker_retrieve_state = {}
    adapter._layerwise_save_storers = {}
    adapter.kv_caches = {
        f"layer_{idx}": layer_kv for idx, layer_kv in enumerate(kvcaches)
    }
    adapter.config = SimpleNamespace(
        get_extra_config_value=lambda key, default=False: default
    )
    adapter.async_loading = False
    return adapter


def _run_wait_for_save(
    adapter: LMCacheConnectorV1Impl, requests: list[ReqMeta]
) -> None:
    """Base wait_for_save pin/unpin logic (CPU path)."""
    metadata_obj = LMCacheConnectorMetadata(requests=requests)
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata_obj)
    LMCacheConnectorV1Impl.wait_for_save(adapter)


def _run_ascend_wait_for_save(
    adapter: LMCacheConnectorV1Impl, requests: list[ReqMeta]
) -> None:
    """Ascend NPU wait_for_save (store + pin defer/unpin)."""
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )

    if not isinstance(adapter, LMCacheAscendConnectorV1Impl):
        raise RuntimeError(
            "NPU wait_for_save requires LMCacheAscendConnectorV1Impl"
        )
    metadata_obj = LMCacheConnectorMetadata(
        requests=_prepare_requests_for_npu_wait_for_save(requests)
    )
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata_obj)
    mock_pp = MagicMock()
    mock_pp.is_last_rank = True
    with patch(
        "lmcache.integration.vllm.vllm_v1_adapter.get_pp_group",
        return_value=mock_pp,
    ):
        adapter.wait_for_save()


def _prime_npu_gpu_connector(connector: Any, kvcaches: list[torch.Tensor]) -> None:
    """Register paged KV caches on the NPU connector before store/retrieve."""
    connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
    connector._initialize_pointers(kvcaches)


def _rebuild_lookup_client_config_and_metadata(
    scheduler_init: dict[str, Any],
) -> tuple[Any, Any]:
    """Rebuild config/metadata in scheduler child (must match worker ZMQ paths)."""
    # Third Party
    from tests.v1.utils import create_test_config, create_test_metadata

    config = create_test_config(
        instance_id=scheduler_init["instance_id"],
        local_cpu=True,
        max_local_cpu_size=0.5,
        rpc_port=scheduler_init["rpc_port"],
    )
    metadata = create_test_metadata(engine_id=scheduler_init["engine_id"])
    metadata.kv_connector_extra_config = config.extra_config
    return config, metadata


def _worker_process(
    instance_id: str,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
    npu_worker: bool,
) -> None:
    _init_spawn_env()

    try:
        _worker_run(instance_id, cmd_queue, result_queue, ready_queue, npu_worker)
    except Exception as exc:
        ready_queue.put({"error": f"{type(exc).__name__}: {exc}"})
        raise


def _worker_run(
    instance_id: str,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
    npu_worker: bool,
) -> None:
    use_npu = npu_worker and npu_available()
    if npu_worker and not use_npu:
        raise RuntimeError("NPU worker requested but torch.npu is not available")

    if use_npu:
        torch.npu.set_device(0)
        _patch_loaded_lmc_ops_bindings()

    # Third Party
    from tests.v1.utils import (
        create_test_config,
        create_test_metadata,
        generate_kv_cache_paged_list_tensors,
        generate_tokens,
        recover_engine_states,
    )

    if use_npu:
        from tests.v1.utils import create_npu_connector

    config = create_test_config(
        instance_id=instance_id,
        local_cpu=True,
        max_local_cpu_size=0.5,
        rpc_port=0,
    )
    metadata = create_test_metadata()
    metadata.engine_id = "mp_test_engine"
    metadata.kv_connector_extra_config = config.extra_config

    device = "npu" if use_npu else "cpu"
    if use_npu:
        gpu_connector = create_npu_connector(HIDDEN_DIM, NUM_LAYERS)
    else:
        gpu_connector = MockGPUConnector(kv_shape=(4, 2, CHUNK_SIZE, 8, 128))

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id=instance_id,
        config=config,
        metadata=metadata,
        gpu_connector=gpu_connector,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )
    engine.post_init()

    transport = LookupClientFactory._create_zmq_server_transport(metadata)
    server = LMCacheLookupServer(engine, metadata, transport)

    kvcaches = generate_kv_cache_paged_list_tensors(
        NUM_BLOCKS,
        device,
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
    )
    adapter = _create_worker_adapter(engine, device=device, kvcaches=kvcaches)

    # CPU tokens/slot_mapping: NPU connector copies slot indices on store_stream.
    tokens = generate_tokens(PROMPT_LEN, "cpu", fixed=True)
    slot_mapping = torch.arange(PROMPT_LEN, dtype=torch.long, device="cpu")
    store_kwargs: dict[str, Any] = {
        "kvcaches": kvcaches,
        "slot_mapping": slot_mapping,
    }
    if use_npu:
        _prime_npu_gpu_connector(engine.gpu_connector, kvcaches)
        connector = engine.gpu_connector
        with torch.npu.stream(connector.store_stream):
            slot_mapping_npu = slot_mapping.to(
                device=kvcaches[0].device,
                dtype=torch.long,
                non_blocking=True,
            )
        connector.store_stream.synchronize()
        store_kwargs["slot_mapping_npu"] = slot_mapping_npu
    store_mask = torch.ones(PROMPT_LEN, dtype=torch.bool)
    engine.store(
        tokens,
        mask=store_mask,
        **store_kwargs,
    )
    recover_engine_states(engine)

    ready_queue.put(
        {
            "instance_id": instance_id,
            "prompt_tokens": tokens.tolist(),
            "engine_id": metadata.engine_id,
            "rpc_port": config.extra_config.get("lmcache_rpc_port", 0),
            "npu_enabled": use_npu,
        }
    )

    try:
        while True:
            try:
                cmd = cmd_queue.get(timeout=120)
            except queue.Empty:
                result_queue.put({"ok": False, "error": "worker timeout"})
                continue

            op = cmd.get("op")
            try:
                if op == "shutdown":
                    break

                if op == "wait_for_save":
                    path = cmd.get("path", "base")
                    requests = cmd["requests"]
                    if path == "npu":
                        _run_ascend_wait_for_save(adapter, requests)
                    else:
                        _run_wait_for_save(adapter, requests)
                    result_queue.put({"ok": True, "op": op, "path": path})
                    continue

                if op == "prune":
                    adapter._prune_worker_retrieve_state(set(cmd["active_req_ids"]))
                    result_queue.put({"ok": True, "op": op})
                    continue

                if op == "drop_state":
                    adapter._drop_worker_retrieve_state(cmd["req_id"])
                    result_queue.put({"ok": True, "op": op})
                    continue

                if op == "preempt":
                    req_ids = set(cmd["req_ids"])
                    if hasattr(adapter, "handle_preemptions"):
                        adapter.handle_preemptions(req_ids)
                    else:
                        for rid in req_ids:
                            adapter._drop_worker_retrieve_state(rid)
                    result_queue.put({"ok": True, "op": op})
                    continue

                if op == "get_finished":
                    adapter.get_finished({cmd["req_id"]})
                    result_queue.put({"ok": True, "op": op})
                    continue

                if op == "save_worker_state":
                    req_id = cmd["req_id"]
                    adapter._worker_retrieve_state[req_id] = WorkerRetrieveState(
                        cached_keys=[["mp-key"]],
                        cached_starts=[0],
                        cached_ends=[PROMPT_LEN],
                        metadata_warm=True,
                        location="LocalCPUBackend",
                        token_count=PROMPT_LEN,
                    )
                    result_queue.put({"ok": True, "op": op})
                    continue

                if op == "has_lookup_pins":
                    req_id = cmd["req_id"]
                    has_pins = req_id in engine.lookup_pins
                    result_queue.put({"ok": True, "has_pins": has_pins, "op": op})
                    continue

                if op == "worker_state_keys":
                    keys = list(adapter._worker_retrieve_state.keys())
                    result_queue.put({"ok": True, "keys": keys, "op": op})
                    continue

                result_queue.put({"ok": False, "error": f"unknown op {op}"})
            except Exception as exc:
                result_queue.put(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}", "op": op}
                )
    finally:
        server.close()
        try:
            engine.close()
        except AttributeError:
            pass
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)


def _scheduler_process(
    scheduler_init: dict[str, Any],
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
) -> None:
    _init_spawn_env()

    config, metadata = _rebuild_lookup_client_config_and_metadata(scheduler_init)
    transport = LookupClientFactory._create_zmq_client_transport(config, metadata)
    client = LMCacheLookupClient(config, metadata, transport)
    ready_queue.put({"ok": True})

    try:
        while True:
            try:
                cmd = cmd_queue.get(timeout=120)
            except queue.Empty:
                result_queue.put({"ok": False, "error": "scheduler timeout"})
                continue

            op = cmd.get("op")
            if op == "shutdown":
                break

            if op == "lookup":
                token_ids = cmd["token_ids"]
                lookup_id = cmd["lookup_id"]
                hits = client.lookup(token_ids, lookup_id=lookup_id)
                result_queue.put(
                    {"ok": True, "op": op, "hits": hits, "lookup_id": lookup_id}
                )
                continue

            result_queue.put({"ok": False, "error": f"unknown op {op}"})
    finally:
        client.close()


@dataclass
class MultiprocessHarness:
    worker: mp.Process
    scheduler: mp.Process
    worker_cmd: mp.Queue
    worker_res: mp.Queue
    scheduler_cmd: mp.Queue
    scheduler_res: mp.Queue
    instance_id: str
    prompt_tokens: list[int]
    npu_enabled: bool = False

    def worker_wait_for_save(
        self,
        requests: list[ReqMeta],
        *,
        path: str = "base",
        timeout: float = 60,
    ) -> dict[str, Any]:
        return self.worker_call(
            {"op": "wait_for_save", "requests": requests, "path": path},
            timeout=timeout,
        )

    def worker_call(self, cmd: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        self.worker_cmd.put(cmd)
        return self._wait_result(self.worker_res, timeout)

    def scheduler_call(
        self,
        cmd: dict[str, Any],
        timeout: float = 60,
    ) -> dict[str, Any]:
        self.scheduler_cmd.put(cmd)
        return self._wait_result(self.scheduler_res, timeout)

    @staticmethod
    def _wait_result(result_queue: mp.Queue, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = result_queue.get(timeout=1)
                if msg.get("ok"):
                    return msg
                raise RuntimeError(msg.get("error", "worker/scheduler error"))
            except queue.Empty:
                continue
        raise TimeoutError("multiprocess harness result timeout")

    def shutdown(self) -> None:
        try:
            self.worker_cmd.put({"op": "shutdown"})
            self.scheduler_cmd.put({"op": "shutdown"})
        except Exception:
            pass
        self.worker.join(timeout=30)
        self.scheduler.join(timeout=30)
        if self.worker.is_alive():
            self.worker.terminate()
        if self.scheduler.is_alive():
            self.scheduler.terminate()


def start_multiprocess_harness(npu_worker: bool = False) -> MultiprocessHarness:
    ctx = mp.get_context("spawn")
    instance_id = f"mp_sched_worker_{uuid.uuid4().hex[:8]}"

    worker_cmd: mp.Queue = ctx.Queue()
    worker_res: mp.Queue = ctx.Queue()
    scheduler_cmd: mp.Queue = ctx.Queue()
    scheduler_res: mp.Queue = ctx.Queue()
    worker_ready: mp.Queue = ctx.Queue()
    scheduler_ready: mp.Queue = ctx.Queue()

    worker = ctx.Process(
        target=_worker_process,
        args=(instance_id, worker_cmd, worker_res, worker_ready, npu_worker),
        name="lmcache-mp-worker",
    )
    worker.start()

    init_info = worker_ready.get(timeout=120)
    if init_info.get("error"):
        worker.join(timeout=10)
        raise RuntimeError(f"worker failed to start: {init_info['error']}")

    scheduler_init = {
        "instance_id": init_info["instance_id"],
        "engine_id": init_info["engine_id"],
        "rpc_port": init_info["rpc_port"],
    }

    scheduler = ctx.Process(
        target=_scheduler_process,
        args=(scheduler_init, scheduler_cmd, scheduler_res, scheduler_ready),
        name="lmcache-mp-scheduler",
    )
    scheduler.start()
    scheduler_ready.get(timeout=120)

    time.sleep(0.5)

    return MultiprocessHarness(
        worker=worker,
        scheduler=scheduler,
        worker_cmd=worker_cmd,
        worker_res=worker_res,
        scheduler_cmd=scheduler_cmd,
        scheduler_res=scheduler_res,
        instance_id=init_info["instance_id"],
        prompt_tokens=init_info["prompt_tokens"],
        npu_enabled=init_info.get("npu_enabled", False),
    )
