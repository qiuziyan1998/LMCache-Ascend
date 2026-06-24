# SPDX-License-Identifier: Apache-2.0
"""
Harness for scheduler/worker split-process integration tests.

Worker process: LMCacheEngine + LMCacheLookupServer + connector lifecycle hooks.
Scheduler process: LMCacheLookupClient (ZMQ IPC, same as production).
"""

# Standard
import multiprocessing as mp
import os
import queue
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
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
    # Local
    from bootstrap import prepare_environment

    prepare_environment()


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


def _create_worker_adapter(engine: Any) -> Any:
    try:
        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            LMCacheAscendConnectorV1Impl,
        )

        adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    except ImportError:
        # First Party
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        adapter = object.__new__(LMCacheConnectorV1Impl)

    adapter.kv_role = "kv_both"
    adapter.use_layerwise = False
    adapter.store_async = False
    adapter.device = "cpu"
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._worker_retrieve_state = {}
    adapter._layerwise_save_storers = {}
    adapter.kv_caches = {"layer0": torch.zeros(1)}
    adapter.config = SimpleNamespace(
        get_extra_config_value=lambda key, default=False: default
    )
    adapter.async_loading = False
    return adapter


def _rebuild_lookup_client_config_and_metadata(scheduler_init: dict[str, Any]) -> tuple[Any, Any]:
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
) -> None:
    _init_spawn_env()

    try:
        _worker_run(instance_id, cmd_queue, result_queue, ready_queue)
    except Exception as exc:
        ready_queue.put({"error": f"{type(exc).__name__}: {exc}"})
        raise


def _worker_run(
    instance_id: str,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
) -> None:
    # Third Party
    from tests.v1.utils import (
        create_test_config,
        create_test_metadata,
        generate_kv_cache_paged_list_tensors,
        generate_tokens,
        recover_engine_states,
    )

    config = create_test_config(
        instance_id=instance_id,
        local_cpu=True,
        max_local_cpu_size=0.5,
        rpc_port=0,
    )
    metadata = create_test_metadata()
    metadata.engine_id = "mp_test_engine"
    metadata.kv_connector_extra_config = config.extra_config

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
    adapter = _create_worker_adapter(engine)

    tokens = generate_tokens(PROMPT_LEN, "cpu", fixed=True)
    kvcaches = generate_kv_cache_paged_list_tensors(
        32,
        "cpu",
        block_size=16,
        num_layers=4,
    )
    slot_mapping = torch.arange(PROMPT_LEN, dtype=torch.long)
    store_mask = torch.ones(PROMPT_LEN, dtype=torch.bool)
    engine.store(
        tokens,
        mask=store_mask,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
    )
    recover_engine_states(engine)

    ready_queue.put(
        {
            "instance_id": instance_id,
            "prompt_tokens": tokens.tolist(),
            "engine_id": metadata.engine_id,
            "rpc_port": config.extra_config.get("lmcache_rpc_port", 0),
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
            if op == "shutdown":
                break

            if op == "wait_for_save":
                requests = cmd["requests"]
                metadata_obj = LMCacheConnectorMetadata(requests=requests)
                adapter._parent = SimpleNamespace(
                    _get_connector_metadata=lambda: metadata_obj
                )
                adapter.wait_for_save()
                result_queue.put({"ok": True, "op": op})
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

            if op == "request_finished":
                # Third Party
                from vllm.v1.request import RequestStatus

                req = SimpleNamespace(
                    request_id=cmd["req_id"],
                    status=RequestStatus.FINISHED,
                )
                adapter.request_finished(req, [])
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
    finally:
        server.close()
        engine.close()
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

    def worker_call(self, cmd: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        self.worker_cmd.put(cmd)
        return self._wait_result(self.worker_res, timeout)

    def scheduler_call(self, cmd: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
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


def start_multiprocess_harness() -> MultiprocessHarness:
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
        args=(instance_id, worker_cmd, worker_res, worker_ready),
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
    )
