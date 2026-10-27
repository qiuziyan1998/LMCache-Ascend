# SPDX-License-Identifier: Apache-2.0
"""Run the vLLM engine and LMCache Python control path with CPU hardware.

This is a scheduling/ownership test, not a numerical model or ACL emulator.
The production modules are imported normally; only device, model execution,
process RPC and network transport are substituted.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import traceback
from types import SimpleNamespace

from cpu_vllm_runtime import CPUHardware, prepare_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=500 * 1024)
    parser.add_argument("--batch-tokens", type=int, default=16 * 1024)
    parser.add_argument("--cache-chunk-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--connector-only", dest="native_runner", action="store_false")
    parser.set_defaults(native_runner=True)
    parser.add_argument(
        "--fault",
        choices=(
            "none",
            "skip-load-wait",
            "skip-save-wait",
            "early-ready-event",
            "corrupt-remote",
        ),
        default="none",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    kv_width = 9
    for name in (
        "banks",
        "tp_size",
        "requests",
        "layers",
        "decode_tokens",
        "rounds",
        "batch_tokens",
        "context_tokens",
        "cache_chunk_tokens",
        "max_num_seqs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    tp_size = args.tp_size
    root = Path(__file__).resolve().parents[1]
    prepare_paths(root)
    os.environ.update(
        VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE="true",
        VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS=str(args.banks),
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_SHARED_POOL="1",
        LMCACHE_PREFILL_STORE_DEBUG="0",
        VLLM_LOGGING_LEVEL="WARNING",
        LMCACHE_TRACK_USAGE="false",
        LMCACHE_LOG_LEVEL="WARNING",
        PYTHONHASHSEED="0",
    )
    hardware = CPUHardware(root, args.seed).install()
    hardware.queue.fault = args.fault
    hardware.install_mooncake()
    import torch

    # Tiny deterministic kernels should not create large OpenMP thread teams.
    torch.set_num_threads(1)
    from vllm.config import (
        ModelConfig,
        VllmConfig,
        SchedulerConfig,
        DeviceConfig,
        CacheConfig,
        CompilationConfig,
        KVTransferConfig,
        ParallelConfig,
    )
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.abstract import Executor
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.distributed.kv_transfer import kv_transfer_state
    from vllm.forward_context import set_forward_context
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.attention.utils import record_layerwise_prefill_bank_use
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic as LMCacheConnectorV1,
    )
    from vllm_ascend.patch.platform.patch_kv_cache_interface import (
        AscendMLAAttentionSpec,
    )
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMPagedMemLayerwiseNPUConnector,
    )
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )
    from lmcache.integration.vllm import vllm_v1_adapter as adapter
    from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
    from vllm_ascend.worker.dsa_shared_pool import reshape_dsa_shared_pool_raw

    services = {}
    broadcasts = {}
    stats = {
        "checked_rows": 0,
        "mixed_batches": 0,
        "P_steps": 0,
        "D_steps": 0,
        "decode_tokens": 0,
        "prefill_chunks": [],
    }
    layers, block_size = args.layers, 128
    max_length = (
        args.context_tokens
        + (args.requests - 1) * 37
        + args.rounds * (args.decode_tokens + 21)
        + block_size
    )
    max_length = (max_length + block_size - 1) // block_size * block_size
    cache_blocks = max(
        64, (max_length // block_size + 4) * min(args.requests, args.max_num_seqs)
    )
    index_at = tuple(sorted(set(range(0, layers, 4)) | {layers - 1}))
    # The production capacity preflight reserves the metadata's maximum head
    # width for both groups and both CPU planes, including indexer objects.
    cpu_bytes = max_length * args.requests * (layers + len(index_at)) * kv_width * 4
    lmconfig = LMCacheEngineConfig.from_defaults(
        chunk_size=args.cache_chunk_tokens,
        local_cpu=True,
        max_local_cpu_size=max(0.1, cpu_bytes * 1.3 / 1024**3 + 0.03),
        use_layerwise=True,
        dsa_two_groups=True,
        enable_sparse_attention=True,
        save_unfull_chunk=True,
        # Full-prefix checking on CPU can outlast the accelerator timeout.
        pin_timeout_sec=3600,
        enable_shared_cpu_cache=True,
        remote_url="mooncakestore://127.0.0.1:50051",
        extra_config={
            "save_only_first_rank": True,
            "mooncake_layer_merged_page_objects": True,
            "save_chunk_meta": False,
            "local_hostname": "cpu-runtime",
            "metadata_server": "cpu",
            "master_server_address": "cpu:50051",
            "mooncake_page_first_multi_buffer": True,
            # CPU simulation time is unrelated to the production network timeout.
            "transfer_timeout": 300,
        },
    )
    lmconfig.store_async = True
    lmconfig.store_async_max_queue_size = 0

    class LocalLookup(LookupClientInterface):
        def __init__(self, engine_id):
            self.engine_id = engine_id
            self.cache = {}

        def lookup(self, token_ids, lookup_id, request_configs=None):
            result = services[self.engine_id, 0].lookup(
                token_ids, lookup_id=lookup_id, request_configs=request_configs
            )
            self.cache[lookup_id] = result
            return result

        def lookup_cache(self, lookup_id):
            return self.cache.get(lookup_id, -1)

        def clear_lookup_status(self, lookup_id):
            self.cache.pop(lookup_id, None)

        def close(self):
            pass

        def supports_producer_reuse(self):
            return True

    class CPUServiceFactory(VllmServiceFactory):
        def get_or_create_metadata(self):
            if self.metadata is None:
                rank = self.vllm_config.parallel_config.rank
                self.metadata = LMCacheMetadata(
                    model_name="cpu-identity",
                    world_size=args.tp_size,
                    local_world_size=args.tp_size,
                    worker_id=rank,
                    local_worker_id=rank,
                    kv_dtype=torch.float16,
                    kv_shape=(layers, 1, args.cache_chunk_tokens, 1, kv_width),
                    use_mla=True,
                    role=self.role,
                    chunk_size=args.cache_chunk_tokens,
                    engine_id=self.vllm_config.kv_transfer_config.engine_id,
                    max_model_len=max_length,
                    runtime_kv_group_layer_counts=self.runtime_kv_group_layer_counts,
                    runtime_kv_group_layer_names=self.runtime_kv_group_layer_names,
                )
            return self.metadata

        def get_engine_instance_id(self):
            return self.vllm_config.kv_transfer_config.engine_id

        def get_or_create_lmcache_engine(self):
            if self.role == "scheduler":
                return None
            metadata = self.get_or_create_metadata()
            gpu = VLLMPagedMemLayerwiseNPUConnector(
                kv_width,
                layers,
                False,
                chunk_size=args.cache_chunk_tokens,
                dtype=torch.float16,
                device=torch.device("cpu"),
                use_mla=True,
                dsa_two_groups=True,
            )
            gpu.runtime_kv_group_layer_counts = metadata.runtime_kv_group_layer_counts
            messages = broadcasts.setdefault(self.get_engine_instance_id(), [])
            self.read_cursor = 0

            def broadcast(obj, src):
                if metadata.worker_id == src:
                    messages.append(copy.deepcopy(obj))
                    return obj
                assert self.read_cursor < len(messages), (
                    "Collective receive preceded rank-0 send"
                )
                value = copy.deepcopy(messages[self.read_cursor])
                self.read_cursor += 1
                return value

            engine = AscendLMCacheEngine(
                self.lmcache_config,
                metadata,
                ChunkedTokenDatabase(self.lmcache_config, metadata),
                gpu,
                broadcast,
                broadcast,
            )
            services[self.get_engine_instance_id(), metadata.worker_id] = engine
            return engine

        def maybe_create_lookup_client(self):
            return LocalLookup(self.get_engine_instance_id())

        def maybe_create_lookup_server(self):
            return None

        def maybe_create_offload_server(self):
            return None

        def maybe_create_runtime_plugin_launcher(self):
            return None

        def maybe_create_internal_api_server(self, lmcache_manager):
            return None

        def maybe_create_prometheus_logger(self):
            return None

        def maybe_create_health_monitor(self, lmcache_manager):
            return None

    adapter.VllmServiceFactory = CPUServiceFactory
    adapter.lmcache_get_or_create_config = lambda: lmconfig
    adapter.LMCacheConnectorV1Impl = LMCacheAscendConnectorV1Impl
    # The worker rank is a process/RPC property, independent of KV decisions.
    import vllm.distributed.parallel_state as parallel_state

    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    adapter.get_tensor_model_parallel_rank = lambda: 0

    class CPUIdentityModel(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()

        def embed_input_ids(self, input_ids):
            return input_ids

        def forward(self, input_ids, positions, **kwargs):
            return input_ids

        def compute_logits(self, hidden_states):
            return hidden_states

    ModelRegistry.register_model("CPUIdentityModel", CPUIdentityModel)

    class CPUExecutor(Executor):
        def _init_executor(self):
            self.lengths = {}
            self.computed = {}
            self.tokens = {}
            self.blocks = {}
            self.worker = None
            self.model = CPUIdentityModel(vllm_config=self.vllm_config)

        def collective_rpc(
            self, method, timeout=None, args=(), kwargs=None, non_block=False
        ):
            if method == "get_kv_cache_spec":
                specs = {
                    f"model.layers.{i}.self_attn.attn": AscendMLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=kv_width,
                        sparse_head_dim=(kv_width - 1, 1),
                        dtype=torch.float16,
                    )
                    for i in range(layers)
                }
                specs.update(
                    {
                        (
                            f"model.layers.{i}.self_attn.indexer.k_cache"
                        ): AscendMLAAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=1,
                            head_size=2,
                            sparse_head_dim=(2,),
                            dtype=torch.float16,
                        )
                        for i in index_at
                    }
                )
                return [specs for _ in range(tp_size)]
            if method == "determine_available_memory":
                return [16 * 1024**3] * tp_size
            if method == "get_supported_tasks":
                return [("generate",)]
            if method == "get_kv_connector_handshake_metadata":
                return [None]
            if method == "initialize_from_config":
                self.workers = []
                for rank, cache_config in enumerate(args[0]):
                    self._initialize_worker(rank, cache_config)
                self.worker = self.workers[0]
                self.kv_aggregator = KVOutputAggregator.from_connector(
                    self.worker, tp_size
                )
                return [None] * len(self.workers)
            if method == "compile_or_warm_up_model":
                return [0.0] * len(self.workers)
            raise NotImplementedError(f"Unsubstituted executor RPC: {method}")

        def _initialize_worker(self, rank, cache_config):
            self.cache_config = cache_config
            worker_config = copy.deepcopy(self.vllm_config)
            worker_config.parallel_config.rank = rank
            self.worker = LMCacheConnectorV1(
                worker_config, KVConnectorRole.WORKER, cache_config
            )
            assert not self.worker._lmcache_engine._manager._init_failed, (
                self.worker._lmcache_engine._manager._init_failed_reason
            )
            if args.native_runner:
                from cpu_native_runner import make_runner

                self.worker.native_runner = make_runner(
                    worker_config, self.worker, cache_config, hardware, stats
                )
                self.worker.cpu_compute_stream = hardware.Stream()
                self.worker.cpu_compute_stream.name = (
                    f"{self.vllm_config.kv_transfer_config.engine_id}-r{rank}/compute"
                )
                self.workers.append(self.worker)
                return
            caches = {}
            raw_tensors = {}
            for spec in self.cache_config.kv_cache_tensors:
                raw = torch.zeros(spec.size, dtype=torch.uint8)
                hardware.queue.memory(raw, raw)
                raw_tensors.update({name: raw for name in spec.shared_by})
            for group in self.cache_config.kv_cache_groups:
                for name in group.layer_names:
                    caches[name] = reshape_dsa_shared_pool_raw(
                        raw_tensors[name],
                        torch.float16,
                        block_size,
                        1,
                        kv_width - 1,
                        1,
                        2,
                        is_indexer="indexer" in name,
                    )
            self.worker.register_kv_caches(caches)
            self.worker.cpu_compute_stream = hardware.Stream()
            self.worker.cpu_compute_stream.name = (
                f"{self.vllm_config.kv_transfer_config.engine_id}-r{rank}/compute"
            )
            self.workers.append(self.worker)

        def check_health(self):
            pass

        def execute_model(self, scheduler_output, non_block=False):
            for req in scheduler_output.scheduled_new_reqs:
                self.lengths[req.req_id] = len(req.prompt_token_ids)
                self.tokens[req.req_id] = list(req.prompt_token_ids)
                self.blocks[req.req_id] = [list(group) for group in req.block_ids]
                self.computed[req.req_id] = req.num_computed_tokens
            cached = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(cached.req_ids):
                new = cached.new_block_ids[i]
                if new is not None:
                    for group, values in enumerate(new):
                        self.blocks[req_id][group].extend(values)
            for worker in self.workers:
                self.worker = worker
                if args.native_runner:
                    from vllm.config import set_current_vllm_config
                    from vllm_ascend import ascend_config

                    runner = worker.native_runner
                    if (
                        worker is self.workers[0]
                        and scheduler_output.total_num_scheduled_tokens
                    ):
                        role = (
                            "P"
                            if worker._lmcache_engine._layerwise_prefill_p_node
                            else "D"
                        )
                        stats[role + "_steps"] += 1
                        if role == "P":
                            scheduled = scheduler_output.num_scheduled_tokens
                            stats["prefill_chunks"].append(
                                {
                                    req: [
                                        self.computed[req],
                                        self.computed[req] + count,
                                    ]
                                    for req, count in scheduled.items()
                                }
                            )
                        stats["mixed_batches"] += (
                            len(scheduler_output.num_scheduled_tokens) > 1
                        )
                    ascend_config._ASCEND_CONFIG = runner.ascend_config
                    previous = kv_transfer_state._KV_CONNECTOR_AGENT
                    kv_transfer_state._KV_CONNECTOR_AGENT = worker
                    try:
                        with (
                            hardware.stream(worker.cpu_compute_stream),
                            set_current_vllm_config(runner.vllm_config),
                        ):
                            worker.native_output = runner.execute_model(
                                copy.deepcopy(scheduler_output)
                            )
                    finally:
                        kv_transfer_state._KV_CONNECTOR_AGENT = previous
                else:
                    self._execute_worker(scheduler_output)
            for req_id, count in scheduler_output.num_scheduled_tokens.items():
                self.computed[req_id] += count
            if non_block:
                future = Future()
                future.set_result(None)
                return future
            return None

        def _execute_worker(self, scheduler_output):
            previous = kv_transfer_state._KV_CONNECTOR_AGENT
            kv_transfer_state._KV_CONNECTOR_AGENT = self.worker
            try:
                with (
                    hardware.stream(self.worker.cpu_compute_stream),
                    set_forward_context({}, self.vllm_config),
                ):
                    return self._execute_worker_impl(scheduler_output)
            finally:
                kv_transfer_state._KV_CONNECTOR_AGENT = previous

        def _execute_worker_impl(self, scheduler_output):
            self.last = scheduler_output
            self.worker.bind_connector_metadata(
                copy.deepcopy(scheduler_output.kv_connector_metadata)
            )
            bridge = self.worker._lmcache_engine
            context = SimpleNamespace(attn_metadata={}, no_compile_layers={})
            self.worker.start_load_kv(context)
            requests = self.worker._get_connector_metadata().requests
            if self.worker is self.workers[0]:
                role = "P" if bridge._layerwise_prefill_p_node else "D"
                stats[role + "_steps"] += bool(requests)
                stats["mixed_batches"] += len(requests) > 1
            # These methods contain no attention/model parameters. Execute the
            # real SFA transfer-window callbacks around the dummy arithmetic.
            callbacks = object.__new__(AscendSFAImpl)
            pending_operations = []
            if bridge._layerwise_prefill_p_node:
                bridge.submit_layerwise_prefill_load(-1)
            for layer in range(layers):
                previous_operations = pending_operations
                previous_names = (
                    callbacks._submit_sfa_layerwise_transfer_window(previous_operations)
                    if previous_operations
                    else []
                )
                pending_operations = []
                names = [f"model.layers.{layer}.self_attn.attn"]
                if layer in index_at:
                    names.append(f"model.layers.{layer}.self_attn.indexer.k_cache")
                for name in names:
                    group = int("indexer" in name)
                    ordinal = index_at.index(layer) if group else layer
                    caches = bridge.kv_caches[name]
                    sparse = [
                        request for request in requests if request.is_sparse_decode
                    ]
                    if sparse:
                        selected = [
                            list(
                                range(
                                    min(self.computed[r.req_id], self.lengths[r.req_id])
                                )
                            )
                            for r in sparse
                        ]
                        width = max(map(len, selected))
                        indices = torch.full((len(sparse), width), -1, dtype=torch.long)
                        destinations = torch.full_like(indices, -1)
                        for row, r in enumerate(sparse):
                            indices[row, : len(selected[row])] = torch.tensor(
                                selected[row]
                            )
                            blocks = self.blocks[r.req_id][group]
                            destinations[row, : len(selected[row])] = torch.tensor(
                                [
                                    blocks[p // block_size] * block_size
                                    + p % block_size
                                    for p in selected[row]
                                ]
                            )
                        self.worker.wait_for_layer_load(
                            name,
                            selected_tokens=indices,
                            request_ids=[r.req_id for r in sparse],
                            target_slot_mapping=destinations,
                            selected_token_counts=torch.tensor(
                                [len(x) for x in selected]
                            ),
                        )
                    else:
                        self.worker.wait_for_layer_load(name)
                    for request in requests:
                        start = self.computed[request.req_id]
                        count = scheduler_output.num_scheduled_tokens[request.req_id]
                        end = start + count
                        bank = (
                            ordinal + request.layerwise_prefill_bank_offset
                        ) % args.banks
                        if request.block_ids_by_bank is not None:
                            blocks = request.block_ids_by_bank[bank][group]
                            slots = tuple(
                                blocks[pos // block_size] * block_size
                                + pos % block_size
                                for pos in range(end)
                            )
                        else:
                            blocks = self.blocks[request.req_id][group]
                            slots = tuple(
                                blocks[pos // block_size] * block_size
                                + pos % block_size
                                for pos in range(end)
                            )
                        tokens = tuple(self.tokens[request.req_id])

                        def compute(
                            caches=caches,
                            slots=slots,
                            tokens=tokens,
                            start=start,
                            end=end,
                            layer=layer,
                            group=group,
                        ):
                            physical = torch.tensor(slots)
                            inputs = torch.tensor(tokens[:end])
                            outputs = self.model(inputs, torch.arange(end))
                            hardware.queue.counts["compute"] += 1
                            for plane, cache in enumerate(caches):
                                view = cache.reshape(-1, cache.shape[-1])
                                expected = (
                                    outputs[:, None] * 13
                                    + torch.arange(end)[:, None] * 7
                                    + layer * 17
                                    + group * 3
                                    + plane
                                    + torch.arange(view.shape[-1])
                                ) % 1024
                                expected = expected.to(view.dtype)
                                if not torch.equal(
                                    view[physical[:start]], expected[:start]
                                ):
                                    raise AssertionError(
                                        f"Historical KV mismatch layer={layer} "
                                        f"group={group} history={start} tokens"
                                    )
                                view[physical[start:end]] = expected[start:end]
                                stats["checked_rows"] += end

                        hardware.queue.current.submit(f"compute {name}", compute)
                    if bridge._layerwise_prefill_p_node:
                        record_layerwise_prefill_bank_use(name)
                        pending_operations.append((name, caches))
                    else:
                        self.worker.save_kv_layer(name, caches, None)
                terminal_names = []
                if layer == layers - 1 and pending_operations:
                    if previous_names:
                        callbacks._finish_sfa_layerwise_transfer_window(
                            previous_operations, previous_names
                        )
                        previous_names = []
                    terminal_names = callbacks._submit_sfa_layerwise_transfer_window(
                        pending_operations
                    )
                hardware.queue.current.submit("compute output projection")
                if previous_names:
                    callbacks._finish_sfa_layerwise_transfer_window(
                        previous_operations, previous_names
                    )
                if terminal_names:
                    callbacks._finish_sfa_layerwise_transfer_window(
                        pending_operations, terminal_names
                    )
                    pending_operations = []
            self.worker.wait_for_save()

        def sample_tokens(self, grammar_output, non_block=False):
            if args.native_runner:
                from vllm.config import set_current_vllm_config

                outputs = []
                for worker in self.workers:
                    with (
                        hardware.stream(worker.cpu_compute_stream),
                        set_current_vllm_config(worker.native_runner.vllm_config),
                    ):
                        hardware.queue.until(worker.cpu_compute_stream.tail)
                        output = worker.native_output
                        if output is None:
                            output = worker.native_runner.sample_tokens(grammar_output)
                        if hasattr(output, "get_output"):
                            output = output.get_output()
                        outputs.append(output)
                return self.kv_aggregator.aggregate(outputs)
            for worker in self.workers:
                hardware.queue.until(worker.cpu_compute_stream.tail)
            ids = list(self.last.num_scheduled_tokens)
            sampled = [[7] if self.computed[x] >= self.lengths[x] else [] for x in ids]
            for req_id, values in zip(ids, sampled, strict=True):
                self.tokens[req_id].extend(values)
            outputs = []
            for worker in self.workers:
                sending, recving = worker.get_finished(self.last.finished_req_ids)
                invalid = worker.get_block_ids_with_load_errors()
                assert not invalid, f"Cache load reported invalid blocks: {invalid}"
                outputs.append(
                    ModelRunnerOutput(
                        req_ids=ids,
                        req_id_to_index={x: i for i, x in enumerate(ids)},
                        sampled_token_ids=sampled,
                        logprobs=None,
                        prompt_logprobs_dict={},
                        kv_connector_output=KVConnectorOutput(
                            finished_sending=sending,
                            finished_recving=recving,
                            invalid_block_ids=invalid,
                            kv_connector_worker_meta=worker.build_connector_worker_meta(),
                        ),
                    )
                )
                worker.clear_connector_metadata()
            return self.kv_aggregator.aggregate(outputs)

    with tempfile.TemporaryDirectory(prefix="vllm-cpu-model-") as directory:
        model = Path(directory)
        (model / "config.json").write_text(
            json.dumps(
                dict(
                    model_type="deepseek_v2",
                    architectures=["CPUIdentityModel"],
                    vocab_size=256,
                    hidden_size=16,
                    num_hidden_layers=layers,
                    num_attention_heads=args.tp_size,
                    num_key_value_heads=1,
                    intermediate_size=32,
                    max_position_embeddings=max_length,
                    kv_lora_rank=kv_width - 1,
                    qk_rope_head_dim=1,
                    qk_nope_head_dim=kv_width - 1,
                    v_head_dim=kv_width - 1,
                    **(
                        {"index_topk": 128, "index_head_dim": 2}
                        if args.native_runner
                        else {}
                    ),
                )
            )
        )
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(model),
                skip_tokenizer_init=True,
                enforce_eager=True,
                max_model_len=max_length,
                dtype="float16",
            ),
            device_config=DeviceConfig("cpu"),
            parallel_config=ParallelConfig(tensor_parallel_size=args.tp_size),
            cache_config=CacheConfig(
                block_size=block_size,
                num_gpu_blocks_override=cache_blocks,
                enable_prefix_caching=False,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=max_length,
                is_encoder_decoder=False,
                max_num_batched_tokens=args.batch_tokens,
                max_num_seqs=args.max_num_seqs,
                enable_chunked_prefill=True,
                scheduler_cls="vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
            ),
            compilation_config=CompilationConfig(mode=0),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheAscendConnectorV1Dynamic",
                kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
                kv_role="kv_producer",
                engine_id="cpu-P",
            ),
        )
        engine = EngineCore(config, CPUExecutor, False)
        print(
            "[CPU_E2E] initialized",
            type(engine.scheduler).__name__,
            "TP",
            tp_size,
            flush=True,
        )

        def run_requests(target, prompts, phase, max_tokens):
            results = {key: [] for key in prompts}
            for req_id, tokens in prompts.items():
                request = EngineCoreRequest(
                    request_id=req_id,
                    prompt_token_ids=tokens,
                    mm_features=None,
                    sampling_params=SamplingParams(
                        max_tokens=max_tokens, ignore_eos=True, temperature=0
                    ),
                    pooling_params=None,
                    arrival_time=0.0,
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                )
                target.add_request(*target.preprocess_add_request(request))
            for step in range(2048):
                if not target.scheduler.has_requests():
                    break
                outputs, _ = target.step_fn()
                if phase.startswith("P ") and (step + 1) % 4 == 0:
                    completed = stats["prefill_chunks"][-1]
                    print(
                        f"[CPU_E2E] {phase}: chunk step={step + 1} spans={completed}",
                        flush=True,
                    )
                for output in outputs.values():
                    for item in output.outputs:
                        results[item.request_id].extend(item.new_token_ids)
            else:
                raise RuntimeError(f"{phase} scheduler did not finish")
            assert all(len(tokens) == max_tokens for tokens in results.values()), (
                results
            )
            assert all(token == 7 for tokens in results.values() for token in tokens), (
                f"{phase}: unexpected identity-model output: {results}"
            )
            print(
                f"[CPU_E2E] {phase}: requests={len(prompts)} steps={step} "
                f"tokens={sum(map(len, results.values()))}",
                flush=True,
            )
            return results

        try:
            sessions = {
                i: [(i * 29 + j) % 251 for j in range(args.context_tokens + i * 37)]
                for i in range(args.requests)
            }
            prompts = {f"P-0-{i}": tokens for i, tokens in sessions.items()}
            run_requests(engine, prompts, "P round 0", 1)
            first_chunks = [
                row["P-0-0"] for row in stats["prefill_chunks"] if "P-0-0" in row
            ]
            if args.native_runner:
                assert (
                    len(first_chunks)
                    >= (args.context_tokens + args.batch_tokens - 1)
                    // args.batch_tokens
                )
                assert (
                    first_chunks[0][0] == 0
                    and first_chunks[-1][1] == args.context_tokens
                )
                assert all(
                    a[1] == b[0]
                    for a, b in zip(first_chunks, first_chunks[1:], strict=False)
                )
            services["cpu-P", 0].wait_for_pending_stores(prompts)
            os.environ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = "false"
            lmconfig = copy.deepcopy(lmconfig)
            lmconfig.store_async = False
            lmconfig.extra_config.pop("shm_name", None)
            lmconfig.extra_config.pop("shared_cpu_cache_resolved_name", None)
            dconfig = copy.deepcopy(config)
            dconfig.kv_transfer_config.kv_role = "kv_consumer"
            dconfig.kv_transfer_config.engine_id = "cpu-D"
            dengine = EngineCore(dconfig, CPUExecutor, False)
            p_slab = services[
                "cpu-P", 0
            ].storage_manager.local_cpu_backend.memory_allocator.buffer
            d_slab = services[
                "cpu-D", 0
            ].storage_manager.local_cpu_backend.memory_allocator.buffer
            assert p_slab.data_ptr() != d_slab.data_ptr(), (
                "P and D must not share their local CPU cache"
            )
            for turn in range(args.rounds):
                if turn:
                    for tokens in sessions.values():
                        tokens.extend([7] * args.decode_tokens + [41, 42, 43] * 7)
                    os.environ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = "true"
                    prompts = {
                        f"P-{turn}-{i}": tokens for i, tokens in sessions.items()
                    }
                    run_requests(engine, prompts, f"P round {turn}", 1)
                    services["cpu-P", 0].wait_for_pending_stores(prompts)
                os.environ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = "false"
                prompts = {f"D-{turn}-{i}": tokens for i, tokens in sessions.items()}
                run_requests(dengine, prompts, f"D round {turn}", args.decode_tokens)
                stats["decode_tokens"] += len(prompts) * args.decode_tokens
            hardware.queue.drain()
            assert hardware.transfer_stats["early_reads"] == 0
            assert (
                hardware.transfer_stats["poisoned_operations"]
                == hardware.transfer_stats["completed_operations"]
            )
            puts = sum(call[0] == "put" for call in hardware.remote_calls)
            gets = sum(call[0] == "get" for call in hardware.remote_calls)
            assert (
                puts
                and gets
                and stats["checked_rows"]
                and stats["P_steps"]
                and stats["D_steps"]
            )
            if args.requests > 1:
                assert stats["mixed_batches"], "No mixed request batch was exercised"
            report = dict(
                status="passed",
                seed=args.seed,
                banks=args.banks,
                tp=tp_size,
                layers=layers,
                kv_width=kv_width,
                native_model_runner=args.native_runner,
                requests=args.requests,
                rounds=args.rounds,
                context_tokens=args.context_tokens,
                batch_tokens=args.batch_tokens,
                cache_chunk_tokens=args.cache_chunk_tokens,
                initial_prefill_chunk_count=len(first_chunks),
                **stats,
                remote_puts=puts,
                remote_gets=gets,
                device_commands=len(hardware.queue.commands),
                device_operations=hardware.queue.counts,
                transfer_validation=hardware.transfer_stats,
                source_revisions={
                    name: subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root.parent / name, text=True
                    ).strip()
                    for name in ("vllm", "vllm-ascend", "LMCache", "LMCache-Ascend")
                },
                scope=(
                    "Real EngineCore/AsyncRecomputeScheduler/"
                    + ("NPUModelRunner/" if args.native_runner else "")
                    + "LMCache Python modules; "
                    "CPU identity model, simulated device and Mooncake SDK; "
                    "no ACL kernels or ACL graph replay"
                ),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
            print("[CPU_E2E] " + json.dumps(report), flush=True)
        finally:
            if sys.exc_info()[0] is not None:
                # Deliberate device failures leave store workers waiting on
                # failed transfers, just like a failed accelerator. This CLI
                # is the isolated fault-test process; do not wait on shutdown.
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(2)
            for service in services.values():
                service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
