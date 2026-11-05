# SPDX-License-Identifier: Apache-2.0
"""CPU identity model hosted by the production Ascend model runner."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def make_runner(config, worker, cache_config, hardware, stats):
    """Initialize the original runner; replace only the model's arithmetic."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.distributed.kv_transfer import kv_transfer_state
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.platforms import current_platform
    from vllm.platforms.interface import PlatformEnum
    from vllm_ascend.ascend_config import init_ascend_config, clear_ascend_config
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
    from vllm_ascend.attention.utils import record_layerwise_prefill_bank_use
    from vllm_ascend.ops.rotary_embedding import _record_cos_and_sin_cache
    from vllm_ascend.platform import NPUPlatform
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        ranks=[0],
        is_first_rank=True,
        is_last_rank=True,
        device=torch.device("cpu"),
        device_group=None,
        cpu_group=None,
    )
    for name in ("_TP", "_PP", "_DP", "_EP", "_DCP", "_PCP"):
        setattr(parallel_state, name, group)
    current_platform._enum = PlatformEnum.CPU
    current_platform.get_attn_backend_cls = NPUPlatform.get_attn_backend_cls
    config.model_config.hf_text_config.index_topk = 2048
    config.model_config.hf_text_config.index_head_dim = 2

    class IdentityAttention(AttentionLayerBase):
        def __init__(self, spec):
            self.spec, self.impl = spec, SimpleNamespace()
            self.kv_cache = [torch.tensor([])]

        def get_attn_backend(self):
            return AscendSFABackend

        def get_kv_cache_spec(self, _config):
            return self.spec

    for cache_group in cache_config.kv_cache_groups:
        for name in cache_group.layer_names:
            config.compilation_config.static_forward_context[name] = IdentityAttention(
                cache_group.kv_cache_spec
            )

    callbacks = object.__new__(AscendSFAImpl)
    bridge = worker._lmcache_engine
    layer_names = tuple(bridge.lmcache_engine.metadata.runtime_kv_group_layer_names[0])
    index_names = set(bridge.lmcache_engine.metadata.runtime_kv_group_layer_names[1])

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def embed_input_ids(self, input_ids):
            return input_ids[:, None].expand(-1, 16).float()

        def forward(self, input_ids, positions, **kwargs):
            context = get_forward_context()
            pending = []
            is_prefill = bridge._layerwise_prefill_p_node
            if is_prefill:
                bridge.submit_layerwise_prefill_load(-1)
            for layer, latent_name in enumerate(layer_names):
                old = pending
                old_names = (
                    callbacks._submit_sfa_layerwise_transfer_window(old) if old else []
                )
                pending = []
                sibling = latent_name.rsplit(".", 1)[0] + ".indexer.k_cache"
                names = (
                    (latent_name, sibling) if sibling in index_names else (latent_name,)
                )
                for name in names:
                    meta = context.attn_metadata[name]
                    group_id = int("indexer" in name)
                    planes = bridge.kv_caches[name]
                    requests = worker._get_connector_metadata().requests
                    sparse = [
                        request for request in requests if request.is_sparse_decode
                    ]
                    selected_by_request = {}
                    if sparse:
                        rows, slots, counts = [], [], []
                        for request in sparse:
                            row = runner.input_batch.req_id_to_index[request.req_id]
                            length = int(runner.input_batch.num_prompt_tokens[row])
                            selected = (
                                torch.arange(length)
                                if group_id or length <= 2048
                                else torch.linspace(0, length - 1, 2048).long()
                            )
                            selected_by_request[request.req_id] = selected
                            rows.append(selected)
                            slots.append(
                                meta.block_table[row][selected // 128].long() * 128
                                + selected % 128
                            )
                            counts.append(len(selected))
                        width = max(counts)
                        indices = torch.full((len(rows), width), -1, dtype=torch.long)
                        target = torch.full_like(indices, -1)
                        for row, (selected, destination) in enumerate(
                            zip(rows, slots, strict=True)
                        ):
                            indices[row, : len(selected)] = selected
                            target[row, : len(selected)] = destination
                        worker.wait_for_layer_load(
                            name,
                            selected_tokens=indices,
                            request_ids=[r.req_id for r in sparse],
                            target_slot_mapping=target,
                            selected_token_counts=torch.tensor(counts),
                        )
                    else:
                        worker.wait_for_layer_load(name)
                    for request in requests:
                        row = runner.input_batch.req_id_to_index[request.req_id]
                        end = int(meta.seq_lens_cpu[row])
                        start = end - int(
                            meta.query_start_loc_cpu[row + 1]
                            - meta.query_start_loc_cpu[row]
                        )
                        tokens = torch.from_numpy(
                            runner.input_batch.token_ids_cpu[row, :end].copy()
                        ).long()
                        table = meta.block_table[row].clone()
                        physical = (
                            table[torch.arange(end) // 128].long() * 128
                            + torch.arange(end) % 128
                        )
                        history = selected_by_request.get(request.req_id)
                        if history is None:
                            history = torch.arange(start)
                        else:
                            history = torch.cat(
                                (
                                    history[history < start],
                                    torch.arange(
                                        int(runner.input_batch.num_prompt_tokens[row]),
                                        start,
                                    ),
                                )
                            ).unique()
                        read_ranges, write_ranges = [], []
                        for cache in planes:
                            read_ranges.extend(
                                hardware.tensor_ranges(cache, physical[history])
                            )
                            write_ranges.extend(
                                hardware.tensor_ranges(cache, physical[start:])
                            )

                        def compute(
                            planes=planes,
                            end=end,
                            start=start,
                            tokens=tokens,
                            physical=physical,
                            history=history,
                            layer=layer,
                            group_id=group_id,
                        ):
                            selected = torch.cat((history, torch.arange(start, end)))
                            for plane, cache in enumerate(planes):
                                view = cache.reshape(-1, cache.shape[-1])
                                for offset in range(0, len(selected), 32768):
                                    pos = selected[offset : offset + 32768]
                                    expected = (
                                        (
                                            tokens[pos, None] * 13
                                            + pos[:, None] * 7
                                            + layer * 17
                                            + group_id * 3
                                            + plane
                                            + torch.arange(view.shape[-1])
                                        )
                                        .remainder(1024)
                                        .to(view.dtype)
                                    )
                                    old = pos < start
                                    assert torch.equal(
                                        view[physical[pos[old]]], expected[old]
                                    ), (
                                        f"Native runner KV mismatch layer={layer} "
                                        f"group={group_id} history={start}"
                                    )
                                    view[physical[pos[~old]]] = expected[~old]
                                    stats["checked_rows"] += len(pos)
                            hardware.queue.counts["compute"] += 1

                        hardware.submit_access(
                            f"compute native {name}",
                            compute,
                            reads=read_ranges,
                            writes=write_ranges,
                        )
                    if is_prefill:
                        record_layerwise_prefill_bank_use(name)
                        pending.append((name, planes))
                terminal = []
                if layer == len(layer_names) - 1 and pending:
                    if old_names:
                        callbacks._finish_sfa_layerwise_transfer_window(old, old_names)
                        old_names = []
                    terminal = callbacks._submit_sfa_layerwise_transfer_window(pending)
                hardware.queue.current.submit("compute native output projection")
                if old_names:
                    callbacks._finish_sfa_layerwise_transfer_window(old, old_names)
                if terminal:
                    callbacks._finish_sfa_layerwise_transfer_window(pending, terminal)
            return input_ids[:, None].expand(-1, 16).to(torch.float16)

        def compute_logits(self, hidden_states):
            logits = torch.zeros(len(hidden_states), 256)
            logits[:, 7] = 1
            return logits

    previous = kv_transfer_state._KV_CONNECTOR_AGENT
    kv_transfer_state._KV_CONNECTOR_AGENT = worker
    try:
        with set_current_vllm_config(config):
            clear_ascend_config()
            init_ascend_config(config)
            runner = NPUModelRunner(config, torch.device("cpu"))
            _record_cos_and_sin_cache(
                torch.ones(config.model_config.max_model_len, 1),
                torch.zeros(config.model_config.max_model_len, 1),
            )
            runner.model = IdentityModel()
            runner.initialize_kv_cache(cache_config)
    finally:
        kv_transfer_state._KV_CONNECTOR_AGENT = previous
    seen = set()
    for entry in runner.kv_caches:
        for tensor in entry if isinstance(entry, (tuple, list)) else (entry,):
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            raw = torch.empty(0, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            hardware.queue.memory(raw, raw)
    return runner
