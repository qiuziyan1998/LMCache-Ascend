# SPDX-License-Identifier: Apache-2.0
"""JSON-only request/config metadata for an explicit D snapshot replay."""

from __future__ import annotations

import json
import os
from typing import Any


def capture_request(request: Any) -> dict[str, Any]:
    """Preserve the actual D prompt, including any token appended by the proxy."""
    try:
        import msgspec

        if (
            request.mm_features
            or request.lora_request
            or request.prompt_embeds is not None
        ):
            raise ValueError("Replay supports text-only requests without LoRA")
        if request.num_output_tokens:
            raise ValueError("Snapshot is not at the first D receive")
        sampling = msgspec.to_builtins(request.sampling_params)
        # Runtime-only fields and transport addresses must never be replayed.
        sampling = {
            key: value
            for key, value in sampling.items()
            if not key.startswith("_")
            and key
            not in (
                "output_kind",
                "output_text_buffer_length",
                "skip_clone",
            )
        }
        extra = sampling.pop("extra_args", None) or {}
        transport = extra.pop("kv_transfer_params", None) or {}
        if extra:
            raise ValueError(
                "Custom sampling extra_args need an explicit replay adapter"
            )
        bootstrap = transport.get("bootstrap_final_hidden")
        if bootstrap is not None:
            sampling["extra_args"] = {
                "kv_transfer_params": {"bootstrap_final_hidden": bootstrap}
            }
        result = dict(
            schema=1,
            prompt_token_ids=list(request.prompt_token_ids),
            sampling=sampling,
            cache_salt=request.cache_salt,
        )
        return json.loads(json.dumps(result))
    except Exception as error:
        # A metadata error must not break the model or invalidate raw KV comparison.
        return {"schema": 1, "error": f"{type(error).__name__}: {error}"}


def capture_runtime(config: Any, lmcache_config: Any) -> dict[str, Any]:
    """Record compute settings, excluding cluster endpoints and credentials."""
    try:
        model, parallel = config.model_config, config.parallel_config
        if parallel.pipeline_parallel_size != 1:
            raise ValueError(
                "D snapshot replay currently requires pipeline_parallel_size=1"
            )
        options = {
            key: getattr(model, key)
            for key in (
                "model",
                "tokenizer",
                "trust_remote_code",
                "revision",
                "tokenizer_revision",
                "max_model_len",
                "quantization",
                "enforce_eager",
                "seed",
            )
        }
        options.update(
            dtype=str(model.dtype).removeprefix("torch."),
            tensor_parallel_size=parallel.tensor_parallel_size,
            enable_expert_parallel=parallel.enable_expert_parallel,
            additional_config=config.additional_config,
            max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
            async_scheduling=config.scheduler_config.async_scheduling,
            gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
        )
        speculative = config.speculative_config
        if speculative is not None:
            if speculative.method != "deepseek_mtp":
                raise ValueError(f"Unsupported replay speculator: {speculative.method}")
            options["speculative_config"] = dict(
                method=speculative.method,
                num_speculative_tokens=speculative.num_speculative_tokens,
            )
        # GPU graph shape/capture lists can change for a single-request replay;
        # keep the captured eager setting and report this scope in the launcher.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("VLLM_ASCEND_")
            and not any(
                word in key for word in ("DEBUG", "STATS", "DUMP", "P_NODE", "BALANCE")
            )
        }
        return json.loads(
            json.dumps(
                dict(
                    engine_options=options,
                    environment=environment,
                    chunk_size=lmcache_config.chunk_size,
                )
            )
        )
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
