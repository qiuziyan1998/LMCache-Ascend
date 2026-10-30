# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import msgspec

SPEC = importlib.util.spec_from_file_location(
    "snapshot_metadata",
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend/v1/decode_snapshot_metadata.py",
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Sampling(msgspec.Struct):
    temperature: float = 0.0
    max_tokens: int = 128
    extra_args: dict | None = None


def test_request_keeps_exact_tokens_and_strips_remote_endpoints():
    request = NS(
        mm_features=[],
        lora_request=None,
        prompt_embeds=None,
        num_output_tokens=0,
        prompt_token_ids=[1, 2, 999],
        cache_salt=None,
        sampling_params=Sampling(
            extra_args={
                "kv_transfer_params": {
                    "remote_engine_id": "old-process",
                    "bootstrap_final_hidden": {"data": "bytes"},
                }
            }
        ),
    )
    result = M.capture_request(request)
    assert result["prompt_token_ids"] == [1, 2, 999]
    assert "old-process" not in json.dumps(result)
    assert result["sampling"]["extra_args"]["kv_transfer_params"] == {
        "bootstrap_final_hidden": {"data": "bytes"}
    }
    request.num_output_tokens = 3
    assert "first D receive" in M.capture_request(request)["error"]


def test_unsupported_request_is_explicit_not_a_production_exception():
    assert "error" in M.capture_request(None)
    assert "error" in M.capture_runtime(NS(), NS())


def test_runtime_keeps_compute_settings_without_remote_credentials(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "false")
    monkeypatch.setenv("LMCACHE_REMOTE_PASSWORD", "not-copied")
    config = NS(
        model_config=NS(
            model="glm",
            tokenizer="glm",
            trust_remote_code=True,
            revision=None,
            tokenizer_revision=None,
            max_model_len=600000,
            quantization="ascend",
            enforce_eager=False,
            seed=7,
            dtype="torch.bfloat16",
        ),
        parallel_config=NS(
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            enable_expert_parallel=True,
        ),
        scheduler_config=NS(max_num_batched_tokens=16384, async_scheduling=True),
        cache_config=NS(gpu_memory_utilization=0.93),
        additional_config={"layer_sharding": ["q_b_proj"]},
        speculative_config=NS(method="deepseek_mtp", num_speculative_tokens=1),
    )
    result = M.capture_runtime(config, NS(chunk_size=1024))
    assert "error" not in result
    assert result["engine_options"]["speculative_config"]["num_speculative_tokens"] == 1
    assert result["engine_options"]["gpu_memory_utilization"] == 0.93
    assert result["engine_options"]["enforce_eager"] is False
    assert "not-copied" not in json.dumps(result)
    assert "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE" not in result["environment"]
