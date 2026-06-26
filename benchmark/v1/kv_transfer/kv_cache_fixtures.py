# SPDX-License-Identifier: Apache-2.0
"""KV cache generators and equality checks for load benchmarks."""
from __future__ import annotations

# Standard
from typing import List, Tuple

# Third Party
import torch


def check_paged_kv_cache_equal(
    left,
    right,
    slot_mapping,
    *,
    num_heads: int = 8,
    head_size: int = 128,
    vllm_two_major: bool = True,
    kv_format: int = 1,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    dsa_head_dim: int = 0,
    label: str = "",
) -> None:
    """Assert two paged KV caches match at ``slot_mapping`` indices."""
    prefix = f"{label}: " if label else ""
    slots = slot_mapping.reshape(-1)
    num_tokens = slots.numel()

    for layer_id, (left_kv, right_kv) in enumerate(zip(left, right, strict=False)):
        if kv_format == 1 and not vllm_two_major:
            left_kv = left_kv.transpose(0, 1)
            right_kv = right_kv.transpose(0, 1)

        if kv_format in (3, 4):
            left_k = left_kv[0].reshape(-1, num_heads, kv_lora_rank)
            left_v = left_kv[1].reshape(-1, num_heads, qk_rope_head_dim)
            right_k = right_kv[0].reshape(-1, num_heads, kv_lora_rank)
            right_v = right_kv[1].reshape(-1, num_heads, qk_rope_head_dim)
        else:
            left_k = left_kv[0].reshape(-1, num_heads, head_size)
            left_v = left_kv[1].reshape(-1, num_heads, head_size)
            right_k = right_kv[0].reshape(-1, num_heads, head_size)
            right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert left_k.shape[0] >= num_tokens, f"{prefix}layer {layer_id} K too small"
        assert left_v.shape[0] >= num_tokens, f"{prefix}layer {layer_id} V too small"
        assert (left_k[slots] == right_k[slots]).all(), (
            f"{prefix}layer {layer_id} K mismatch"
        )
        assert (left_v[slots] == right_v[slots]).all(), (
            f"{prefix}layer {layer_id} V mismatch"
        )

        if kv_format == 4:
            left_dsa = left_kv[2].reshape(-1, dsa_head_dim)
            right_dsa = right_kv[2].reshape(-1, dsa_head_dim)
            assert (left_dsa[slots] == right_dsa[slots]).all(), (
                f"{prefix}layer {layer_id} DSA mismatch"
            )


def generate_mla_kv_cache(
    num_blocks: int,
    device: str,
    num_layers: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    k_shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v_shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    return [
        (
            torch.rand(k_shape, dtype=dtype, device=device),
            torch.rand(v_shape, dtype=dtype, device=device),
        )
        for _ in range(num_layers)
    ]


def generate_dsa_kv_cache(
    num_blocks: int,
    device: str,
    num_layers: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dsa_head_dim: int = 128,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    k_shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v_shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    dsa_k_shape = [num_blocks, block_size, 1, dsa_head_dim]
    return [
        (
            torch.rand(k_shape, dtype=dtype, device=device),
            torch.rand(v_shape, dtype=dtype, device=device),
            torch.rand(dsa_k_shape, dtype=dtype, device=device),
        )
        for _ in range(num_layers)
    ]
