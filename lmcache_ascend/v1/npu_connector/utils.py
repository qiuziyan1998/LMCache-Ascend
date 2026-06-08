# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import lru_cache
from typing import List, Sequence, Tuple, Union
import inspect

# Third Party
from lmcache.v1.gpu_connector.utils import permute_to_contiguous
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

_KVTupleTwoOrMore = Tuple[torch.Tensor, ...]
_KVLayer = Union[torch.Tensor, _KVTupleTwoOrMore]


def permute_kv_caches_to_contiguous(
    kv_caches: List[_KVLayer],
) -> List[_KVLayer]:
    """Apply :func:`permute_to_contiguous` to each tensor in *kv_caches*.

    Each entry is either a single ``torch.Tensor`` (merged KV) or a tuple of
    two or more tensors (e.g. K/V, or more parts). The returned list has the
    same length and
    structure; tensors are metadata-only permutes where applicable and may
    share storage with the inputs (see upstream ``permute_to_contiguous``).
    """
    results: List[_KVLayer] = []
    for layer in kv_caches:
        if isinstance(layer, torch.Tensor):
            results.append(permute_to_contiguous(layer))
        elif isinstance(layer, tuple):
            if len(layer) < 2:
                raise ValueError(
                    "Tuple KV entries must contain at least two tensors; "
                    f"got len={len(layer)}"
                )
            permuted: List[torch.Tensor] = []
            for t in layer:
                if not isinstance(t, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor inside KV tuple, got {type(t)}"
                    )
                permuted.append(permute_to_contiguous(t))
            results.append(tuple(permuted))
        else:
            raise ValueError(
                f"Unsupported KV cache entry type: {type(layer)} "
                "(expected Tensor or tuple of Tensors)"
            )
    return results


@lru_cache(maxsize=1)
def _op_param_names(op_name: str) -> frozenset[str]:
    fn = getattr(lmc_ops, op_name)
    return frozenset(inspect.signature(fn).parameters.keys())


def _call_fused_op(op_name: str, kwargs: dict) -> None:
    fn = getattr(lmc_ops, op_name)
    params = _op_param_names(op_name)
    if "k_hidden_dims" in params:
        fn(**{name: kwargs[name] for name in params})
        return

    if op_name == "batched_fused_sparse_single_layer_kv_transfer":
        fn(
            kwargs["lmc_tensors"],
            kwargs["staging_cache"],
            kwargs["vllm_kv_caches"],
            kwargs["slot_mapping"],
            kwargs["selected_token_idx"],
            kwargs["chunk_offsets"],
            kwargs["chunk_sizes"],
            kwargs["kvcache_format_raw"],
            kwargs["token_major"],
            kwargs["vllm_two_major"],
        )
        return

    fn(
        kwargs["lmc_tensors"],
        kwargs["staging_cache"],
        kwargs["vllm_kv_caches"],
        kwargs["slot_mapping"],
        kwargs["chunk_offsets"],
        kwargs["chunk_sizes"],
        kwargs["direction"],
        kwargs["kvcache_format_raw"],
        kwargs["token_major"],
        kwargs["vllm_two_major"],
    )


def batched_fused_single_layer_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    vllm_kv_caches: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    slot_mapping_full: torch.Tensor,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
    direction: bool,
    kvcache_format_raw: int,
    token_major: bool,
    vllm_two_major: bool,
    k_hidden_dims: int = 0,
    v_hidden_dims: int = 0,
    dsa_hidden_dims: int = 0,
) -> None:
    _call_fused_op(
        "batched_fused_single_layer_kv_transfer",
        {
            "lmc_tensors": lmc_tensors,
            "staging_cache": staging_cache,
            "vllm_kv_caches": vllm_kv_caches,
            "slot_mapping": slot_mapping_full,
            "chunk_offsets": chunk_offsets,
            "chunk_sizes": chunk_sizes,
            "direction": direction,
            "kvcache_format_raw": kvcache_format_raw,
            "token_major": token_major,
            "vllm_two_major": vllm_two_major,
            "k_hidden_dims": k_hidden_dims,
            "v_hidden_dims": v_hidden_dims,
            "dsa_hidden_dims": dsa_hidden_dims,
        },
    )


def batched_fused_sparse_single_layer_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    vllm_kv_caches: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
    kvcache_format_raw: int,
    token_major: bool,
    vllm_two_major: bool,
    k_hidden_dims: int = 0,
    v_hidden_dims: int = 0,
    dsa_hidden_dims: int = 0,
) -> None:
    _call_fused_op(
        "batched_fused_sparse_single_layer_kv_transfer",
        {
            "lmc_tensors": lmc_tensors,
            "staging_cache": staging_cache,
            "vllm_kv_caches": vllm_kv_caches,
            "slot_mapping": slot_mapping_packed,
            "selected_token_idx": selected_token_idx,
            "chunk_offsets": chunk_offsets,
            "chunk_sizes": chunk_sizes,
            "kvcache_format_raw": kvcache_format_raw,
            "token_major": token_major,
            "vllm_two_major": vllm_two_major,
            "k_hidden_dims": k_hidden_dims,
            "v_hidden_dims": v_hidden_dims,
            "dsa_hidden_dims": dsa_hidden_dims,
        },
    )
