# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Literal, Optional, Sequence, Tuple, Union

# Third Party
from lmcache.v1.gpu_connector.utils import permute_to_contiguous
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

_KVTupleTwoOrMore = Tuple[torch.Tensor, ...]
_KVLayer = Union[torch.Tensor, _KVTupleTwoOrMore]
_KVCacheArg = Union[torch.Tensor, Sequence[torch.Tensor]]


def _normalize_vllm_kv_caches(
    vllm_kv_caches: _KVCacheArg,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if isinstance(vllm_kv_caches, torch.Tensor):
        return vllm_kv_caches
    if isinstance(vllm_kv_caches, tuple):
        normalized = vllm_kv_caches
    elif isinstance(vllm_kv_caches, list):
        normalized = tuple(vllm_kv_caches)
    else:
        raise ValueError(
            f"Unsupported vLLM KV cache type: {type(vllm_kv_caches)} "
            "(expected Tensor or list/tuple of Tensors)"
        )
    if len(normalized) == 0:
        raise ValueError("vLLM KV cache tuple/list must not be empty")
    for tensor in normalized:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Expected torch.Tensor inside vLLM KV cache, got {type(tensor)}"
            )
    return normalized


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
            if len(layer) == 0:
                raise ValueError("Tuple KV entries must not be empty")
            if len(layer) == 1:
                t = layer[0]
                if not isinstance(t, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor inside KV tuple, got {type(t)}"
                    )
                # DSA_INDEX: (indexer_k,) single-plane tuple
                results.append((permute_to_contiguous(t),))
                continue
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


_FusedOpMode = Literal["extended", "legacy"]
_FUSED_OP_MODES: dict[str, _FusedOpMode | None] = {
    "batched_fused_single_layer_kv_transfer": None,
    "batched_fused_sparse_single_layer_kv_transfer": None,
}


def _is_pybind_arity_type_error(exc: TypeError) -> bool:
    msg = str(exc)
    return "incompatible function arguments" in msg or "takes" in msg and "positional" in msg


def _call_dense_extended(fn, kwargs: dict) -> None:
    fn(
        kwargs["lmc_tensors"],
        kwargs["staging_cache"],
        _normalize_vllm_kv_caches(kwargs["vllm_kv_caches"]),
        kwargs["slot_mapping"],
        kwargs["chunk_offsets"],
        kwargs["chunk_sizes"],
        kwargs["direction"],
        kwargs["kvcache_format_raw"],
        kwargs["token_major"],
        kwargs["vllm_two_major"],
        kwargs["k_hidden_dims"],
        kwargs["v_hidden_dims"],
        kwargs["dsa_hidden_dims"],
    )


def _call_dense_legacy(fn, kwargs: dict) -> None:
    fn(
        kwargs["lmc_tensors"],
        kwargs["staging_cache"],
        _normalize_vllm_kv_caches(kwargs["vllm_kv_caches"]),
        kwargs["slot_mapping"],
        kwargs["chunk_offsets"],
        kwargs["chunk_sizes"],
        kwargs["direction"],
        kwargs["kvcache_format_raw"],
        kwargs["token_major"],
        kwargs["vllm_two_major"],
    )


def _call_sparse_extended(fn, kwargs: dict) -> None:
    fn(
        kwargs["lmc_tensors"],
        kwargs["staging_cache"],
        _normalize_vllm_kv_caches(kwargs["vllm_kv_caches"]),
        kwargs["slot_mapping"],
        kwargs["selected_token_idx"],
        kwargs["chunk_offsets"],
        kwargs["chunk_sizes"],
        kwargs["kvcache_format_raw"],
        kwargs["token_major"],
        kwargs["vllm_two_major"],
        kwargs["k_hidden_dims"],
        kwargs["v_hidden_dims"],
        kwargs["dsa_hidden_dims"],
        kwargs.get("sparse_indices_cpu"),
    )


def _call_sparse_legacy(fn, kwargs: dict) -> None:
    fn(
        kwargs["lmc_tensors"],
        kwargs["staging_cache"],
        _normalize_vllm_kv_caches(kwargs["vllm_kv_caches"]),
        kwargs["slot_mapping"],
        kwargs["selected_token_idx"],
        kwargs["chunk_offsets"],
        kwargs["chunk_sizes"],
        kwargs["kvcache_format_raw"],
        kwargs["token_major"],
        kwargs["vllm_two_major"],
    )


def _call_fused_op(op_name: str, kwargs: dict) -> None:
    """Invoke a pybind fused op, probing extended vs legacy arity once."""
    fn = getattr(lmc_ops, op_name)
    is_sparse = op_name == "batched_fused_sparse_single_layer_kv_transfer"
    call_extended = _call_sparse_extended if is_sparse else _call_dense_extended
    call_legacy = _call_sparse_legacy if is_sparse else _call_dense_legacy

    mode = _FUSED_OP_MODES.get(op_name)
    if mode == "extended":
        call_extended(fn, kwargs)
        return
    if mode == "legacy":
        call_legacy(fn, kwargs)
        return

    try:
        call_extended(fn, kwargs)
        _FUSED_OP_MODES[op_name] = "extended"
    except TypeError as exc:
        if not _is_pybind_arity_type_error(exc):
            raise
        call_legacy(fn, kwargs)
        _FUSED_OP_MODES[op_name] = "legacy"


def batched_fused_single_layer_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    vllm_kv_caches: _KVCacheArg,
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
    vllm_kv_caches = _normalize_vllm_kv_caches(vllm_kv_caches)
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


def sparse_mla_dsa_batched_direct_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    vllm_kv_caches: _KVCacheArg,
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    kvcache_format_raw: int,
    token_major: bool,
    vllm_two_major: bool,
    k_hidden_dims: int = 0,
    v_hidden_dims: int = 0,
    dsa_hidden_dims: int = 0,
    lmc_host_interleaved: bool = False,
    chunk_ptrs_npu: Optional[torch.Tensor] = None,
) -> None:
    vllm_kv_caches = _normalize_vllm_kv_caches(vllm_kv_caches)
    lmc_ops.sparse_mla_dsa_batched_direct_kv_transfer(
        lmc_tensors,
        vllm_kv_caches,
        slot_mapping_packed,
        selected_token_idx,
        chunk_size,
        total_tokens,
        kvcache_format_raw,
        token_major,
        vllm_two_major,
        k_hidden_dims,
        v_hidden_dims,
        dsa_hidden_dims,
        lmc_host_interleaved,
        chunk_ptrs_npu,
    )


def prepare_sparse_direct_layer_state(
    lmc_layout_sample: torch.Tensor,
    vllm_kv_caches: _KVCacheArg,
    slot_mapping_ref: torch.Tensor,
    token_major: bool,
    vllm_two_major: bool,
    kvcache_format_raw: int,
    k_hidden_dims: int,
    v_hidden_dims: int,
    dsa_hidden_dims: int,
    lmc_num_tokens: int,
):
    vllm_kv_caches = _normalize_vllm_kv_caches(vllm_kv_caches)
    return lmc_ops.prepare_sparse_direct_layer_state(
        lmc_layout_sample,
        vllm_kv_caches,
        slot_mapping_ref,
        token_major,
        vllm_two_major,
        kvcache_format_raw,
        k_hidden_dims,
        v_hidden_dims,
        dsa_hidden_dims,
        lmc_num_tokens,
    )


def sparse_mla_dsa_batched_direct_kv_transfer_fast(
    layer_state,
    slot_mapping_packed: torch.Tensor,
    selected_token_idx: torch.Tensor,
    chunk_ptrs_npu: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
    lmc_host_interleaved: bool,
    validate_inputs: bool = False,
) -> None:
    lmc_ops.sparse_mla_dsa_batched_direct_kv_transfer_fast(
        layer_state,
        slot_mapping_packed,
        selected_token_idx,
        chunk_ptrs_npu,
        chunk_size,
        total_tokens,
        lmc_host_interleaved,
        validate_inputs,
    )


def dense_mla_dsa_batched_direct_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    vllm_kv_caches: _KVCacheArg,
    slot_mapping_full: torch.Tensor,
    chunk_offsets_npu: torch.Tensor,
    chunk_sizes_npu: torch.Tensor,
    total_tokens: int,
    kvcache_format_raw: int,
    token_major: bool,
    vllm_two_major: bool,
    k_hidden_dims: int = 0,
    v_hidden_dims: int = 0,
    dsa_hidden_dims: int = 0,
    lmc_host_interleaved: bool = False,
    direction: bool = False,
    chunk_ptrs_npu: Optional[torch.Tensor] = None,
    fixed_chunk_size: int = 0,
) -> None:
    vllm_kv_caches = _normalize_vllm_kv_caches(vllm_kv_caches)
    lmc_ops.dense_mla_dsa_batched_direct_kv_transfer(
        lmc_tensors,
        vllm_kv_caches,
        slot_mapping_full,
        chunk_offsets_npu,
        chunk_sizes_npu,
        total_tokens,
        kvcache_format_raw,
        token_major,
        vllm_two_major,
        k_hidden_dims,
        v_hidden_dims,
        dsa_hidden_dims,
        lmc_host_interleaved,
        direction,
        chunk_ptrs_npu,
        fixed_chunk_size,
    )


def dense_mla_dsa_batched_direct_kv_transfer_fast(
    layer_state,
    slot_mapping_full: torch.Tensor,
    chunk_ptrs_npu: torch.Tensor,
    chunk_offsets_npu: torch.Tensor,
    chunk_sizes_npu: torch.Tensor,
    total_tokens: int,
    lmc_host_interleaved: bool,
    direction: bool,
    validate_inputs: bool = False,
    fixed_chunk_size: int = 0,
) -> None:
    lmc_ops.dense_mla_dsa_batched_direct_kv_transfer_fast(
        layer_state,
        slot_mapping_full,
        chunk_ptrs_npu,
        chunk_offsets_npu,
        chunk_sizes_npu,
        total_tokens,
        lmc_host_interleaved,
        direction,
        validate_inputs,
        fixed_chunk_size,
    )


def batched_fused_sparse_single_layer_kv_transfer(
    lmc_tensors: Sequence[torch.Tensor],
    staging_cache: torch.Tensor,
    vllm_kv_caches: _KVCacheArg,
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
    sparse_indices_cpu: Optional[torch.Tensor] = None,
) -> None:
    vllm_kv_caches = _normalize_vllm_kv_caches(vllm_kv_caches)
    mode = _FUSED_OP_MODES["batched_fused_sparse_single_layer_kv_transfer"]
    if mode == "extended":
        lmc_ops.batched_fused_sparse_single_layer_kv_transfer(
            lmc_tensors,
            staging_cache,
            vllm_kv_caches,
            slot_mapping_packed,
            selected_token_idx,
            chunk_offsets,
            chunk_sizes,
            kvcache_format_raw,
            token_major,
            vllm_two_major,
            k_hidden_dims,
            v_hidden_dims,
            dsa_hidden_dims,
            sparse_indices_cpu,
        )
        return
    if mode == "legacy":
        lmc_ops.batched_fused_sparse_single_layer_kv_transfer(
            lmc_tensors,
            staging_cache,
            vllm_kv_caches,
            slot_mapping_packed,
            selected_token_idx,
            chunk_offsets,
            chunk_sizes,
            kvcache_format_raw,
            token_major,
            vllm_two_major,
        )
        return
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
            "sparse_indices_cpu": sparse_indices_cpu,
        },
    )
