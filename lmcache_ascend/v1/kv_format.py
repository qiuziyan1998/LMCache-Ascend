# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import List, Tuple, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish
    the KV cache data structures of different versions of vLLM.

    The order of enum values MUST match the KVCacheFormat
    definition in kernels/types.h to ensure correct interoperability
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """Merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    MLA_KV = auto()
    """MLA format for DeepSeek V2/V3 models
    layer: tuple: (k_cache, v_cache) where K and V have different dimensions
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    This format is used when K/V shapes differ (detected automatically).
    """

    DSA_KV = auto()
    """DSA (Deep Sparse Attention) format for DeepSeek V3.2 sparse models
    layer: tuple: (k_cache, v_cache, dsa_k_cache)
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    - dsa_k_cache.shape = [num_blocks, block_size, 1, 128]

    This format is used for sparse attention with lightning indexer.
    Legacy bundled path: latent and indexer packed into one 3-tuple.
    """

    MLA_LATENT = auto()
    """MLA latent-only format for the two-group DSA path.
    layer: tuple: (k_nope, k_pe) where the two have different dimensions
    - k_nope.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - k_pe.shape   = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    The indexer key is stored separately (see DSA_INDEX). CPU chunk layout is
    1D stacked planes: [k_nope plane: N*kH | k_pe plane: N*vH].
    """

    DSA_INDEX = auto()
    """DSA indexer-only format for the two-group DSA path.
    layer: tuple: (indexer_k,) — a single tensor
    - indexer_k.shape = [num_blocks, block_size, 1, index_head_dim]

    CPU chunk layout is 1D single plane: [indexer plane: N*dsaH].
    Naturally append-friendly (single plane, no V-plane shifting).
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_mla_format(self) -> bool:
        return self in (KVCacheFormat.MLA_KV, KVCacheFormat.MLA_LATENT)

    def is_mla_latent_format(self) -> bool:
        return self == KVCacheFormat.MLA_LATENT

    def is_dsa_format(self) -> bool:
        return self in (KVCacheFormat.DSA_KV, KVCacheFormat.DSA_INDEX)

    def is_dsa_index_format(self) -> bool:
        return self == KVCacheFormat.DSA_INDEX

    def is_tuple_format(self) -> bool:
        return self in (
            KVCacheFormat.SEPARATE_KV,
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.MLA_LATENT,
            KVCacheFormat.DSA_INDEX,
        )

    def get_kv_size(self) -> int:
        if self == KVCacheFormat.DSA_KV:
            return 3
        elif self in (KVCacheFormat.SEPARATE_KV,
                      KVCacheFormat.MLA_KV,
                      KVCacheFormat.MLA_LATENT):
            return 2
        elif self in (KVCacheFormat.MERGED_KV, KVCacheFormat.DSA_INDEX):
            return 1
        return 0

    @property
    def kv_group(self) -> int:
        """0 = latent group, 1 = indexer group. Used in LayerCacheEngineKey."""
        if self == KVCacheFormat.DSA_INDEX:
            return 1
        return 0

    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        use_mla: bool = False,
        dsa_two_groups: bool = False,
    ) -> "KVCacheFormat":
        """
        Automatically detect KV cache format based on data structure.

        Detection logic:
        1. DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache) — legacy bundled
        2. DSA_INDEX: 1-tuple (indexer_k,) — two-group indexer (only when dsa_two_groups)
        3. MLA_LATENT: tuple with 2 elements in MLA two-group mode
        4. MLA_KV: tuple with 2 elements in MLA mode
        5. SEPARATE_KV: tuple with 2 elements where K/V shapes are same and
           use_mla is false
        6. MERGED_KV: single tensor with specific shape patterns
        """
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, tuple):
            tuple_len = len(first_cache)

            # DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache) — legacy bundled
            if tuple_len == 3:
                k_cache, v_cache, dsa_k_cache = first_cache
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected DSA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}, dsa_k_shape={dsa_k_cache.shape}"
                        )
                        return KVCacheFormat.DSA_KV

            # DSA_INDEX: 1-tuple (indexer_k,) — two-group indexer
            if tuple_len == 1 and dsa_two_groups:
                indexer_k = first_cache[0]
                if isinstance(indexer_k, torch.Tensor):
                    logger.debug(
                        f"Detected DSA_INDEX format: "
                        f"indexer_k_shape={indexer_k.shape}"
                    )
                    return KVCacheFormat.DSA_INDEX

            # MLA_LATENT / MLA_KV / SEPARATE_KV: tuple with 2 elements
            if tuple_len == 2:
                k_cache, v_cache = first_cache
                if isinstance(k_cache, torch.Tensor) and isinstance(
                    v_cache, torch.Tensor
                ):
                    # MLA_LATENT or MLA_KV: the tensor pair is semantically MLA.
                    # Shape alone is not enough: at some TP degrees the
                    # sharded latent width can equal the PE width.
                    if use_mla or k_cache.shape != v_cache.shape:
                        if dsa_two_groups:
                            logger.debug(
                                f"Detected MLA_LATENT format (two-group): "
                                f"k_shape={k_cache.shape}, "
                                f"v_shape={v_cache.shape}"
                            )
                            return KVCacheFormat.MLA_LATENT
                        logger.debug(
                            f"Detected MLA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}"
                        )
                        return KVCacheFormat.MLA_KV
                    # SEPARATE_KV: K/V shapes are same
                    return KVCacheFormat.SEPARATE_KV

            return KVCacheFormat.SEPARATE_KV

        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED
