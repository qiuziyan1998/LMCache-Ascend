# SPDX-License-Identifier: Apache-2.0
"""Fixed-address, device-selected sparse MLA transfers for ACL graph replay."""

# Standard
from typing import Any

# Third Party
import torch
from lmcache.v1.gpu_connector.sparse import PreparedSparseSource

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.utils import (
    prepare_sparse_direct_destination_state,
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
)


class SparseGraphTransfer:
    """One layer's stable source tables and process-owned destinations.

    ``bind`` is graph-external; ``load`` is captured on the compute stream.
    Source ownership is provided by LMCache's request lease, NOT this object.
    Callers must fence replay before finalizing or replacing that lease.

    K and PE use two single-plane kernels. This deliberately avoids capturing
    a host-side tail length into the two-plane kernel's PE offset. Pointer
    tables, token limits and tail offsets can change without recapture.
    """

    def __init__(
        self,
        kv_caches: tuple[torch.Tensor, torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_size: int,
        max_tokens: int,
    ) -> None:
        if chunk_size <= 0 or max_tokens <= 0:
            raise ValueError("chunk_size and max_tokens must be positive")
        if len(kv_caches) != 2 or any(t.ndim != 4 for t in kv_caches):
            raise ValueError("Full graph requires unbundled paged K/PE caches")
        if any(t.dtype != kv_caches[0].dtype for t in kv_caches):
            raise ValueError("K and PE must have the same dtype")
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.capacity = (max_tokens + chunk_size - 1) // chunk_size
        self.device = kv_caches[0].device
        self.slot_dtype = slot_mapping.dtype
        self.k_bytes = (
            kv_caches[0].shape[-2]
            * kv_caches[0].shape[-1]
            * kv_caches[0].element_size()
        )
        self.ptrs = torch.zeros(
            (2, self.capacity), dtype=torch.int64, device=self.device
        )
        self.valid_tokens = torch.zeros((), dtype=torch.int32, device=self.device)
        self.states: tuple[Any, ...] = tuple(
            prepare_sparse_direct_destination_state(
                [cache], slot_mapping, KVCacheFormat.DSA_INDEX.value, 0, 0, 0
            )
            for cache in kv_caches
        )

    def bind(self, source: PreparedSparseSource, layer_id: int) -> None:
        """Upload request pointers before forward, without changing addresses."""
        counts = source.chunk_token_counts
        if (
            not counts
            or len(counts) > self.capacity
            or not 0 < source.total_tokens <= self.max_tokens
            or any(n != self.chunk_size for n in counts[:-1])
            or not 0 < counts[-1] <= self.chunk_size
            or sum(counts) < source.total_tokens
        ):
            raise ValueError(
                "Full graph source is not a contiguous bounded chunk prefix"
            )
        layer = source.layers[layer_id]
        if (
            layer.chunk_ptrs_npu.ndim != 1
            or layer.chunk_ptrs_npu.dtype != torch.int64
            or not layer.chunk_ptrs_npu.is_contiguous()
        ):
            raise ValueError("Source pointer table must be contiguous int64")
        if layer.chunk_ptrs_npu.numel() != len(counts):
            raise ValueError("Source pointer table and physical chunk lengths differ")
        if layer.chunk_ptrs_npu.device != self.device:
            raise ValueError("Source pointer table is on the wrong device")
        # Use device aliases produced by the shared allocator, not CPU data_ptr.
        count_tensor = torch.tensor(counts, dtype=torch.int64, device=self.device)
        self.ptrs.zero_()
        self.ptrs[0, : len(counts)].copy_(layer.chunk_ptrs_npu)
        self.ptrs[1, : len(counts)].copy_(
            layer.chunk_ptrs_npu + count_tensor * self.k_bytes
        )
        self.valid_tokens.fill_(source.total_tokens)

    def load(
        self,
        selected: torch.Tensor,
        counts: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Capture device top-k -> masked sparse copy, with no host inspection."""
        valid = (selected >= 0) & (selected < self.valid_tokens)
        safe_selected = torch.where(valid, selected, 0).to(torch.int32).contiguous()
        safe_slots = torch.where(valid, slots, -1).to(self.slot_dtype).contiguous()
        safe_counts = counts.to(torch.int32).contiguous()
        for plane, state in enumerate(self.states):
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                state,
                safe_slots,
                safe_selected,
                self.ptrs[plane],
                self.chunk_size,
                self.capacity * self.chunk_size,
                False,
                safe_counts,
            )

    def clear_source(self) -> None:
        """Disable transfers for no-offload steps without changing graph inputs."""
        self.valid_tokens.zero_()
