# SPDX-License-Identifier: Apache-2.0
"""Fixed-address, device-selected sparse MLA transfers for ACL graph replay."""

# Standard
from collections.abc import Sequence
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
        request_capacity: int = 1,
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
        if (
            request_capacity <= 0
            or request_capacity * self.capacity * chunk_size >= 2**31
        ):
            raise ValueError("Graph request capacity exceeds int32 token addressing")
        self.request_capacity = request_capacity
        self.device = kv_caches[0].device
        self.slot_dtype = slot_mapping.dtype
        self.k_bytes = (
            kv_caches[0].shape[-2]
            * kv_caches[0].shape[-1]
            * kv_caches[0].element_size()
        )
        self.ptrs = torch.zeros(
            (2, request_capacity * self.capacity), dtype=torch.int64, device=self.device
        )
        self.valid_tokens = torch.zeros(
            (request_capacity, 1), dtype=torch.int32, device=self.device
        )
        self.lane_offsets = torch.arange(
            request_capacity, dtype=torch.int32, device=self.device
        ).view(-1, 1)
        self.lane_offsets.mul_(self.capacity * chunk_size)
        self.states: tuple[Any, ...] = tuple(
            prepare_sparse_direct_destination_state(
                [cache], slot_mapping, KVCacheFormat.DSA_INDEX.value, 0, 0, 0
            )
            for cache in kv_caches
        )

    def bind(self, source: PreparedSparseSource, layer_id: int) -> None:
        """Upload request pointers before forward, without changing addresses."""
        if self.request_capacity != 1:
            raise ValueError("Use bind_batch for multiple request lanes")
        self.bind_batch((source,), layer_id)

    def _validate_source(self, source: PreparedSparseSource, layer_id: int) -> None:
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

    def bind_batch(
        self, sources: Sequence[PreparedSparseSource | None], layer_id: int
    ) -> None:
        """Bind ordered request lanes; None and padded lanes cannot transfer KV."""
        if len(sources) > self.request_capacity:
            raise ValueError("Graph sources exceed request capacity")
        # Validate every lane before overwriting any captured state.
        for source in sources:
            if source is not None:
                self._validate_source(source, layer_id)
        self.ptrs.zero_()
        self.valid_tokens.zero_()
        for lane, source in enumerate(sources):
            if source is None:
                continue
            counts = source.chunk_token_counts
            layer = source.layers[layer_id]
            count_tensor = torch.tensor(counts, dtype=torch.int64, device=self.device)
            start = lane * self.capacity
            end = start + len(counts)
            self.ptrs[0, start:end].copy_(layer.chunk_ptrs_npu)
            self.ptrs[1, start:end].copy_(
                layer.chunk_ptrs_npu + count_tensor * self.k_bytes
            )
            self.valid_tokens[lane].fill_(source.total_tokens)

    def load(
        self,
        selected: torch.Tensor,
        counts: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Capture device top-k -> masked sparse copy, with no host inspection."""
        if (
            selected.shape[0] != self.request_capacity
            or slots.shape != selected.shape
            or counts.shape[0] != self.request_capacity
        ):
            raise ValueError("Graph payload does not match request capacity")
        valid = (selected >= 0) & (selected < self.valid_tokens)
        virtual_selected = selected.to(torch.int32) + self.lane_offsets
        safe_selected = torch.where(valid, virtual_selected, 0).contiguous()
        safe_slots = torch.where(valid, slots, -1).to(self.slot_dtype).contiguous()
        active = self.valid_tokens if counts.ndim == 2 else self.valid_tokens.view(-1)
        safe_counts = torch.where(active > 0, counts, 0).to(torch.int32).contiguous()
        for plane, state in enumerate(self.states):
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                state,
                safe_slots,
                safe_selected,
                self.ptrs[plane],
                self.chunk_size,
                self.request_capacity * self.capacity * self.chunk_size,
                False,
                safe_counts,
            )

    def clear_source(self) -> None:
        """Disable transfers for no-offload steps without changing graph inputs."""
        self.valid_tokens.zero_()
