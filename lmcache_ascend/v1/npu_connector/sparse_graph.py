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
        request_capacity: int | None = None,
    ) -> None:
        if chunk_size <= 0 or max_tokens <= 0:
            raise ValueError("chunk_size and max_tokens must be positive")
        if len(kv_caches) != 2 or any(t.ndim != 4 for t in kv_caches):
            raise ValueError("Full graph requires unbundled paged K/PE caches")
        if any(t.dtype != kv_caches[0].dtype for t in kv_caches):
            raise ValueError("K and PE must have the same dtype")
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.request_capacity = int(
            request_capacity
            if request_capacity is not None
            else slot_mapping.shape[0]
        )
        if self.request_capacity <= 0:
            raise ValueError("request_capacity must be positive")
        self.capacity = (max_tokens + chunk_size - 1) // chunk_size
        self.device = kv_caches[0].device
        self.slot_dtype = slot_mapping.dtype
        self.k_bytes = (
            kv_caches[0].shape[-2]
            * kv_caches[0].shape[-1]
            * kv_caches[0].element_size()
        )
        self.ptrs = torch.zeros(
            (2, self.request_capacity * self.capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self.valid_tokens = torch.zeros(
            (self.request_capacity, 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.lane_offsets = (
            torch.arange(
                self.request_capacity,
                dtype=torch.int32,
                device=self.device,
            ).view(-1, 1)
            * (self.capacity * self.chunk_size)
        )
        self.states: tuple[Any, ...] = tuple(
            prepare_sparse_direct_destination_state(
                [cache], slot_mapping, KVCacheFormat.DSA_INDEX.value, 0, 0, 0
            )
            for cache in kv_caches
        )
        self._bound_source_ids: tuple[int, ...] = ()
        self._bound_layer_id = -1

    def bind(self, source: PreparedSparseSource, layer_id: int) -> None:
        """Bind one request; retained for singleton graph callers."""
        if self.request_capacity != 1:
            raise ValueError("bind() is only valid for singleton graph transfers")
        self.bind_batch((source,), layer_id)

    def bind_batch(
        self,
        sources: Sequence[PreparedSparseSource],
        layer_id: int,
    ) -> None:
        """Upload one source per request lane without changing graph addresses."""
        source_tuple = tuple(sources)
        if not source_tuple or len(source_tuple) > self.request_capacity:
            raise ValueError("Graph sources exceed the fixed request capacity")
        source_ids = tuple(source.binding_id for source in source_tuple)
        if (
            source_ids == self._bound_source_ids
            and layer_id == self._bound_layer_id
        ):
            return

        normalized: list[tuple[PreparedSparseSource, torch.Tensor]] = []
        for source in source_tuple:
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
                raise ValueError(
                    "Source pointer table and physical chunk lengths differ"
                )
            if layer.chunk_ptrs_npu.device != self.device:
                raise ValueError("Source pointer table is on the wrong device")
            count_tensor = torch.as_tensor(
                counts,
                dtype=torch.int64,
                device=self.device,
            )
            normalized.append((source, count_tensor))

        self.ptrs.zero_()
        self.valid_tokens.zero_()
        valid_tokens = torch.zeros(
            (self.request_capacity, 1),
            dtype=torch.int32,
            device=self.device,
        )
        for lane, (source, count_tensor) in enumerate(normalized):
            layer = source.layers[layer_id]
            count = len(source.chunk_token_counts)
            start = lane * self.capacity
            end = start + count
            self.ptrs[0, start:end].copy_(layer.chunk_ptrs_npu)
            self.ptrs[1, start:end].copy_(
                layer.chunk_ptrs_npu + count_tensor * self.k_bytes
            )
            valid_tokens[lane, 0] = source.total_tokens
        self.valid_tokens.copy_(valid_tokens)
        self._bound_source_ids = source_ids
        self._bound_layer_id = layer_id

    def load(
        self,
        selected: torch.Tensor,
        counts: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Capture device top-k -> masked sparse copy, with no host inspection."""
        if selected.shape[0] != self.request_capacity:
            raise ValueError("Selected-token rows must equal request_capacity")
        valid = (selected >= 0) & (selected < self.valid_tokens)
        selected_i32 = selected.to(torch.int32)
        virtual_selected = selected_i32 + self.lane_offsets
        safe_selected = torch.where(valid, virtual_selected, 0).contiguous()
        safe_slots = torch.where(valid, slots, -1).to(self.slot_dtype).contiguous()
        active_rows = self.valid_tokens.view(-1) > 0
        safe_counts = torch.where(
            active_rows,
            counts.to(torch.int32),
            0,
        ).contiguous()
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
        self._bound_source_ids = ()
        self._bound_layer_id = -1
