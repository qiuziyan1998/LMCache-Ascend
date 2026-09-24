# SPDX-License-Identifier: Apache-2.0
"""Fixed-address, device-selected sparse MLA transfers for ACL graph replay."""

# Standard
from collections.abc import Sequence

# Third Party
import torch
from lmcache.v1.gpu_connector.sparse import PreparedSparseSource

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.utils import (
    prepare_sparse_direct_destination_state,
    sparse_graph_kv_transfer,
)


class SparseGraphTransfer:
    """One layer's stable source tables and process-owned destinations.

    ``bind`` is graph-external; ``load`` is captured on the compute stream.
    Source ownership is provided by LMCache's request lease, NOT this object.
    Callers must fence replay before finalizing or replacing that lease.

    One kernel copies K and PE using separate device pointer tables. Physical
    tail offsets and logical token limits can change without recapture. It also
    handles request lanes and masks invalid selections, with no torch where,
    index conversion or temporary payload tensors in the captured graph.
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
        self.state = prepare_sparse_direct_destination_state(
            list(kv_caches), slot_mapping, KVCacheFormat.MLA_LATENT.value, 0, 0, 0
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

    def plan_bind_update(
        self,
        sources: Sequence[PreparedSparseSource | None],
        changed_lanes: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Choose fewer tensor writes; None preserves the full-bind path."""
        if not changed_lanes:
            return ()
        full_cost = 2  # Whole pointer and limit clears.
        lane_costs = []
        for source in sources:
            if source is None:
                lane_costs.append(2)
                continue
            counts = source.chunk_token_counts
            if not counts:
                return None  # Let normal validation report malformed geometry.
            writes = 3 + int(counts[-1] != self.chunk_size)
            full_cost += writes
            lane_costs.append(writes + int(len(counts) < self.capacity))
        cost = sum(lane_costs[i] if i < len(lane_costs) else 2 for i in changed_lanes)
        return changed_lanes if cost < full_cost else None

    def bind_batch(
        self,
        sources: Sequence[PreparedSparseSource | None],
        layer_id: int,
        *,
        lanes: tuple[int, ...] | None = None,
    ) -> None:
        """Bind ordered request lanes; None and padded lanes cannot transfer KV."""
        if len(sources) > self.request_capacity:
            raise ValueError("Graph sources exceed request capacity")
        if lanes is not None and (
            len(set(lanes)) != len(lanes)
            or any(
                not isinstance(i, int) or not 0 <= i < self.request_capacity
                for i in lanes
            )
        ):
            raise ValueError("Invalid or duplicate graph source lane")
        selected = range(len(sources)) if lanes is None else lanes
        # Validate all affected lanes before overwriting this layer's state.
        for lane in selected:
            source = sources[lane] if lane < len(sources) else None
            if source is not None:
                self._validate_source(source, layer_id)
        if lanes is None:
            self.ptrs.zero_()
            self.valid_tokens.zero_()
        for lane in selected:
            source = sources[lane] if lane < len(sources) else None
            if source is None:
                if lanes is not None:
                    self.ptrs[
                        :, lane * self.capacity : (lane + 1) * self.capacity
                    ].zero_()
                    self.valid_tokens[lane].zero_()
                continue
            counts = source.chunk_token_counts
            layer = source.layers[layer_id]
            start = lane * self.capacity
            end = start + len(counts)
            if lanes is not None and len(counts) < self.capacity:
                self.ptrs[:, end : start + self.capacity].zero_()
            self.ptrs[0, start:end].copy_(layer.chunk_ptrs_npu)
            # Validation guarantees full physical chunks except possibly the tail.
            torch.add(
                layer.chunk_ptrs_npu,
                self.chunk_size * self.k_bytes,
                out=self.ptrs[1, start:end],
            )
            if counts[-1] != self.chunk_size:
                self.ptrs[1, end - 1 : end].add_(
                    (counts[-1] - self.chunk_size) * self.k_bytes
                )
            self.valid_tokens[lane].fill_(source.total_tokens)

    def load(
        self,
        selected: torch.Tensor,
        counts: torch.Tensor,
        slots: torch.Tensor,
        *,
        max_aiv_cores: int = 0,
    ) -> None:
        """Capture device top-k -> one K/PE copy, without payload preprocessing."""
        if (
            selected.shape[0] != self.request_capacity
            or slots.shape != selected.shape
            or counts.shape[0] != self.request_capacity
        ):
            raise ValueError("Graph payload does not match request capacity")
        sparse_graph_kv_transfer(
            self.state,
            slots,
            selected,
            counts,
            self.ptrs,
            self.valid_tokens,
            self.chunk_size,
            **({"max_aiv_cores": max_aiv_cores} if max_aiv_cores else {}),
        )

    def clear_source(self) -> None:
        """Disable transfers for no-offload steps without changing graph inputs."""
        self.valid_tokens.zero_()
