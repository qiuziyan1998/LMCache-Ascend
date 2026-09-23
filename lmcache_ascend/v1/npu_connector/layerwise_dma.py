# SPDX-License-Identifier: Apache-2.0
"""CPU tensor-only planning and address binding for layerwise prefill DMA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


def cache_page_size_bytes(cache) -> int:
    """Return one allocator page's bytes for tensor or tuple-plane caches."""
    if isinstance(cache, (tuple, list)):
        first = cache[0]
        page_bytes = first[0].numel() * first.element_size()
        if len(cache) > 1:
            second = cache[1]
            page_bytes += second[0].numel() * second.element_size()
        return int(page_bytes)
    return int(cache[0].numel() * cache.element_size())


def build_group_cycles(
    latent, indexer, chunk_tokens, multiplier, latent_plane_widths=None
):
    """Derive the same byte bundle as the shared-block allocator at startup."""
    latent_page = cache_page_size_bytes(latent)
    index_page = cache_page_size_bytes(indexer)
    bundle_bytes = math.lcm(latent_page, index_page) * multiplier
    latent_block_tokens = (
        latent[0].shape[1] if isinstance(latent, (tuple, list)) else latent.shape[1]
    )
    indexer_block_tokens = (
        indexer[0].shape[1]
        if isinstance(indexer, (tuple, list))
        else indexer.shape[1]
    )
    index_tokens = bundle_bytes // index_page * indexer_block_tokens
    widths = torch.as_tensor(
        latent_plane_widths
        or ([latent[0].shape[-1], latent[1].shape[-1]]
            if isinstance(latent, (tuple, list))
            else [latent.shape[-1]]),
        dtype=torch.long,
    )
    index_splits = (widths.cumsum(0)[:-1] * index_tokens) // widths.sum()
    return (
        DmaCycle.build(
            bundle_bytes // latent_page * latent_block_tokens, chunk_tokens
        ),
        DmaCycle.build(index_tokens, chunk_tokens, index_splits),
    )


@dataclass(frozen=True)
class DmaPlan:
    chunk: torch.Tensor
    slot: torch.Tensor
    chunk_token: torch.Tensor
    tokens: torch.Tensor

    def __len__(self):
        return self.tokens.numel()


@dataclass(frozen=True)
class DmaCycle:
    bundle_tokens: int
    chunk_tokens: int
    period: int
    boundaries: torch.Tensor

    @classmethod
    def build(cls, bundle_tokens: int, chunk_tokens: int, internal_offsets=None):
        if min(bundle_tokens, chunk_tokens) <= 0:
            raise ValueError("DMA bundle/chunk sizes must be positive")
        period = math.lcm(bundle_tokens, chunk_tokens)
        offsets = torch.zeros(1, dtype=torch.long)
        if internal_offsets is not None:
            offsets = torch.cat(
                (offsets, torch.as_tensor(internal_offsets, dtype=torch.long))
            )
        boundaries = torch.unique(
            torch.cat(
                (
                    (
                        torch.arange(0, period, bundle_tokens, dtype=torch.long)[
                            :, None
                        ]
                        + offsets
                    ).flatten(),
                    torch.arange(0, period, chunk_tokens, dtype=torch.long),
                )
            ),
            sorted=True,
        )
        return cls(bundle_tokens, chunk_tokens, period, boundaries)

    def plan(self, slots: torch.Tensor, start: int, end: int) -> DmaPlan:
        """Full CPU slot map; allocator guarantees contiguous logical bundles."""
        if start < 0 or end < start or end > slots.numel():
            raise ValueError("DMA interval exceeds bank slot map")
        periods = torch.arange(
            start // self.period, (end + self.period - 1) // self.period
        )
        edges = (periods[:, None] * self.period + self.boundaries[None, :]).reshape(-1)
        edges = torch.cat(
            (
                torch.tensor([start]),
                edges[(edges > start) & (edges < end)],
                torch.tensor([end]),
            )
        )
        if start == end:
            edges = edges[:1]
        begin = edges[:-1]
        host_begin = torch.maximum(
            begin // self.chunk_tokens * self.chunk_tokens, torch.tensor(start)
        )
        return DmaPlan(
            begin // self.chunk_tokens - start // self.chunk_tokens,
            slots[begin],
            begin - host_begin,
            edges[1:] - begin,
        )

    def plan_ranges(self, slots, starts, ends, slot_mapping_base=0):
        """Expand periodic boundaries; allow skipped chunks and partial tails."""
        starts = torch.as_tensor(starts, dtype=torch.long)
        ends = torch.as_tensor(ends, dtype=torch.long)
        periods = torch.arange(
            int(starts[0]) // self.period,
            (int(ends[-1]) + self.period - 1) // self.period,
        )
        periodic = (periods[:, None] * self.period + self.boundaries).flatten()
        edges = torch.unique(torch.cat((periodic, starts, ends)), sorted=True)
        begin, stop = edges[:-1], edges[1:]
        chunk = torch.searchsorted(ends, begin, right=True).clamp(max=ends.numel() - 1)
        active = (begin >= starts[chunk]) & (stop <= ends[chunk])
        begin, stop, chunk = begin[active], stop[active], chunk[active]
        return DmaPlan(
            chunk, slots[begin - slot_mapping_base], begin - starts[chunk], stop - begin
        )

    def plan_block_id_ranges(
        self, block_ids, block_size: int, starts, ends, slot_mapping_base=0
    ):
        """Plan ranges from block IDs without materializing per-token slots."""
        if block_size <= 0:
            raise ValueError("DMA block size must be positive")
        block_ids = torch.as_tensor(block_ids, dtype=torch.long)
        starts = torch.as_tensor(starts, dtype=torch.long)
        ends = torch.as_tensor(ends, dtype=torch.long)
        if block_ids.numel() == 0 or starts.numel() == 0:
            return DmaPlan(
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        if bool((starts < slot_mapping_base).any()) or bool((ends <= starts).any()):
            raise ValueError("DMA block-id ranges are invalid")
        logical_end = int(ends[-1]) - slot_mapping_base
        if logical_end > block_ids.numel() * block_size:
            raise ValueError("DMA block IDs do not cover the requested range")
        periods = torch.arange(
            int(starts[0] - slot_mapping_base) // self.period,
            (logical_end + self.period - 1) // self.period,
        )
        periodic = periods[:, None] * self.period + self.boundaries[None, :]
        block_starts = torch.arange(
            0, block_ids.numel() * block_size, block_size, dtype=torch.long
        )
        block_gaps = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool),
                block_ids[1:] != block_ids[:-1] + 1,
            )
        )
        gap_edges = block_starts[block_gaps]
        edges = torch.unique(
            torch.cat(
                (
                    periodic.flatten() + slot_mapping_base,
                    starts,
                    ends,
                    gap_edges + slot_mapping_base,
                )
            ),
            sorted=True,
        )
        edges = edges[(edges >= starts[0]) & (edges <= ends[-1])]
        begin, stop = edges[:-1], edges[1:]
        chunk = torch.searchsorted(ends, begin, right=True)
        active = (begin >= starts[chunk]) & (stop <= ends[chunk])
        begin, stop, chunk = begin[active], stop[active], chunk[active]
        relative = begin - slot_mapping_base
        block_index = relative // block_size
        slots = block_ids[block_index] * block_size + relative % block_size
        return DmaPlan(
            chunk,
            slots,
            begin - starts[chunk],
            stop - begin,
        )


@dataclass(frozen=True)
class BoundCopyPrefix:
    """Previously bound H2D rows for one request/group/layer/bank."""

    owners: tuple[object, ...]
    owner_ids: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    npu_ptrs: tuple[int, ...]
    plane_widths: tuple[int, ...]
    element_bytes: int
    segment_chunks: torch.Tensor
    rows: list[list[int]]
    reused_chunks: int = 0


def bind_incremental_copy_addresses(
    plan: DmaPlan,
    source_objs: Sequence[object],
    starts: Sequence[int],
    ends: Sequence[int],
    npu_ptrs: Sequence[int],
    plane_widths: Sequence[int],
    element_bytes: int,
    host_ptr: Callable[[object], int],
    host_tokens: Callable[[object], int],
    previous: BoundCopyPrefix | None,
    *,
    slot_prefix_unchanged: bool,
) -> BoundCopyPrefix:
    """Keep stable rows; replace a grown tail and bind newly appended chunks."""
    npu_ptrs = tuple(npu_ptrs)
    plane_widths = tuple(plane_widths)
    old_count = len(previous.owners) if previous is not None else 0
    compatible = bool(
        slot_prefix_unchanged
        and previous is not None
        and old_count <= len(source_objs)
        and npu_ptrs == previous.npu_ptrs
        and plane_widths == previous.plane_widths
        and element_bytes == previous.element_bytes
    )
    current_ids = tuple(map(id, source_objs))
    stable_before_tail = bool(
        compatible
        and old_count > 0
        and current_ids[: old_count - 1] == previous.owner_ids[: old_count - 1]
        and tuple(starts[: old_count - 1]) == previous.starts[: old_count - 1]
        and tuple(ends[: old_count - 1]) == previous.ends[: old_count - 1]
    )
    tail_unchanged = (
        bool(
            stable_before_tail
            and current_ids[old_count - 1] == previous.owner_ids[old_count - 1]
            and starts[old_count - 1] == previous.starts[old_count - 1]
            and ends[old_count - 1] == previous.ends[old_count - 1]
        )
        if old_count
        else False
    )
    begin_chunk = (
        old_count if tail_unchanged else old_count - 1 if stable_before_tail else 0
    )
    first_segment = int(torch.searchsorted(plan.chunk, begin_chunk))
    previous_segments = (
        int(torch.searchsorted(previous.segment_chunks, begin_chunk))
        if begin_chunk and previous is not None
        else 0
    )
    suffix_plan = DmaPlan(
        plan.chunk[first_segment:] - begin_chunk,
        plan.slot[first_segment:],
        plan.chunk_token[first_segment:],
        plan.tokens[first_segment:],
    )
    suffix_objs = source_objs[begin_chunk:]
    suffix_rows = (
        bind_copy_addresses(
            suffix_plan,
            list(map(host_ptr, suffix_objs)),
            npu_ptrs,
            (
                torch.as_tensor(ends[begin_chunk:])
                - torch.as_tensor(starts[begin_chunk:])
            ).tolist(),
            plane_widths,
            element_bytes,
            device_to_host=False,
            host_chunk_tokens=list(map(host_tokens, suffix_objs)),
        )
        if suffix_objs
        else []
    )
    rows = (
        previous.rows[: previous_segments * len(plane_widths)] + suffix_rows
        if begin_chunk and previous is not None
        else suffix_rows
    )
    return BoundCopyPrefix(
        owners=tuple(source_objs),
        owner_ids=current_ids,
        starts=tuple(starts),
        ends=tuple(ends),
        npu_ptrs=npu_ptrs,
        plane_widths=plane_widths,
        element_bytes=element_bytes,
        segment_chunks=plan.chunk,
        rows=rows,
        reused_chunks=begin_chunk,
    )


def plan_bundle_copies(slots, chunk_sizes, bundle_tokens) -> DmaPlan:
    """Vectorized compatibility/reference path for arbitrary maps and chunks."""
    slots = torch.as_tensor(slots, dtype=torch.long)
    sizes = torch.as_tensor(chunk_sizes, dtype=torch.long)
    if bundle_tokens <= 0 or sizes.numel() == 0 or bool((sizes <= 0).any()):
        raise ValueError("DMA needs positive bundle and chunk token counts")
    if int(sizes.sum()) != slots.numel() or bool((slots < 0).any()):
        raise ValueError("DMA sizes/slots invalid")
    ends = sizes.cumsum(0)
    gaps = (
        torch.nonzero(
            (slots[1:] != slots[:-1] + 1) | (slots[1:] % bundle_tokens == 0)
        ).flatten()
        + 1
    )
    edges = torch.unique(
        torch.cat((torch.zeros(1, dtype=torch.long), ends, gaps)), sorted=True
    )
    begin = edges[:-1]
    chunk = torch.searchsorted(ends, begin, right=True)
    return DmaPlan(
        chunk, slots[begin], begin - (ends - sizes)[chunk], edges[1:] - begin
    )


def bind_copy_addresses(
    plan,
    host_ptrs,
    npu_ptrs,
    chunk_sizes,
    plane_widths,
    element_bytes,
    *,
    device_to_host,
    host_chunk_tokens=None,
):
    """Broadcast segments x planes, returning native (dst, src, bytes) rows."""
    host = torch.as_tensor(host_ptrs, dtype=torch.long)
    npu = torch.as_tensor(npu_ptrs, dtype=torch.long)
    widths = torch.as_tensor(plane_widths, dtype=torch.long)
    sizes = torch.as_tensor(chunk_sizes, dtype=torch.long)
    physical = torch.as_tensor(
        chunk_sizes if host_chunk_tokens is None else host_chunk_tokens,
        dtype=torch.long,
    )
    if host.numel() != sizes.numel() or npu.numel() != widths.numel():
        raise ValueError("DMA pointer count differs from prepared layout")
    if physical.numel() != sizes.numel() or bool((physical < sizes).any()):
        raise ValueError("DMA host chunk shorter than copied range")
    if element_bytes <= 0 or bool((widths <= 0).any()):
        raise ValueError("DMA element and plane widths must be positive")
    byte_width = widths * element_bytes
    plane_offset = (widths.cumsum(0) - widths) * element_bytes
    cpu = (
        host[plan.chunk, None]
        + physical[plan.chunk, None] * plane_offset[None, :]
        + plan.chunk_token[:, None] * byte_width[None, :]
    )
    device = npu[None, :] + plan.slot[:, None] * byte_width[None, :]
    size = plan.tokens[:, None] * byte_width[None, :]
    dst, src = (cpu, device) if device_to_host else (device, cpu)
    return torch.stack((dst, src, size), dim=-1).reshape(-1, 3).tolist()
