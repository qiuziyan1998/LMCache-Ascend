# SPDX-License-Identifier: Apache-2.0
"""Prepared Group-0 shard copies into canonical, service-owned CPU pages."""

from dataclasses import dataclass
from typing import Any

import torch

from lmcache.integration.vllm.dcp import DcpWriteLayout
from lmcache.v1.memory_management import LayerPageMemoryObj


@dataclass
class PreparedDcpCapture:
    """Keep actual DMA operands alive through the caller's final-access fence."""

    copies: list[tuple[int, int, int]]
    owners: tuple[Any, ...]


def prepare_dcp_capture(
    pages: list[Any],
    starts: list[int],
    ends: list[int],
    caches: list,
    layout: DcpWriteLayout,
    rank: int,
    *,
    memory_objs: list[list[Any]] | None = None,
    capture_start: int = 0,
) -> PreparedDcpCapture:
    """Map owned stripes directly to CPU planes without a full-prefix HBM copy.

    The existing layerwise DMA API accepts host destinations, unlike the AIV
    transfer API's mapped device pointers. All pages and HBM tensors are retained.
    Group 1 continues to use the existing BF16/C8 group-capture implementation.
    """
    if not pages or len(pages) != len(starts) or len(pages) != len(ends) or not caches:
        raise ValueError("DCP capture requires matching pages, intervals and layers")
    if type(capture_start) is not int or not 0 <= capture_start <= ends[-1]:
        raise ValueError("Invalid DCP resident capture frontier")
    if any(a >= b for a, b in zip(starts, ends)) or any(
        a != b for a, b in zip(starts[1:], ends[:-1])
    ):
        raise ValueError("DCP capture intervals must be contiguous and nonempty")
    copies, owners = [], list(pages)
    if memory_objs is not None and (
        len(memory_objs) != len(caches)
        or any(len(row) != len(pages) for row in memory_objs)
    ):
        raise ValueError("DCP capture per-layer page cardinality differs")
    widths = (512, 64)
    for layer, tensors in enumerate(caches):
        if len(tensors) != 2:
            raise ValueError("DCP capture requires separate latent and RoPE planes")
        dtype = tensors[0].dtype
        if dtype not in (torch.bfloat16, torch.float16) or any(
            tensor.dtype != dtype
            or tensor.device != tensors[0].device
            or tensor.ndim != 4
            or tensor.shape[1:] != (layout.block_size, 1, width)
            or not tensor.is_contiguous()
            for tensor, width in zip(tensors, widths)
        ):
            raise ValueError("DCP capture has incompatible HBM plane geometry")
        owners.extend(tensors)
        plane_bytes = [width * tensors[0].element_size() for width in widths]
        objects = memory_objs[layer] if memory_objs is not None else pages
        for page, start, end in zip(objects, starts, ends):
            owners.append(page)
            if not page.is_valid():
                raise ValueError("DCP capture has an invalid CPU owner")
            if isinstance(page, LayerPageMemoryObj):
                if page.num_layers != len(caches) or not page.layer_layout_is_valid():
                    raise ValueError(
                        "DCP capture has invalid canonical CPU page ownership"
                    )
                raw = page.raw_data
                if (
                    raw is None
                    or raw.device.type != "cpu"
                    or raw.dtype != torch.uint8
                    or not raw.is_contiguous()
                ):
                    raise ValueError("DCP capture requires canonical CPU layer pages")
                size = page.layer_size_bytes(layer)
                offset = page.group_prefix_sum[layer]
                if offset < 0 or offset + size > raw.numel():
                    raise ValueError("DCP capture CPU layer exceeds its allocation")
                base, page_dtype = raw.data_ptr() + offset, page.get_dtypes()[layer]
            else:
                raw = page.tensor
                if raw is None or raw.device.type != "cpu" or not raw.is_contiguous():
                    raise ValueError(
                        "DCP capture requires contiguous CPU layer storage"
                    )
                base, size, page_dtype = (
                    raw.data_ptr(),
                    raw.numel() * raw.element_size(),
                    raw.dtype,
                )
            owners.append(raw)
            capacity, remainder = divmod(size, sum(plane_bytes))
            if remainder or capacity < end - start or page_dtype != dtype:
                raise ValueError("DCP capture page precision or token extent differs")
            for token, slot, count in layout.runs(
                max(start, min(capture_start, end)), end, rank
            ):
                plane_offset = 0
                for tensor, token_bytes in zip(tensors, plane_bytes):
                    if slot + count > tensor.shape[0] * layout.block_size:
                        raise ValueError(
                            "DCP capture source exceeds its physical allocation"
                        )
                    copies.append(
                        (
                            base + plane_offset + (token - start) * token_bytes,
                            tensor.data_ptr() + slot * token_bytes,
                            count * token_bytes,
                        )
                    )
                    plane_offset += capacity * token_bytes
    return PreparedDcpCapture(copies, tuple(owners))
