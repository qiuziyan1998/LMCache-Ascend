# SPDX-License-Identifier: Apache-2.0
"""Bounded, asynchronous byte readback for direct Mooncake NPU imports.

Uses the existing load stream and DMA operator: at most three keys, 192 sample
bytes and 3 KiB of aligned DMA per request/import. The caller records its normal
readiness event after this readback, so cache reuse cannot overtake it. No host
wait, new thread, full tensor copy, or device arithmetic is introduced.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import ctypes
import zlib

# First Party
from lmcache.v1.prefill_audit import audit_record, sample_windows, sampled_key_indices


@dataclass
class _Readback:
    request: str
    keys: tuple[str, ...]
    layout: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    buffer: Any
    event: Any
    armed: bool = False


class DirectNpuAudit:
    """Retain only small host buffers until their existing-stream DMA ends."""

    def __init__(self, torch_api: Any, copy: Callable[..., Any]) -> None:
        self.torch, self.copy = torch_api, copy
        self.pending: list[_Readback] = []

    def submit(
        self,
        request: str,
        keys: Sequence[str],
        pointers: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
        stream: Any,
        owners: Sequence[Any] = (),
    ) -> None:
        """Enqueue a bounded readback before the caller records readiness.

        Errors are diagnostic records, not serving exceptions. If submission
        partially succeeds, retain its buffer rather than free in-flight DMA.
        """
        try:
            self.poll()
            selected = sampled_key_indices(keys, limit=3)
            if not selected:
                return
            bounds = []
            for owner in owners:
                storage = owner.untyped_storage()
                bounds.append(
                    (
                        int(storage.data_ptr()),
                        int(storage.data_ptr()) + storage.nbytes(),
                    )
                )
            if not bounds:
                bounds = sorted(
                    (int(p), int(p) + int(n))
                    for row, lengths in zip(pointers, sizes, strict=True)
                    for p, n in zip(row, lengths, strict=True)
                )
                merged = []
                for start, end in bounds:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else:
                        merged.append((start, end))
                bounds = merged
            segments, layout, aligned_regions = [], [], {}
            cursor = 0
            for index in selected:
                fragments = []
                for source, count in sample_windows(pointers[index], sizes[index]):
                    aligned_start = source // 64 * 64
                    aligned_end = (source + count + 63) // 64 * 64
                    if not any(
                        start <= aligned_start and aligned_end <= end
                        for start, end in bounds
                    ):
                        raise ValueError(
                            "aligned readback is outside owned NPU storage"
                        )
                    region = (aligned_start, aligned_end)
                    if region not in aligned_regions:
                        aligned_regions[region] = cursor
                        segments.append(
                            (cursor, aligned_start, aligned_end - aligned_start)
                        )
                        cursor += aligned_end - aligned_start
                    fragments.append(
                        (aligned_regions[region] + source - aligned_start, count)
                    )
                layout.append((sum(sizes[index]), tuple(fragments)))
            allocation = self.torch.empty(
                cursor + 63, dtype=self.torch.uint8, device="cpu", pin_memory=True
            )
            offset = (-int(allocation.data_ptr())) % 64
            host = allocation[offset : offset + cursor]
            event = self.torch.npu.Event()
            probe = _Readback(
                str(request),
                tuple(keys[i] for i in selected),
                tuple(layout),
                host,
                event,
            )
            self.pending.append(probe)
            with self.torch.npu.stream(stream):
                base = int(host.data_ptr())
                self.copy(
                    [
                        (base + offset, source, count)
                        for offset, source, count in segments
                    ],
                    True,
                )
                event.record(stream)
                probe.armed = True
        except Exception as error:
            audit_record(
                "coverage",
                stage="get_npu",
                req=request,
                error="direct_readback_submit_failed",
                detail=str(error)[:160],
            )

    def poll(self) -> None:
        """Emit completed probes only; never synchronize or spin on an event."""
        keep = []
        for probe in self.pending:
            if not probe.armed:
                keep.append(probe)
                continue
            try:
                if not probe.event.query():
                    keep.append(probe)
                    continue
                base = int(probe.buffer.data_ptr())
                samples = []
                for total, fragments in probe.layout:
                    checksum = 0
                    for offset, size in fragments:
                        checksum = zlib.crc32(
                            ctypes.string_at(base + offset, size), checksum
                        )
                    samples.append((total, f"{checksum:08x}"))
                audit_record(
                    "payload",
                    stage="get_npu",
                    req=probe.request,
                    keys=probe.keys,
                    samples=samples,
                )
            except Exception as error:
                # Preserve possibly in-flight storage on a device query error.
                probe.armed = False
                keep.append(probe)
                audit_record(
                    "coverage",
                    stage="get_npu",
                    req=probe.request,
                    error="direct_readback_query_failed",
                    detail=str(error)[:160],
                )
        self.pending = keep
