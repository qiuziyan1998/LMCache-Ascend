# SPDX-License-Identifier: Apache-2.0
"""Synchronous, lossless D-receive archives. No hashing or diagnostic queue."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, Iterable, Sequence

import torch

FORMAT = "decode-kv-raw-v1"
ROWS_PER_COPY = 4096


def request_key(value: str) -> str:
    """Keep the public completion UUID even when PD adds private suffixes."""
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value)
    result = match[0].lower() if match else value
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", result) or result in (".", ".."):
        raise ValueError("Invalid snapshot request ID")
    return result


def packet_planes(tensor: torch.Tensor, widths: Sequence[int], interleaved: bool):
    """Expose raw per-token planes without copying/decoding floating values."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Expected a contiguous, received CPU packet")
    raw = tensor.detach().view(torch.uint8).reshape(-1)
    capacity, remainder = divmod(raw.numel(), sum(widths))
    if remainder or capacity == 0:
        raise ValueError("Packet bytes disagree with the registered layout")
    if interleaved:
        return raw.reshape(capacity, sum(widths)).split(tuple(widths), dim=1)
    return tuple(
        part.reshape(capacity, width)
        for part, width in zip(
            raw.split([capacity * w for w in widths]), widths, strict=True
        )
    )


def physical_indexer_slots(
    slots: torch.Tensor, *, c8: bool, factor: int, block_map: torch.Tensor | None
) -> torch.Tensor:
    """Translate logical 128-token C8 blocks exactly as the load planner does."""
    slots = slots.detach().to(device="cpu", dtype=torch.long)
    if slots.ndim != 1 or bool((slots < 0).any()):
        raise ValueError("Negative or non-vector indexer slot mapping")
    if not c8:
        return slots
    block, offset = slots // 128, slots % 128
    if block_map is not None:
        mapping = block_map.detach().to(device="cpu", dtype=torch.long)
        if len(block) and int(block.max()) >= mapping.numel():
            raise ValueError("C8 block is outside the physical block map")
        return mapping[block] * 128 + offset
    return block * (128 * factor) + offset


class RawSnapshot:
    """Write bounded chunks; publish complete only after every file is closed."""

    def __init__(
        self,
        root: str,
        req_id: str,
        tokens: Sequence[int],
        metadata: dict[str, Any],
        groups: Sequence[int],
    ):
        host = re.sub(r"[^a-zA-Z0-9_.-]", "_", socket.gethostname())
        worker = (
            f"{host}-dp{metadata['dp_rank']}-tp{metadata['tp_rank']}-p{os.getpid()}"
        )
        self.directory = Path(root) / request_key(req_id) / worker
        # An existing capture is never silently replaced by a later observation.
        self.directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(metadata)
        replay = metadata.pop("replay_request", None)
        runtime = metadata.pop("runtime", None)
        if replay is not None:
            (self.directory / "replay.json").write_text(
                json.dumps(dict(request=replay, runtime=runtime)), encoding="utf-8"
            )
            metadata["replay_file"] = "replay.json"
        self.meta = dict(
            metadata,
            format=FORMAT,
            request_id=req_id,
            host=host,
            pid=os.getpid(),
            byteorder=sys.byteorder,
            status="writing",
            tokens=len(tokens),
            groups=list(groups),
            files=[],
            errors=[],
        )
        (self.directory / "tokens.json").write_text(
            json.dumps(list(tokens)), encoding="utf-8"
        )
        self._publish()

    def _publish(self) -> None:
        temporary = self.directory / "manifest.tmp"
        temporary.write_text(
            json.dumps(self.meta, separators=(",", ":")), encoding="utf-8"
        )
        temporary.replace(self.directory / "manifest.json")

    def write_layer(
        self,
        group: int,
        layer: int,
        batches: Iterable[Sequence[torch.Tensor]],
        dtypes: Sequence[str],
        source: str,
    ) -> None:
        """Write each component in logical-token order, preserving every raw bit."""
        records = []
        with ExitStack() as stack:
            files = []
            widths = None
            count = 0
            for planes in batches:
                if len(planes) != len(dtypes) or not planes:
                    raise ValueError("Missing component in received layer")
                raw = [
                    p.detach().contiguous().view(torch.uint8).reshape(p.shape[0], -1)
                    for p in planes
                ]
                current_widths = [p.shape[1] for p in raw]
                if widths is None:
                    widths = current_widths
                    for component, (width, dtype) in enumerate(
                        zip(widths, dtypes, strict=True)
                    ):
                        name = f"g{group}-l{layer}-c{component}.bin"
                        files.append(
                            stack.enter_context((self.directory / name).open("xb"))
                        )
                        records.append(
                            dict(
                                group=group,
                                layer=layer,
                                component=component,
                                row_bytes=width,
                                dtype=dtype,
                                path=name,
                                source=source,
                            )
                        )
                if current_widths != widths or len({p.shape[0] for p in raw}) != 1:
                    raise ValueError("Inconsistent component rows or widths")
                for stream, plane in zip(files, raw, strict=True):
                    if plane.device.type != "cpu":
                        raise ValueError("Readback must complete before file writing")
                    # memoryview avoids another whole-layer byte-string allocation.
                    data = memoryview(plane.numpy()).cast("B")
                    if stream.write(data) != len(data):
                        raise OSError("Short snapshot write")
                count += raw[0].shape[0]
            if count != self.meta["tokens"]:
                raise ValueError(
                    f"Incomplete layer: got {count}, expected {self.meta['tokens']}"
                )
        for record in records:
            record.update(start=0, end=count, bytes=count * record["row_bytes"])
        self.meta["files"].extend(records)
        self._publish()

    def finish(self) -> None:
        expected = {
            (g, layer)
            for g, layers in enumerate(self.meta["groups"])
            for layer in range(layers)
        }
        actual = {(r["group"], r["layer"]) for r in self.meta["files"]}
        if actual != expected:
            raise ValueError("Not all registered layers were archived")
        self.meta["status"] = "complete"
        self._publish()

    def fail(self, error: Exception) -> None:
        self.meta["status"] = "error"
        self.meta["errors"].append(f"{type(error).__name__}: {error}")
        self._publish()


def cpu_batches(
    tensors: Iterable[torch.Tensor],
    counts: Sequence[int],
    widths: Sequence[int],
    interleaved: bool,
):
    """Exclude unused packet capacity but keep all valid token/component bytes."""
    for tensor, count in zip(tensors, counts, strict=True):
        planes = packet_planes(tensor, widths, interleaved)
        if count <= 0 or count > planes[0].shape[0]:
            raise ValueError("Chunk coverage exceeds packet capacity")
        for start in range(0, count, ROWS_PER_COPY):
            yield tuple(p[start : min(count, start + ROWS_PER_COPY)] for p in planes)


def paged_batches(planes: Sequence[torch.Tensor], slots: torch.Tensor):
    """Bound NPU readback to 4096 rows; never copy the entire physical cache."""
    if not planes or any(p.ndim < 3 for p in planes):
        raise ValueError("Expected block/token/component cache planes")
    if len(slots) and (
        int(slots.min()) < 0
        or any(int(slots.max()) >= p.shape[0] * p.shape[1] for p in planes)
    ):
        raise ValueError("Physical indexer slot outside its cache allocation")
    for start in range(0, len(slots), ROWS_PER_COPY):
        indices = slots[start : start + ROWS_PER_COPY]
        result = []
        for plane in planes:
            device_indices = indices.to(plane.device)
            # Advanced indexing also handles strided plane views. Flattening a
            # strided view could accidentally copy the entire physical cache.
            result.append(
                plane[
                    device_indices // plane.shape[1], device_indices % plane.shape[1]
                ].to("cpu")
            )
        yield tuple(result)
