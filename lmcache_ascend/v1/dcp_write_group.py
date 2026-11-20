# SPDX-License-Identifier: Apache-2.0
"""Foreground-only agreement for cooperative writes into canonical CPU pages."""

from datetime import timedelta
from threading import get_ident
from typing import Any, Callable

import torch.distributed as dist


class DcpWriteGroup:
    """Isolate write control from model and asynchronous lookup collectives.

    No collective is issued on ordinary decode steps. Callers enter the same
    ordered save/checkpoint operation on every TP/DCP rank, on the model thread.
    Communication failure is fatal: it never proves native access has stopped.
    """

    def __init__(self, tp: Any, timeout: float) -> None:
        self.rank = tp.rank_in_group
        self.world_size = tp.world_size
        self.sessions = tuple(f"worker-{rank}" for rank in range(self.world_size))
        self.thread = get_ident()
        self.sequence = 0
        self.failed = False
        self.group = dist.new_group(
            ranks=tp.ranks,
            backend="gloo",
            timeout=timedelta(seconds=timeout),
            use_local_synchronization=True,
            group_desc="lmcache_dcp_write",
        )

    def exchange(self, phase: str, identity: tuple, value: Any) -> list[Any]:
        """Agree operation/phase and return value-only peer completion evidence."""
        if self.failed or get_ident() != self.thread:
            raise RuntimeError(
                "DCP write agreement is failed or called outside its model thread"
            )
        self.sequence += 1
        packet = (self.sequence, phase, identity, value)
        packets = [None] * self.world_size
        try:
            dist.all_gather_object(packets, packet, group=self.group)
            if any(
                not isinstance(p, tuple) or len(p) != 4 or p[:3] != packet[:3]
                for p in packets
            ):
                raise RuntimeError(
                    "DCP write participants disagree on operation or phase"
                )
            return [p[3] for p in packets]
        except BaseException:
            self.failed = True
            raise

    def leader(self, phase: str, identity: tuple, action: Callable[[], Any]) -> Any:
        """Publish rank-zero success or failure before a peer can wait on data.

        The original exception stays local; tracebacks/Futures/native objects
        must never be included in the returned action value.
        """
        error, value = None, None
        if self.rank == 0:
            try:
                value = (True, action())
            except BaseException as exc:
                error = exc
                value = (False, f"{type(exc).__name__}: {exc}")
        values = self.exchange(phase, identity, value)
        if any(v is not None for v in values[1:]) or not isinstance(values[0], tuple):
            self.failed = True
            raise RuntimeError("DCP write has an invalid leader outcome")
        success, result = values[0]
        if not success:
            raise RuntimeError(f"DCP {phase} failed: {result}") from error
        return result
