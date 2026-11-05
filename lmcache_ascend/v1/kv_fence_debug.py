# SPDX-License-Identifier: Apache-2.0
"""Opt-in, payload-free isolation of P/D device dependency boundaries.

LMCACHE_PREFILL_FENCE_DEBUG=check|p_load|p_use|p_save|d_load|all
Comma-separated stages are accepted. ``check`` never queries/waits on a device.
LMCACHE_PREFILL_FENCE_GROUPS=0,1 selects KV groups (default: both).
LMCACHE_PREFILL_FENCE_LAYERS=start:end selects group-local layer ordinals,
with an exclusive end (default: all). No tensors, files or worker threads are
created. Incompatible heavy diagnostics are rejected, not silently overridden.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Callable, Mapping, Sequence


_STAGES = frozenset(("p_load", "p_use", "p_save", "d_load"))
_ENV = "LMCACHE_PREFILL_FENCE_DEBUG"


class KVFenceDebug:
    """Check host handoffs and optionally synchronize one existing boundary."""

    @classmethod
    def from_environment(cls, role: str, rank: int) -> KVFenceDebug | None:
        """Return None when disabled; reject ambiguous/invalid experiments."""
        raw = os.environ.get(_ENV, "").strip().lower()
        if raw in ("", "0", "off"):
            return None
        return cls(role, rank, os.environ)

    def __init__(
        self,
        role: str,
        rank: int,
        env: Mapping[str, str],
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        raw = env.get(_ENV, "check").strip().lower()
        stages = set(raw.split(","))
        if stages == {"all"}:
            stages = set(_STAGES)
        if not stages or stages - (_STAGES | {"check"}):
            raise ValueError(f"Invalid {_ENV}: {raw!r}")
        for name in ("LMCACHE_PREFILL_STORE_DEBUG", "LMCACHE_PREFILL_AUDIT"):
            if env.get(name, "0") == "1":
                raise ValueError(f"Set {name}=0 when using {_ENV}")
        self.stages = frozenset(stages - {"check"})
        self.check_protocol = "check" in stages
        groups = env.get("LMCACHE_PREFILL_FENCE_GROUPS", "0,1")
        try:
            self.groups = frozenset(int(value) for value in groups.split(","))
            bounds = env.get("LMCACHE_PREFILL_FENCE_LAYERS", "0:").split(":")
            if len(bounds) != 2:
                raise ValueError("expected start:end")
            self.first = int(bounds[0]) if bounds[0] else 0
            self.end = int(bounds[1]) if bounds[1] else None
            if (
                not self.groups
                or self.groups - {0, 1}
                or self.first < 0
                or (self.end is not None and self.end <= self.first)
            ):
                raise ValueError("invalid range")
        except ValueError as exc:
            raise ValueError("Invalid LMCACHE_PREFILL_FENCE_GROUPS/LAYERS") from exc
        self.role = role
        self.base = dict(
            host=socket.gethostname(), pid=os.getpid(), rank=rank, role=role
        )
        self.emit = emit or self._print
        self.seen: set[tuple[str, int]] = set()
        self.expected: dict[tuple[int, int, int], set[str]] = {}
        self.submitted: dict[tuple[int, int, int], set[str]] = {}
        self.latest: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.consumed: set[tuple[int, int, int]] = set()
        self.step = 0
        self.begun = False
        self.emit(
            dict(
                self.base,
                event="enabled",
                stages=sorted(self.stages),
                groups=sorted(self.groups),
                layers=[self.first, self.end],
                payload_sampling=False,
                protocol_check=self.check_protocol,
            )
        )

    @staticmethod
    def _print(record: dict[str, Any]) -> None:
        print("[KVFENCE] " + json.dumps(record, separators=(",", ":")), flush=True)

    def fail(self, code: str, group: int, layer: int, bank: int, **fields: Any) -> None:
        """Emit one actionable protocol error and stop before using the bank."""
        record = dict(
            self.base,
            event="error",
            code=code,
            step=self.step,
            group=group,
            layer=layer,
            bank=bank,
            **fields,
        )
        self.emit(record)
        raise RuntimeError("KVFENCE " + json.dumps(record, separators=(",", ":")))

    def begin_step(
        self,
        requests: Sequence[Any],
        layer_counts: Sequence[int],
        bank_count: int,
        chunk_size: int,
    ) -> None:
        """Declare required loads from scheduler metadata before any submission.

        A true zero-load request is allowed to wait on an earlier save only.
        A declared hit must submit every request's load before the shared bank
        is consumed. Counts/indices use the same aligned mask as the adapter.
        """
        if self.role != "P" or not self.check_protocol:
            return
        if bank_count <= 0 or chunk_size <= 0:
            raise ValueError("KVFENCE requires positive bank/chunk sizes")
        self.step += 1
        self.begun = True
        self.expected.clear()
        self.submitted.clear()
        self.latest.clear()
        self.consumed.clear()
        for request in requests:
            spec = request.load_spec
            if (
                spec is None
                or not spec.can_load
                or request.is_sparse_decode
                or getattr(spec, "dsa_cold_compact_load", False)
            ):
                continue
            count = min(int(spec.lmcache_cached_tokens), len(request.token_ids))
            masked = min(int(spec.vllm_cached_tokens), count) // chunk_size * chunk_size
            if count <= masked:
                continue
            for group, layers in enumerate(layer_counts):
                for layer in range(layers):
                    key = (group, layer, layer % bank_count)
                    self.expected.setdefault(key, set()).add(request.req_id)

    def loaded(
        self,
        req: str,
        group: int,
        layer: int,
        bank: int,
        event: Any,
    ) -> None:
        """Remember actual submission order, including all requests on one FIFO."""
        if not self.check_protocol:
            return
        key = (group, layer, bank)
        if not self.begun:
            self.fail("missing_step_declaration", group, layer, bank, req=req)
        if key in self.consumed:
            self.fail("load_after_consume", group, layer, bank, req=req)
        self.submitted.setdefault(key, set()).add(req)
        self.latest.setdefault(key, {})[req] = event

    def before_read(
        self, group: int, layer: int, bank: int, events: Mapping[str, Any]
    ) -> None:
        """Reject absent/stale/wrong handoffs without inspecting NPU contents."""
        if not self.check_protocol:
            return
        key = (group, layer, bank)
        if not self.begun:
            self.fail("missing_step_declaration", group, layer, bank)
        if key in self.consumed:
            return  # The adapter can join a layer twice on the same compute stream.
        expected = self.expected.get(key, set())
        submitted = self.submitted.get(key, set())
        if expected != submitted:
            self.fail(
                "request_load_mismatch",
                group,
                layer,
                bank,
                missing=sorted(expected - submitted),
                unexpected=sorted(submitted - expected),
            )
        latest = self.latest.get(key, {})
        wrong = sorted(
            req for req in set(latest) | set(events)
            if latest.get(req) is not events.get(req)
        )
        if wrong:
            self.fail(
                "wrong_load_event", group, layer, bank,
                requests=wrong, expected_events=len(latest), actual_events=len(events),
            )
        self.consumed.add(key)
        self.latest.pop(key, None)
        self.submitted.pop(key, None)

    def wait(
        self,
        stage: str,
        group: int,
        layer: int,
        fence: Any,
        *,
        req: str | None = None,
        bank: int | None = None,
    ) -> None:
        """Wait only at the selected boundary, never by taking a global device fence.

        ``fence`` is an existing event or the exact transfer stream. Callers
        must place this immediately after that operation, before later work.
        """
        if (
            stage not in self.stages
            or not stage.startswith(self.role.lower())
            or group not in self.groups
            or layer < self.first
            or (self.end is not None and layer >= self.end)
        ):
            return
        marker = (stage, group)
        if marker not in self.seen:
            self.seen.add(marker)
            self.emit(
                dict(
                    self.base,
                    event="active",
                    stage=stage,
                    group=group,
                    layer=layer,
                    bank=bank,
                    req=req,
                )
            )
        try:
            fence.synchronize()
        except Exception as exc:
            self.fail(
                "device_fence_failed",
                group,
                layer,
                -1 if bank is None else bank,
                req=req,
                stage=stage,
                detail=str(exc)[:400],
            )
