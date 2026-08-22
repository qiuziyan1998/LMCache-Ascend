# SPDX-License-Identifier: Apache-2.0
"""Opt-in content fingerprints for Group-1 NPU cache diagnostics.

The production data path must never copy NPU tensors to the host for
diagnostics.  All public entry points in this module are therefore guarded by
``enable_npu_content_diagnostics`` and are no-ops until the LMCache connector
configures that single knob.  Inline fingerprints explicitly report that the
host readback may synchronize the NPU.  Attention-side snapshots are cloned on
the current NPU stream and read back only after sampling, draft proposal, and
async state updates, so diagnostics cannot synchronize the serving workflow.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
import hashlib
import importlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)

CONTENT_DIAGNOSTIC_SCHEMA = 1
CONTENT_DIAGNOSTIC_ALGORITHM = "blake2b-128"
MAX_LOGICAL_TOKEN_SAMPLES = 24
MAX_LAYER_SAMPLES = 3
MAX_TRACKED_REQUEST_LAYERS = 4096
MAX_PENDING_SNAPSHOTS = 512

_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
_STATE_LOCK = threading.Lock()
_ENABLED = False
_PENDING: list["_DeferredSnapshot"] = []
_SEEN: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_REQUEST_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_GROUP0_SOURCE_PROBES: OrderedDict[
    tuple[str, int], "_Group0SourceProbe"
] = OrderedDict()
_STORE_PROBE_STREAMS: dict[int, Any] = {}
_PENDING_STORE_PROBES: list["_PendingStoreProbe"] = []


@dataclass(slots=True)
class _Group0SourceProbe:
    req_id: str
    layer_id: int
    source_chunks: tuple[torch.Tensor, ...]
    selected_tokens: torch.Tensor
    selected_count: torch.Tensor | None
    chunk_size: int
    total_tokens: int
    token_major: bool
    plane_widths: tuple[int, ...]
    destination_stride: int


@dataclass(slots=True)
class _DeferredSnapshot:
    event: str
    fields: dict[str, Any]
    values: torch.Tensor | None = None
    physical_slots: torch.Tensor | None = None
    expected_rows: dict[tuple[int, int], str] | None = None
    components: dict[str, torch.Tensor] | None = None
    comparison_components: dict[str, torch.Tensor] | None = None
    group0_source_probes: tuple[_Group0SourceProbe, ...] | None = None


@dataclass(slots=True)
class _PendingStoreProbe:
    """Deferred store-time source probe captured on an independent stream.

    ``as_put`` rows mirror the persistent put's actual fence contract (the
    producer events handed to the connector), while ``producer_fenced`` rows
    additionally wait on the attention producer events the put could have
    used.  Both captures are device-side only; host readback happens at flush
    time on the probe's own stream, so the probe can neither synchronize nor
    reorder the serving or store workflow.
    """

    req_id: str
    stream: Any
    page_ranges: tuple[tuple[int, int], ...]
    put_producer_event_count: int
    producer_fence_events: tuple[Any, ...]
    ready_event_source: str
    tokens_by_group: dict[int, list[int]]
    slot_tensors: dict[int, Any]
    rows: dict[tuple[str, int, int, int], torch.Tensor]


def configure_npu_content_diagnostics(enabled: bool) -> None:
    """Configure the process-local diagnostic gate from LMCache config.

    Args:
        enabled: Exact value of ``enable_npu_content_diagnostics``.

    Notes:
        Disabling the knob also releases all request-scoped diagnostic state.
        This function is called while the worker connector is initialized,
        before inference begins.
    """
    global _ENABLED
    with _STATE_LOCK:
        _ENABLED = bool(enabled)
        if not _ENABLED:
            _PENDING.clear()
            _SEEN.clear()
            _REQUEST_SPECS.clear()
            _GROUP0_SOURCE_PROBES.clear()
            _PENDING_STORE_PROBES.clear()
    _configure_vllm_diagnostic_bridge(_ENABLED)
    if _ENABLED:
        _content_log(
            "content_diagnostics_enabled",
            default_enabled=False,
            inline_readback_may_mask_ordering_issue=True,
            deferred_attention_readback=True,
        )


def npu_content_diagnostics_enabled() -> bool:
    """Return whether content diagnostics are enabled in this worker."""
    return _ENABLED


def _configure_vllm_diagnostic_bridge(enabled: bool) -> None:
    """Install callbacks without importing LMCache from vLLM model modules."""
    module_name = "vllm_ascend.lmcache_diagnostics"
    try:
        if enabled:
            bridge = importlib.import_module(module_name)
            bridge.install_npu_content_diagnostic_callbacks(
                bridge.NPUContentDiagnosticCallbacks(
                    begin_deferred_step=begin_deferred_diagnostic_step,
                    flush_deferred=flush_deferred_diagnostics,
                    fingerprint_compact_group1=fingerprint_compact_group1,
                    register_group1_source=register_group1_source_fingerprint,
                    queue_group1_first_consume=queue_group1_first_consume,
                    queue_cache_tail=queue_cache_tail_fingerprint,
                    queue_selected_topk=queue_selected_topk_fingerprint,
                    queue_staged_graph_stage=(
                        queue_staged_graph_stage_fingerprint
                    ),
                )
            )
        else:
            bridge = sys.modules.get(module_name)
            if bridge is not None:
                bridge.clear_npu_content_diagnostic_callbacks()
    except Exception:
        # Diagnostics are optional and must never prevent worker startup.
        logger.warning(
            "Failed to configure optional vLLM-Ascend content diagnostics",
            exc_info=True,
        )


def _field(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _content_log(event: str, **fields: Any) -> None:
    if not _ENABLED:
        return
    payload = {
        "schema": CONTENT_DIAGNOSTIC_SCHEMA,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        **fields,
    }
    logger.info(
        "[LMCACHE_NPU_CONTENT_DIAG] %s",
        json.dumps(payload, default=str, separators=(",", ":")),
    )


def log_npu_content_diagnostic_event(event: str, **fields: Any) -> None:
    """Emit a metadata-only diagnostic event behind the single opt-in gate.

    Callers must not pass tensors.  This helper exists for causal-order
    diagnostics (for example NPU event readiness) that do not require device
    readback and therefore cannot themselves repair a missing synchronization.
    """
    _content_log(event, **fields)


def _nonfatal_diagnostic(
    event: str,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Prevent optional diagnostic failures from escaping into inference."""

    def decorate(function: Callable[..., None]) -> Callable[..., None]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> None:
            if not _ENABLED:
                return
            try:
                function(*args, **kwargs)
            except Exception as error:
                error_event = (
                    "group0_source_probe_error"
                    if event == "group0_source_payload"
                    else "group1_fingerprint_error"
                )
                context = (
                    {
                        "req_id": kwargs.get("req_id"),
                        "layer_id": kwargs.get("layer_id"),
                    }
                    if event == "group0_source_payload"
                    else {}
                )
                _content_log(
                    error_event,
                    requested_event=event,
                    error_type=type(error).__name__,
                    error=str(error),
                    readback_mode="deferred_after_model_forward",
                    readback_may_synchronize=False,
                    **context,
                )

        return wrapped

    return decorate


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _hash_bytes(value: bytes) -> str:
    return hashlib.blake2b(value, digest_size=16).hexdigest()


def _hash_cpu_tensor(tensor: torch.Tensor) -> str:
    return _hash_bytes(_raw_bytes(tensor))


def _ordered_sample(values: Iterable[int], limit: int) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if len(ordered) <= limit:
        return ordered
    # Keep both ends and sample the interior evenly.  This is deterministic
    # and retains early/late discontinuities without growing wire metadata.
    indices = {
        round(position * (len(ordered) - 1) / (limit - 1)) for position in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def compact_sample_spec(
    layers: Sequence[Any],
    runs: Sequence[Any],
    token_count: int,
    chunk_size: int,
) -> tuple[list[int], list[int]]:
    """Choose bounded layer/token samples around page and slot-run boundaries.

    Args:
        layers: Ordered compact-layout layer descriptions.
        runs: Ordered logical-to-physical slot runs.
        token_count: Number of logical tokens covered by the layout.
        chunk_size: LMCache page/chunk size.

    Returns:
        A pair of sampled layer IDs and logical token positions.
    """
    if token_count <= 0 or not layers or not runs:
        return [], []
    layer_ids = [int(_field(layer, "layer_id")) for layer in layers]
    sampled_layers = _ordered_sample(
        (layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]),
        MAX_LAYER_SAMPLES,
    )
    candidates = {0, 1, token_count - 2, token_count - 1}
    if chunk_size > 0:
        for boundary in range(chunk_size, token_count, chunk_size):
            candidates.update((boundary - 1, boundary))
    run_boundaries = [int(_field(run, "logical_token_start")) for run in runs[1:]]
    # Prefer discontinuities at both ends when a request has many compact
    # runs.  The final bounded sampler still covers the interior.
    for boundary in run_boundaries:
        candidates.update((boundary - 1, boundary))
    sampled_tokens = _ordered_sample(
        (value for value in candidates if 0 <= value < token_count),
        MAX_LOGICAL_TOKEN_SAMPLES,
    )
    return sampled_layers, sampled_tokens


def _logical_to_physical(
    runs: Sequence[Any], logical_tokens: Sequence[int]
) -> list[int]:
    result: list[int] = []
    run_index = 0
    for logical_token in logical_tokens:
        while run_index < len(runs):
            run = runs[run_index]
            start = int(_field(run, "logical_token_start"))
            count = int(_field(run, "token_count"))
            if logical_token < start + count:
                if logical_token < start:
                    raise ValueError("Compact diagnostic token precedes run")
                result.append(
                    int(_field(run, "physical_slot_start")) + logical_token - start
                )
                break
            run_index += 1
        else:
            raise ValueError("Compact diagnostic token is outside run coverage")
    return result


def _owner_by_base(owners: Iterable[torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(owner.data_ptr()): owner for owner in owners}


def _wire_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded, portable portion carried with a live descriptor."""
    return {
        "schema": result["schema"],
        "algorithm": result["algorithm"],
        "sample_layer_ids": result["sample_layer_ids"],
        "sample_logical_tokens": result["sample_logical_tokens"],
        "row_hashes": result["row_hashes"],
        "layer_hashes": result["layer_hashes"],
        "combined_hash": result["combined_hash"],
        "readback_may_synchronize": True,
    }


def _store_probe_stream(device: torch.device) -> Any:
    """Return the per-device stream used for store-time source probes.

    The stream is deliberately independent of every compute stream: a probe
    enqueued on it carries no ordering dependency on attention scatter work,
    which is exactly the read contract of the transfer engine's fenced DMA.
    """
    index = device.index if device.index is not None else torch.npu.current_device()
    with _STATE_LOCK:
        stream = _STORE_PROBE_STREAMS.get(int(index))
        if stream is None:
            stream = torch.npu.Stream(device=int(index))
            _STORE_PROBE_STREAMS[int(index)] = stream
        return stream


def _wait_probe_event(stream: Any, event: Any) -> None:
    """Order the probe stream behind a producer event without host sync."""
    wait_event = getattr(stream, "wait_event", None)
    if callable(wait_event):
        wait_event(event)
        return
    event.wait(stream)


def _store_probe_layers(layers: Sequence[Any]) -> list[int]:
    """Return the bounded first/middle/last layer sample for a KV group."""
    num_layers = len(layers)
    if num_layers <= 0:
        return []
    return _ordered_sample(
        (0, num_layers // 2, num_layers - 1),
        MAX_LAYER_SAMPLES,
    )


def _store_probe_planes(entry: Any) -> tuple[Any, ...]:
    """Normalize one per-layer cache entry into its tensor planes."""
    if isinstance(entry, (tuple, list)):
        return tuple(entry)
    return (entry,)


def queue_store_time_source_fingerprint(
    *,
    req_id: str,
    group_caches: Mapping[int, Sequence[Any]],
    slot_mappings: Mapping[int, Any],
    slot_mapping_base: int,
    page_ranges: Sequence[tuple[int, int]],
    put_producer_events: Any = None,
    producer_fence_events: Any = None,
    ready_event_source: str = "missing",
) -> None:
    """Queue a store-time fingerprint of the direct-store NPU source rows.

    The persistent direct-NPU put synchronizes only the producer events it is
    handed and then reads the registered source memory without any torch
    stream ordering.  This probe mirrors that contract: ``as_put`` rows are
    captured on an independent stream gated by exactly the events the put
    receives, and ``producer_fenced`` rows wait on the attention producer
    fence the put could have used.  Both captures are device-side clones, so
    an ordering race is observed rather than masked; host readback is
    deferred to :func:`flush_deferred_diagnostics` on the probe stream alone.

    Args:
        req_id: Request identifier for the store batch.
        group_caches: Per-KV-group list of per-layer cache tensors or plane
            tuples.
        slot_mappings: Per-KV-group token-to-slot mapping for this store
            window.
        slot_mapping_base: Absolute token index of the mapping window start.
        page_ranges: Absolute ``(start, end)`` token ranges of the pages in
            the batch.
        put_producer_events: The fence events the persistent put will
            actually synchronize on.
        producer_fence_events: The attention producer fence the put could
            have used (counterfactual capture).
        ready_event_source: Provenance of the resolved producer event.

    Notes:
        The function is a strict no-op unless the single content-diagnostic
        config knob is enabled, and any failure is logged and swallowed.
    """
    if not _ENABLED or not group_caches or not page_ranges:
        return
    started = time.perf_counter()
    try:
        candidates: set[int] = set()
        for range_start, range_end in page_ranges:
            range_start = int(range_start)
            range_end = int(range_end)
            candidates.update((range_start, range_end - 1))
            if range_end - range_start > 2:
                candidates.add((range_start + range_end - 1) // 2)
        tokens = _ordered_sample(candidates, MAX_LOGICAL_TOKEN_SAMPLES)
        if not tokens:
            return
        device = None
        for layers in group_caches.values():
            for entry in layers:
                tensor = _store_probe_planes(entry)[0]
                if isinstance(tensor, torch.Tensor):
                    device = tensor.device
                    break
            if device is not None:
                break
        if device is None or device.type != "npu":
            return
        stream = _store_probe_stream(device)
        base = int(slot_mapping_base)

        def _as_events(value: Any) -> tuple[Any, ...]:
            if isinstance(value, (tuple, list)):
                return tuple(event for event in value if event is not None)
            return () if value is None else (value,)

        put_events = _as_events(put_producer_events)
        fence_events = _as_events(producer_fence_events)

        tokens_by_group: dict[int, list[int]] = {}
        slot_tensors: dict[int, Any] = {}
        rows: dict[tuple[str, int, int, int], torch.Tensor] = {}
        captures: list[tuple[int, int, int, torch.Tensor, Any]] = []
        with torch.npu.stream(stream):
            # Mirror the put's actual fence: it synchronizes exactly these
            # events before the DMA reads registered source memory.
            for event in put_events:
                _wait_probe_event(stream, event)
            for group, layers in sorted(group_caches.items()):
                mapping = (
                    slot_mappings.get(group)
                    if hasattr(slot_mappings, "get")
                    else None
                )
                if not isinstance(mapping, torch.Tensor):
                    continue
                group_tokens = [
                    token
                    for token in tokens
                    if 0 <= token - base < int(mapping.numel())
                ]
                if not group_tokens:
                    continue
                index_tensor = torch.tensor(
                    [token - base for token in group_tokens],
                    dtype=torch.long,
                    device=mapping.device,
                )
                slot_tensor = (
                    mapping.detach()
                    .reshape(-1)
                    .index_select(0, index_tensor)
                    .long()
                )
                slot_gpu = slot_tensor.to(device)
                tokens_by_group[int(group)] = group_tokens
                slot_tensors[int(group)] = slot_gpu
                for layer_id in _store_probe_layers(layers):
                    entry = layers[int(layer_id)]
                    for plane_index, plane in enumerate(
                        _store_probe_planes(entry)
                    ):
                        if (
                            not isinstance(plane, torch.Tensor)
                            or plane.ndim < 2
                        ):
                            continue
                        # Match the attention scatter's own flattened view
                        # so slot indices address exactly the bytes the DMA
                        # reads for this (layer, token).
                        flat = plane.detach().reshape(-1, plane.shape[-1])
                        captures.append(
                            (
                                int(group),
                                int(layer_id),
                                plane_index,
                                flat,
                                slot_gpu,
                            )
                        )
                        rows[("as_put", int(group), int(layer_id), plane_index)] = (
                            flat.index_select(0, slot_gpu).detach()
                        )
            if fence_events and captures:
                for event in fence_events:
                    _wait_probe_event(stream, event)
                for group, layer_id, plane_index, flat, slot_gpu in captures:
                    rows[
                        ("producer_fenced", group, layer_id, plane_index)
                    ] = flat.index_select(0, slot_gpu).detach()

        probe = _PendingStoreProbe(
            req_id=str(req_id),
            stream=stream,
            page_ranges=tuple(
                (int(range_start), int(range_end))
                for range_start, range_end in page_ranges
            ),
            put_producer_event_count=len(put_events),
            producer_fence_events=fence_events,
            ready_event_source=str(ready_event_source),
            tokens_by_group=tokens_by_group,
            slot_tensors=slot_tensors,
            rows=rows,
        )
        dropped = None
        with _STATE_LOCK:
            if len(_PENDING_STORE_PROBES) >= MAX_PENDING_SNAPSHOTS:
                dropped = _PENDING_STORE_PROBES.pop(0)
            _PENDING_STORE_PROBES.append(probe)
        if dropped is not None:
            _content_log(
                "content_diagnostic_snapshot_dropped",
                dropped_event="store_time_source_fingerprint",
                reason="pending_limit",
            )
        _content_log(
            "store_time_source_probe_queued",
            req_id=str(req_id),
            tokens=tokens,
            groups=sorted(tokens_by_group),
            put_producer_event_count=len(put_events),
            producer_fence_event_count=len(fence_events),
            ready_event_source=str(ready_event_source),
            capture_order=(
                "as_put_on_probe_stream_then_producer_fenced_on_probe_stream"
            ),
            readback_mode="deferred_after_model_forward_on_probe_stream",
            readback_may_synchronize=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    except Exception as error:
        _content_log(
            "store_time_source_probe_error",
            req_id=str(req_id),
            stage="queue",
            error_type=type(error).__name__,
            error=str(error),
        )


def _flush_store_probes() -> None:
    """Read queued store probes on their own stream and log fingerprints."""
    with _STATE_LOCK:
        probes = list(_PENDING_STORE_PROBES)
        _PENDING_STORE_PROBES.clear()
    for probe in probes:
        started = time.perf_counter()
        try:
            rows_cpu: dict[tuple[str, int, int, int], torch.Tensor] = {}
            slots_cpu: dict[int, list[int]] = {}
            # Readback rides the probe stream so it can never order or
            # synchronize the compute stream; the synchronize below covers
            # only the probe's own tiny copies.
            with torch.npu.stream(probe.stream):
                for key, value in probe.rows.items():
                    rows_cpu[key] = value.detach().to("cpu").contiguous()
                for group, tensor in probe.slot_tensors.items():
                    slots_cpu[int(group)] = (
                        tensor.detach().to("cpu").reshape(-1).tolist()
                    )
            probe.stream.synchronize()
            summaries: list[dict[str, Any]] = []
            zero_rows = {"as_put": 0, "producer_fenced": 0}
            total_rows = {"as_put": 0, "producer_fenced": 0}
            for key in sorted(rows_cpu):
                variant, group, layer_id, plane_index = key
                group_tokens = probe.tokens_by_group.get(group, ())
                rows = rows_cpu[key]
                row_hashes: dict[str, str] = {}
                zero_tokens: list[int] = []
                for index, token in enumerate(group_tokens):
                    if index >= int(rows.shape[0]):
                        break
                    row = rows[index]
                    # Disambiguated key so prefiller-stored and decoder-loaded
                    # rows compare mechanically across hosts: group 0 has two
                    # planes and group 1 one, which collide under a bare
                    # "{layer}:{token}" key.
                    row_hashes[
                        f"{group}:{layer_id}:{plane_index}:{int(token)}"
                    ] = _hash_cpu_tensor(row)
                    if int(torch.count_nonzero(row).item()) == 0:
                        zero_tokens.append(int(token))
                zero_rows[variant] += len(zero_tokens)
                total_rows[variant] += len(row_hashes)
                summaries.append(
                    {
                        "variant": variant,
                        "kv_group": group,
                        "layer_id": layer_id,
                        "plane": plane_index,
                        "tokens": [int(token) for token in group_tokens],
                        "row_hashes": row_hashes,
                        "zero_tokens": zero_tokens,
                        "all_zero": bool(
                            zero_tokens
                            and len(zero_tokens) == len(row_hashes)
                        ),
                    }
                )
            _content_log(
                "store_time_source_fingerprint",
                req_id=probe.req_id,
                page_ranges=[list(r) for r in probe.page_ranges],
                put_producer_event_count=probe.put_producer_event_count,
                producer_fence_event_count=len(probe.producer_fence_events),
                ready_event_source=probe.ready_event_source,
                slots={
                    str(group): values
                    for group, values in sorted(slots_cpu.items())
                },
                layer_summaries=summaries,
                as_put_zero_rows=zero_rows["as_put"],
                as_put_total_rows=total_rows["as_put"],
                producer_fenced_zero_rows=zero_rows["producer_fenced"],
                producer_fenced_total_rows=total_rows["producer_fenced"],
                readback_mode="deferred_after_model_forward_on_probe_stream",
                readback_may_synchronize=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as error:
            _content_log(
                "store_time_source_probe_error",
                req_id=probe.req_id,
                stage="flush",
                error_type=type(error).__name__,
                error=str(error),
            )


def _page_layer_tensor(
    page: Any,
    layer_id: int,
    slab: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return one layer's CPU tensor view for a shared layer page.

    Rank0 page objects carry their own ``raw_data`` (the allocated slab
    slice); passive page views have ``raw_data=None`` and address the shared
    slab through ``meta.address`` plus the page-relative group prefix sums.
    """
    prefix = getattr(page, "group_prefix_sum", ())
    meta = getattr(page, "meta", None)
    if (
        meta is None
        or layer_id + 1 >= len(prefix)
        or not meta.dtypes
        or not meta.shapes
        or layer_id >= len(meta.dtypes)
        or layer_id >= len(meta.shapes)
    ):
        return None
    begin, end = int(prefix[layer_id]), int(prefix[layer_id + 1])
    raw_data = getattr(page, "raw_data", None)
    if isinstance(raw_data, torch.Tensor):
        base = raw_data
    else:
        if not isinstance(slab, torch.Tensor):
            return None
        address = int(meta.address)
        base = slab[address : address + int(meta.phy_size)]
    try:
        return base[begin:end].view(meta.dtypes[layer_id]).view(
            meta.shapes[layer_id]
        )
    except (RuntimeError, ValueError):
        return None


def log_shared_page_source_fingerprint(
    *,
    req_id: str,
    phase: str,
    kv_group: int,
    pages: Sequence[Any],
    chunk_starts: Sequence[int] | None = None,
    slab: torch.Tensor | None = None,
    rank: int,
    passive: bool,
) -> None:
    """Fingerprint shared-slab page sources before the H2D copy reads them.

    The slab is host memory and the producing GET/broadcast completed before
    materialization, so this reads CPU bytes only: it never touches an NPU
    stream and cannot perturb load/compute ordering.  Zero-content findings at
    this point therefore reflect slab content or view offset arithmetic, not a
    synchronization race.  Row hashes use the ``"{layer}:{token}"`` key format
    of the store-time source fingerprint when ``chunk_starts`` is provided,
    so prefiller-stored and decoder-loaded rows compare directly.

    Args:
        req_id: Request identifier.
        phase: Shared-CPU phase (e.g. ``sparse_decode_bootstrap``).
        kv_group: KV group of the pages (probe targets group 1).
        pages: Layer page objects materialized for this request.
        chunk_starts: Absolute first-token index per page, when aligned.
        slab: Passive shared slab tensor, required for passive views.
        rank: Reporting worker rank.
        passive: Whether this rank materialized passive views.

    Notes:
        Strict no-op unless the single content-diagnostic config knob is
        enabled; failures are logged and swallowed.
    """
    if not _ENABLED or not pages or int(kv_group) != 1:
        return
    started = time.perf_counter()
    try:
        first = pages[0]
        num_layers = int(getattr(first, "num_layers", 0) or 0)
        if num_layers <= 0:
            return
        layer_ids = _ordered_sample(
            (0, num_layers // 2, num_layers - 1),
            MAX_LAYER_SAMPLES,
        )
        zero_pages_by_layer = {int(layer): 0 for layer in layer_ids}
        page_reports: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages):
            valid_tokens = int(getattr(page, "valid_tokens", 0) or 0)
            local_tokens = (
                _ordered_sample(
                    (0, valid_tokens // 2, valid_tokens - 1),
                    3,
                )
                if valid_tokens > 0
                else []
            )
            chunk_start = (
                int(chunk_starts[page_index])
                if chunk_starts is not None and page_index < len(chunk_starts)
                else None
            )
            layer_reports: dict[str, Any] = {}
            for layer_id in layer_ids:
                tensor = _page_layer_tensor(page, int(layer_id), slab)
                if tensor is None:
                    layer_reports[str(int(layer_id))] = {
                        "error": "unresolved_layer_view"
                    }
                    continue
                nonzero = int(torch.count_nonzero(tensor).item())
                elements = int(tensor.numel())
                row_hashes: dict[str, str] = {}
                if local_tokens:
                    try:
                        rows = tensor.reshape(valid_tokens, -1)
                        for local_token in local_tokens:
                            digest = _hash_cpu_tensor(rows[local_token])
                            if chunk_start is not None:
                                # Same disambiguated key format as the
                                # prefiller store-time fingerprint so the two
                                # hosts' row hashes compare directly.
                                row_hashes[
                                    f"{int(kv_group)}:{int(layer_id)}:0:"
                                    f"{chunk_start + local_token}"
                                ] = digest
                            else:
                                row_hashes[
                                    f"L{int(layer_id)}:p{page_index}"
                                    f"t{local_token}"
                                ] = digest
                    except (RuntimeError, ValueError):
                        row_hashes = {}
                if nonzero == 0:
                    zero_pages_by_layer[int(layer_id)] += 1
                layer_reports[str(int(layer_id))] = {
                    "nonzero": nonzero,
                    "elements": elements,
                    "layer_size": int(getattr(page, "layer_size", 0) or 0),
                    "prefix": int(page.group_prefix_sum[int(layer_id)]),
                    "address": int(page.meta.address),
                    "valid_tokens": valid_tokens,
                    "row_hashes": row_hashes,
                    "all_zero": nonzero == 0,
                }
            page_reports.append(
                {
                    "page": page_index,
                    "chunk_start": chunk_start,
                    "layers": layer_reports,
                }
            )
        _content_log(
            "group1_page_source_fingerprint",
            req_id=str(req_id),
            phase=str(phase),
            kv_group=int(kv_group),
            rank=int(rank),
            passive=bool(passive),
            pages=len(pages),
            sampled_layers=[int(layer) for layer in layer_ids],
            zero_pages_by_layer={
                str(int(layer)): count
                for layer, count in zero_pages_by_layer.items()
            },
            page_reports=page_reports,
            readback_mode="inline_cpu_slab_only",
            readback_may_synchronize=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    except Exception as error:
        _content_log(
            "group1_page_source_probe_error",
            req_id=str(req_id),
            stage="log",
            error_type=type(error).__name__,
            error=str(error),
        )


def log_group1_pointer_table(
    *,
    req_id: str,
    phase: str,
    kv_group: int,
    pages: Sequence[Any],
    dev_ptr_rows: Sequence[Sequence[int]],
    rank: int,
) -> None:
    """Validate the dense-direct H2D per-layer source pointer table.

    The dense-direct kernel is layer-agnostic: for one layer it reads
    ``dev_ptr_rows[layer][chunk]`` and scatters to that layer's destination.
    A layer-specific corruption (e.g. only the last DSA index layer reading
    zeros while other layers of the same pages read real bytes) therefore
    localizes to this table's per-layer rows, or to the device registration
    behind them.  This check validates, per chunk:

    - row deltas: ``row[layer][k] - row[0][k]`` must equal the page's own
      ``group_prefix_sum[layer]`` (the same offset the CPU views use);
    - registration offset: ``row[0][k] - page_k.layer_data_ptr(0)`` (device
      minus host address) must be constant across chunks — a varying offset
      means the device pointers do not map 1:1 onto the host slab;
    - raggedness: every layer row must have the same chunk count.

    All values are host-side integers captured after the table is built;
    nothing here touches the device.

    Args:
        req_id: Request identifier.
        phase: Shared-CPU phase of the load.
        kv_group: KV group of the pages (probe targets group 1).
        pages: Materialized layer page objects (chunk order).
        dev_ptr_rows: Per-layer source device pointer rows.
        rank: Reporting worker rank.

    Notes:
        Strict no-op unless the content-diagnostic knob is enabled.
    """
    if not _ENABLED or int(kv_group) != 1 or not pages or not dev_ptr_rows:
        return
    started = time.perf_counter()
    try:
        num_layers = len(dev_ptr_rows)
        num_chunks = len(pages)
        ragged = sorted(
            {
                len(row) if row is not None else -1
                for row in dev_ptr_rows
            }
        )
        layer_ids = _ordered_sample(
            (0, num_layers // 2, num_layers - 1),
            MAX_LAYER_SAMPLES,
        )
        host_bases = [int(page.layer_data_ptr(0)) for page in pages]
        registration_offsets = []
        table_ok = True
        layer_reports: dict[str, Any] = {}
        for layer_id in layer_ids:
            row = (
                dev_ptr_rows[int(layer_id)]
                if int(layer_id) < len(dev_ptr_rows)
                else None
            )
            if row is None or len(row) < num_chunks:
                layer_reports[str(int(layer_id))] = {
                    "error": "missing_or_short_row",
                    "row_len": len(row) if row is not None else None,
                }
                table_ok = False
                continue
            # Each page carries its own prefix sums: full pages share the
            # chunk-size stride, but the partial tail page has a smaller
            # per-layer stride. Compare every chunk against its own page's
            # prefix, not page 0's.
            deltas = [int(row[k]) - int(dev_ptr_rows[0][k]) for k in range(num_chunks)]
            delta_mismatches = [
                k
                for k, delta in enumerate(deltas)
                if delta != int(pages[k].group_prefix_sum[int(layer_id)])
            ]
            if int(layer_id) == 0:
                registration_offsets = [
                    int(row[k]) - host_bases[k] for k in range(num_chunks)
                ]
            if delta_mismatches:
                table_ok = False
            layer_reports[str(int(layer_id))] = {
                "expected_delta": expected_delta,
                "deltas_preview": deltas[:4],
                "delta_mismatch_chunks": delta_mismatches[:8],
                "row_preview": [int(value) for value in row[:4]],
            }
        host_strides = [
            host_bases[k + 1] - host_bases[k] for k in range(num_chunks - 1)
        ]
        _content_log(
            "group1_pointer_table",
            req_id=str(req_id),
            phase=str(phase),
            kv_group=int(kv_group),
            rank=int(rank),
            num_layers=num_layers,
            num_chunks=num_chunks,
            ragged_row_lengths=ragged,
            host_base_preview=host_bases[:4],
            host_strides_preview=host_strides[:4],
            registration_offsets=registration_offsets,
            registration_offset_constant=(
                len(set(registration_offsets)) == 1
                if registration_offsets
                else None
            ),
            layer_reports=layer_reports,
            table_consistent=table_ok,
            readback_mode="host_integers_only",
            readback_may_synchronize=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    except Exception as error:
        _content_log(
            "group1_pointer_table_probe_error",
            req_id=str(req_id),
            error_type=type(error).__name__,
            error=str(error),
        )


def fingerprint_compact_group1(
    *,
    event: str,
    req_id: str,
    owners: Iterable[torch.Tensor],
    layers: Sequence[Any],
    runs: Sequence[Any],
    token_count: int,
    chunk_size: int,
    tp_rank: int,
    dp_rank: int,
    sample_layer_ids: Sequence[int] | None = None,
    sample_logical_tokens: Sequence[int] | None = None,
    expected: dict[str, Any] | None = None,
    transfer_id: str | None = None,
) -> dict[str, Any] | None:
    """Fingerprint sampled Group-1 rows from a compact slot layout.

    This function performs an inline device-to-host readback and can therefore
    synchronize outstanding NPU work.  It is a strict no-op unless the single
    content-diagnostic config knob is enabled.
    """
    if not _ENABLED:
        return None
    started = time.perf_counter()
    try:
        default_layers, default_tokens = compact_sample_spec(
            layers, runs, token_count, chunk_size
        )
        sampled_layers = list(
            default_layers if sample_layer_ids is None else sample_layer_ids
        )
        sampled_tokens = list(
            default_tokens if sample_logical_tokens is None else sample_logical_tokens
        )
        if (
            not sampled_layers
            or not sampled_tokens
            or len(sampled_layers) > MAX_LAYER_SAMPLES
            or len(sampled_tokens) > MAX_LOGICAL_TOKEN_SAMPLES
        ):
            raise ValueError("Invalid compact diagnostic sample specification")
        layer_by_id = {int(_field(layer, "layer_id")): layer for layer in layers}
        owner_by_base = _owner_by_base(owners)
        physical_slots = _logical_to_physical(runs, sampled_tokens)
        row_hashes: dict[str, str] = {}
        layer_hashes: dict[str, str] = {}
        combined = hashlib.blake2b(digest_size=16)
        row_bytes_by_layer: dict[str, int] = {}
        dtype_by_layer: dict[str, str] = {}
        for layer_id in sampled_layers:
            layer = layer_by_id[int(layer_id)]
            owner = owner_by_base[int(_field(layer, "buffer_base"))]
            flat = owner.reshape(-1, *owner.shape[2:])
            slot_tensor = torch.tensor(
                physical_slots,
                dtype=torch.long,
                device=owner.device,
            )
            rows_cpu = flat.index_select(0, slot_tensor).detach().to("cpu")
            layer_digest = hashlib.blake2b(digest_size=16)
            for index, logical_token in enumerate(sampled_tokens):
                row = rows_cpu[index].contiguous()
                raw = _raw_bytes(row)
                digest = _hash_bytes(raw)
                row_hashes[f"{int(layer_id)}:{int(logical_token)}"] = digest
                layer_digest.update(int(logical_token).to_bytes(8, "little"))
                layer_digest.update(raw)
                combined.update(int(layer_id).to_bytes(4, "little"))
                combined.update(int(logical_token).to_bytes(8, "little"))
                combined.update(raw)
            layer_hashes[str(int(layer_id))] = layer_digest.hexdigest()
            row_bytes_by_layer[str(int(layer_id))] = int(
                rows_cpu[0].numel() * rows_cpu[0].element_size()
            )
            dtype_by_layer[str(int(layer_id))] = str(owner.dtype)
        result = {
            "schema": CONTENT_DIAGNOSTIC_SCHEMA,
            "algorithm": CONTENT_DIAGNOSTIC_ALGORITHM,
            "sample_layer_ids": [int(value) for value in sampled_layers],
            "sample_logical_tokens": [int(value) for value in sampled_tokens],
            "physical_slots": [int(value) for value in physical_slots],
            "row_hashes": row_hashes,
            "layer_hashes": layer_hashes,
            "combined_hash": combined.hexdigest(),
            "row_bytes_by_layer": row_bytes_by_layer,
            "dtype_by_layer": dtype_by_layer,
        }
        expected_hash = expected.get("combined_hash") if expected else None
        expected_rows = expected.get("row_hashes", {}) if expected else {}
        expected_row_matches = {
            key: (digest == expected_rows[key])
            for key, digest in row_hashes.items()
            if key in expected_rows
        }
        _content_log(
            event,
            req_id=req_id,
            transfer_id=transfer_id,
            tp_rank=int(tp_rank),
            dp_rank=int(dp_rank),
            **result,
            expected_combined_hash=expected_hash,
            matches_expected=(
                result["combined_hash"] == expected_hash
                if expected_hash is not None
                else None
            ),
            expected_row_matches=expected_row_matches,
            mismatched_rows=[
                key for key, matches in expected_row_matches.items() if not matches
            ],
            readback_mode="inline_device_to_host",
            readback_may_synchronize=True,
            explicit_device_synchronize=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return _wire_fingerprint(result)
    except Exception as error:
        _content_log(
            "group1_fingerprint_error",
            req_id=req_id,
            transfer_id=transfer_id,
            requested_event=event,
            tp_rank=int(tp_rank),
            dp_rank=int(dp_rank),
            error_type=type(error).__name__,
            error=str(error),
            readback_mode="inline_device_to_host",
            readback_may_synchronize=True,
        )
        return None


def register_group1_source_fingerprint(
    req_id: str, fingerprint: dict[str, Any] | None
) -> None:
    """Register a validated source sample spec for first-consume comparison."""
    if not _ENABLED or not fingerprint:
        return
    with _STATE_LOCK:
        _REQUEST_SPECS[str(req_id)] = dict(fingerprint)
        _REQUEST_SPECS.move_to_end(str(req_id))
        while len(_REQUEST_SPECS) > MAX_TRACKED_REQUEST_LAYERS:
            _REQUEST_SPECS.popitem(last=False)


@_nonfatal_diagnostic("group0_source_payload")
def register_group0_source_probe(
    *,
    req_id: str | None,
    layer_id: int,
    num_layers: int,
    source_chunks: Sequence[torch.Tensor],
    selected_tokens: torch.Tensor,
    selected_count: torch.Tensor | None,
    chunk_size: int,
    total_tokens: int,
    token_major: bool,
    layer_cache: Any,
) -> None:
    """Retain a bounded Group-0 source observation for post-join comparison.

    Only device-to-device clones of at most two integer indices are queued
    here. Source extraction and every device-to-host read happen later in
    :func:`flush_deferred_diagnostics`, after behavior-critical sampling.
    """
    if not _ENABLED:
        return
    if int(num_layers) <= 0:
        _content_log(
            "group0_source_probe_error",
            req_id=req_id,
            layer_id=int(layer_id),
            reason="invalid_num_layers",
            num_layers=int(num_layers),
        )
        return
    if int(layer_id) not in (0, int(num_layers) // 2):
        return
    if req_id is None:
        _content_log(
            "group0_source_probe_error",
            req_id=None,
            layer_id=int(layer_id),
            reason="missing_req_id",
        )
        return
    if not source_chunks or int(chunk_size) <= 0 or int(total_tokens) <= 0:
        _content_log(
            "group0_source_probe_error",
            req_id=str(req_id),
            layer_id=int(layer_id),
            reason="invalid_source_context",
            source_chunk_count=len(source_chunks),
            chunk_size=int(chunk_size),
            total_tokens=int(total_tokens),
        )
        return
    planes = (
        tuple(layer_cache)
        if isinstance(layer_cache, (tuple, list))
        else (layer_cache,)
    )
    if not planes or not all(isinstance(plane, torch.Tensor) for plane in planes):
        _content_log(
            "group0_source_probe_error",
            req_id=str(req_id),
            layer_id=int(layer_id),
            reason="invalid_destination_cache",
        )
        return
    if any(plane.ndim < 2 for plane in planes):
        _content_log(
            "group0_source_probe_error",
            req_id=str(req_id),
            layer_id=int(layer_id),
            reason="invalid_destination_cache_shape",
            shapes=[list(plane.shape) for plane in planes],
        )
        return
    slot_capacity = int(planes[0].shape[0]) * int(planes[0].shape[1])
    if slot_capacity <= 0:
        _content_log(
            "group0_source_probe_error",
            req_id=str(req_id),
            layer_id=int(layer_id),
            reason="invalid_destination_capacity",
            slot_capacity=slot_capacity,
        )
        return
    plane_widths: list[int] = []
    for plane in planes:
        if int(plane.shape[0]) * int(plane.shape[1]) != slot_capacity:
            _content_log(
                "group0_source_probe_error",
                req_id=str(req_id),
                layer_id=int(layer_id),
                reason="inconsistent_destination_planes",
            )
            return
        plane_widths.append(int(plane.numel()) // slot_capacity)
    selected_snapshot = selected_tokens.detach().reshape(-1)[:2].clone()
    count_snapshot = (
        selected_count.detach().reshape(-1)[:1].clone()
        if isinstance(selected_count, torch.Tensor) and selected_count.numel()
        else None
    )
    probe = _Group0SourceProbe(
        req_id=str(req_id),
        layer_id=int(layer_id),
        source_chunks=tuple(source_chunks),
        selected_tokens=selected_snapshot,
        selected_count=count_snapshot,
        chunk_size=int(chunk_size),
        total_tokens=int(total_tokens),
        token_major=bool(token_major),
        plane_widths=tuple(plane_widths),
        destination_stride=min(
            8,
            int(selected_tokens.shape[-1])
            if selected_tokens.ndim
            else int(selected_tokens.numel()),
        ),
    )
    key = (probe.req_id, probe.layer_id)
    with _STATE_LOCK:
        _GROUP0_SOURCE_PROBES[key] = probe
        _GROUP0_SOURCE_PROBES.move_to_end(key)
        while len(_GROUP0_SOURCE_PROBES) > MAX_TRACKED_REQUEST_LAYERS:
            _GROUP0_SOURCE_PROBES.popitem(last=False)
    _content_log(
        "group0_source_probe_registered",
        req_id=probe.req_id,
        layer_id=probe.layer_id,
        source_chunk_count=len(probe.source_chunks),
        total_tokens=probe.total_tokens,
        selected_shape=list(selected_tokens.shape),
    )


def _take_group0_source_probes(
    request_ids: Sequence[str], layer_id: int
) -> tuple[_Group0SourceProbe, ...] | None:
    probes: list[_Group0SourceProbe | None] = []
    with _STATE_LOCK:
        for req_id in request_ids:
            probes.append(
                _GROUP0_SOURCE_PROBES.pop(
                    (str(req_id), int(layer_id)), None
                )
            )
    if not probes or any(probe is None for probe in probes):
        return None
    return tuple(probe for probe in probes if probe is not None)


def _claim_once(stage: str, req_id: str, layer_name: str) -> bool:
    key = (stage, str(req_id), str(layer_name))
    with _STATE_LOCK:
        if key in _SEEN:
            return False
        _SEEN[key] = None
        while len(_SEEN) > MAX_TRACKED_REQUEST_LAYERS:
            _SEEN.popitem(last=False)
    return True


def _layer_id(layer_name: str) -> int | None:
    match = _LAYER_PATTERN.search(str(layer_name))
    return int(match.group(1)) if match else None


def _request_row_counts(row_request_indices: Any, request_count: int) -> list[int]:
    counts = [1] * request_count
    if row_request_indices is None:
        return counts
    if isinstance(row_request_indices, torch.Tensor):
        if row_request_indices.device.type != "cpu":
            return counts
        values = row_request_indices.detach().reshape(-1).tolist()
    else:
        values = list(row_request_indices)
    counts = [0] * request_count
    for value in values:
        index = int(value)
        if 0 <= index < request_count:
            counts[index] += 1
    return [max(value, 1) for value in counts]


def _persistent_observation_spec(
    *, num_hidden_layers: int | None, context_tokens: int
) -> dict[str, Any] | None:
    """Build a bounded sample spec when no live source fingerprint exists.

    Persistent-only and live-to-persistent-fallback requests have no wire
    fingerprint to register.  They still need observable prefix rows and
    top-k choices so two hosts/runs can be compared.  This function selects a
    deterministic first/middle/last layer set and prefix positions while
    intentionally leaving expected hashes empty.
    """
    if (
        num_hidden_layers is None
        or isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
        or context_tokens <= 0
    ):
        return None
    layers = _ordered_sample(
        (0, num_hidden_layers // 2, num_hidden_layers - 1),
        MAX_LAYER_SAMPLES,
    )
    tokens = _ordered_sample(
        (
            value
            for value in (
                0,
                1,
                context_tokens // 2,
                context_tokens - 2,
                context_tokens - 1,
            )
            if 0 <= value < context_tokens
        ),
        MAX_LOGICAL_TOKEN_SAMPLES,
    )
    return {
        "sample_layer_ids": layers,
        "sample_logical_tokens": tokens,
        "row_hashes": {},
        "diagnostic_origin": "persistent_observed_only",
    }


def _cpu_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        values = values.detach().reshape(-1).tolist()
    return [int(value) for value in values]


def _log_snapshot_skip_once(
    *,
    requested_event: str,
    req_ids: Sequence[str],
    layer_name: str,
    reason: str,
    **fields: Any,
) -> None:
    for req_id in req_ids:
        if not _claim_once(f"skip:{requested_event}", str(req_id), reason):
            continue
        _content_log(
            "content_diagnostic_snapshot_skipped",
            requested_event=requested_event,
            req_id=str(req_id),
            layer_name=str(layer_name),
            reason=reason,
            **fields,
        )


@_nonfatal_diagnostic("cache_tail_fingerprint")
def queue_cache_tail_fingerprint(
    *,
    req_ids: Sequence[str] | None,
    layer_name: str,
    kv_group: int,
    stage: str,
    cache_parts: dict[str, torch.Tensor],
    slot_mapping: torch.Tensor | None,
    query_start_loc_cpu: Any,
    seq_lens_cpu: Any,
    produced_parts: dict[str, torch.Tensor] | None = None,
    num_actual_tokens: int | None = None,
    attn_state: Any = None,
) -> None:
    """Queue each request's cached-prefix boundary row.

    The snapshot is device-to-device only. Host hashing is deferred until the
    complete model forward returns, so the diagnostic cannot insert a sync
    between attention layers. ``produced_parts`` records the freshly computed
    row beside the cache row, allowing the log to prove whether scatter wrote
    the intended value to the intended physical slot.
    """
    if not _ENABLED or not req_ids:
        return
    request_ids = [str(req_id) for req_id in req_ids]
    query_starts = _cpu_int_list(query_start_loc_cpu)
    seq_lens = _cpu_int_list(seq_lens_cpu)
    if (
        slot_mapping is None
        or query_starts is None
        or seq_lens is None
        or len(query_starts) < len(request_ids) + 1
        or len(seq_lens) < len(request_ids)
        or not cache_parts
    ):
        _log_snapshot_skip_once(
            requested_event="cache_tail_fingerprint",
            req_ids=request_ids,
            layer_name=layer_name,
            reason="missing_cache_boundary_metadata",
            kv_group=int(kv_group),
            stage=str(stage),
            has_slot_mapping=slot_mapping is not None,
            query_start_count=(
                len(query_starts) if query_starts is not None else None
            ),
            seq_len_count=len(seq_lens) if seq_lens is not None else None,
            cache_components=sorted(cache_parts),
        )
        return
    layer_id = _layer_id(layer_name)
    if layer_id is None:
        _log_snapshot_skip_once(
            requested_event="cache_tail_fingerprint",
            req_ids=request_ids,
            layer_name=layer_name,
            reason="unrecognized_cache_layer_name",
            kv_group=int(kv_group),
            stage=str(stage),
        )
        return
    actual_limit = (
        min(int(num_actual_tokens), int(slot_mapping.shape[0]))
        if num_actual_tokens is not None
        else int(slot_mapping.shape[0])
    )
    for req_index, req_id in enumerate(request_ids):
        query_start = int(query_starts[req_index])
        query_end = min(int(query_starts[req_index + 1]), actual_limit)
        if query_end <= query_start:
            continue
        query_length = int(query_starts[req_index + 1]) - query_start
        sequence_length = int(seq_lens[req_index])
        cached_prefix_tokens = max(sequence_length - query_length, 0)
        if "before_current_scatter" in str(stage):
            # The before-scatter row sits inside the current query window and
            # is only comparable when it was previously populated by a cache
            # load (the request's first resume forward with a cached prefix).
            # A cold request's first forward (cached_prefix == 0) samples an
            # unwritten row, and its chunked-prefill continuations sample the
            # first new token of the chunk; both compare unequal by
            # construction and would drown the verdict histogram in
            # guaranteed-false noise. Emit exactly once, and only when the
            # first forward carried a cached prefix.
            first_window_key = (
                f"tail_before:{int(kv_group)}",
                str(req_id),
                str(layer_name),
            )
            with _STATE_LOCK:
                first_window = first_window_key not in _SEEN
                if first_window:
                    _SEEN[first_window_key] = None
                    while len(_SEEN) > MAX_TRACKED_REQUEST_LAYERS:
                        _SEEN.popitem(last=False)
            if not first_window or not cached_prefix_tokens:
                _log_snapshot_skip_once(
                    requested_event="cache_tail_fingerprint",
                    req_ids=[req_id],
                    layer_name=layer_name,
                    reason=(
                        "first_forward_without_cached_prefix"
                        if not cached_prefix_tokens
                        else "later_window_before_scatter"
                    ),
                    kv_group=int(kv_group),
                    stage=str(stage),
                )
                continue
        if not _claim_once(
            f"tail:{int(kv_group)}:{stage}", req_id, layer_name
        ):
            continue
        # On a prefix hit, the first query row is the mandatory boundary token
        # that replaces the final cached row. On a full prefill there is no
        # cached boundary, so sample the final prompt row for cross-run/source
        # comparison instead.
        row_index = query_start if cached_prefix_tokens else query_end - 1
        row_index_device = torch.tensor(
            [row_index], dtype=torch.long, device=slot_mapping.device
        )
        physical_slot = slot_mapping.index_select(
            0, row_index_device
        ).long()
        selected_cache: dict[str, torch.Tensor] = {}
        for name, cache in cache_parts.items():
            flat = cache.reshape(-1, *cache.shape[2:])
            selected_cache[str(name)] = flat.index_select(
                0, physical_slot
            ).reshape(1, -1)
        selected_produced: dict[str, torch.Tensor] | None = None
        if produced_parts is not None:
            selected_produced = {}
            for name, produced in produced_parts.items():
                produced_flat = produced.reshape(produced.shape[0], -1)
                selected_produced[str(name)] = produced_flat.index_select(
                    0, row_index_device.to(produced.device)
                )
            if set(selected_cache) != set(selected_produced):
                raise ValueError(
                    "Cache and produced diagnostic components differ: "
                    f"cache={sorted(selected_cache)}, "
                    f"produced={sorted(selected_produced)}"
                )
        logical_token = (
            cached_prefix_tokens
            if cached_prefix_tokens
            else sequence_length - 1
        )
        _queue_snapshot(
            _DeferredSnapshot(
                event="cache_tail_fingerprint",
                fields={
                    "req_id": req_id,
                    "layer_name": str(layer_name),
                    "layer_id": layer_id,
                    "kv_group": int(kv_group),
                    "stage": str(stage),
                    "logical_tokens": [logical_token],
                    "request_seq_len": sequence_length,
                    "query_start": query_start,
                    "query_end": int(query_starts[req_index + 1]),
                    "query_length": query_length,
                    "selected_row_index": row_index,
                    "cached_prefix_tokens": cached_prefix_tokens,
                    "prefix_hit": bool(cached_prefix_tokens),
                    "selection_reason": (
                        "first_query_row_at_cached_prefix_boundary"
                        if cached_prefix_tokens
                        else "last_full_prefill_row"
                    ),
                    # Explicitly scope what a matches_produced verdict here
                    # can and cannot prove. After-scatter comparisons ride the
                    # compute stream's own ordering, so they validate the
                    # scatter writeback only: they cannot observe what an
                    # unfenced concurrent DMA (direct store) reads from the
                    # same slots. Store-time source fingerprints cover that.
                    "comparison_scope": (
                        "scatter_writeback_stream_ordered"
                        if "after_current_scatter" in str(stage)
                        else "loaded_boundary_row_before_replacement"
                    ),
                    "num_actual_tokens": (
                        int(num_actual_tokens)
                        if num_actual_tokens is not None
                        else None
                    ),
                    "attn_state": str(attn_state),
                    "cache_components": sorted(selected_cache),
                    "has_produced_comparison": (
                        selected_produced is not None
                    ),
                    "snapshot_monotonic_ms": round(
                        time.perf_counter() * 1000, 3
                    ),
                    "capture_order": str(stage),
                },
                physical_slots=physical_slot,
                components=selected_cache,
                comparison_components=selected_produced,
            )
        )


@_nonfatal_diagnostic("group1_first_decode_consume")
def queue_group1_first_consume(
    *,
    req_ids: Sequence[str] | None,
    layer_name: str,
    indexer_cache: torch.Tensor,
    indexer_block_table: torch.Tensor | None,
    seq_lens_cpu: Any,
    block_size: int,
    row_request_indices: Any = None,
    num_decode_tokens: int | None = None,
    num_actual_tokens: int | None = None,
    attn_state: Any = None,
    decode_valid_rows_all: bool | None = None,
    group1_connector_wait_called: bool | None = None,
    num_hidden_layers: int | None = None,
) -> None:
    """Queue exact Group-1 rows consumed by the first decoder forward.

    The selected rows are copied device-to-device on the current stream.  Host
    readback and hashing are deferred until :func:`flush_deferred_diagnostics`.
    """
    if not _ENABLED or not req_ids:
        return
    request_ids = [str(req_id) for req_id in req_ids]
    if indexer_block_table is None or seq_lens_cpu is None or block_size <= 0:
        _log_snapshot_skip_once(
            requested_event="group1_first_decode_consume",
            req_ids=request_ids,
            layer_name=layer_name,
            reason="missing_attention_metadata",
            has_indexer_block_table=indexer_block_table is not None,
            has_seq_lens=seq_lens_cpu is not None,
            block_size=int(block_size),
        )
        return
    layer_id = _layer_id(layer_name)
    if layer_id is None:
        _log_snapshot_skip_once(
            requested_event="group1_first_decode_consume",
            req_ids=request_ids,
            layer_name=layer_name,
            reason="unrecognized_indexer_layer_name",
        )
        return
    seq_lens = _cpu_int_list(seq_lens_cpu)
    if seq_lens is None:
        _log_snapshot_skip_once(
            requested_event="group1_first_decode_consume",
            req_ids=request_ids,
            layer_name=layer_name,
            reason="sequence_lengths_not_on_cpu",
        )
        return
    row_owners = _cpu_int_list(row_request_indices)
    request_count = len(request_ids)
    block_table_rows = int(indexer_block_table.shape[0])
    if len(seq_lens) < request_count or block_table_rows < request_count:
        _log_snapshot_skip_once(
            requested_event="group1_first_decode_consume",
            req_ids=request_ids,
            layer_name=str(layer_name),
            reason="incomplete_request_index_metadata",
            request_count=request_count,
            seq_len_count=len(seq_lens),
            block_table_rows=block_table_rows,
            row_owners=row_owners,
        )
        return
    row_counts = _request_row_counts(row_request_indices, len(req_ids))
    flat = indexer_cache.reshape(-1, *indexer_cache.shape[2:])
    for req_index, req_id in enumerate(request_ids):
        with _STATE_LOCK:
            spec = _REQUEST_SPECS.get(str(req_id))
        context_tokens = max(seq_lens[req_index] - row_counts[req_index], 0)
        if not spec:
            spec = _persistent_observation_spec(
                num_hidden_layers=num_hidden_layers,
                context_tokens=context_tokens,
            )
            if spec is None:
                _log_snapshot_skip_once(
                    requested_event="group1_first_decode_consume",
                    req_ids=[req_id],
                    layer_name=layer_name,
                    reason="missing_persistent_observation_metadata",
                    num_hidden_layers=num_hidden_layers,
                    context_tokens=context_tokens,
                )
                continue
        if layer_id not in spec.get("sample_layer_ids", ()):
            continue
        logical_tokens = [
            int(value)
            for value in spec.get("sample_logical_tokens", ())
            if 0 <= int(value) < context_tokens
        ]
        if not logical_tokens:
            continue
        if not _claim_once("consume", str(req_id), layer_name):
            continue
        logical = torch.tensor(
            logical_tokens,
            dtype=torch.long,
            device=indexer_block_table.device,
        )
        block_indices = torch.div(logical, block_size, rounding_mode="floor")
        offsets = torch.remainder(logical, block_size)
        physical_slots = (
            indexer_block_table[req_index].index_select(0, block_indices).long()
            * block_size
            + offsets
        )
        rows = flat.index_select(0, physical_slots)
        expected_rows = {
            (layer_id, token): str(
                spec.get("row_hashes", {}).get(f"{layer_id}:{token}", "")
            )
            for token in logical_tokens
        }
        snapshot = _DeferredSnapshot(
            event="group1_first_decode_consume",
            fields={
                "req_id": str(req_id),
                "layer_name": str(layer_name),
                "layer_id": layer_id,
                "logical_tokens": logical_tokens,
                "context_tokens": context_tokens,
                "request_seq_len": seq_lens[req_index],
                "request_row_count": row_counts[req_index],
                "seq_lens": seq_lens,
                "row_owners": row_owners,
                "valid_row_indices": (
                    [
                        index
                        for index, owner in enumerate(row_owners)
                        if 0 <= owner < request_count
                    ]
                    if row_owners is not None
                    else None
                ),
                "padding_row_indices": (
                    [
                        index
                        for index, owner in enumerate(row_owners)
                        if owner < 0
                    ]
                    if row_owners is not None
                    else None
                ),
                "invalid_row_indices": (
                    [
                        index
                        for index, owner in enumerate(row_owners)
                        if owner >= request_count
                    ]
                    if row_owners is not None
                    else None
                ),
                "num_decode_tokens": (
                    int(num_decode_tokens)
                    if num_decode_tokens is not None
                    else None
                ),
                "num_actual_tokens": (
                    int(num_actual_tokens)
                    if num_actual_tokens is not None
                    else None
                ),
                "attn_state": str(attn_state),
                "decode_valid_rows_all": (
                    bool(decode_valid_rows_all)
                    if decode_valid_rows_all is not None
                    else None
                ),
                "group1_connector_wait_called": (
                    bool(group1_connector_wait_called)
                    if group1_connector_wait_called is not None
                    else None
                ),
                "source_fingerprint_registered": (
                    spec.get("diagnostic_origin")
                    != "persistent_observed_only"
                ),
                "comparison_mode": spec.get(
                    "diagnostic_origin", "live_source_expected"
                ),
                "dtype": str(indexer_cache.dtype),
                "snapshot_monotonic_ms": round(time.perf_counter() * 1000, 3),
                "capture_order": (
                    "after_group1_connector_wait_call_before_current_token_scatter"
                    if group1_connector_wait_called
                    else "before_current_token_scatter_without_group1_connector_wait"
                ),
            },
            values=rows,
            physical_slots=physical_slots,
            expected_rows=expected_rows,
        )
        _queue_snapshot(snapshot)


@_nonfatal_diagnostic("group1_selected_topk_fingerprint")
def queue_selected_topk_fingerprint(
    *,
    req_ids: Sequence[str] | None,
    layer_name: str,
    topk_indices: torch.Tensor,
    row_request_indices: Any = None,
    seq_lens_cpu: Any = None,
    num_decode_tokens: int | None = None,
    num_actual_tokens: int | None = None,
    attn_state: Any = None,
    decode_valid_rows_all: bool | None = None,
    num_hidden_layers: int | None = None,
) -> None:
    """Queue selected top-k token rows per request for deferred hashing.

    ``row_request_indices`` is CPU scheduler metadata.  It lets the diagnostic
    preserve row ownership without reading an NPU tensor back to the host.  A
    single-request batch may omit it; multi-request batches are skipped rather
    than emitting an ambiguous fingerprint.
    """
    if not _ENABLED or not req_ids:
        return
    layer_id = _layer_id(layer_name)
    if layer_id is None:
        return
    request_ids = [str(req_id) for req_id in req_ids]
    seq_lens = _cpu_int_list(seq_lens_cpu)
    row_counts = _request_row_counts(row_request_indices, len(request_ids))
    eligible: set[int] = set()
    comparison_modes: dict[int, str] = {}
    for req_index, req_id in enumerate(request_ids):
        with _STATE_LOCK:
            spec = _REQUEST_SPECS.get(req_id)
        if not spec and seq_lens is not None and req_index < len(seq_lens):
            spec = _persistent_observation_spec(
                num_hidden_layers=num_hidden_layers,
                context_tokens=max(
                    seq_lens[req_index] - row_counts[req_index], 0
                ),
            )
        if spec and layer_id in spec.get("sample_layer_ids", ()):
            eligible.add(req_index)
            comparison_modes[req_index] = str(
                spec.get("diagnostic_origin", "live_source_expected")
            )
    if not eligible:
        return
    row_owners: list[int]
    if row_request_indices is None:
        if len(request_ids) != 1:
            _content_log(
                "content_diagnostic_snapshot_skipped",
                requested_event="group1_selected_topk_fingerprint",
                layer_name=str(layer_name),
                reason="ambiguous_multi_request_row_ownership",
            )
            return
        row_owners = [0] * int(topk_indices.shape[0])
    elif isinstance(row_request_indices, torch.Tensor):
        if row_request_indices.device.type != "cpu":
            return
        row_owners = [
            int(value) for value in row_request_indices.detach().reshape(-1).tolist()
        ]
    else:
        row_owners = [int(value) for value in row_request_indices]
    row_limit = min(len(row_owners), int(topk_indices.shape[0]))
    valid_row_indices = [
        index
        for index, owner in enumerate(row_owners[:row_limit])
        if 0 <= owner < len(request_ids)
    ]
    padding_row_indices = [
        index for index, owner in enumerate(row_owners[:row_limit]) if owner < 0
    ]
    invalid_row_indices = [
        index
        for index, owner in enumerate(row_owners[:row_limit])
        if owner >= len(request_ids)
    ]
    unowned_topk_row_indices = list(
        range(row_limit, int(topk_indices.shape[0]))
    )
    for req_index in sorted(eligible):
        req_id = request_ids[req_index]
        row_indices = [
            row_index
            for row_index, owner in enumerate(row_owners[:row_limit])
            if owner == req_index
        ]
        if not row_indices or not _claim_once("topk", req_id, layer_name):
            continue
        rows = topk_indices.index_select(
            0,
            torch.tensor(
                row_indices,
                dtype=torch.long,
                device=topk_indices.device,
            ),
        )
        _queue_snapshot(
            _DeferredSnapshot(
                event="group1_selected_topk_fingerprint",
                fields={
                    "req_id": req_id,
                    "layer_name": str(layer_name),
                    "layer_id": layer_id,
                    "row_indices": row_indices,
                    "request_seq_len": (
                        seq_lens[req_index]
                        if seq_lens is not None and req_index < len(seq_lens)
                        else None
                    ),
                    "seq_lens": seq_lens,
                    "row_owners": row_owners[:row_limit],
                    "valid_row_indices": valid_row_indices,
                    "padding_row_indices": padding_row_indices,
                    "invalid_row_indices": invalid_row_indices,
                    "unowned_topk_row_indices": unowned_topk_row_indices,
                    "topk_row_count": int(topk_indices.shape[0]),
                    "row_owner_count": len(row_owners),
                    "num_decode_tokens": (
                        int(num_decode_tokens)
                        if num_decode_tokens is not None
                        else None
                    ),
                    "num_actual_tokens": (
                        int(num_actual_tokens)
                        if num_actual_tokens is not None
                        else None
                    ),
                    "attn_state": str(attn_state),
                    "decode_valid_rows_all": (
                        bool(decode_valid_rows_all)
                        if decode_valid_rows_all is not None
                        else None
                    ),
                    "source_fingerprint_registered": (
                        comparison_modes[req_index]
                        != "persistent_observed_only"
                    ),
                    "comparison_mode": comparison_modes[req_index],
                    "shape": list(rows.shape),
                    "dtype": str(topk_indices.dtype),
                    "snapshot_monotonic_ms": round(time.perf_counter() * 1000, 3),
                    "capture_order": ("after_topk_selection_before_compact_remap"),
                },
                values=rows,
            )
        )


@_nonfatal_diagnostic("staged_sfa_graph_fingerprint")
def queue_staged_graph_stage_fingerprint(
    *,
    req_ids: Sequence[str] | None,
    layer_name: str,
    stage: str,
    components: dict[str, torch.Tensor],
    row_request_indices: Any = None,
    seq_lens_cpu: Any = None,
    num_decode_tokens: int | None = None,
    num_actual_tokens: int | None = None,
    attn_state: Any = None,
    num_hidden_layers: int | None = None,
    graph_key: Any = None,
) -> None:
    """Queue a bounded snapshot at a staged-SFA replay boundary.

    Callers provide tensors that already represent the small row/slot sample
    relevant to the boundary.  Cloning happens on the current NPU stream, so
    the captured values cannot be overwritten by a later graph island.  Host
    readback remains deferred until behavior-critical sampling work is queued.
    """
    if not _ENABLED or not req_ids or not components:
        return
    layer_id = _layer_id(layer_name)
    if layer_id is None:
        return
    request_ids = [str(req_id) for req_id in req_ids]
    seq_lens = _cpu_int_list(seq_lens_cpu)
    row_counts = _request_row_counts(
        row_request_indices,
        len(request_ids),
    )
    sampled = False
    for req_index, req_id in enumerate(request_ids):
        with _STATE_LOCK:
            spec = _REQUEST_SPECS.get(req_id)
        if not spec and seq_lens is not None and req_index < len(seq_lens):
            spec = _persistent_observation_spec(
                num_hidden_layers=num_hidden_layers,
                context_tokens=max(
                    int(seq_lens[req_index]) - row_counts[req_index],
                    0,
                ),
            )
        if spec and layer_id in spec.get("sample_layer_ids", ()):
            sampled = True
            break
    if not sampled:
        return
    request_key = "|".join(request_ids)
    if not _claim_once(
        f"staged_graph:{stage}",
        request_key,
        layer_name,
    ):
        return
    snapshots = {
        str(name): value.detach().clone()
        for name, value in components.items()
        if isinstance(value, torch.Tensor)
    }
    if not snapshots:
        return
    row_owners = _cpu_int_list(row_request_indices)
    group0_source_probes = (
        _take_group0_source_probes(request_ids, layer_id)
        if stage == "after_selective_retrieve_before_graph_post"
        else None
    )
    _queue_snapshot(
        _DeferredSnapshot(
            event="staged_sfa_graph_fingerprint",
            fields={
                "request_ids": request_ids,
                "layer_name": str(layer_name),
                "layer_id": layer_id,
                "stage": str(stage),
                "graph_key": str(graph_key),
                "seq_lens": seq_lens,
                "row_owners": row_owners,
                "valid_row_indices": (
                    [
                        index
                        for index, owner in enumerate(row_owners)
                        if 0 <= owner < len(request_ids)
                    ]
                    if row_owners is not None
                    else None
                ),
                "padding_row_indices": (
                    [
                        index
                        for index, owner in enumerate(row_owners)
                        if owner < 0
                    ]
                    if row_owners is not None
                    else None
                ),
                "num_decode_tokens": (
                    int(num_decode_tokens)
                    if num_decode_tokens is not None
                    else None
                ),
                "num_actual_tokens": (
                    int(num_actual_tokens)
                    if num_actual_tokens is not None
                    else None
                ),
                "attn_state": str(attn_state),
                "component_names": sorted(snapshots),
                "snapshot_monotonic_ms": round(
                    time.perf_counter() * 1000,
                    3,
                ),
                "capture_order": str(stage),
                "device_snapshot_may_perturb_timing": True,
            },
            components=snapshots,
            group0_source_probes=group0_source_probes,
        )
    )


def _queue_snapshot(snapshot: _DeferredSnapshot) -> None:
    dropped: _DeferredSnapshot | None = None
    with _STATE_LOCK:
        if len(_PENDING) >= MAX_PENDING_SNAPSHOTS:
            dropped = _PENDING.pop(0)
        _PENDING.append(snapshot)
    if dropped is not None:
        _content_log(
            "content_diagnostic_snapshot_dropped",
            dropped_event=dropped.event,
            reason="pending_limit",
        )


def begin_deferred_diagnostic_step() -> None:
    """Discard snapshots left by an interrupted prior model forward."""
    if not _ENABLED:
        return
    with _STATE_LOCK:
        stale = len(_PENDING)
        _PENDING.clear()
        _GROUP0_SOURCE_PROBES.clear()
        stale_probes = len(_PENDING_STORE_PROBES)
        _PENDING_STORE_PROBES.clear()
    if stale or stale_probes:
        _content_log(
            "content_diagnostic_snapshot_dropped",
            dropped_count=stale + stale_probes,
            reason="previous_forward_did_not_flush",
        )


def _materialize_group0_source_samples(
    probes: Sequence[_Group0SourceProbe],
) -> tuple[
    dict[str, torch.Tensor], list[dict[str, Any]], list[int]
]:
    """Gather the registered LocalCPU records after sampling has completed."""
    plane_rows: list[list[torch.Tensor]] = []
    details: list[dict[str, Any]] = []
    destination_rows: list[int] = []
    destination_offset = 0
    expected_plane_count: int | None = None
    for probe in probes:
        selected = [
            int(value)
            for value in probe.selected_tokens.detach()
            .to("cpu")
            .reshape(-1)
            .tolist()
        ]
        selected_count = (
            int(probe.selected_count.detach().to("cpu").reshape(-1)[0])
            if probe.selected_count is not None
            else len(selected)
        )
        sample_count = min(len(selected), max(selected_count, 0))
        source_chunks = tuple(probe.source_chunks)
        detail: dict[str, Any] = {
            "req_id": probe.req_id,
            "layer_id": probe.layer_id,
            "selected_token_preview": selected[:sample_count],
            "selected_count": selected_count,
            "sample_count": sample_count,
            "source_chunk_count": len(source_chunks),
            "source_chunk_data_ptr_preview": [
                int(tensor.data_ptr()) for tensor in source_chunks[:2]
            ],
            "token_major": probe.token_major,
        }
        if expected_plane_count is None:
            expected_plane_count = len(probe.plane_widths)
            plane_rows = [[] for _ in range(expected_plane_count)]
        if len(probe.plane_widths) != expected_plane_count:
            detail.update(supported=False, reason="plane_count_mismatch")
            details.append(detail)
            destination_offset += probe.destination_stride
            continue
        if any(tensor.device.type != "cpu" for tensor in source_chunks):
            detail.update(supported=False, reason="source_not_cpu")
            details.append(detail)
            destination_offset += probe.destination_stride
            continue
        record_width = sum(probe.plane_widths)
        if record_width <= 0:
            detail.update(supported=False, reason="invalid_record_width")
            details.append(detail)
            destination_offset += probe.destination_stride
            continue
        gathered: list[list[torch.Tensor]] = [
            [] for _ in range(expected_plane_count)
        ]
        reason = None
        for source_token in selected[:sample_count]:
            if source_token < 0 or source_token >= probe.total_tokens:
                reason = "source_token_oob"
                break
            chunk_index, local_token = divmod(source_token, probe.chunk_size)
            if chunk_index >= len(source_chunks):
                reason = "source_chunk_oob"
                break
            flat = source_chunks[chunk_index].detach().reshape(-1)
            if int(flat.numel()) % record_width:
                reason = "source_chunk_layout_mismatch"
                break
            chunk_tokens = int(flat.numel()) // record_width
            if local_token >= chunk_tokens:
                reason = "source_local_token_oob"
                break
            plane_offset = 0
            for plane_index, width in enumerate(probe.plane_widths):
                start = (
                    local_token * record_width + plane_offset
                    if probe.token_major
                    else plane_offset * chunk_tokens + local_token * width
                )
                gathered[plane_index].append(flat[start : start + width])
                plane_offset += width
        if reason is not None:
            detail.update(supported=False, reason=reason)
            details.append(detail)
            destination_offset += probe.destination_stride
            continue
        for plane_index, rows in enumerate(gathered):
            plane_rows[plane_index].extend(rows)
        destination_rows.extend(
            range(destination_offset, destination_offset + sample_count)
        )
        destination_offset += probe.destination_stride
        detail.update(supported=True)
        details.append(detail)
    names = (
        ("retrieved_nope_sample", "retrieved_pe_sample")
        if len(plane_rows) == 2
        else tuple(
            f"retrieved_plane_{index}_sample"
            for index in range(len(plane_rows))
        )
    )
    components = {
        name: torch.stack(rows)
        for name, rows in zip(names, plane_rows, strict=True)
        if rows
    }
    return components, details, destination_rows


def flush_deferred_diagnostics() -> None:
    """Read snapshots after behavior-critical work and log fingerprints."""
    if not _ENABLED:
        return
    _flush_store_probes()
    with _STATE_LOCK:
        pending = list(_PENDING)
        _PENDING.clear()
    for snapshot in pending:
        started = time.perf_counter()
        try:
            fields = dict(snapshot.fields)
            if snapshot.components is not None:
                components_cpu = {
                    name: value.detach().to("cpu").contiguous()
                    for name, value in snapshot.components.items()
                }
                comparison_cpu = (
                    {
                        name: value.detach().to("cpu").contiguous()
                        for name, value in snapshot.comparison_components.items()
                    }
                    if snapshot.comparison_components is not None
                    else None
                )
                component_summaries: dict[str, dict[str, Any]] = {}
                for name, value in components_cpu.items():
                    comparison = (
                        comparison_cpu.get(name)
                        if comparison_cpu is not None
                        else None
                    )
                    summary: dict[str, Any] = {
                        "content_hash": _hash_cpu_tensor(value),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "value_count": int(value.numel()),
                    }
                    if (
                        snapshot.event == "staged_sfa_graph_fingerprint"
                        and not value.is_floating_point()
                        and not value.is_complex()
                    ):
                        flat = value.reshape(-1)
                        summary["value_preview"] = [
                            int(item)
                            for item in flat[: min(8, int(flat.numel()))].tolist()
                        ]
                    if comparison is not None:
                        summary["produced_hash"] = _hash_cpu_tensor(comparison)
                        summary["matches_produced"] = torch.equal(
                            value, comparison
                        )
                        if value.shape == comparison.shape:
                            delta = value.to(torch.float32) - comparison.to(
                                torch.float32
                            )
                            summary["max_abs_delta"] = float(
                                delta.abs().max().item()
                            ) if delta.numel() else 0.0
                            summary["differing_element_count"] = int(
                                torch.count_nonzero(delta).item()
                            )
                    component_summaries[name] = summary
                fields["component_summaries"] = component_summaries
                matches = [
                    summary.get("matches_produced")
                    for summary in component_summaries.values()
                    if "matches_produced" in summary
                ]
                fields["all_components_match_produced"] = (
                    all(value is True for value in matches)
                    if matches
                    else None
                )
                if snapshot.group0_source_probes is not None:
                    try:
                        (
                            source_components,
                            source_details,
                            destination_rows,
                        ) = _materialize_group0_source_samples(
                            snapshot.group0_source_probes
                        )
                        source_summaries: dict[str, dict[str, Any]] = {}
                        source_matches: list[bool] = []
                        for name, source in source_components.items():
                            summary = {
                                "content_hash": _hash_cpu_tensor(source),
                                "shape": list(source.shape),
                                "dtype": str(source.dtype),
                                "value_count": int(source.numel()),
                            }
                            destination = components_cpu.get(name)
                            if (
                                destination is not None
                                and destination_rows
                                and max(destination_rows)
                                < int(destination.shape[0])
                            ):
                                row_index = torch.tensor(
                                    destination_rows, dtype=torch.long
                                )
                                destination = destination.index_select(
                                    0, row_index
                                )
                                if int(destination.numel()) == int(
                                    source.numel()
                                ):
                                    source_view = source.reshape(
                                        destination.shape
                                    )
                                    matches_destination = torch.equal(
                                        source_view, destination
                                    )
                                    summary.update(
                                        destination_hash=_hash_cpu_tensor(
                                            destination
                                        ),
                                        matches_destination=(
                                            matches_destination
                                        ),
                                    )
                                    delta = source_view.to(
                                        torch.float32
                                    ) - destination.to(torch.float32)
                                    summary["max_abs_delta"] = (
                                        float(delta.abs().max().item())
                                        if delta.numel()
                                        else 0.0
                                    )
                                    summary["differing_element_count"] = int(
                                        torch.count_nonzero(delta).item()
                                    )
                                    source_matches.append(matches_destination)
                            source_summaries[name] = summary
                        _content_log(
                            "group0_source_destination_fingerprint",
                            request_ids=fields.get("request_ids"),
                            layer_name=fields.get("layer_name"),
                            layer_id=fields.get("layer_id"),
                            graph_key=fields.get("graph_key"),
                            source_details=source_details,
                            component_summaries=source_summaries,
                            all_components_match=(
                                all(source_matches)
                                if source_matches
                                else None
                            ),
                            readback_mode="deferred_after_sampling_workflow",
                            readback_may_synchronize=True,
                            explicit_device_synchronize=False,
                        )
                    except Exception as error:
                        _content_log(
                            "group0_source_probe_error",
                            request_ids=fields.get("request_ids"),
                            layer_id=fields.get("layer_id"),
                            error_type=type(error).__name__,
                            error=str(error),
                        )
            else:
                if snapshot.values is None:
                    raise ValueError("Deferred snapshot has no values")
                values_cpu = snapshot.values.detach().to("cpu").contiguous()
                fields["content_hash"] = _hash_cpu_tensor(values_cpu)
                fields["value_count"] = int(values_cpu.numel())
            if snapshot.physical_slots is not None:
                slots = [
                    int(value)
                    for value in snapshot.physical_slots.detach()
                    .to("cpu")
                    .reshape(-1)
                    .tolist()
                ]
                fields["physical_slots"] = slots
                if snapshot.components is None:
                    logical_tokens = fields.get("logical_tokens", ())
                    row_hashes: dict[str, str] = {}
                    expected_matches: dict[str, bool | None] = {}
                    for index, logical_token in enumerate(logical_tokens):
                        digest = _hash_cpu_tensor(values_cpu[index])
                        key = f"{fields['layer_id']}:{int(logical_token)}"
                        row_hashes[key] = digest
                        expected = (
                            snapshot.expected_rows.get(
                                (int(fields["layer_id"]), int(logical_token))
                            )
                            if snapshot.expected_rows
                            else None
                        )
                        expected_matches[key] = (
                            digest == expected if expected else None
                        )
                    fields["row_hashes"] = row_hashes
                    fields["expected_matches"] = expected_matches
                    comparable = [
                        value
                        for value in expected_matches.values()
                        if value is not None
                    ]
                    fields["all_expected_rows_match"] = (
                        all(value is True for value in comparable)
                        if comparable
                        else None
                    )
            elif snapshot.components is None:
                flat = values_cpu.reshape(-1)
                preview_count = min(8, int(flat.numel()))
                fields["value_preview"] = [
                    int(value) for value in flat[:preview_count].tolist()
                ]
                if snapshot.event == "group1_selected_topk_fingerprint":
                    rows = values_cpu.reshape(values_cpu.shape[0], -1)
                    sequence_length = fields.get("request_seq_len")
                    row_summaries: list[dict[str, Any]] = []
                    for offset, row in enumerate(rows):
                        row_flat = row.reshape(-1)
                        count = int(row_flat.numel())
                        row_index = int(fields["row_indices"][offset])
                        summary: dict[str, Any] = {
                            "row_index": row_index,
                            "value_count": count,
                            "content_hash": _hash_cpu_tensor(row_flat),
                            "min": int(row_flat.min().item()) if count else None,
                            "max": int(row_flat.max().item()) if count else None,
                            "unique_count": (
                                int(torch.unique(row_flat).numel()) if count else 0
                            ),
                            "identity_prefix": bool(
                                count
                                and torch.equal(
                                    row_flat,
                                    torch.arange(
                                        count,
                                        dtype=row_flat.dtype,
                                    ),
                                )
                            ),
                        }
                        if sequence_length is not None:
                            valid = (row_flat >= 0) & (
                                row_flat < int(sequence_length)
                            )
                            summary["out_of_range_count"] = int(
                                count - int(valid.sum().item())
                            )
                        row_summaries.append(summary)
                    fields["row_summaries"] = row_summaries
            _content_log(
                snapshot.event,
                **fields,
                readback_mode=(
                    "deferred_after_sampling_workflow"
                    if snapshot.event == "staged_sfa_graph_fingerprint"
                    else "deferred_after_model_forward"
                ),
                readback_may_synchronize=True,
                explicit_device_synchronize=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as error:
            _content_log(
                "group1_fingerprint_error",
                requested_event=snapshot.event,
                error_type=type(error).__name__,
                error=str(error),
                readback_mode=(
                    "deferred_after_sampling_workflow"
                    if snapshot.event == "staged_sfa_graph_fingerprint"
                    else "deferred_after_model_forward"
                ),
                readback_may_synchronize=True,
            )
