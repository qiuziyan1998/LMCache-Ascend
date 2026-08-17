# SPDX-License-Identifier: Apache-2.0
"""Opt-in content fingerprints for Group-1 NPU cache diagnostics.

The production data path must never copy NPU tensors to the host for
diagnostics.  All public entry points in this module are therefore guarded by
``enable_npu_content_diagnostics`` and are no-ops until the LMCache connector
configures that single knob.  Inline fingerprints explicitly report that the
host readback may synchronize the NPU.  Attention-side snapshots are cloned on
the current NPU stream and read back only after the model forward returns so a
diagnostic does not introduce a host synchronization between attention layers.
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
from typing import Any, Iterable, Sequence

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


@dataclass(slots=True)
class _DeferredSnapshot:
    event: str
    fields: dict[str, Any]
    values: torch.Tensor | None = None
    physical_slots: torch.Tensor | None = None
    expected_rows: dict[tuple[int, int], str] | None = None
    components: dict[str, torch.Tensor] | None = None
    comparison_components: dict[str, torch.Tensor] | None = None


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
                _content_log(
                    "group1_fingerprint_error",
                    requested_event=event,
                    error_type=type(error).__name__,
                    error=str(error),
                    readback_mode="deferred_after_model_forward",
                    readback_may_synchronize=False,
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
        if not _claim_once(
            f"tail:{int(kv_group)}:{stage}", req_id, layer_name
        ):
            continue
        query_length = int(query_starts[req_index + 1]) - query_start
        sequence_length = int(seq_lens[req_index])
        cached_prefix_tokens = max(sequence_length - query_length, 0)
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
        if not spec:
            _log_snapshot_skip_once(
                requested_event="group1_first_decode_consume",
                req_ids=[req_id],
                layer_name=layer_name,
                reason="source_fingerprint_not_registered",
            )
            continue
        if layer_id not in spec.get("sample_layer_ids", ()):
            continue
        context_tokens = max(seq_lens[req_index] - row_counts[req_index], 0)
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
    eligible: set[int] = set()
    registered: set[int] = set()
    for req_index, req_id in enumerate(request_ids):
        with _STATE_LOCK:
            spec = _REQUEST_SPECS.get(req_id)
        if spec:
            registered.add(req_index)
            if layer_id in spec.get("sample_layer_ids", ()):
                eligible.add(req_index)
    missing = [
        req_id
        for req_index, req_id in enumerate(request_ids)
        if req_index not in registered
    ]
    if missing:
        _log_snapshot_skip_once(
            requested_event="group1_selected_topk_fingerprint",
            req_ids=missing,
            layer_name=layer_name,
            reason="source_fingerprint_not_registered",
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
    seq_lens = _cpu_int_list(seq_lens_cpu)
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
                    "shape": list(rows.shape),
                    "dtype": str(topk_indices.dtype),
                    "snapshot_monotonic_ms": round(time.perf_counter() * 1000, 3),
                    "capture_order": ("after_topk_selection_before_compact_remap"),
                },
                values=rows,
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
    if stale:
        _content_log(
            "content_diagnostic_snapshot_dropped",
            dropped_count=stale,
            reason="previous_forward_did_not_flush",
        )


def flush_deferred_diagnostics() -> None:
    """Read queued NPU snapshots after model forward and emit fingerprints."""
    if not _ENABLED:
        return
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
                readback_mode="deferred_after_model_forward",
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
                readback_mode="deferred_after_model_forward",
                readback_may_synchronize=True,
            )
