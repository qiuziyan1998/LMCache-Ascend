# SPDX-License-Identifier: Apache-2.0
"""Worker D2H into service-owned destinations using actual shared slab views."""

# Standard
from dataclasses import dataclass, field
from dataclasses import replace
from contextlib import nullcontext
from types import GeneratorType
from typing import Any
from uuid import uuid4
from weakref import proxy
import time

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.cache_engine import LayerwiseStoreResult
from lmcache.utils import CacheStoreEvent, convert_tokens_to_list
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.rpc.storage_protocol import SourceGeometry, unpack_uints, wire_hash
from lmcache.v1.rpc.storage_protocol import content_digest
from lmcache.integration.vllm.storage_process import cancel_unadmitted_operation
from lmcache.v1.shared_cpu_cache import (
    LocalLeaseHandle,
    SharedChunkHandle,
    SharedHandleBatch,
)


@dataclass
class _Write:
    future: Any
    reply: dict | None = None
    holds: list[Any] = field(default_factory=list)
    views: list[Any] = field(default_factory=list)
    retired: list[Any] = field(default_factory=list)
    imported: bool = False
    fenced: bool = False
    committed: bool = False
    unknown: bool = False
    result: Any = None


def _profile(stats: Any, phase: str) -> Any:
    return getattr(stats, phase)() if stats is not None else nullcontext()


def _observe(
    engine: Any,
    candidates: list,
    tokens: Any,
    stats: Any,
    *,
    hashes: Any = None,
) -> None:
    """Preserve enabled store counters/events for the actually committed subset."""
    if stats is not None:
        engine.stats_monitor.on_store_finished(
            stats, sum(b - a for a, b, _ in candidates)
        )
    if not getattr(engine, "kv_events_enabled", False):
        return
    previous = 0
    for start, end, key in candidates:
        event = CacheStoreEvent(
            block_hashes=[key.chunk_hash],
            parent_block_hash=None if start == 0 else previous,
            token_ids=convert_tokens_to_list(tokens, start, end)
            if tokens is not None
            else hashes[start : end + 1]
            if hashes is not None
            else [],
            block_size=end - start,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        if isinstance(tokens, torch.Tensor):
            event.medium = tokens.device
        engine.kv_events.append(event)
        previous = key.chunk_hash


def _request(engine: Any, state: _Write, method: str, payload: Any) -> Any:
    original = state.future.identity
    deadline = time.monotonic() + engine.storage_process.timeout_s
    while True:
        future = engine.storage_process.client.request(
            method,
            payload,
            request_id=original.request_id,
            request_attempt=original.request_attempt,
            load_generation=original.load_generation,
            purpose="store",
        )
        try:
            result = future.result(max(0, deadline - time.monotonic()))
            if result.status == "BUSY":
                result = cancel_unadmitted_operation(
                    engine.storage_process.client,
                    future,
                    max(0, deadline - time.monotonic()),
                )
                if result.status == "CANCELLED_NOT_STARTED":
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "CPU store control admission deadline expired"
                        )
                    time.sleep(min(0.005, max(0, deadline - time.monotonic())))
                    continue
            break
        except BaseException:
            state.unknown = True
            raise
    if result.status not in ("COMPLETED", "FAILED_SAFE", "CANCELLED_NOT_STARTED"):
        state.unknown = True
        raise RuntimeError(result.error or f"CPU store returned {result.status}")
    acknowledged = engine.storage_process.client.acknowledge_terminal(
        future.identity,
        {
            "terminal_result_digest": content_digest(
                (result.status, result.value, result.error)
            )
        },
    ).result(engine.storage_process.timeout_s)
    if acknowledged.status != "ACKNOWLEDGED":
        state.unknown = True
        raise RuntimeError("CPU store control did not acknowledge its terminal result")
    if result.status != "COMPLETED" or result.error:
        raise RuntimeError(result.error or f"CPU store returned {result.status}")
    if isinstance(result.value, dict) and result.value.get("status") == "PENDING":
        state.unknown = True
        raise RuntimeError("CPU store publication is still active")
    return result.value


def _reserve(
    engine: Any,
    req_id: str,
    candidates: list,
    group: int,
    kwargs: dict,
    representation: str,
) -> _Write | None:
    plan = engine.storage_layout
    entry = dict(
        group_id=group,
        representation=representation,
        starts=tuple(a for a, _, _ in candidates),
        ends=tuple(b for _, b, _ in candidates),
        hashes=tuple(
            wire_hash(key.chunk_hash, plan.hash_width) for _, _, key in candidates
        ),
    )
    payload = dict(
        purpose="store",
        groups=(entry,),
        request_configs=dict(kwargs.get("request_configs") or {}),
        location=engine.store_location,
        transfer_spec=kwargs.get("transfer_spec"),
    )
    future = engine.storage_process.client.request(
        "reserve_write",
        payload,
        request_id=req_id,
        request_attempt=str(kwargs.get("process_request_attempt") or uuid4().hex),
        load_generation=int(kwargs.get("process_load_generation", 0)),
        purpose="store",
    )
    state = _Write(future)
    # A lost reply remains a live operation, never permission to reserve again.
    engine.__dict__.setdefault("_process_store_writes", {})[future.identity.key] = state
    if not getattr(engine, "_process_store_guard_installed", False):
        receiver = proxy(engine)
        engine.storage_process.drain_guards.append(
            lambda owner=receiver: not owner._process_store_writes
        )
        engine._process_store_guard_installed = True
    safe_failure = False
    try:
        result = future.result(engine.storage_process.timeout_s)
        if result.status == "BUSY":
            result = cancel_unadmitted_operation(
                engine.storage_process.client, future, engine.storage_process.timeout_s
            )
            if result.status == "CANCELLED_NOT_STARTED":
                engine._process_store_writes.pop(future.identity.key)
                return None
        state.result = result
        if result.status == "FAILED_SAFE":
            safe_failure = True
            engine.storage_process.client.acknowledge_terminal(future.identity).result(
                engine.storage_process.timeout_s
            )
            engine._process_store_writes.pop(future.identity.key)
            details = result.value if isinstance(result.value, dict) else {}
            if details.get("fallback") == "layers" and representation == "page":
                return _reserve(engine, req_id, candidates, group, kwargs, "layers")
            if details.get("store_refused"):
                return None
        if result.status != "COMPLETED" or result.error:
            raise RuntimeError(
                result.error or f"CPU store reservation: {result.status}"
            )
        state.reply = result.value
        if state.reply.get("status") == "SKIPPED":
            engine.storage_process.client.acknowledge_terminal(future.identity).result(
                engine.storage_process.timeout_s
            )
            engine._process_store_writes.pop(future.identity.key)
            return None
        mapping = engine.storage_process.mapping
        for item in state.reply["groups"]:
            receiver = proxy(state)
            client = engine.storage_process.client

            def retire(
                lease: str, proof: str, receiver: Any = receiver, client: Any = client
            ) -> None:
                receiver.retired.append(
                    client.retire(
                        future.identity,
                        future.payload_digest,
                        lease,
                        proof,
                    )
                )

            state.holds.append(
                LocalLeaseHandle(
                    lease_id=item["lease_id"],
                    mapping=mapping,
                    retire=retire,
                )
            )
        if (
            state.reply["slab_name"] != mapping.shm_name
            or state.reply["slab_size"] != mapping.size
            or state.reply["slab_generation"] != mapping.generation
        ):
            raise ValueError("CPU store destination belongs to a stale slab")
        return state
    except BaseException as error:
        if safe_failure:
            engine._process_store_writes.pop(future.identity.key, None)
        elif state.reply is not None:
            try:
                _release(engine, state)
            except BaseException as cleanup_error:
                cleanup_error.process_store_state = state
                raise
        else:
            state.unknown = True
            engine.storage_process.client.cancel(future.identity)
            error.process_store_state = state
        raise


def _flat_view(
    engine: Any,
    state: _Write,
    key: Any,
    item: dict,
    start: int,
    end: int,
    group: int,
    layer: int,
    index: int,
) -> Any:
    shapes = [torch.Size(shape) for shape in item["shapes"]]
    dtypes = [getattr(torch, dtype.removeprefix("torch.")) for dtype in item["dtypes"]]
    logical = sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )
    if (
        logical != item["logical_size"]
        or logical > item["extent"]
        or item["valid_tokens"] != end - start
    ):
        raise ValueError("CPU store destination logical geometry disagrees")
    handle = SharedChunkHandle(
        state.future.identity.request_id,
        "process_store",
        key,
        layer,
        group,
        index,
        engine.storage_process.mapping.shm_name,
        item["offset"],
        item["extent"],
        logical,
        shapes[0],
        dtypes[0],
        MemoryFormat(item["memory_format"]),
        engine.storage_process.mapping.generation,
        0,
        shapes=shapes,
        dtypes=dtypes,
    )
    view = engine.shared_cpu_cache_passive_allocator.create_view(
        handle,
        expected_request_id=handle.request_id,
        expected_phase=handle.phase,
        expected_layer_id=layer,
        expected_kv_group=group,
        expected_chunk_index=index,
        expected_key=key,
    )
    view.metadata.valid_tokens = end - start
    state.views.append(view)
    return view


def _destinations(
    engine: Any, state: _Write, candidates: list, group: int, layerwise: bool
) -> tuple[list, list, list]:
    layout = engine.storage_layout.groups[group]
    layers = len(layout.layer_names)
    if len(state.reply["groups"]) != 1:
        raise ValueError("CPU store destination has unexpected groups")
    item = state.reply["groups"][0]
    if item["group_id"] != group:
        raise ValueError("CPU store destination has the wrong KV group")
    starts, ends = list(item["starts"]), list(item["ends"])
    if item["representation"] == "layers":
        count = len(starts) // layers
        if (
            len(starts) != count * layers
            or len(item["objects"]) != len(starts)
            or tuple(item["layer_ids"]) != tuple(range(layers)) * count
        ):
            raise ValueError("CPU store layer fallback has invalid cardinality")
        rows = [[] for _ in range(layers)]
        for i, (start, end, obj) in enumerate(
            zip(starts, ends, item["objects"], strict=True)
        ):
            chunk, layer = divmod(i, layers)
            if (start, end) != candidates[chunk][:2]:
                raise ValueError("CPU store layer fallback changed canonical ranges")
            rows[layer].append(
                _flat_view(
                    engine,
                    state,
                    candidates[chunk][2].split_layers(layers)[layer],
                    obj,
                    start,
                    end,
                    group,
                    layer,
                    chunk,
                )
            )
        return rows, starts[::layers], ends[::layers]
    if (
        tuple(zip(starts, ends, strict=True))
        != tuple((a, b) for a, b, _ in candidates[: len(starts)])
        or not starts
    ):
        raise ValueError("CPU store destination changed canonical ranges")
    if item["representation"] == "page":
        count = len(starts)
        offsets, extents = (
            unpack_uints(item["page_offsets"], count),
            unpack_uints(item["page_extents"], count),
        )
        indices = unpack_uints(item["geometry_ids"], count, 32)
        geometries = msgspec.convert(
            item["geometries"], type=tuple[SourceGeometry, ...]
        )
        pages = []
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            geometry = geometries[indices[index]]
            geometry.validate(layers, extents[index])
            if (
                not end - start
                <= geometry.token_capacity
                <= engine.storage_layout.chunk_size
                or tuple(geometry.layer_dtypes) != tuple(layout.layer_dtypes)
                or geometry.layer_starts
                != tuple(sum(geometry.layer_sizes[:n]) for n in range(layers))
            ):
                raise ValueError(
                    "CPU store page geometry disagrees with registered layers"
                )
            for layer in range(layers):
                widths = layout.plane_token_bytes[layer]
                if geometry.layer_sizes[layer] != geometry.token_capacity * sum(widths):
                    raise ValueError("CPU store page precision or sidecars changed")
            shapes = [torch.Size(shape) for shape in geometry.layer_shapes]
            batch = SharedHandleBatch(
                engine.storage_process.mapping.shm_name,
                0,
                layers,
                1,
                [geometry.layer_sizes[0]],
                [0],
                [],
                [offsets[index]],
                [extents[index]],
            )
            page = engine.shared_cpu_cache_passive_allocator.create_page_view(
                batch,
                chunk_index=0,
                shape=shapes[0],
                dtype=getattr(torch, geometry.layer_dtypes[0]),
                fmt=MemoryFormat(layout.memory_format),
                cached_positions=range(start, end),
                layer_shapes=shapes,
            )
            state.views.append(page)
            pages.append(page)
        return [list(pages) for _ in range(layers)], starts, ends
    if item["representation"] != "flat" or len(item["objects"]) != len(starts):
        raise ValueError("CPU store destination has an unknown representation")
    objects = [
        _flat_view(
            engine, state, candidates[i][2], obj, starts[i], ends[i], group, 0, i
        )
        for i, obj in enumerate(item["objects"])
    ]
    if layerwise:
        raise ValueError("Layerwise process stores require page or layer destinations")
    return objects, starts, ends


def _release(engine: Any, state: _Write) -> None:
    if state.unknown:
        raise RuntimeError("CPU store completion is unknown; destinations retained")
    proof = "fenced" if state.imported else "never_imported"
    if state.imported and not state.fenced:
        try:
            engine.gpu_connector.store_stream.synchronize()
            state.fenced = True
        except BaseException:
            state.unknown = True
            raise
    if not state.committed:
        _request(
            engine,
            state,
            "cancel_write",
            {
                "reservation_id": state.reply["reservation_id"],
                "proof": proof,
            },
        )
    for view in state.views:
        view.ref_count_down()
    state.views.clear()
    for hold in state.holds:
        hold.release(proof)
    try:
        for retirement in state.retired:
            if retirement.result(engine.storage_process.timeout_s).status != "RETIRED":
                raise RuntimeError("CPU store destination retirement is pending")
        acknowledged = engine.storage_process.client.acknowledge_terminal(
            state.future.identity,
            {
                "terminal_result_digest": content_digest(
                    (state.result.status, state.result.value, state.result.error)
                )
            },
        ).result(engine.storage_process.timeout_s)
    except BaseException:
        state.unknown = True
        raise
    if acknowledged.status != "ACKNOWLEDGED":
        state.unknown = True
        raise RuntimeError("CPU store destination ownership was not acknowledged")
    engine._process_store_writes.pop(state.future.identity.key, None)


def _candidates(
    engine: Any, tokens: Any, hashes: Any, offsets: Any, mask: Any, kwargs: dict
) -> list:
    return list(
        engine.token_database.process_tokens(
            tokens,
            hashes,
            offsets,
            mask,
            request_configs=kwargs.get("request_configs"),
            kv_group=kwargs.get("kv_group", 0),
        )
    )


def _unknown_completion(engine: Any, state: _Write) -> None:
    """Quarantine only after normal TP failure publication, or during abort."""
    try:
        if state.reply is None:
            engine.storage_process.client.cancel(state.future.identity)
            return
        _request(
            engine,
            state,
            "cancel_write",
            {
                "reservation_id": state.reply["reservation_id"],
                "proof": "unknown",
            },
        )
    finally:
        engine.storage_process.mark_failed("CPU store native completion is unknown")


def _publish_completion(
    engine: Any,
    result: Any,
    tokens: Any,
    kwargs: dict,
    error: BaseException | None = None,
    state: _Write | None = None,
) -> None:
    """Publish terminal store status before any rank may promote this group."""
    predecessor = kwargs.get("process_collective_predecessor")
    if predecessor is not None:
        try:
            predecessor.exception()
        except BaseException:
            pass
    identity = dict(
        req_id=result.request_id,
        phase="storage_process_store_write",
        request_ordinal=int(kwargs.get("decode_window_end") or len(tokens)),
        layer_id=0,
        kv_group=result.kv_group,
    )
    if not engine._is_passive():
        envelope = engine._shared_layerwise_error_envelope(
            **identity,
            message=str(error) if error else "",
            details={
                "plan": result.process_store_plan,
                "committed_end": result.committed_end,
                "ownership_unknown": bool(state is not None and state.unknown),
            },
        )
        if error is None:
            envelope = replace(envelope, status="skipped")
        engine._broadcast_shared_envelope(envelope)
    else:
        envelope = engine._receive_matching_shared_envelope(**identity)
    engine._validate_shared_layerwise_envelope(
        replace(envelope, status="skipped"), **identity
    )
    if envelope.error_details.get("ownership_unknown"):
        if state is not None:
            _unknown_completion(engine, state)
        else:
            engine.storage_process.mark_failed("CPU store native completion is unknown")
    engine._validate_shared_layerwise_envelope(envelope, **identity)
    result.process_store_plan = envelope.error_details["plan"]
    result.committed_end = envelope.error_details["committed_end"]


def run_store(
    engine: Any,
    req_id: str,
    tokens: Any,
    hashes: Any,
    offsets: Any,
    mask: Any,
    kwargs: dict,
    num_to_store_tokens: int | None = None,
) -> None:
    """Preserve ordinary whole-group flat storage with an M-owned allocation."""
    if engine.is_frozen():
        return
    monitor = getattr(engine, "stats_monitor", None)
    stats = (
        monitor.on_store_request(
            num_to_store_tokens
            if num_to_store_tokens is not None
            else (len(tokens) if tokens is not None else sum(offsets))
        )
        if monitor is not None
        else None
    )
    with _profile(stats, "profile_process_tokens"):
        candidates = _candidates(engine, tokens, hashes, offsets, mask, kwargs)
    if not candidates:
        return
    group = int(kwargs.get("kv_group", 0))
    try:
        state = _reserve(engine, req_id, candidates, group, kwargs, "flat")
    except BaseException as error:
        failed = getattr(error, "process_store_state", None)
        if failed is not None and failed.unknown:
            _unknown_completion(engine, failed)
        raise
    if state is None:
        return
    try:
        objects, starts, ends = _destinations(engine, state, candidates, group, False)
        if engine.use_layerwise:
            rows = [[] for _ in engine.storage_layout.groups[group].layer_names]
            for chunk, (start, end, item) in enumerate(
                zip(
                    starts,
                    ends,
                    state.reply["groups"][0]["objects"],
                    strict=True,
                )
            ):
                position = 0
                for layer, row in enumerate(rows):
                    shape, dtype, fmt = engine._expected_shared_cpu_chunk_metadata(
                        kv_group=group,
                        num_tokens=end - start,
                        layer_id=layer,
                    )
                    size = shape.numel() * dtype.itemsize
                    row.append(
                        _flat_view(
                            engine,
                            state,
                            candidates[chunk][2].split_layers(len(rows))[layer],
                            dict(
                                offset=item["offset"] + position,
                                extent=size,
                                logical_size=size,
                                shapes=(tuple(shape),),
                                dtypes=(str(dtype),),
                                memory_format=fmt.value,
                                valid_tokens=end - start,
                            ),
                            start,
                            end,
                            group,
                            layer,
                            chunk,
                        )
                    )
                    position += size
                if position != item["logical_size"]:
                    raise ValueError(
                        "Flat store group does not match exact layer offsets"
                    )
            objects = rows
        for hold in state.holds:
            hold.mark_imported()
        state.imported = True
        with _profile(stats, "profile_from_gpu"):
            result = engine.gpu_connector.batched_from_gpu(
                objects, starts, ends, **kwargs
            )
            if isinstance(result, GeneratorType):
                try:
                    for _ in result:
                        pass
                finally:
                    result.close()
            else:
                engine.gpu_connector.store_stream.synchronize()
        state.fenced = True
        with _profile(stats, "profile_put"):
            result = _request(
                engine,
                state,
                "commit_write",
                {
                    "reservation_id": state.reply["reservation_id"],
                    "proof": "fenced",
                },
            )
        state.committed = result.get("status") == "COMMITTED"
        if not state.committed:
            raise RuntimeError("CPU store did not commit")
        _observe(engine, candidates[: len(starts)], tokens, stats, hashes=hashes)
    finally:
        try:
            _release(engine, state)
        except BaseException:
            if state.unknown:
                _unknown_completion(engine, state)
            raise


def store_layer(engine: Any, tokens: Any, mask: Any, kwargs: dict) -> Any:
    """Keep the existing native per-layer/group transfer and suspension points."""
    req_id, group = engine._get_req_id(kwargs), int(kwargs.get("kv_group", 0))
    layers = engine._num_layers_for_kv_group(group)
    result = LayerwiseStoreResult(req_id, group)
    result.process_store_plan = {}
    if not engine.is_healthy() or engine._is_passive() or engine.is_frozen():
        for _ in range(layers):
            yield
        _publish_completion(engine, result, tokens, kwargs)
        yield result
        return
    try:
        engine._ensure_layerwise_connector_layout(**kwargs)
        layers = engine._num_transfer_layers_for_call(group, kwargs)
        monitor = getattr(engine, "stats_monitor", None)
        requested = len(tokens) if mask is None else int(mask.sum().item())
        stats = monitor.on_store_request(requested) if monitor is not None else None
        with _profile(stats, "profile_process_tokens"):
            candidates = _candidates(engine, tokens, None, None, mask, kwargs)
        state = (
            _reserve(engine, req_id, candidates, group, kwargs, "page")
            if candidates
            else None
        )
    except BaseException as error:
        for _ in range(layers):
            yield
        _publish_completion(
            engine,
            result,
            tokens,
            kwargs,
            error,
            getattr(error, "process_store_state", None),
        )
        raise
    if state is None:
        for _ in range(layers):
            yield
        _publish_completion(engine, result, tokens, kwargs)
        yield result
        return
    consumer = None
    failure = None
    closing = False
    try:
        objects, starts, ends = _destinations(engine, state, candidates, group, True)
        for hold in state.holds:
            hold.mark_imported()
        state.imported = True
        connector = engine.gpu_connector
        group_store = getattr(connector, "batched_from_gpu_group", None)
        supports = getattr(connector, "supports_batched_from_gpu_group", None)
        use_group = bool(
            (kwargs.get("decode_window_save") or kwargs.get("all_layers_ready"))
            and callable(group_store)
            and callable(supports)
            and supports(group)
            and (
                not kwargs.get("windowed_sparse_save")
                or all(
                    end - start == engine.config.chunk_size
                    for start, end in zip(starts, ends, strict=True)
                )
            )
        )
        if use_group:
            for _ in range(layers):
                yield
            with _profile(stats, "profile_from_gpu"):
                group_store(objects, starts, ends, **kwargs)
        else:
            consumer = connector.batched_from_gpu(objects, starts, ends, **kwargs)
            with _profile(stats, "profile_from_gpu"):
                next(consumer)
            for _ in range(layers):
                yield
                with _profile(stats, "profile_from_gpu"):
                    next(consumer)
        # Both production native paths already synchronize their final D2H.
        state.fenced = True
        with _profile(stats, "profile_put"):
            committed = _request(
                engine,
                state,
                "commit_write",
                {
                    "reservation_id": state.reply["reservation_id"],
                    "proof": "fenced",
                },
            )
        state.committed = committed.get("status") == "COMMITTED"
        if not state.committed:
            raise RuntimeError("CPU layerwise store did not commit")
        result.starts, result.ends = starts, ends
        result.keys = [
            list(row)
            for row in zip(
                *[
                    candidate[2].split_layers(layers)
                    for candidate in candidates[: len(starts)]
                ],
                strict=True,
            )
        ]
        result.committed_end = int(committed["committed_end"])
        _observe(engine, candidates[: len(starts)], tokens, stats)
        result.process_store_plan = {
            "starts": tuple(starts),
            "ends": tuple(ends),
            "hashes": tuple(
                wire_hash(candidate[2].chunk_hash, engine.storage_layout.hash_width)
                for candidate in candidates[: len(starts)]
            ),
            "identity": state.future.identity,
        }
    except GeneratorExit:
        closing = True
    except BaseException as error:
        failure = error
    finally:
        try:
            _release(engine, state)
        except BaseException as error:
            failure = error
        finally:
            if consumer is not None:
                consumer.close()
    if closing:
        if failure is not None:
            if state.unknown:
                _unknown_completion(engine, state)
            raise failure
        return
    _publish_completion(engine, result, tokens, kwargs, failure, state)
    if failure is not None:
        raise failure
    yield result
