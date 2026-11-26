# SPDX-License-Identifier: Apache-2.0
"""Cooperative decoder saves using existing CPU reservations and persistence."""

from types import SimpleNamespace
from weakref import proxy

import msgspec

from lmcache.v1.cache_engine import LayerwiseStoreResult
from lmcache.v1.rpc.storage_protocol import OperationIdentity, wire_hash
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle
from lmcache_ascend.v1 import process_store


def _adopt(engine, envelope):
    original = SimpleNamespace(
        identity=msgspec.convert(envelope["identity"], type=OperationIdentity),
        payload_digest=envelope["digest"],
    )
    state = process_store._Write(original, reply=envelope["reply"])
    engine.__dict__.setdefault("_process_store_writes", {})[original.identity.key] = (
        state
    )
    if not getattr(engine, "_process_store_guard_installed", False):
        owner = proxy(engine)
        engine.storage_process.drain_guards.append(
            lambda: not owner._process_store_writes
        )
        engine._process_store_guard_installed = True
    client, receiver = engine.storage_process.client, proxy(state)

    def retire(lease, proof):
        receiver.retired.append(
            client.retire(original.identity, original.payload_digest, lease, proof)
        )

    for item in state.reply["groups"]:
        state.holds.append(
            LocalLeaseHandle(
                lease_id=item["lease_id"],
                mapping=engine.storage_process.mapping,
                retire=retire,
            )
        )
    return state


def _release_peer(engine, state):
    proof = "fenced" if state.imported else "never_imported"
    for view in state.views:
        view.ref_count_down()
    state.views.clear()
    for hold in state.holds:
        hold.release(proof)
    for future in state.retired:
        if future.result(engine.storage_process.timeout_s).status != "RETIRED":
            raise RuntimeError("Cooperative store participant retirement is pending")
    engine._process_store_writes.pop(state.future.identity.key)


def store_group(engine, request, kv_group, caches, mask, kwargs, *, input_error=None):
    """Save one canonical group; never publish a partially populated DCP page.

    Called only by the foreground connector after every layer has finished.
    No operation is submitted for ordinary decode steps with no save request.
    """
    group = engine.dcp_write_group
    tokens = request.token_ids
    options = dict(kwargs, kv_group=kv_group, dcp_write_participants=group.sessions)
    candidates, error = [], None
    try:
        if input_error is not None:
            raise ValueError(input_error)
        if kv_group == 0 and (
            request.dcp_write_layout is None
            or request.dcp_write_layout.world_size != group.world_size
        ):
            raise ValueError("DCP save source layout differs from its producer group")
        candidates = process_store._candidates(
            engine, tokens, None, None, mask, options
        )
        signature = tuple(
            (a, b, wire_hash(k.chunk_hash, engine.storage_layout.hash_width))
            for a, b, k in candidates
        )
    except Exception as exc:
        signature, error = (), str(exc)
    identity = (request.req_id, kv_group, len(tokens))
    inputs = group.exchange("store_plan", identity, (signature, error))
    if any(item != inputs[0] for item in inputs) or any(item[1] for item in inputs):
        raise RuntimeError("Cooperative store request/key plans differ across ranks")
    result = LayerwiseStoreResult(request.req_id, kv_group)
    result.process_store_plan = {}
    if not candidates:
        return result
    monitor = getattr(engine, "stats_monitor", None)
    stats = (
        monitor.on_store_request(len(tokens) if mask is None else int(mask.sum()))
        if group.rank == 0 and monitor is not None
        else None
    )
    state = None

    def reserve():
        nonlocal state
        state = process_store._reserve(
            engine, request.req_id, candidates, kv_group, options, "page"
        )
        if state is None:
            return None
        source = options.get("dcp_prefix_source") if kv_group == 0 else None
        return {
            "identity": msgspec.to_builtins(state.future.identity),
            "digest": state.future.payload_digest,
            "reply": state.reply,
            "capture_start": min(source.total_tokens, len(tokens))
            if source is not None
            else 0,
        }

    try:
        envelope = group.leader("store_reserve", identity, reserve)
    except BaseException:
        engine.storage_process.mark_failed(
            "Cooperative store reservation outcome is unknown"
        )
        raise
    if envelope is None:
        return result
    plan, error = None, None
    try:
        try:
            if group.rank:
                state = _adopt(engine, envelope)
                mapping = engine.storage_process.mapping
                if (
                    state.reply["slab_name"],
                    state.reply["slab_size"],
                    state.reply["slab_generation"],
                ) != (
                    mapping.shm_name,
                    mapping.size,
                    mapping.generation,
                ):
                    raise ValueError(
                        "Cooperative store descriptor belongs to a stale slab"
                    )
            objects, starts, ends = process_store._destinations(
                engine, state, candidates, kv_group, True
            )
            for hold in state.holds:
                hold.mark_imported()
            state.imported = True
            if kv_group == 0:
                if group.rank == 0:
                    from lmcache_ascend.v1.dcp_save import seed_prefix

                    seed_prefix(
                        engine, options.get("dcp_prefix_source"), objects, starts, ends
                    )
                plan = engine.gpu_connector.prepare_dcp_group_capture(
                    objects[0],
                    starts,
                    ends,
                    request.dcp_write_layout,
                    group.rank,
                    kvcaches=caches,
                    memory_objs=objects,
                    capture_start=min(envelope["capture_start"], ends[-1]),
                )
            elif group.rank == 0:
                base = kwargs.get("slot_mapping_base", 0)
                slots = kwargs["slot_mapping"]
                if starts[0] < base or ends[-1] - base > len(slots):
                    raise ValueError(
                        "Cooperative indexer save exceeds its slot mapping"
                    )
                plan = engine.gpu_connector.prepare_group_capture(
                    objects,
                    starts,
                    ends,
                    kv_group=1,
                    kvcaches=caches,
                    slot_mapping=slots[starts[0] - base : ends[-1] - base],
                    slot_mapping_base=starts[0],
                )
        except Exception as exc:
            error = str(exc)
        # Retain operands even when a later submission/fence fails.
        if state is not None:
            state.dcp_plan = plan
        errors = group.exchange("store_prepared", identity, error)
        unknown = False
        if not any(errors) and plan is not None:
            try:
                enqueue = (
                    engine.gpu_connector.enqueue_dcp_group_capture
                    if kv_group == 0
                    else engine.gpu_connector.enqueue_group_capture
                )
                state.dcp_event = enqueue(plan)
            except Exception as exc:
                error = str(exc)
        try:
            engine.gpu_connector.finish_checkpoint_capture()
        except Exception as exc:
            unknown, error = True, str(exc)
        outcomes = group.exchange("store_fenced", identity, (unknown, error))
        if any(item[0] for item in outcomes):
            raise RuntimeError("Cooperative store native completion is unknown")
        state.fenced = True
        state.dcp_plan = None
        state.dcp_event = None
        failed = any(errors) or any(item[1] for item in outcomes)
        if group.rank:
            _release_peer(engine, state)
        group.exchange("store_peers_retired", identity, None)

        def publish():
            try:
                if failed:
                    return None
                committed = process_store._request(
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
                    raise RuntimeError("Cooperative store publication did not commit")
                process_store._observe(engine, candidates[: len(starts)], tokens, stats)
                return committed
            finally:
                process_store._release(engine, state)

        committed = group.leader("store_published", identity, publish)
        if failed:
            raise RuntimeError(
                f"Cooperative store capture failed: {errors}, {outcomes}"
            )
        result.starts, result.ends = starts, ends
        layers = engine.num_layers_for_group(kv_group)
        result.keys = [
            list(row)
            for row in zip(
                *(c[2].split_layers(layers) for c in candidates[: len(starts)])
            )
        ]
        result.committed_end = int(committed["committed_end"])
        result.process_store_plan = {
            "starts": tuple(starts),
            "ends": tuple(ends),
            "hashes": tuple(
                wire_hash(c[2].chunk_hash, engine.storage_layout.hash_width)
                for c in candidates[: len(starts)]
            ),
            "identity": state.future.identity,
        }
        return result
    except BaseException:
        if state is not None and state.future.identity.key in getattr(
            engine, "_process_store_writes", {}
        ):
            state.unknown = True
            engine.storage_process.mark_failed(
                "Cooperative store did not reach a shared terminal outcome"
            )
        elif state is None or group.failed:
            engine.storage_process.mark_failed("Cooperative store agreement failed")
        raise
