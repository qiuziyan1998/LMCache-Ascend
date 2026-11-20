# SPDX-License-Identifier: Apache-2.0
"""Cooperative foreground capture; canonical publication stays with TP0."""

from types import SimpleNamespace
import time

import msgspec

from lmcache.integration.vllm.preemption_checkpoint import CheckpointResult
from lmcache.v1.rpc.storage_protocol import OperationIdentity, content_digest
from lmcache_ascend.v1.preemption_checkpoint import CaptureJob, _checkpoint_slots


def capture_checkpoint(worker, spec, caches, block_size, prefix_state=None):
    """Capture each rank's Group-0 stripes before acknowledging HBM reuse.

    All control exchange runs on the worker model thread. Original native plans
    and service reservations remain explicitly owned on any uncertain failure.
    """
    group = worker.engine.dcp_write_group
    layout = spec.dcp_write_layout
    key = (spec.req_id, spec.generation)
    # Physical block IDs can differ between owners. Agree logical coverage and
    # configuration, while each participant validates its own physical table.
    identity = (
        *key,
        spec.base,
        spec.end,
        spec.prefix_end,
        spec.resident_start,
        content_digest(spec.request_configs),
    )
    job = CaptureJob(spec)
    state = None
    error = None
    try:
        if (
            layout is None
            or layout.world_size != group.world_size
            or layout.block_size != block_size
        ):
            raise ValueError("Checkpoint DCP layout differs from its worker group")
        if set(caches) != {0, 1} or not 0 <= spec.base <= spec.prefix_end < spec.end:
            raise ValueError("Invalid cooperative checkpoint coverage")
        if not spec.base <= spec.resident_start <= spec.end:
            raise ValueError("Invalid cooperative checkpoint resident frontier")
        tuple(layout.runs(spec.resident_start, spec.end, group.rank))
        if group.rank == 0:
            # Group 1 remains fully resident and uses its original slot mapping.
            if (spec.end + block_size - 1) // block_size > len(spec.blocks[1]) or any(
                b <= 0
                for b in spec.blocks[1][
                    spec.base // block_size : (spec.end + block_size - 1) // block_size
                ]
            ):
                raise ValueError(
                    "Checkpoint indexer source includes null/missing blocks"
                )
    except Exception as exc:
        error = str(exc)
    errors = group.exchange("checkpoint_preflight", identity, error)
    if any(errors):
        if worker.writer:
            worker.results.append(CheckpointResult(*key, "failed", reason=str(errors)))
        return

    def reserve():
        if key in worker.jobs:
            old = worker.jobs[key]
            return {
                "duplicate": True,
                "end": old.captured_end,
                "cancelled": old.cancelled,
            }
        if (
            key in worker.reservations
            or len(set(worker.jobs) | worker.reservations.keys()) >= worker.max_jobs
        ):
            return {"refused": True}
        source = getattr(prefix_state, "prepared_sparse_sources", {}).get(0)
        try:
            try:
                if source is not None:
                    worker._sources[key] = (
                        worker.engine.compact_source_importer.source_descriptor(source)
                    )
                worker._reserve_fragments(job, caches)
            except Exception as error:
                # An explicit pre-export refusal has no peer descriptors or DMA.
                # Preserve the ordinary retry/recompute outcome, not fatal stop.
                pending = worker.reservations.get(key)
                if pending is None:
                    return {"refused": True, "reason": str(error)}
                if (
                    pending is not None
                    and pending.reply is None
                    and pending.future.done()
                ):
                    outcome = pending.future.result()
                    if outcome.status in ("FAILED_SAFE", "CANCELLED_NOT_STARTED") or (
                        outcome.status == "COMPLETED"
                        and outcome.value.get("status") == "SKIPPED"
                    ):
                        pending.cooperative_pending = False
                        worker._ack_released(pending.future.identity, no_leases=True)
                        return {"refused": True}
                raise
        finally:
            worker._sources.pop(key, None)
        original = worker.reservations[key]
        return {
            "identity": msgspec.to_builtins(original.future.identity),
            "digest": original.future.payload_digest,
            "reply": original.reply,
        }

    try:
        envelope = group.leader("checkpoint_reserve", identity, reserve)
    except BaseException:
        # A lost reservation envelope can hide exported peer leases. Neither
        # timeout nor a failed collective is evidence that those leases retired.
        if key in worker.reservations:
            worker.reservations[key].job.quarantined = True
        worker.engine.storage_process.mark_failed(
            "Cooperative checkpoint reservation outcome is unknown"
        )
        raise
    if envelope.get("duplicate") or envelope.get("refused"):
        if worker.writer:
            worker.results.append(
                CheckpointResult(
                    *key,
                    "failed"
                    if envelope.get("refused") or envelope.get("cancelled")
                    else "captured",
                    envelope.get("end", 0),
                    envelope.get("reason", "duplicate or busy cooperative checkpoint"),
                )
            )
        return

    try:
        if not worker.writer:
            from lmcache_ascend.v1.process_checkpoint import _Reservation

            original = SimpleNamespace(
                identity=msgspec.convert(envelope["identity"], type=OperationIdentity),
                payload_digest=envelope["digest"],
            )
            state = _Reservation(
                original, job, reply=envelope["reply"], cooperative_pending=True
            )
            worker.reservations[key] = state
        else:
            state = worker.reservations[key]
            state.cooperative_pending = True
        error = None
        try:
            if not worker.writer:
                worker._import_fragments(job, state)
            job.captured_end = int(state.reply["captured_end"])
            worker._begin_capture(job)
            for kv_group, fragments in job.fragments.items():
                if kv_group == 1 and not worker.writer:
                    continue
                starts, ends, pages = map(list, zip(*fragments))
                connector = worker.engine.gpu_connector
                if kv_group == 0:
                    plan = connector.prepare_dcp_group_capture(
                        pages,
                        starts,
                        ends,
                        layout,
                        group.rank,
                        kvcaches=caches[0],
                    )
                else:
                    slots = _checkpoint_slots(
                        spec.blocks[1], starts[0], ends[-1], block_size
                    )
                    plan = connector.prepare_group_capture(
                        [pages] * worker.engine.num_layers_for_group(1),
                        starts,
                        ends,
                        slot_mapping=slots,
                        slot_mapping_base=starts[0],
                        kv_group=1,
                        kvcaches=caches[1],
                    )
                job.plans.append((kv_group, plan))
        except Exception as exc:
            error = str(exc)
        errors = group.exchange("checkpoint_prepared", identity, error)
        unknown = False
        if not any(errors):
            try:
                state.dcp_events = []
                for kv_group, plan in job.plans:
                    enqueue = (
                        worker.engine.gpu_connector.enqueue_dcp_group_capture
                        if kv_group == 0
                        else worker.engine.gpu_connector.enqueue_group_capture
                    )
                    state.dcp_events.append(enqueue(plan))
            except Exception as exc:
                error = str(exc)
        # Preparation can already enqueue metadata uploads. Fence also after a
        # partial submission exception, before advertising any safe outcome.
        try:
            worker.engine.gpu_connector.finish_checkpoint_capture()
        except Exception as exc:
            unknown, error = True, str(exc)
        outcomes = group.exchange("checkpoint_fenced", identity, (unknown, error))
        if any(outcome[0] for outcome in outcomes):
            raise RuntimeError("Cooperative checkpoint has unknown native completion")
        state.fenced = True
        state.dcp_events = []
        job.plans.clear()
        state.cooperative_pending = False
        failed = any(errors) or any(outcome[1] for outcome in outcomes)
        # Peers release only their own participant holds. TP0's producer/consumer
        # holds retain canonical pages until accepted-token seal or cancellation.
        if not worker.writer:
            for fragments in job.fragments.values():
                for _, _, page in fragments:
                    page.ref_count_down()
            job.fragments.clear()
            for hold in state.holds:
                hold.release("fenced" if state.imported else "never_imported")
            deadline = time.monotonic() + worker.timeout
            while any(hold.lease_id not in state.retired for hold in state.holds):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Cooperative checkpoint participant retirement timed out"
                    )
                time.sleep(0.001)
            worker.reservations.pop(key)
        group.exchange("checkpoint_peers_retired", identity, None)
        if worker.writer:
            if failed or job.cancelled:
                worker._release(job)
                worker.results.append(
                    CheckpointResult(*key, "failed", reason=str((errors, outcomes)))
                )
            else:
                worker.jobs[key] = job
                worker.results.append(
                    CheckpointResult(*key, "captured", job.captured_end)
                )
    except BaseException:
        job.quarantined = True
        worker.jobs[key] = job
        worker.engine.storage_process.mark_failed(
            "Cooperative checkpoint ownership is unresolved"
        )
        raise
