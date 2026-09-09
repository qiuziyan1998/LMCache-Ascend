# SPDX-License-Identifier: Apache-2.0
"""Final-call coalescing through the real engine and producer admission path.

Token identities, tensor planning, persistence and the network are host fakes.
No native buffers are allocated or transferred by these tests.
"""

from asyncio import CancelledError
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition
from types import SimpleNamespace
import ctypes
import gc
import weakref

from lmcache.v1.remote_fill import (
    PROTOCOL_VERSION,
    ControlPage,
    OperationIdentity,
    ProtocolLimits,
    ReserveWindowRequest,
    decode_request,
    encode_request,
    manifest_digest,
    request_payload_digest,
)
import msgspec
import pytest

from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine, _DirectStoreRequestState
from lmcache_ascend.v1.remote_fill_coordinator import (
    ProducerRequestState,
    RemoteFillCoordinator,
)
from lmcache_ascend.v1.remote_fill_producer import (
    RemoteFillFatalError,
    RemoteFillWindowResult,
)


@dataclass
class Key:
    kv_group: int
    chunk_hash: int
    valid: int

    def to_string(self):
        return f"{self.kv_group}:{self.chunk_hash}:{self.valid}"


class Tokens:
    def process_tokens(
        self, tokens=None, hashes=None, offsets=None, kv_group=0, **kwargs
    ):
        if hashes is not None:
            offset = 0
            for key_hash, size in zip(hashes, offsets, strict=True):
                yield offset, offset + size, Key(kv_group, key_hash, size)
                offset += size
        else:
            yield from self.process_tokens_from_prefix(tokens, kv_group=kv_group)

    def process_tokens_from_prefix(
        self, tokens, prefix_token_count=0, kv_group=0, **kwargs
    ):
        for start in range(prefix_token_count, len(tokens), 1024):
            end = min(start + 1024, len(tokens))
            yield start, end, Key(kv_group, start, end - start)


class Owner:
    pass


def fixture(
    *,
    groups=(0,),
    mode="per_chunk",
    byte_limit=8 << 30,
    prefix=1024,
    tail_behavior="normal",
    remote=True,
    job_limit=2,
    layers=79,
    payload_widths=None,
):
    engine = object.__new__(AscendLMCacheEngine)
    config = SimpleNamespace(
        chunk_size=1024,
        dsa_two_groups=True,
        save_unfull_chunk=True,
        remote_fill_submission_mode=mode,
        remote_fill_window_tokens=4096,
        remote_fill_max_inflight_windows_per_request=job_limit,
        remote_fill_max_inflight_bytes=byte_limit,
        remote_fill_max_bytes_per_request=16 << 30,
        remote_fill_circuit_breaker_enabled=False,
        get_extra_config_value=lambda name, default: default,
    )
    state = _DirectStoreRequestState(
        remote_fill=ProducerRequestState(handoff=SimpleNamespace(transfer_id="t"))
    )
    jobs, persistent, transfers, admissions, owner_refs = [], [], [], [], []
    engine.config = config
    engine._direct_store_enabled = True
    engine._direct_store_states = {"r": state}
    engine._store_cv = Condition()
    engine._pending_store_reqs = {}
    engine.token_database = Tokens()
    engine._remote_fill_direct_groups = lambda: groups
    engine._remote_fill_session_context = lambda: None
    engine._remote_fill_pages_per_window = lambda: 4 * len(groups)
    engine._remote_fill_prepare_request = (
        lambda *a: remote and not state.remote_fill.disabled_reason
    )
    engine._record_live_source_pages = lambda *a: None
    engine._finalize_direct_store = lambda *a, **k: setattr(state, "finalized", True)
    byte_width = payload_widths or {0: layers * 576 * 2, 1: layers * 128 * 2}
    engine._remote_fill_immutable_layout = lambda: (
        "fixture",
        SimpleNamespace(
            num_layers=layers,
            group=lambda group: SimpleNamespace(
                expected_bytes=lambda tokens, layers: tokens * byte_width[group]
            ),
        ),
    )

    def exists(keys):
        if tail_behavior == "exists_error" and any(k.valid < 1024 for k in keys):
            raise RuntimeError("tail lookup failed")
        return [
            k.chunk_hash + k.valid <= prefix
            or (tail_behavior == "existing" and k.valid < 1024)
            for k in keys
        ]

    def planner(caches, slots, starts, ends, group, **kwargs):
        if tail_behavior == "unsupported" and any(
            end - start < 1024 for start, end in zip(starts, ends)
        ):
            return None
        owner = Owner()
        owner_refs.append((group, weakref.ref(owner)))
        return (
            [[100 + start] for start in starts],
            [[(end - start) * byte_width[group]] for start, end in zip(starts, ends)],
            (owner,),
        )

    engine.gpu_connector = SimpleNamespace(plan_direct_page_sources=planner)
    engine.storage_manager = SimpleNamespace(
        batched_external_pages_exist=exists,
        submit_remote_fill_direct_push=object(),
        prepare_remote_fill_source=object(),
    )
    coordinator = RemoteFillCoordinator(
        config=config,
        tp_size=4,
        storage_manager=engine.storage_manager,
        fatal_reporter=engine._remote_fill_require_paired_restart,
    )
    engine._get_remote_fill_coordinator = lambda: coordinator

    def submit(job):
        future = Future()
        jobs.append((job, future))
        return future

    coordinator._executor = SimpleNamespace(submit=submit)

    def control_pages(batch):
        return tuple(
            ControlPage(
                canonical_key=k.to_string(),
                kv_group=k.kv_group,
                chunk_index=start // 1024,
                chunk_start=start,
                chunk_end=end,
                valid_tokens=end - start,
                destination_tp_rank=0,
                expected_bytes=sum(sizes),
                layer_count=layers,
                layout_tag="fixture",
            )
            for k, (start, end), sizes in zip(
                batch.keys, batch.ranges, batch.sizes, strict=True
            )
        )

    engine._remote_fill_control_pages = control_pages
    engine._remote_fill_probe_control_pages = lambda plans, start, end: tuple(
        ControlPage(
            canonical_key=key.to_string(),
            kv_group=g,
            chunk_index=a // 1024,
            chunk_start=a,
            chunk_end=b,
            valid_tokens=b - a,
            destination_tp_rank=0,
            expected_bytes=(b - a) * byte_width[g],
            layer_count=layers,
            layout_tag="fixture",
        )
        for g in groups
        for a, b, key in plans[g]
        if start <= a and b <= end
    )

    def transfer(**kwargs):
        plan = kwargs["source_plan"]
        assert plan.owners and all(
            any(ref() is owner for g, ref in owner_refs if g in groups)
            for owner in plan.owners
        )
        transfers.append(
            (
                kwargs["control_pages"],
                plan.producer_events,
                tuple(
                    (p.canonical_key, p.source_ptrs, p.source_lengths)
                    for p in plan.pages
                ),
            )
        )
        return RemoteFillWindowResult(kwargs["window_id"], True, True)

    state.remote_fill.session = SimpleNamespace(
        direct_viable=True,
        request_id="r",
        probe_window=lambda **kw: RemoteFillWindowResult(kw["window_id"], True, False),
        transfer_window=transfer,
    )

    def persist(batch, retry):
        persistent.append(
            (
                tuple(k.to_string() for k in batch.keys),
                batch.ranges,
                tuple(tuple(v) for v in batch.sizes),
                batch.ready_events,
            )
        )
        state.submitted_end.update(batch.group_ends)
        future = Future()
        future.set_result(None)
        state.futures.append(future)
        return future

    engine._track_direct_batch = persist

    def drain(reqs):
        admissions.append(
            (
                state.remote_fill.queued_windows,
                state.remote_fill.queued_bytes,
                state.remote_fill.disabled_reason,
            )
        )
        while jobs:
            job, future = jobs.pop(0)
            future.set_result(job())
        state.committed_end.update(state.submitted_end)
        return set(reqs)

    engine.wait_for_direct_stores = drain
    engine._store_direct_cpu_group = lambda req, tokens, *a: len(tokens)
    fences = (object(), object())

    def invoke(end=2116, final=True, events=None):
        source_events = fences if events is None else events
        return engine.store_direct_prefill(
            "r",
            list(range(end)),
            {0: [object()], 1: [object()]},
            {0: object(), 1: object()},
            final=final,
            slot_mapping_base=prefix,
            verified_prefix_end=prefix,
            source_ready_event=fences[0],
            source_ready_event_source="forward_context.sfa_reshape_cache_event",
            source_ready_events=source_events,
        )

    return SimpleNamespace(
        engine=engine,
        config=config,
        state=state,
        coordinator=coordinator,
        jobs=jobs,
        persistent=persistent,
        transfers=transfers,
        admissions=admissions,
        owner_refs=owner_refs,
        fences=fences,
        invoke=invoke,
        drain=drain,
    )


@pytest.mark.parametrize("groups", [(0,), (0, 1)])
@pytest.mark.parametrize("tail_tokens", [1, 68, 1023])
def test_final_call_coalesces_tail_with_full_pages_under_two_job_limit(
    groups, tail_tokens
):
    f = fixture(groups=groups)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert f.invoke(end=2048 + tail_tokens)
        assert f.admissions[0] == (
            2,
            (1024 + tail_tokens) * sum({0: 91008, 1: 20224}[g] for g in groups),
            "",
        )
        assert (
            len(f.persistent) == 2
        )  # Persistent full and tail writes remain separate.
        assert f.persistent[0][1] == ((1024, 2048), (1024, 2048))
        assert f.persistent[1][1] == ((2048, 2048 + tail_tokens),) * 2
        assert len(f.transfers) == 1
        pages, fences, sources = f.transfers[0]
        assert [(p.chunk_start, p.chunk_end, p.kv_group) for p in pages] == [
            (a, b, g)
            for a, b in [(1024, 2048), (2048, 2048 + tail_tokens)]
            for g in groups
        ]
        assert set(p.canonical_key for p in pages) == set(key for key, _, _ in sources)
        assert all(
            p.expected_bytes == sum(sizes)
            for p in pages
            for key, _, sizes in sources
            if key == p.canonical_key
        )
        assert fences == f.fences
        request = ReserveWindowRequest(
            common=OperationIdentity(
                protocol_version=PROTOCOL_VERSION,
                operation_id="reserve",
                operation_sequence=1,
                payload_digest="0" * 64,
                transfer_id="t",
                request_attempt=0,
                destination_engine_epoch=1,
                shared_cache_generation=1,
            ),
            window_id=0,
            source_generation=1,
            control_pages=pages,
            manifest_digest=manifest_digest(pages),
        )
        request = msgspec.structs.replace(
            request,
            common=msgspec.structs.replace(
                request.common, payload_digest=request_payload_digest(request)
            ),
        )
        limits = ProtocolLimits(max_window_bytes=8 << 30, max_reserved_bytes=16 << 30)
        assert decode_request(encode_request(request, limits), limits) == request
        assert (
            f.state.remote_fill.queued_windows == f.state.remote_fill.queued_bytes == 0
        )
        assert f.coordinator._queued_bytes == 0
        assert f.state.committed_end == {0: 2048 + tail_tokens, 1: 2048 + tail_tokens}
        assert all(ref() is None for _, ref in f.owner_refs)
    finally:
        if was_enabled:
            gc.enable()


def test_nonfinal_chunks_still_stream_and_are_not_held_for_the_tail():
    f = fixture()
    assert f.invoke(end=2048, final=False)
    assert len(f.jobs) == 2  # Probe and full-page job already submitted.
    f.drain(("r",))
    assert f.invoke()
    assert len(f.transfers) == 2
    assert [len(t[0]) for t in f.transfers] == [1, 1]
    assert f.state.remote_fill.disabled_reason == ""


@pytest.mark.parametrize("end", [2048, 1024 + 17])
def test_aligned_and_tail_only_final_calls_keep_single_data_submission(end):
    f = fixture()
    f.invoke(end=end)
    assert len(f.transfers) == 1
    assert len(f.persistent) == 1


@pytest.mark.parametrize("tail_behavior", ["unsupported", "exists_error"])
def test_tail_fallback_preserves_persistent_completion(tail_behavior):
    f = fixture(tail_behavior=tail_behavior)
    f.invoke()
    assert f.state.committed_end == {0: 2116, 1: 2116}
    assert f.state.remote_fill.queued_windows == f.state.remote_fill.queued_bytes == 0
    assert f.coordinator._queued_bytes == 0
    assert not any(p.valid_tokens < 1024 for t in f.transfers for p in t[0])


def test_existing_persistent_tail_still_participates_in_remote_fill():
    f = fixture(tail_behavior="existing")
    f.invoke()
    assert len(f.persistent) == 1
    assert [p.valid_tokens for p in f.transfers[0][0]] == [1024, 68]


@pytest.mark.parametrize(
    "bound", ["remote_fill_max_inflight_bytes", "remote_fill_max_bytes_per_request"]
)
def test_oversized_combination_keeps_separate_bounded_admission(bound):
    f = fixture(job_limit=4)
    setattr(f.config, bound, 93192192)
    f.invoke()
    assert f.admissions[0][1] <= 93192192
    assert f.state.remote_fill.disabled_reason == "producer_backpressure"
    assert f.state.committed_end == {0: 2116, 1: 2116}


def test_combined_admission_respects_other_requests_retained_bytes():
    f = fixture()
    other = ProducerRequestState()
    retained = f.config.remote_fill_max_inflight_bytes - 95000000
    assert f.coordinator._acquire_queue_capacity(other, retained)
    f.invoke()
    assert f.state.remote_fill.disabled_reason == "producer_backpressure"
    assert f.state.remote_fill.queued_bytes == 0
    assert f.coordinator._queued_bytes == retained
    f.coordinator._release_queue_capacity(other, retained)
    assert f.coordinator._queued_bytes == 0


@pytest.mark.parametrize("groups", [(0,), (0, 1)])
def test_combined_job_keeps_each_control_window_bounded(groups):
    f = fixture(groups=groups)
    f.invoke(end=1024 + 4096 + 68)
    assert f.admissions[0][0] == 2
    assert len(f.transfers) == 2
    assert [len(t[0]) for t in f.transfers] == [4 * len(groups), len(groups)]
    assert all(p.valid_tokens == 68 for p in f.transfers[-1][0])
    assert f.state.remote_fill.disabled_reason == ""


def test_full_and_tail_fences_are_both_retained():
    f = fixture()
    tail_event = object()
    calls = []

    def ready_events(state, token_end):
        calls.append(token_end)
        return f.fences if len(calls) == 1 else (f.fences[0], tail_event)

    f.engine._remote_fill_source_ready_events = ready_events
    f.invoke()
    assert f.transfers[0][1] == (*f.fences, tail_event)


def test_absent_complete_fence_never_enqueues_remote_fill():
    f = fixture()
    f.invoke(events=())
    assert not f.transfers
    assert len(f.persistent) == 2
    assert f.state.remote_fill.disabled_reason == "incomplete_producer_fence"


def test_remote_fill_disabled_preserves_persistent_full_and_tail():
    f = fixture(remote=False)
    f.invoke()
    assert not f.transfers
    assert len(f.persistent) == 2
    assert f.state.committed_end == {0: 2116, 1: 2116}


def test_final_deferred_mode_keeps_existing_submission_policy():
    f = fixture(mode="final_deferred")
    f.invoke()
    assert f.state.remote_fill.disabled_reason == "producer_backpressure"
    assert f.admissions[0][:2] == (2, 93192192)
    assert len(f.persistent) == 2


def test_new_final_batch_can_follow_a_pending_earlier_chunk():
    f = fixture()
    f.invoke(end=2048, final=False)
    probe, done = f.jobs.pop(0)
    done.set_result(probe())
    assert f.state.remote_fill.queued_windows == 1
    f.invoke(end=3140)
    assert f.admissions[0][0] == 2
    assert [len(t[0]) for t in f.transfers] == [1, 2]
    assert f.state.remote_fill.disabled_reason == ""


def test_native_failure_does_not_lose_persistent_completion():
    f = fixture()

    def fail(**kwargs):
        raise RuntimeError("transfer failed before arm")

    f.state.remote_fill.session.transfer_window = fail
    f.invoke()
    assert f.state.remote_fill.disabled_reason == "RuntimeError"
    assert f.state.remote_fill.queued_windows == f.state.remote_fill.queued_bytes == 0
    assert f.state.committed_end == {0: 2116, 1: 2116}


def test_repeated_finalization_does_not_resubmit_the_combined_batch():
    f = fixture()
    f.invoke()
    f.invoke()
    assert len(f.persistent) == 2
    assert len(f.transfers) == 1


def test_tail_disabled_keeps_the_existing_full_page_submission():
    f = fixture()
    f.config.save_unfull_chunk = False
    f.invoke()
    assert len(f.persistent) == 1
    assert [p.valid_tokens for p in f.transfers[0][0]] == [1024]


def test_fatal_combined_transfer_propagates_without_finalizing_request():
    f = fixture()
    f.engine._remote_fill_fatal_transfers = ()
    f.engine.mark_init_failed = lambda reason: None

    def fail(**kwargs):
        raise RemoteFillFatalError("uncertain armed transfer")

    f.state.remote_fill.session.transfer_window = fail
    with pytest.raises(RemoteFillFatalError, match="uncertain armed transfer"):
        f.invoke()
    assert f.engine._remote_fill_fatal_transfers == ("t",)
    assert not f.state.finalized
    assert f.engine._pending_store_reqs == {}


@pytest.mark.parametrize("job_limit,prefix", [(4, 1024), (2, 0)])
def test_spare_job_capacity_keeps_full_transfer_ahead_of_tail_planning(
    job_limit, prefix
):
    f = fixture(job_limit=job_limit, prefix=prefix)
    planner = f.engine.gpu_connector.plan_direct_page_sources
    observed = []

    def check_order(caches, slots, starts, ends, group, **kwargs):
        if ends[0] - starts[0] < 1024:
            observed.append(f.state.remote_fill.queued_windows)
        return planner(caches, slots, starts, ends, group, **kwargs)

    f.engine.gpu_connector.plan_direct_page_sources = check_order
    f.invoke()
    assert observed and observed[0] == int(prefix > 0) + 1
    assert f.state.remote_fill.disabled_reason == ""


@pytest.mark.parametrize("pressure", ["batch", "request", "global"])
def test_streaming_keeps_success_when_combined_bytes_would_not_fit(pressure):
    # One full page fits, full+tail do not; the full transfer can finish while
    # the caller is preparing the tail. Preserve this previously valid order.
    f = fixture()
    other = ProducerRequestState()
    retained = 0
    if pressure == "batch":
        f.config.remote_fill_max_inflight_bytes = 95000000
    elif pressure == "request":
        f.config.remote_fill_max_bytes_per_request = 95000000
    else:
        retained = f.config.remote_fill_max_inflight_bytes - 95000000
        assert f.coordinator._acquire_queue_capacity(other, retained)
    planner = f.engine.gpu_connector.plan_direct_page_sources
    drained = False

    def progress_full(caches, slots, starts, ends, group, **kwargs):
        nonlocal drained
        if not drained and ends[0] - starts[0] < 1024:
            drained = True
            f.drain(("r",))
        return planner(caches, slots, starts, ends, group, **kwargs)

    f.engine.gpu_connector.plan_direct_page_sources = progress_full
    f.invoke()
    assert f.state.remote_fill.disabled_reason == ""
    assert [len(t[0]) for t in f.transfers] == [1, 1]
    assert f.coordinator._queued_bytes == retained
    if retained:
        f.coordinator._release_queue_capacity(other, retained)


def test_coalescing_layout_failure_keeps_existing_persistent_fallback():
    f = fixture()

    def invalid_layout():
        raise ValueError("invalid remote layout")

    f.engine._remote_fill_immutable_layout = invalid_layout
    # The existing admission wrapper catches this same layout failure when
    # building control pages. Early coalescing inspection must do so as well.
    f.engine._remote_fill_control_pages = lambda batch: invalid_layout()
    f.invoke()
    assert f.state.committed_end == {0: 2116, 1: 2116}
    assert f.state.remote_fill.disabled_reason == "ValueError"
    assert not f.transfers


@pytest.mark.parametrize("groups", [(0,), (0, 1)])
def test_coalesced_sources_preserve_fragmented_payload_bytes(groups):
    # Small synthetic two-layer payloads; actual pointers/lengths are read to
    # catch dropped/reordered fragments without allocating production NPU pages.
    widths = {0: 32, 1: 16}
    f = fixture(groups=groups, layers=2, payload_widths=widths)
    expected, persisted, received = {}, {}, {}

    def planner(caches, slots, starts, ends, group, **kwargs):
        owner = Owner()
        owner.buffers = []
        f.owner_refs.append((group, weakref.ref(owner)))
        ptrs, lengths = [], []
        for start, end in zip(starts, ends, strict=True):
            key = Key(group, start, end - start).to_string()
            payload = bytes(
                (group * 37 + start + i) % 251
                for i in range((end - start) * widths[group])
            )
            expected[key] = payload
            cuts = (0, len(payload) // 3, 2 * len(payload) // 3, len(payload))
            parts = [payload[a:b] for a, b in zip(cuts, cuts[1:])]
            buffers = [ctypes.create_string_buffer(part) for part in parts]
            owner.buffers.extend(buffers)
            ptrs.append([ctypes.addressof(buffer) for buffer in buffers])
            lengths.append([len(part) for part in parts])
        return ptrs, lengths, (owner,)

    persist = f.engine._track_direct_batch

    def persist_bytes(batch, retry):
        assert batch.ready_events == f.fences
        for key, ptrs, lengths in zip(batch.keys, batch.ptrs, batch.sizes, strict=True):
            persisted[key.to_string()] = b"".join(
                ctypes.string_at(p, n) for p, n in zip(ptrs, lengths, strict=True)
            )
        return persist(batch, retry)

    def transfer_bytes(**kwargs):
        plan = kwargs["source_plan"]
        assert plan.owners and plan.producer_events == f.fences
        for page in plan.pages:
            assert page.canonical_key not in received
            received[page.canonical_key] = b"".join(
                ctypes.string_at(p, n)
                for p, n in zip(page.source_ptrs, page.source_lengths, strict=True)
            )
        return RemoteFillWindowResult(kwargs["window_id"], True, True)

    f.engine.gpu_connector.plan_direct_page_sources = planner
    f.engine._track_direct_batch = persist_bytes
    f.state.remote_fill.session.transfer_window = transfer_bytes
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        f.invoke(end=1024 + 4096 + 68)
        assert persisted == expected
        assert received == {
            k: v for k, v in expected.items() if int(k.split(":")[0]) in groups
        }
        assert f.state.remote_fill.disabled_reason == ""
        assert all(ref() is None for _, ref in f.owner_refs)
    finally:
        if was_enabled:
            gc.enable()


def test_coalescing_hint_never_reserves_or_bypasses_live_admission():
    f = fixture()
    s = f.state.remote_fill
    assert f.coordinator._acquire_queue_capacity(s, 0)  # A pending probe.
    before = (s.queued_windows, s.queued_bytes, f.coordinator._queued_bytes)
    assert f.coordinator.can_coalesce_final_batch(s, 99380736)
    assert before == (s.queued_windows, s.queued_bytes, f.coordinator._queued_bytes)
    other = ProducerRequestState()
    assert f.coordinator._acquire_queue_capacity(
        other, f.config.remote_fill_max_inflight_bytes
    )
    assert not f.coordinator._acquire_queue_capacity(s, 99380736)
    f.coordinator._release_queue_capacity(
        other, f.config.remote_fill_max_inflight_bytes
    )
    f.coordinator._release_queue_capacity(s, 0)
    assert f.coordinator._queued_bytes == 0


def test_cancel_during_tail_planning_does_not_leave_unsubmitted_source_owners():
    f = fixture()
    planner = f.engine.gpu_connector.plan_direct_page_sources

    def cancel_tail(caches, slots, starts, ends, group, **kwargs):
        if ends[0] - starts[0] < 1024:
            raise CancelledError("cancelled while preparing tail")
        return planner(caches, slots, starts, ends, group, **kwargs)

    f.engine.gpu_connector.plan_direct_page_sources = cancel_tail
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            f.invoke()
        except CancelledError:
            pass
        else:
            pytest.fail("cancellation was swallowed")
        assert not f.state.finalized
        assert len(f.persistent) == 1
        assert f.state.remote_fill.queued_windows == 1  # Only the existing probe.
        f.drain(("r",))
        assert not f.transfers
        assert (
            f.state.remote_fill.queued_windows == f.state.remote_fill.queued_bytes == 0
        )
        assert all(ref() is None for _, ref in f.owner_refs)
    finally:
        if was_enabled:
            gc.enable()
