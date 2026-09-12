# SPDX-License-Identifier: Apache-2.0
"""Expected local loss must never authorize uninitialized KV or fail the request."""

import ast
from concurrent.futures import Future
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest

from test_local_checkpoint_restore import ROOT, implementation
from test_preemption_checkpoint import api as checkpoint_api, start_capture, publish

api = checkpoint_api
REMOTE, PREEMPTED = "remote", "preempted"


def scheduler_adapter(api):
    control, _ = api
    calls = []
    cls = implementation(
        "../LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py",
        "LMCacheConnectorV1Impl",
        {
            "accept_preemption_result",
            "_complete_checkpoint_restore_miss",
            "update_connector_output",
            "_clear_request_marker",
            "_discard_request_set",
            "get_num_new_matched_tokens",
        },
        object,
        serving_perf_enabled=lambda: False,
        logger=NS(debug=lambda *a: None),
        _lmcache_nvtx_annotate=lambda fn: fn,
        LOCAL_CHECKPOINT_CONFIG=control.LOCAL_CHECKPOINT_CONFIG,
        RequestStatus=NS(PREEMPTED=PREEMPTED),
        extract_mm_features=lambda request: ([], []),
        extract_request_configs=lambda params: None,
    )
    adapter = cls()
    request = NS(
        request_id="r",
        num_preemptions=1,
        num_tokens=12,
        num_prompt_tokens=3,
        all_token_ids=list(range(12)),
        prompt_token_ids=[0, 1, 2],
        sampling_params=None,
        status=REMOTE,
        num_computed_tokens=11,
        num_external_computed_tokens=11,
        num_cached_tokens=3,
        bootstrap_sample_pending=False,
        dsa_compact_allocated=False,
        kv_resume_checkpoint=(1, 12, 11),
    )
    request.is_finished = lambda: request.status == "finished"
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 0, 13, 3, ((1,), (2,)), prefix_end=3), "ready", 11
    )
    adapter._preemption_checkpoints = {"r": pending}
    adapter._checkpoint_restore_attempts = {"r": (1, 7)}
    adapter._unfinished_requests = {"r": request}
    adapter._dsa_cold_indexer_block_ids = {"r": {101, 102}}
    adapter._lmcache_chunk_size = 4
    adapter.load_specs = {"r": NS(lmcache_cached_tokens=11, dsa_cold_compact_load=True)}
    adapter._resume_lookup_queries = {"r": (11, "preemption_checkpoint", 0)}
    adapter.lookup_client = NS(
        clear_lookup_status=lambda rid: calls.append(("clear", rid))
    )
    adapter._arm_preemption_controls = lambda: calls.append(("release",))
    adapter.kv_role = "kv_both"
    adapter._request_trackers = {}
    adapter._requests_priority = {}
    adapter.config = NS(blocking_timeout_secs=30)
    adapter.skip_last_n_tokens = 0
    return adapter, request, pending, calls


def receipt(ids=()):
    return NS(
        finished_recving=set(ids),
        invalid_block_ids=set(),
        completed_decode_window_saves={},
    )


def promote(request, retry=True):
    statuses = NS(WAITING_FOR_REMOTE_KVS=REMOTE, PREEMPTED=PREEMPTED, WAITING="waiting")
    base = implementation(
        "../vllm/vllm/v1/core/sched/scheduler.py",
        "Scheduler",
        {"_update_waiting_for_remote_kv", "_try_promote_blocked_waiting_request"},
        object,
        RequestStatus=statuses,
        SERVING_PERF_ENABLED=False,
        logger=NS(warning=lambda *a: None),
    )
    cls = implementation(
        "../vllm-ascend/vllm_ascend/core/recompute_scheduler.py",
        "RecomputeScheduler",
        {"_update_waiting_for_remote_kv"},
        base,
    )
    scheduler = cls()
    scheduler.connector = object()
    scheduler.recompute_kv_load_failures = False
    scheduler.finished_recving_kv_req_ids = {"r"}
    scheduler.failed_recving_kv_req_ids = set()
    calls = []
    scheduler.kv_cache_manager = NS(
        free=lambda r: calls.append("free"),
        cache_blocks=lambda *a: calls.append("cache"),
    )
    assert scheduler._try_promote_blocked_waiting_request(request)
    assert calls == (["free"] if retry else ["cache"])
    assert (
        not scheduler.finished_recving_kv_req_ids
        and not scheduler.failed_recving_kv_req_ids
    )
    assert request.status == PREEMPTED
    if retry:
        assert (
            request.num_computed_tokens == 0
            and request.num_external_computed_tokens == 0
        )
        assert request.num_cached_tokens == 0
        assert not request.bootstrap_sample_pending and request.bootstrap_final_hidden is None
        assert not request.dsa_compact_allocated


@pytest.mark.parametrize("available", [0, 3, 7, 8])
def test_miss_retries_only_after_all_workers_finish_and_preserves_history(
    api, available
):
    adapter, request, pending, calls = scheduler_adapter(api)
    before = list(request.all_token_ids)
    result = api[0].CheckpointResult(
        "r", 1, "restore_miss", available, "evicted", load_generation=7
    )
    adapter.accept_preemption_result(result)
    adapter.update_connector_output(receipt())
    assert pending.end == 11 and not hasattr(request, "kv_resume_checkpoint_retry")
    assert request.num_computed_tokens == 11 and not calls
    adapter.update_connector_output(receipt({"r"}))
    assert request.kv_resume_checkpoint is None
    assert request.kv_resume_checkpoint_retry == (1, 7)
    assert "r" not in adapter._dsa_cold_loaded_req_ids
    assert adapter._checkpoint_restore_releases == [("r", 1, 7)]
    assert pending.restore_retries == 1
    assert pending.status == ("ready" if available > 3 else "failed")
    if available > 3:
        assert pending.end == available
    request.bootstrap_sample_pending = True
    request.bootstrap_final_hidden = {"stale": "hidden"}
    request.dsa_compact_allocated = True
    promote(request)
    looked_up = []
    adapter.lookup_client.lookup_cache = lambda **kw: -1
    adapter.lookup_client.lookup = lambda ids, **kw: looked_up.append(list(ids))
    assert adapter.get_num_new_matched_tokens(request, 0) is None
    assert looked_up == [before[: available if available > 3 else 3]]
    assert request.all_token_ids == before


def test_repeated_eviction_exhausts_local_retry_then_uses_prompt(api):
    adapter, request, pending, _ = scheduler_adapter(api)
    for generation, offered, available in [(7, 11, 8), (8, 8, 7)]:
        adapter._checkpoint_restore_attempts = {"r": (1, generation)}
        adapter._dsa_cold_indexer_block_ids = {"r": {101}}
        adapter.load_specs["r"].lmcache_cached_tokens = offered
        request.status, request.num_computed_tokens = REMOTE, offered
        adapter.accept_preemption_result(
            api[0].CheckpointResult(
                "r", 1, "restore_miss", available, load_generation=generation
            )
        )
        adapter.update_connector_output(receipt({"r"}))
        promote(request)
    assert pending.status == "failed" and pending.restore_retries == 2
    assert request.all_token_ids == list(range(12))


@pytest.mark.parametrize(
    "generation,load_generation,finished", [(2, 7, False), (1, 6, False), (1, 7, True)]
)
def test_stale_or_cancelled_miss_cannot_arm_retry(
    api, generation, load_generation, finished
):
    adapter, request, pending, _ = scheduler_adapter(api)
    if finished:
        request.status = "finished"
    adapter.accept_preemption_result(
        api[0].CheckpointResult(
            "r", generation, "restore_miss", 8, load_generation=load_generation
        )
    )
    assert pending.restore_miss is None
    assert not hasattr(request, "kv_resume_checkpoint_retry")


def test_local_eviction_reports_longest_paired_frontier_before_device_work(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 11)
    engine.backend.evict(("local", "r", 1, 1, 8, 12))
    assert store.local.available("r", 1, 11) == 8
    before = list(engine.calls)
    with pytest.raises(api[0].CheckpointRestoreMiss) as caught:
        store.local.normalize("r", 1, list(range(11)), None)
    assert caught.value.available_end == 8 and engine.calls == before
    assert all(page.refs == 1 for page in engine.backend.pages.values())
    store.close()


@pytest.mark.parametrize("kind", ["miss", "bad_data"])
def test_only_expected_pretransfer_miss_avoids_invalid_blocks_and_stack(api, kind):
    control = api[0]
    calls = []
    logger = NS(
        info=lambda *a: calls.append("info"),
        exception=lambda *a: calls.append("exception"),
    )
    base = implementation(
        "../LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py",
        "LMCacheConnectorV1Impl",
        {"_fail_completed_cold_load", "_record_checkpoint_restore_miss"},
        object,
        logger=logger,
        _clear_terminal_load_tracebacks=lambda *a: calls.append("clear"),
    )
    cls = implementation(
        "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        {"_record_checkpoint_restore_miss"},
        base,
        CheckpointRestoreMiss=control.CheckpointRestoreMiss,
        CheckpointResult=control.CheckpointResult,
    )
    adapter = cls()
    adapter.lmcache_engine = NS(checkpoint_worker=NS(results=[]))
    adapter._synchronize_dsa_cold_dense_load = lambda: calls.append("fence")
    adapter._invalid_block_ids = set()
    adapter._release_request_lookup_pins = lambda rid: calls.append("unpin")
    request = NS(
        req_id="r", load_spec=NS(checkpoint_generation=1, dsa_group1_direct_hbm=False)
    )
    error = (
        control.CheckpointRestoreMiss(8, "evicted")
        if kind == "miss"
        else ValueError("bad layout")
    )
    entry = (7, Future(), request, [101, 102], 0, Future())
    assert adapter._fail_completed_cold_load("r", entry, None, error, False)
    assert "unpin" in calls
    if kind == "miss":
        assert "fence" not in calls  # No restore device work was submitted.
        assert not adapter._invalid_block_ids and "exception" not in calls
        result = adapter.lmcache_engine.checkpoint_worker.results[0]
        assert (
            result.status,
            result.generation,
            result.load_generation,
            result.end,
        ) == ("restore_miss", 1, 7, 8)
    else:
        assert calls[0] == "fence"
        assert adapter._invalid_block_ids == {101, 102}
        assert (
            "exception" in calls
            and not adapter.lmcache_engine.checkpoint_worker.results
        )


def test_all_tp_ranks_receive_the_same_typed_pretransfer_miss(api):
    control = api[0]
    ln = ROOT.parent / "LMCache-NPU"
    nodes = []
    for file, names in [
        (ln / "lmcache/v1/shared_cpu_cache.py", {"SharedHandleEnvelope"}),
        (
            ln / "lmcache/v1/cache_engine.py",
            {"_shared_layerwise_error_envelope", "_validate_shared_layerwise_envelope"},
        ),
    ]:
        nodes.extend(
            n
            for n in ast.walk(ast.parse(file.read_text(encoding="utf-8")))
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        )
    ns = dict(dataclass=dataclass)
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, *nodes], type_ignores=[])
            ),
            "envelope",
            "exec",
        ),
        ns,
    )
    base = type(
        "Base",
        (),
        {
            name: ns[name]
            for name in (
                "_shared_layerwise_error_envelope",
                "_validate_shared_layerwise_envelope",
            )
        },
    )
    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"prepare_checkpoint_restore"},
        base,
        CheckpointRestoreMiss=control.CheckpointRestoreMiss,
    )
    wire = []
    request = NS(
        req_id="r",
        load_spec=NS(checkpoint_generation=1),
        token_ids=list(range(11)),
        request_configs=None,
    )

    def normalize(*args):
        raise control.CheckpointRestoreMiss(8, "evicted")

    for rank in range(4):
        engine = cls()
        engine._is_passive = lambda rank=rank: rank != 0
        engine.shared_cpu_cache_generation, engine.shared_cpu_cache_strict = 1, True
        engine.checkpoint_worker = NS(local=NS(normalize=normalize))
        engine._broadcast_shared_envelope = wire.append
        engine._receive_matching_shared_envelope = lambda **kw: wire[0]
        with pytest.raises(control.CheckpointRestoreMiss) as caught:
            engine.prepare_checkpoint_restore(request)
        assert caught.value.available_end == 8
    assert len(wire) == 1 and wire[0].status == "skipped" and not wire[0].handles
    engine._receive_matching_shared_envelope = lambda **kw: replace(
        wire[0], request_id="stale"
    )
    with pytest.raises(ValueError, match="Invalid shared"):
        engine.prepare_checkpoint_restore(request)


def test_worker_metadata_can_arrive_before_all_worker_completion(api):
    cls = implementation(
        "../vllm/vllm/distributed/kv_transfer/kv_connector/utils.py",
        "KVOutputAggregator",
        {"__init__", "aggregate"},
        object,
        KVConnectorOutput=NS,
    )
    aggregator = cls(4)
    adapter, request, pending, _ = scheduler_adapter(api)
    result = api[0].CheckpointResult("r", 1, "restore_miss", 8, load_generation=7)

    def output(done=False, metadata=None):
        return NS(
            kv_connector_output=NS(
                finished_recving={"r"} if done else None,
                finished_sending=None,
                kv_connector_stats=None,
                kv_connector_worker_meta=metadata,
                kv_cache_events=None,
                invalid_block_ids=set(),
                completed_decode_window_saves={},
                expected_finished_count=0,
            )
        )

    first = aggregator.aggregate(
        [output(True, result), output(), output(), output()]
    ).kv_connector_output
    assert first.finished_recving is None
    adapter.accept_preemption_result(first.kv_connector_worker_meta)
    adapter.update_connector_output(first)
    assert not hasattr(request, "kv_resume_checkpoint_retry") and pending.end == 11
    last = aggregator.aggregate(
        [output(), output(True), output(True), output(True)]
    ).kv_connector_output
    assert last.finished_recving == {"r"} and last.kv_connector_worker_meta is None
    adapter.update_connector_output(last)
    assert pending.end == 8
    promote(request)


def test_cancel_after_miss_still_releases_owners_without_claiming_ready(api):
    adapter, request, pending, _ = scheduler_adapter(api)
    adapter.accept_preemption_result(
        api[0].CheckpointResult("r", 1, "restore_miss", 8, load_generation=7)
    )
    request.status = "finished"
    adapter.update_connector_output(receipt({"r"}))
    assert "r" not in adapter._dsa_cold_loaded_req_ids
    assert not hasattr(request, "kv_resume_checkpoint_retry")
    assert adapter._checkpoint_restore_releases == [("r", 1, 7)]
    assert pending.restore_miss is None


def test_successful_shorter_restore_becomes_ready_without_replaying_old_miss(api):
    adapter, request, pending, _ = scheduler_adapter(api)
    adapter.accept_preemption_result(
        api[0].CheckpointResult("r", 1, "restore_miss", 8, load_generation=7)
    )
    adapter.update_connector_output(receipt({"r"}))
    promote(request)
    adapter._checkpoint_restore_attempts = {"r": (1, 8)}
    adapter._dsa_cold_indexer_block_ids = {"r": {201, 202}}
    adapter.load_specs["r"].lmcache_cached_tokens = 8
    (
        request.status,
        request.num_computed_tokens,
        request.num_external_computed_tokens,
    ) = REMOTE, 8, 8
    adapter.update_connector_output(receipt({"r"}))
    assert adapter._dsa_cold_loaded_req_ids == {"r"}
    assert not hasattr(request, "kv_resume_checkpoint_retry")
    promote(request, retry=False)
    assert request.num_computed_tokens == 8
    assert request.all_token_ids == list(range(12)) and pending.restore_retries == 1
