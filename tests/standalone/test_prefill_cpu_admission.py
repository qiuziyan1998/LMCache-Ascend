# SPDX-License-Identifier: Apache-2.0
"""Run actual connector admission/retirement methods with CPU-only workers."""

import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
LM = ROOT.parent / "LMCache"
spec = importlib.util.spec_from_file_location(
    "admission_connector_tested", LM / "lmcache/v1/prefill_admission.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def implementation():
    class Base:
        def get_num_new_matched_tokens(self, request, computed):
            self.lookups.append(request.request_id)
            return 499712

        def has_pending_control(self):
            return False

        def _release_finished_worker_requests(self, ids):
            self.events.append(("release", tuple(ids)))

        def _drain_dense_load_retirements(self):
            self.events.append("poll-retirements")

    methods = {
        "get_num_new_matched_tokens",
        "has_pending_control",
        "_build_prefill_admission_report",
        "_release_finished_worker_requests",
        "_forget_layerwise_store_results",
    }
    path = ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in (
            "LiveSourceWorkerMetadata",
            "CheckpointWorkerMetadata",
        ):
            nodes.append(node)
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "LMCacheAscendConnectorV1Impl"
        ):
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in methods
            ]
            nodes.append(node)
    ns = dict(
        LMCacheConnectorV1Impl=Base,
        KVConnectorWorkerMetadata=object,
        dataclass=dataclass,
        field=field,
        time=time,
        logger=NS(info=lambda *a: None),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    obj = ns["LMCacheAscendConnectorV1Impl"]()
    obj.lookups, obj.events = [], []
    obj._prefill_admission = module.PrefillAdmissionBudget(4)
    obj._prefill_admission_enabled = True
    obj._force_layerwise_prefill_store = True
    obj._prefill_admission_wait_logged = set()
    obj._prefill_admission_retired = set()
    obj._prefill_admission_last_report = None
    obj._prefill_admission_report_time = 0.0
    obj._vllm_config = NS(scheduler_config=NS(max_num_batched_tokens=16384))
    obj.config = NS(enable_shared_cpu_cache=True, get_extra_config_value=lambda k, d: d)
    pool = dict(capacity=100 * 1024**3, available=100 * 1024**3)
    obj.lmcache_engine = NS(
        metadata=NS(
            worker_id=0,
            is_first_rank=lambda: True,
            chunk_size=1024,
            runtime_kv_group_layer_counts=(79, 22),
            get_shapes=lambda: [torch.Size([93192192 // 2]), torch.Size([3702784])],
            get_dtypes=lambda: [torch.bfloat16, torch.uint8],
        ),
        storage_manager=NS(
            local_cpu_backend=NS(
                get_prefill_admission_capacity=lambda: dict(pool),
                get_memory_allocator=lambda: NS(align_bytes=512),
            )
        ),
        poll_layerwise_prefill_puts=lambda **kw: obj.events.append("drain-stores"),
    )
    return obj, pool, ns


def request(name, tokens=500000):
    return NS(request_id=name, num_tokens=tokens)


def initialize_reports(obj):
    rank0 = obj._build_prefill_admission_report()[0]
    obj._prefill_admission.update(
        tuple(
            dict(rank0, rank=r, pool=rank0["pool"] if r == 0 else None)
            for r in range(4)
        )
    )


def test_no_lookup_or_pinning_before_allocator_bootstrap_or_when_third_does_not_fit():
    obj, _, _ = implementation()
    assert obj.has_pending_control()
    assert obj.get_num_new_matched_tokens(request("a"), 0) is None
    assert not obj.lookups
    initialize_reports(obj)
    assert obj.get_num_new_matched_tokens(request("a"), 0) == 499712
    assert obj.get_num_new_matched_tokens(request("b"), 0) == 499712
    assert obj.get_num_new_matched_tokens(request("c"), 0) is None
    assert obj.lookups == ["a", "b"]
    # Three independently admitted full prompts would exceed the real 100GiB pool.
    assert sum(obj._prefill_admission.reservations.values()) < 100 * 1024**3


def test_oversized_request_marks_error_without_executing_lookup():
    obj, pool, _ = implementation()
    pool.update(capacity=1024**3, available=1024**3)
    initialize_reports(obj)
    req = request("oversized")
    assert obj.get_num_new_matched_tokens(req, 0) is None
    assert "LMCache CPU capacity" in req.kv_admission_error
    assert not obj.lookups


def test_only_waiting_requests_keep_capacity_reports_running():
    obj, pool, _ = implementation()
    pool["available"] = 0
    initialize_reports(obj)
    assert obj.get_num_new_matched_tokens(request("waiter"), 0) is None
    assert obj.has_pending_control()
    assert not obj.lookups and not obj._prefill_admission.reservations
    pool["available"] = pool["capacity"]
    obj._prefill_admission_report_time -= 1
    initialize_reports(obj)
    assert obj.get_num_new_matched_tokens(request("waiter"), 0) == 499712
    assert not obj.has_pending_control()


def test_abort_before_worker_execution_can_retire_reserved_credit():
    obj, _, _ = implementation()
    initialize_reports(obj)
    assert obj.get_num_new_matched_tokens(request("aborted"), 0) == 499712
    obj._prefill_admission.finish("aborted")
    assert obj.has_pending_control()
    obj._release_finished_worker_requests(iter(["aborted"]))
    rank0 = obj._build_prefill_admission_report()[0]
    assert rank0["retired"] == ("aborted",)
    obj._prefill_admission.update(tuple(dict(rank0, rank=r) for r in range(4)))
    assert not obj._prefill_admission.reservations
    assert not obj.has_pending_control()


def test_retirement_waits_for_store_cleanup_and_deferred_dense_owners():
    obj, _, _ = implementation()
    obj._release_finished_worker_requests(["a"])
    assert obj.events == ["drain-stores", ("release", ("a",))]
    obj._cold_load_coordinator = NS(retirements={1: NS(req_id="a")})
    assert obj._build_prefill_admission_report()[0]["retired"] == ()
    obj._cold_load_coordinator.retirements.clear()
    assert obj._build_prefill_admission_report()[0]["retired"] == ("a",)
    assert obj._build_prefill_admission_report()[0]["retired"] == ()


def test_shared_cpu_followers_do_not_charge_their_tiny_local_allocator():
    obj, _, _ = implementation()
    obj.lmcache_engine.metadata.worker_id = 3
    obj.lmcache_engine.metadata.is_first_rank = lambda: False
    obj.lmcache_engine.storage_manager.local_cpu_backend = None
    assert obj._build_prefill_admission_report()[0]["pool"] is None


def test_decoder_skips_accounting_and_reports():
    obj, _, _ = implementation()
    obj._prefill_admission = None
    obj._prefill_admission_enabled = False
    assert obj._build_prefill_admission_report() == ()
    assert not obj.has_pending_control()
    assert obj.get_num_new_matched_tokens(request("d"), 0) == 499712


def test_tp_aggregation_keeps_cpu_reports_and_checkpoint_positional_arguments():
    _, _, ns = implementation()
    plain = ns["LiveSourceWorkerMetadata"]({}, {}, prefill_admission=({"rank": 0},))
    checkpoint = ns["CheckpointWorkerMetadata"](
        {}, {}, ("ready",), prefill_admission=({"rank": 1},)
    )
    for merged in (plain.aggregate(checkpoint), checkpoint.aggregate(plain)):
        assert merged.checkpoint_results == ("ready",)
        assert {r["rank"] for r in merged.prefill_admission} == {0, 1}


def test_capacity_metadata_does_not_change_model_or_bank_code():
    # The report builder works with no NPU stream/event/tensor methods available.
    obj, _, _ = implementation()
    report = obj._build_prefill_admission_report()[0]
    assert report["page_bytes"] == 93192192 + 3702784 + (79 + 22) * 512


def test_oversize_error_ends_only_the_request_in_real_scheduler_method():
    path = ROOT.parent / "vllm/vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_reject_kv_admission_error"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = dict(
        logger=NS(warning=lambda *a: None),
        RequestStatus=NS(FINISHED_ERROR="error"),
        EngineCoreOutput=NS,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    finishes = []
    sched = NS(finish_requests=lambda *args: finishes.append(args))
    req = request("oversized")
    assert not ns[fn.name](sched, req)
    req.kv_admission_error, req.client_index = "not enough CPU capacity", 0
    req.get_finished_reason = lambda: "error"
    assert ns[fn.name](sched, req)
    assert finishes == [("oversized", "error")]
    assert sched._dsa_admission_errors[0][1].stop_reason == req.kv_admission_error


@pytest.mark.parametrize(
    "path",
    [
        "vllm/vllm/v1/core/sched/scheduler.py",
        "vllm-ascend/vllm_ascend/core/recompute_scheduler.py",
    ],
)
def test_both_schedulers_handle_permanent_refusal_before_deferral(path):
    tree = ast.parse((ROOT.parent / path).read_text(encoding="utf-8"))
    gates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "ext_tokens is None"
    ]
    assert gates
    assert "_reject_kv_admission_error" in ast.unparse(gates[0].body[0])
