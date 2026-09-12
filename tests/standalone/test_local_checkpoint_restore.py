# SPDX-License-Identifier: Apache-2.0
"""Exercise production mixed-source dispatch and local-only lookup contracts."""

import ast
from dataclasses import replace
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_preemption_checkpoint import api as checkpoint_api, start_capture, publish

api = checkpoint_api

ROOT = Path(__file__).resolve().parents[2]


def implementation(path, class_name, names, base, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = dict(namespace, Base=base, torch=torch, replace=replace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[class_name]


def lookup_engine(store, engine, control, calls):
    class Base:
        def _lookup_remote_fill_two_group_prefix(self, chunks, **kw):
            calls.append((chunks, kw))
            return chunks[-1][1] if chunks else 0

    cls = implementation(
        "lmcache_ascend/v1/cache_engine.py",
        "AscendLMCacheEngine",
        {"_lookup_remote_fill_two_group_prefix"},
        Base,
        LOCAL_CHECKPOINT_CONFIG=control.LOCAL_CHECKPOINT_CONFIG,
    )
    obj = cls()
    obj.checkpoint_worker, obj.token_database = store, engine.token_database
    return obj


def test_lookup_proves_original_prefix_and_probes_local_tail_without_waiting_pins(
    api, monkeypatch
):
    control, _ = api
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    calls = []
    obj = lookup_engine(store, engine, control, calls)
    chunks = list(engine.token_database.process_tokens(tokens=list(range(12))))
    assert (
        obj._lookup_remote_fill_two_group_prefix(
            chunks,
            search_range=["RemoteBackend"],
            lookup_id="r",
            pin=True,
            request_configs={control.LOCAL_CHECKPOINT_CONFIG: 1},
        )
        == 12
    )
    assert calls[0][0][-1][1] == 3
    assert calls[0][1]["pin"] is False and calls[0][1]["lookup_id"] is None
    assert all(p.refs <= 1 for p in engine.allocated)
    engine.backend.evict(("local", "r", 1, 0, 8, 12))
    assert (
        obj._lookup_remote_fill_two_group_prefix(
            chunks,
            search_range=["RemoteBackend"],
            lookup_id="r",
            pin=True,
            request_configs={control.LOCAL_CHECKPOINT_CONFIG: 1},
        )
        == 8
    )
    store.close()


def test_lookup_does_not_accept_another_history_or_generation(api, monkeypatch):
    control, _ = api
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    obj = lookup_engine(store, engine, control, [])
    chunks = list(engine.token_database.process_tokens(tokens=[99] * 12))
    for generation in (1, 2):
        assert (
            obj._lookup_remote_fill_two_group_prefix(
                chunks,
                search_range=["RemoteBackend"],
                lookup_id="r",
                pin=True,
                request_configs={control.LOCAL_CHECKPOINT_CONFIG: generation},
            )
            == 0
        )
    store.close()


def test_ordinary_lookup_keeps_original_arguments():
    calls = []
    obj = lookup_engine(
        None, NS(token_database=None), NS(LOCAL_CHECKPOINT_CONFIG="local"), calls
    )
    chunks = [(0, 8, "key")]
    obj._lookup_remote_fill_two_group_prefix(
        chunks,
        search_range=["RemoteBackend"],
        lookup_id="ordinary",
        pin=True,
        request_configs={"original": 1},
    )
    assert calls == [
        (
            chunks,
            dict(
                search_range=["RemoteBackend"],
                lookup_id="ordinary",
                pin=True,
                request_configs={"original": 1},
                diagnostics=None,
            ),
        )
    ]


def adapter(calls):
    class Base:
        def _run_dsa_cold_indexer_load(self, plan, device):
            calls.append(("cpu_tail", plan["token_count"], plan["token_mask"].tolist()))
            return plan["token_mask"], "completed-cpu-fence", 0.1, 0.02

        def _run_dsa_cold_compact_load(self, plan, device, indexer, previous, live):
            calls.append(("ordinary_latent", previous))
            return "state"

        def _submit_dsa_cold_compact_load(self, request):
            calls.append(("submit", request.load_spec.dsa_group1_direct_hbm))

    cls = implementation(
        "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        "LMCacheAscendConnectorV1Impl",
        {
            "_run_dsa_cold_indexer_load",
            "_run_dsa_cold_compact_load",
            "_submit_dsa_cold_compact_load",
        },
        Base,
    )
    obj = cls()
    obj._lmcache_chunk_size = 4
    obj.lmcache_engine = NS(
        load_group1_pages_direct=lambda *a: calls.append(
            ("persistent_prefix", len(a[0]))
        )
    )
    return obj


def plan():
    gate = Future()
    gate.set_result(None)
    return dict(
        request=NS(
            req_id="r",
            request_configs=None,
            token_ids=list(range(11)),
            load_spec=NS(
                checkpoint_generation=1,
                checkpoint_prefix_end=7,
                dsa_group1_direct_hbm=False,
            ),
        ),
        latent_shared_ready=gate,
        token_mask=torch.ones(11, dtype=torch.bool),
        token_count=11,
        tokens=list(range(11)),
        indexer_slots_cpu=torch.arange(11),
        indexer_kvcaches=[object()],
    )


def test_mixed_indexer_only_loads_original_prefix_from_mooncake_and_tail_from_cpu():
    calls = []
    obj, p = adapter(calls), plan()
    result = obj._run_dsa_cold_indexer_load(p, None)
    assert calls == [
        ("persistent_prefix", 4),
        ("cpu_tail", 7, [False] * 4 + [True] * 7),
    ]
    assert result[0].all() and p["token_mask"].all() and p["token_count"] == 11
    assert result[1] == "completed-cpu-fence"


def test_failed_latent_admission_unblocks_indexer_without_any_io():
    calls = []
    obj, p = adapter(calls), plan()
    p["latent_shared_ready"] = Future()
    obj.lmcache_engine.prepare_checkpoint_restore = lambda request: (
        _ for _ in ()
    ).throw(ValueError("evicted"))
    with pytest.raises(ValueError, match="evicted"):
        obj._run_dsa_cold_compact_load(p, None, Future())
    with pytest.raises(ValueError, match="evicted"):
        obj._run_dsa_cold_indexer_load(p, None)
    assert calls == []


def test_normalized_owners_outlive_the_whole_paired_load_and_previous_cancel_does_not_poison_it():
    calls = []
    obj, p = adapter(calls), plan()
    owner = NS(ref_count_down=lambda: calls.append(("release",)))
    obj.lmcache_engine.prepare_checkpoint_restore = lambda req: (4, [owner])
    previous = Future()
    previous.cancel()
    assert obj._run_dsa_cold_compact_load(p, None, Future(), previous) == "state"
    assert calls == [("ordinary_latent", None), ("release",)]


def test_local_h2d_keeps_existing_runtime_graph_capture_safety_guard():
    calls = []
    obj, p = adapter(calls), plan()
    p["request"].load_spec.dsa_group1_direct_hbm = True
    obj._submit_dsa_cold_compact_load(p["request"])
    assert calls == [("submit", False)]


def test_normalization_does_not_leave_unevictable_duplicate_cache_aliases(
    api, monkeypatch
):
    store, _, engine = start_capture(api, monkeypatch)
    store.poll()
    publish(api, store, 12)
    _, owners = store.local.normalize("r", 1, list(range(12)), None)
    for page in owners:
        page.ref_count_down()
    assert len({id(p) for p in engine.backend.pages.values()}) == len(
        engine.backend.pages
    )
    assert all(p.refs == 1 for p in engine.backend.pages.values())
    assert store.local.available("r", 1, 12) == 12
    store.close()


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_control_acknowledgement_passes_real_shared_envelope_validation(
    fail,
):
    from dataclasses import dataclass

    ln = ROOT.parent / "LMCache-NPU"
    nodes = []
    for file, names in [
        (ln / "lmcache/v1/shared_cpu_cache.py", {"SharedHandleEnvelope"}),
        (
            ln / "lmcache/v1/cache_engine.py",
            {"_shared_layerwise_error_envelope", "_validate_shared_layerwise_envelope"},
        ),
    ]:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        nodes.extend(
            n
            for n in ast.walk(tree)
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
            "envelopes",
            "exec",
        ),
        ns,
    )
    Base = type(
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
        Base,
        NativeExternalPageTransferUnknownError=type("Unknown", (RuntimeError,), {}),
    )
    wire = []
    leader, passive = cls(), cls()
    for obj, role in [(leader, False), (passive, True)]:
        obj._is_passive = lambda role=role: role
        obj.shared_cpu_cache_generation = 1
        obj.shared_cpu_cache_strict = True

    def normalize(*args):
        if fail:
            raise ValueError("evicted")
        return 4, []

    leader.checkpoint_worker = NS(local=NS(normalize=normalize))
    leader._broadcast_shared_envelope = lambda envelope: wire.append(envelope)
    passive._receive_matching_shared_envelope = lambda **kw: wire[0]
    request = plan()["request"]
    if fail:
        with pytest.raises(RuntimeError, match="unavailable"):
            leader.prepare_checkpoint_restore(request)
        with pytest.raises(ValueError, match="rank0 error"):
            passive.prepare_checkpoint_restore(request)
    else:
        assert leader.prepare_checkpoint_restore(request) == (4, [])
        assert passive.prepare_checkpoint_restore(request) == (4, [])
        assert wire[0].status == "skipped" and not wire[0].handles


@pytest.mark.parametrize("resumed,ready", [(False, False), (True, False), (True, True)])
def test_index_tail_ownership_retires_only_at_the_completed_resume_branch(
    resumed, ready
):
    file = ROOT.parent / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(file.read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "request.resumed_from_preemption"
        and "owned_groups" in ast.unparse(n)
    )
    calls = []

    def completed(request):
        assert resumed
        calls.append("check")
        return ready

    scope = dict(
        self=NS(
            _completed_cold_resume=completed,
            lmcache_engine=NS(
                register_shared_cpu_sparse_request=lambda *a, **kw: calls.append(kw)
            ),
        ),
        request=NS(
            resumed_from_preemption=resumed,
            req_id="r",
            load_spec=NS(checkpoint_generation=1),
        ),
        resumed_req_ids=set(),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
            str(file),
            "exec",
        ),
        scope,
    )
    assert calls == (
        []
        if not resumed
        else ["check"] + ([{"owned_groups": {1: []}}] if ready else [])
    )
