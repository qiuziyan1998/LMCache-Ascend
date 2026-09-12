# SPDX-License-Identifier: Apache-2.0
"""Exercise production checkpoint dispatch without starting the NPU runtime."""

import ast
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
LN = ROOT.parent / "LMCache-NPU"
VA = ROOT.parent / "vllm-ascend"


def declarations(path, names, namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    for n in nodes:
        wanted = names[n.name]
        if wanted is not None:
            n.body = [
                child for child in n.body if getattr(child, "name", None) in wanted
            ] or [ast.Pass()]
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )


def fixture():
    calls = []

    class BaseImpl:
        def start_load_kv(self, ctx, **kw):
            calls.append("ordinary-load")

        def build_connector_worker_meta(self):
            calls.append("ordinary-metadata")
            return None

    class BaseAPI:
        def bind_connector_metadata(self, metadata):
            self.metadata = metadata

        def clear_connector_metadata(self):
            self.metadata = None

        def has_connector_metadata(self):
            return self.metadata is not None

        def _get_connector_metadata(self):
            return self.metadata

    class SupportsHMA:
        pass

    class Engine:
        def get_finished_stores(self, ids):
            calls.append(("ordinary-finish", ids))
            return set()

    ns = dict(
        LMCacheConnectorV1Impl=BaseImpl,
        KVConnectorBase_V1=BaseAPI,
        SupportsHMA=SupportsHMA,
        MultiConnector=BaseAPI,
        KVConnectorWorkerMetadata=object,
        dataclass=dataclass,
        field=field,
        logger=NS(debug=lambda *a: None, info=lambda *a: None),
    )
    declarations(
        ROOT / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
        {
            "LMCacheAscendConnectorV1Impl": {
                "handle_preemptions",
                "_activate_checkpoint_io",
                "_checkpoint_start_load",
                "_checkpoint_finished_stores",
                "_checkpoint_worker_meta",
            },
            "LiveSourceWorkerMetadata": None,
            "CheckpointWorkerMetadata": None,
        },
        ns,
    )
    declarations(
        LN / "lmcache/integration/vllm/lmcache_connector_v1.py",
        {
            "LMCacheConnectorV1Dynamic": {
                "supports_preemption_checkpoint",
                "handle_preemptions",
                "handle_preemptions_with_metadata",
                "prepare_preemption_checkpoint",
            }
        },
        ns,
    )
    # Use the deployed subclass declaration, preserving its real base class.
    declarations(
        ROOT / "lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py",
        {"LMCacheAscendConnectorV1Dynamic": {"supports_dsa_compact_load"}},
        ns,
    )
    declarations(
        VA / "vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        {
            "AscendMultiConnector": {
                "supports_preemption_checkpoint",
                "handle_preemptions_with_metadata",
                "prepare_preemption_checkpoint",
            }
        },
        ns,
    )
    engine = Engine()
    replies = []
    worker = NS(
        jobs={},
        restore_owners={},
        poll=lambda: tuple(replies),
        seal=lambda seal: calls.append(("seal", seal)),
        cancel=lambda *ids: calls.append(("cancel", ids)),
    )

    def capture(spec, caches, block_size, prefix_state=None):
        assert "drop-state" not in calls
        calls.append("capture")
        worker.jobs[(spec.req_id, spec.generation)] = object()

    worker.capture = capture
    engine.checkpoint_worker = worker
    engine.enable_checkpoint_prefix_agreement = lambda: None
    engine.wait_for_pending_stores = lambda ids: calls.append("drain")
    engine.wait_for_direct_stores = lambda ids: calls.append("direct-drain")
    engine.drop_direct_store_states = lambda ids: calls.append("drop-store")
    impl = ns["LMCacheAscendConnectorV1Impl"]()
    impl.lmcache_engine = engine
    impl.config = NS(decode_preemption_checkpoint=True)
    impl.store_async, impl.kv_role = True, "kv_both"
    impl._unfenced_live_stores = {}
    impl._block_size = 16
    impl._worker_retrieve_state = {}
    impl._direct_group_caches = lambda: {0: [object()], 1: [object()]}
    impl._drop_worker_retrieve_state = lambda req: calls.append("drop-state")
    dynamic = ns["LMCacheAscendConnectorV1Dynamic"]()
    dynamic._lmcache_engine, dynamic.metadata = impl, None
    impl._parent = dynamic
    multi = ns["AscendMultiConnector"]()
    multi._connectors = [dynamic]

    class MultiMetadata:
        def __init__(self, metadata):
            self.metadata = [metadata]

    ns["MultiKVConnectorMetadata"] = MultiMetadata
    return impl, dynamic, multi, MultiMetadata, calls, replies, ns


def test_actual_ascend_dynamic_mro_enters_capture_and_restores_idle_dispatch():
    impl, dynamic, multi, MultiMetadata, calls, replies, ns = fixture()
    assert type(dynamic).__mro__[1].__name__ == "LMCacheConnectorV1Dynamic"
    assert multi.supports_preemption_checkpoint
    worker = impl.lmcache_engine.checkpoint_worker
    worker.poll = lambda: pytest.fail("checkpoint polling in ordinary decode")
    for _ in range(20):
        impl.start_load_kv(None)
        assert impl.build_connector_worker_meta() is None
        impl.lmcache_engine.get_finished_stores(set())
    assert len(calls) == 60
    calls.clear()
    worker.poll = lambda: tuple(replies)
    capture = NS(req_id="r", generation=1)
    metadata = NS(
        preemption_captures=(capture,), preemption_seals=(), preemption_cancels=()
    )
    multi.handle_preemptions_with_metadata({"r"}, MultiMetadata(metadata))
    assert calls[:4] == ["drain", "direct-drain", "capture", "drop-state"]
    assert dynamic.metadata is None
    assert "start_load_kv" in impl.__dict__
    replies.append(NS(req_id="r", generation=1, status="captured"))
    assert impl.build_connector_worker_meta().checkpoint_results == tuple(replies)
    dynamic.metadata = NS(
        preemption_seals=("accepted-history",),
        preemption_cancels=(),
        preemption_releases=(),
    )
    impl.start_load_kv(None)
    assert calls[-2:] == [("seal", "accepted-history"), "ordinary-load"]
    impl.lmcache_engine.get_finished_stores({"r"})
    assert calls[-2:] == [("cancel", ("r",)), ("ordinary-finish", {"r"})]
    worker.jobs.clear()
    impl.build_connector_worker_meta()
    assert "start_load_kv" not in impl.__dict__
    assert "build_connector_worker_meta" not in impl.__dict__
    assert "get_finished_stores" not in impl.lmcache_engine.__dict__
    worker.poll = lambda: pytest.fail("checkpoint polling after retirement")
    assert impl.build_connector_worker_meta() is None


def test_checkpoint_results_survive_both_tp_aggregation_orders():
    *_, ns = fixture()
    plain = ns["LiveSourceWorkerMetadata"]({"a": [1]}, {})
    result = ns["CheckpointWorkerMetadata"]({}, {}, ("checkpoint-ready",))
    for merged in (plain.aggregate(result), result.aggregate(plain)):
        assert merged.descriptors == {"a": [1]}
        assert merged.checkpoint_results == ("checkpoint-ready",)
    assert [f.name for f in fields(plain)] == ["descriptors", "remote_fill_results"]


def test_release_only_control_frame_retires_restore_owners_and_restores_idle_methods():
    impl, dynamic, _, _, calls, _, _ = fixture()
    worker = impl.lmcache_engine.checkpoint_worker
    key = ("r", 1, 7)
    worker.restore_owners[key] = [object()]
    worker.release_restore = lambda *ids: (
        calls.append(("release", ids)),
        worker.restore_owners.pop(ids, None),
    )
    impl._activate_checkpoint_io()
    impl.build_connector_worker_meta()
    assert "start_load_kv" in impl.__dict__
    dynamic.metadata = NS(
        preemption_cancels=(),
        preemption_seals=(),
        preemption_releases=(key,),
        requests=[],
    )
    impl.start_load_kv(None)
    assert calls[-2:] == [("release", key), "ordinary-load"]
    impl.build_connector_worker_meta()
    assert "start_load_kv" not in impl.__dict__
