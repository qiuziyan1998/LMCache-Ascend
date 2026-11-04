# SPDX-License-Identifier: Apache-2.0
"""V-up fence selection and connector routing without an NPU runtime."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "kv_fence_vup_test", _ROOT / "lmcache_ascend/v1/kv_fence_debug.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
KVFenceDebug = _MODULE.KVFenceDebug


def make_debug(mode="p_vup", role="P", **env):
    records = []
    debug = KVFenceDebug(
        role, 2, {"LMCACHE_PREFILL_FENCE_DEBUG": mode, **env}, records.append
    )
    return debug, records


@pytest.mark.parametrize("mode", ["p_vup", "check,p_vup", "p_vup,check"])
def test_vup_accepts_only_its_fence_and_optional_host_checks(mode):
    debug, records = make_debug(mode)
    assert debug.stages == {"p_vup"}
    assert debug.check_protocol == ("check" in mode)
    assert records[0]["stages"] == ["p_vup"]


@pytest.mark.parametrize(
    "other", ["p_load", "p_sfa", "p_use", "p_reuse", "p_submit", "p_save", "d_load"]
)
def test_vup_rejects_other_device_fences(other):
    with pytest.raises(ValueError):
        make_debug("check,p_vup," + other)


def test_all_keeps_the_existing_p_use_control():
    debug, _ = make_debug("all")
    assert debug.stages == {"p_load", "p_use", "p_save", "d_load"}


def test_vup_rejects_indexer_only_selection():
    with pytest.raises(ValueError, match="p_vup requires latent KV group 0"):
        make_debug(LMCACHE_PREFILL_FENCE_GROUPS="1")


@pytest.mark.parametrize(
    "mode,role,group,layer",
    [
        ("p_vup", "D", 0, 4),
        ("p_vup", "P", 1, 4),
        ("p_vup", "P", 0, 3),
        ("p_vup", "P", 0, 6),
        ("p_use", "P", 0, 4),
        ("check", "P", 0, 4),
    ],
)
def test_filtering_never_resolves_or_waits_on_a_stream(mode, role, group, layer):
    debug, records = make_debug(
        mode,
        role,
        LMCACHE_PREFILL_FENCE_GROUPS="0,1",
        LMCACHE_PREFILL_FENCE_LAYERS="4:6",
    )

    def forbidden():
        pytest.fail("An unselected fence must not resolve the NPU stream")

    debug.after_vup(group, layer, 0, forbidden)
    assert [record["event"] for record in records] == ["enabled"]


def test_each_selected_vup_waits_but_activation_is_logged_once():
    debug, records = make_debug(LMCACHE_PREFILL_FENCE_LAYERS="4:6")
    actions = []

    class Stream:
        def synchronize(self):
            actions.append("wait")

        def query(self):
            pytest.fail("V-up fence must not query the stream")

    def current_stream():
        actions.append("resolve")
        return Stream()

    debug.after_vup(0, 4, 0, current_stream)
    debug.after_vup(0, 5, 1, current_stream)
    assert actions == ["resolve", "wait", "resolve", "wait"]
    active = [record for record in records if record["event"] == "active"]
    assert len(active) == 1
    assert {key: active[0][key] for key in ("stage", "group", "layer", "bank")} == {
        "stage": "p_vup",
        "group": 0,
        "layer": 4,
        "bank": 0,
    }
    assert debug._reuse_fences == {}


def test_wait_failure_uses_the_existing_fence_error_report():
    debug, records = make_debug()

    class BrokenStream:
        def synchronize(self):
            raise RuntimeError("V-up did not finish")

    with pytest.raises(RuntimeError, match="device_fence_failed"):
        debug.after_vup(0, 7, 3, BrokenStream)
    error = records[-1]
    assert error["event"] == "error"
    assert error["stage"] == "p_vup"
    assert error["group"] == 0 and error["layer"] == 7 and error["bank"] == 3
    assert error["detail"] == "V-up did not finish"


@pytest.fixture
def connector_method():
    """Execute the real connector method without importing its NPU dependencies."""
    source = _ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    connector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "synchronize_prefill_vup_output"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[method.name], namespace


def test_connector_disabled_does_not_resolve_bank_or_torch(connector_method):
    method, _ = connector_method
    # No torch global or bank resolver is installed: touching either fails.
    method(SimpleNamespace(), 7, 0)
    method(SimpleNamespace(fence_debug=None), 7, 0)


def test_connector_routes_the_current_stream_callback_and_selected_bank(
    connector_method,
):
    method, namespace = connector_method
    debug, records = make_debug()
    actions = []
    stream = SimpleNamespace(synchronize=lambda: actions.append("wait"))
    namespace["torch"] = SimpleNamespace(
        npu=SimpleNamespace(current_stream=lambda: stream)
    )

    def bank(layer, group):
        actions.append(("bank", layer, group))
        return 3

    original_event = object()
    connector = SimpleNamespace(
        fence_debug=debug,
        _layerwise_prefill_bank=bank,
        existing_bank_event=original_event,
    )
    method(connector, 7, 0)
    assert actions == [("bank", 7, 0), "wait"]
    assert records[-1]["stage"] == "p_vup"
    assert records[-1]["bank"] == 3
    assert connector.existing_bank_event is original_event
