# SPDX-License-Identifier: Apache-2.0
"""Check the isolated pre-SFA fence without loading the NPU runtime."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prefill_sfa_fence", ROOT / "lmcache_ascend/v1/kv_fence_debug.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_debug(mode="p_sfa", role="P", **env):
    records = []
    debug = MODULE.KVFenceDebug(
        role, 1, {"LMCACHE_PREFILL_FENCE_DEBUG": mode, **env}, records.append
    )
    return debug, records


def test_fence_finishes_input_work_but_does_not_wait_for_future_sfa():
    queue = ["load", "query", "scatter KV", "indexer"]
    completed = []

    def synchronize():
        completed.extend(queue)
        queue.clear()

    debug, records = make_debug()
    debug.before_sfa(0, 7, 3, lambda: NS(synchronize=synchronize))
    queue.append("SFA")
    assert completed == ["load", "query", "scatter KV", "indexer"]
    assert queue == ["SFA"]
    assert records[-1]["event"] == "active"
    assert records[-1]["stage"] == "p_sfa"
    assert records[-1]["layer"] == 7
    debug.before_sfa(0, 8, 0, lambda: NS(synchronize=synchronize))
    assert len(records) == 2  # Startup + first activation only.


@pytest.mark.parametrize("mode", ["p_sfa", "p_use", "p_load", "check", "all"])
@pytest.mark.parametrize("role", ["P", "D"])
@pytest.mark.parametrize("group,layer", [(0, 3), (0, 4), (0, 5), (1, 4)])
def test_filters_before_touching_the_npu(mode, role, group, layer):
    touched = []
    debug, _ = make_debug(mode, role, LMCACHE_PREFILL_FENCE_LAYERS="4:5")
    debug.before_sfa(
        group, layer, 0, lambda: NS(synchronize=lambda: touched.append(True))
    )
    assert bool(touched) == (
        mode == "p_sfa" and role == "P" and group == 0 and layer == 4
    )


@pytest.mark.parametrize(
    "extra", ["p_load", "p_use", "p_reuse", "p_submit", "p_save", "d_load"]
)
def test_rejects_combined_device_fences(extra):
    with pytest.raises(ValueError, match="Use p_sfa alone"):
        make_debug("p_sfa," + extra)


def test_check_can_be_combined_and_group_zero_is_required():
    debug, _ = make_debug("p_sfa,check")
    assert debug.check_protocol
    with pytest.raises(ValueError, match="requires latent KV group 0"):
        make_debug(LMCACHE_PREFILL_FENCE_GROUPS="1")
    debug, _ = make_debug("all")
    assert "p_sfa" not in debug.stages


def test_device_failure_is_reported_and_propagated():
    def synchronize():
        raise RuntimeError("injected stream failure")

    debug, records = make_debug()
    with pytest.raises(RuntimeError, match="device_fence_failed"):
        debug.before_sfa(0, 2, 2, lambda: NS(synchronize=synchronize))
    assert records[-1]["stage"] == "p_sfa"
    assert "injected stream failure" in records[-1]["detail"]


@pytest.mark.parametrize("enabled", [False, True])
def test_connector_does_not_replace_bank_use_or_load_events(enabled):
    tree = ast.parse(
        (ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py").read_text(
            encoding="utf-8"
        )
    )
    cls = next(
        n
        for n in tree.body
        if getattr(n, "name", None) == "VLLMPagedMemLayerwiseNPUConnector"
    )
    method = next(
        n
        for n in cls.body
        if getattr(n, "name", None) == "synchronize_prefill_sfa_inputs"
    )
    calls = []
    namespace = {
        "torch": NS(
            npu=NS(current_stream=lambda: NS(synchronize=lambda: calls.append("wait")))
        )
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "<sfa-fence>",
            "exec",
        ),
        namespace,
    )
    original = object()
    connector = NS(
        fence_debug=make_debug()[0] if enabled else None,
        _layerwise_prefill_bank=lambda layer, group: layer % 4,
        _layerwise_prefill_bank_use_events={(0, 2): original},
        _layerwise_prefill_load_done_events={(0, 6): original},
    )
    namespace[method.name](connector, 6, 0)
    assert calls == (["wait"] if enabled else [])
    assert connector._layerwise_prefill_bank_use_events == {(0, 2): original}
    assert connector._layerwise_prefill_load_done_events == {(0, 6): original}
