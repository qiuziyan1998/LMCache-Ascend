# SPDX-License-Identifier: Apache-2.0
"""An incomplete banked prefix must not be published alongside a healthy one."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_only_complete_requests_reach_remote_fill_and_final_store():
    # The adapter requires the serving SDK; exercise its real method on CPU.
    path = Path(__file__).parents[2] / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    method = next(
        node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "_finish_layerwise_prefill_batch"
    )
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    requests = [
        SimpleNamespace(req_id=name, token_ids=[1], is_last_prefill=True,
                        is_sparse_decode=False, request_configs={},
                        incomplete_prefill_load=failed)
        for name, failed in (("retry", True), ("ready", False))
    ]
    engine = Mock()
    adapter = SimpleNamespace(
        kv_role="kv_producer", lmcache_engine=engine, _completed_layerwise_stores={},
        _parent=SimpleNamespace(_get_connector_metadata=lambda: SimpleNamespace(requests=requests)),
    )
    scope[method.name](adapter)
    args, kwargs = engine.submit_layerwise_prefill_fills.call_args
    assert list(args[0]) == ["ready"]
    assert [entry[0] for entry in kwargs["request_metadata"]] == ["ready"]
    engine.poll_layerwise_prefill_puts.assert_called_once_with(final=True, req_ids=("ready",))
    engine.finish_layerwise_prefill_store.assert_called_once_with(
        "ready", {}, required_store_end=1, persistence_fenced=True, tokens=[1]
    )
