# SPDX-License-Identifier: Apache-2.0
"""Execute the save boundary without model/device/package initialization."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import torch
import pytest


def boundary(store):
    source = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/dcp_save.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if getattr(n, "name", "") == "save_batch"
    )
    ns = dict(Any=Any, torch=torch, store_group=store)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    return ns["save_batch"]


def adapter(requests):
    log = []
    value = NS(
        kv_role="kv_both",
        lmcache_engine=NS(),
        device="cpu",
        _worker_retrieve_state={},
        _parent=NS(_get_connector_metadata=lambda: NS(requests=requests)),
        _direct_group_caches=lambda: {0: "latent", 1: "indexer"},
        _note_decode_window_save_seen=lambda r: None,
        _layerwise_save_range=lambda r: (4, 8),
        _layerwise_store_kwargs=lambda r, g: {},
        _is_decode_window_save_request=lambda r: True,
        _consume_completed_layerwise_store=lambda r, g, ok, result: log.append(
            ("consume", g, result)
        ),
        _mark_prefill_committed=lambda r: log.append("commit"),
        _mark_initial_sparse_release_ready=lambda r: None,
        _maybe_lookup_unpin_for_request=lambda r: None,
    )
    return value, log


def test_warm_decode_boundary_adds_no_store_collective_or_mapping_work():
    def unexpected(*args, **kwargs):
        raise AssertionError("no work is permitted for a step with no saves")

    owner, log = adapter([NS(save_spec=None), NS(save_spec=NS(can_save=False))])
    owner._direct_group_caches = owner._layerwise_store_kwargs = unexpected
    boundary(unexpected)(owner, {})
    assert log == []


def test_save_preserves_both_groups_and_window_relative_indexer_mapping():
    request = NS(
        req_id="r",
        token_ids=list(range(8)),
        dcp_write_layout=object(),
        save_spec=NS(can_save=True, can_save_latent=True, can_save_indexer=True),
        save_indexer_slot_mapping=[torch.arange(104, 108)],
        save_slot_mapping_base=4,
    )
    owner, log = adapter([request])

    def store(engine, req, group, caches, mask, options, *, input_error):
        assert input_error is None
        assert mask.tolist() == [False] * 4 + [True] * 4
        if group:
            assert options["slot_mapping_base"] == 4
            assert options["slot_mapping"].tolist() == list(range(104, 108))
        return group

    context = {}
    boundary(store)(owner, context)
    assert log == [("consume", 0, 0), ("consume", 1, 1), "commit"]
    assert context["decode_window_saves"] == [request]


def test_rank_local_input_failure_is_forwarded_to_shared_preflight():
    request = NS(
        req_id="r",
        token_ids=list(range(8)),
        dcp_write_layout=None,
        save_spec=NS(can_save=True, can_save_latent=True, can_save_indexer=True),
    )
    owner, _ = adapter([request])
    errors = []

    def store(*args, input_error, **kwargs):
        errors.append(input_error)

    boundary(store)(owner, {})
    assert len(errors) == 2 and all("snapshot" in e for e in errors)


def test_unresolved_save_cannot_acknowledge_the_final_fence():
    source = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    node = next(
        n
        for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "_finish_save_batch"
    )
    ns = dict(Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    owner = NS(
        _dsa_dcp_size=2, lmcache_engine=NS(_process_store_writes={"original": object()})
    )
    with pytest.raises(RuntimeError, match="ownership remains unresolved"):
        ns["_finish_save_batch"](owner, {})
    owner.lmcache_engine._process_store_writes.clear()
    ns["_finish_save_batch"](owner, {})
