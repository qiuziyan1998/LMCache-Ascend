# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in Group-1 NPU content fingerprints."""

# Standard
from collections.abc import Iterator
import sys
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1 import content_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def reset_content_diagnostics() -> Iterator[None]:
    diagnostics.configure_npu_content_diagnostics(False)
    yield
    diagnostics.configure_npu_content_diagnostics(False)


def _layout(
    owners: list[torch.Tensor], physical_starts: tuple[int, int]
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    layers = [
        {
            "layer_id": layer_id,
            "buffer_base": int(owner.data_ptr()),
            "token_bytes": int(owner[0, 0].numel() * owner.element_size()),
            "slot_capacity": int(owner.shape[0] * owner.shape[1]),
        }
        for layer_id, owner in enumerate(owners)
    ]
    runs = [
        {
            "logical_token_start": 0,
            "physical_slot_start": physical_starts[0],
            "token_count": 2,
        },
        {
            "logical_token_start": 2,
            "physical_slot_start": physical_starts[1],
            "token_count": 2,
        },
    ]
    return layers, runs


def _owners(physical_starts: tuple[int, int]) -> list[torch.Tensor]:
    owners = [torch.zeros((2, 4, 1, 2)) for _ in range(3)]
    slots = (
        physical_starts[0],
        physical_starts[0] + 1,
        physical_starts[1],
        physical_starts[1] + 1,
    )
    for layer_id, owner in enumerate(owners):
        flat = owner.reshape(8, 1, 2)
        for logical_token, physical_slot in enumerate(slots):
            flat[physical_slot].fill_(layer_id * 100 + logical_token)
    return owners


def test_content_fingerprint_is_strictly_disabled_by_default() -> None:
    owners = _owners((0, 4))
    layers, runs = _layout(owners, (0, 4))

    assert (
        diagnostics.fingerprint_compact_group1(
            event="source",
            req_id="request",
            owners=owners,
            layers=layers,
            runs=runs,
            token_count=4,
            chunk_size=2,
            tp_rank=0,
            dp_rank=0,
        )
        is None
    )


def test_metadata_only_event_uses_same_diagnostic_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )

    diagnostics.log_npu_content_diagnostic_event("ready", req_id="request")
    assert events == []

    diagnostics.configure_npu_content_diagnostics(True)
    diagnostics.log_npu_content_diagnostic_event(
        "ready", req_id="request", ready=False
    )
    assert events[-1] == ("ready", {"req_id": "request", "ready": False})


def test_group1_pointer_table_reports_per_page_expected_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)

    class Page:
        def __init__(self, base: int, group_prefix_sum: tuple[int, ...]) -> None:
            self.base = base
            self.group_prefix_sum = group_prefix_sum

        def layer_data_ptr(self, layer_id: int) -> int:
            return self.base + self.group_prefix_sum[layer_id]

    pages = [Page(1_000, (0, 16)), Page(2_000, (0, 8))]
    dev_ptr_rows = [[1_100, 2_100], [1_116, 2_108]]

    diagnostics.log_group1_pointer_table(
        req_id="request",
        phase="cold_bootstrap",
        kv_group=1,
        pages=pages,
        dev_ptr_rows=dev_ptr_rows,
        rank=0,
    )

    assert len(events) == 1
    event, fields = events[0]
    assert event == "group1_pointer_table"
    assert fields["table_consistent"] is True
    layer_report = fields["layer_reports"]["1"]
    assert layer_report["expected_deltas_preview"] == [16, 8]
    assert layer_report["deltas_preview"] == [16, 8]
    assert layer_report["delta_mismatch_chunks"] == []


def test_configure_installs_and_clears_vllm_callback_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[bool] = []

    class CallbackBundle:
        def __init__(self, **callbacks: object) -> None:
            self.callbacks = callbacks

    installed: list[CallbackBundle] = []

    fake_bridge = SimpleNamespace(
        NPUContentDiagnosticCallbacks=CallbackBundle,
        install_npu_content_diagnostic_callbacks=installed.append,
        clear_npu_content_diagnostic_callbacks=lambda: cleared.append(True),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.lmcache_diagnostics",
        fake_bridge,
    )

    diagnostics.configure_npu_content_diagnostics(True)
    assert len(installed) == 1
    callbacks = installed[0].callbacks
    assert (
        callbacks["begin_deferred_step"]
        is diagnostics.begin_deferred_diagnostic_step
    )
    assert callbacks["flush_deferred"] is diagnostics.flush_deferred_diagnostics
    assert (
        callbacks["fingerprint_compact_group1"]
        is diagnostics.fingerprint_compact_group1
    )
    assert (
        callbacks["register_group1_source"]
        is diagnostics.register_group1_source_fingerprint
    )
    assert (
        callbacks["queue_group1_first_consume"]
        is diagnostics.queue_group1_first_consume
    )
    assert (
        callbacks["queue_cache_tail"]
        is diagnostics.queue_cache_tail_fingerprint
    )
    assert (
        callbacks["queue_selected_topk"]
        is diagnostics.queue_selected_topk_fingerprint
    )
    assert (
        callbacks["queue_staged_graph_stage"]
        is diagnostics.queue_staged_graph_stage_fingerprint
    )

    diagnostics.configure_npu_content_diagnostics(False)
    assert cleared == [True]


def test_fingerprint_uses_logical_not_physical_slot_identity() -> None:
    diagnostics.configure_npu_content_diagnostics(True)
    source_owners = _owners((0, 4))
    destination_owners = _owners((2, 6))
    source_layers, source_runs = _layout(source_owners, (0, 4))
    destination_layers, destination_runs = _layout(destination_owners, (2, 6))

    source = diagnostics.fingerprint_compact_group1(
        event="source",
        req_id="request",
        owners=source_owners,
        layers=source_layers,
        runs=source_runs,
        token_count=4,
        chunk_size=2,
        tp_rank=0,
        dp_rank=0,
    )
    assert source is not None
    destination = diagnostics.fingerprint_compact_group1(
        event="destination",
        req_id="request",
        owners=destination_owners,
        layers=destination_layers,
        runs=destination_runs,
        token_count=4,
        chunk_size=2,
        tp_rank=0,
        dp_rank=0,
        sample_layer_ids=source["sample_layer_ids"],
        sample_logical_tokens=source["sample_logical_tokens"],
        expected=source,
    )

    assert destination is not None
    assert destination["combined_hash"] == source["combined_hash"]
    assert destination["row_hashes"] == source["row_hashes"]

    destination_owners[1].reshape(8, 1, 2)[6].add_(1)
    corrupted = diagnostics.fingerprint_compact_group1(
        event="destination",
        req_id="request",
        owners=destination_owners,
        layers=destination_layers,
        runs=destination_runs,
        token_count=4,
        chunk_size=2,
        tp_rank=0,
        dp_rank=0,
        sample_layer_ids=source["sample_layer_ids"],
        sample_logical_tokens=source["sample_logical_tokens"],
        expected=source,
    )
    assert corrupted is not None
    assert corrupted["combined_hash"] != source["combined_hash"]
    assert corrupted["row_hashes"]["1:2"] != source["row_hashes"]["1:2"]


def test_first_consume_and_topk_readback_are_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)
    owners = _owners((0, 4))
    layers, runs = _layout(owners, (0, 4))
    source = diagnostics.fingerprint_compact_group1(
        event="source",
        req_id="request",
        owners=owners,
        layers=layers,
        runs=runs,
        token_count=4,
        chunk_size=2,
        tp_rank=0,
        dp_rank=0,
        sample_layer_ids=(0,),
        sample_logical_tokens=(0, 1),
    )
    assert source is not None
    diagnostics.register_group1_source_fingerprint("request", source)
    diagnostics.begin_deferred_diagnostic_step()

    diagnostics.queue_group1_first_consume(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        indexer_cache=owners[0],
        indexer_block_table=torch.tensor([[0, 1]]),
        seq_lens_cpu=torch.tensor([3]),
        block_size=4,
        row_request_indices=torch.tensor([0, -1]),
        num_decode_tokens=1,
        num_actual_tokens=1,
        attn_state="DecodeOnly",
        decode_valid_rows_all=False,
        group1_connector_wait_called=True,
    )
    diagnostics.queue_selected_topk_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        topk_indices=torch.tensor([[7, 3, 1]]),
        row_request_indices=torch.tensor([0, -1]),
        seq_lens_cpu=torch.tensor([3]),
        num_decode_tokens=1,
        num_actual_tokens=1,
        attn_state="DecodeOnly",
        decode_valid_rows_all=False,
    )
    assert [event for event, _ in events] == [
        "content_diagnostics_enabled",
        "source",
    ]

    diagnostics.flush_deferred_diagnostics()

    by_event = {event: fields for event, fields in events}
    consume = by_event["group1_first_decode_consume"]
    assert consume["all_expected_rows_match"] is True
    assert consume["readback_mode"] == "deferred_after_model_forward"
    assert consume["capture_order"] == (
        "after_group1_connector_wait_call_before_current_token_scatter"
    )
    assert consume["group1_connector_wait_called"] is True
    assert consume["request_seq_len"] == 3
    assert consume["request_row_count"] == 1
    assert consume["row_owners"] == [0, -1]
    assert consume["valid_row_indices"] == [0]
    assert consume["padding_row_indices"] == [1]
    assert consume["invalid_row_indices"] == []
    assert consume["num_decode_tokens"] == 1
    assert consume["num_actual_tokens"] == 1
    assert consume["attn_state"] == "DecodeOnly"
    assert consume["decode_valid_rows_all"] is False
    topk = by_event["group1_selected_topk_fingerprint"]
    assert topk["value_preview"] == [7, 3, 1]
    assert topk["readback_mode"] == "deferred_after_model_forward"
    assert topk["capture_order"] == ("after_topk_selection_before_compact_remap")
    assert topk["request_seq_len"] == 3
    assert topk["row_owners"] == [0]
    assert topk["valid_row_indices"] == [0]
    assert topk["padding_row_indices"] == []
    assert topk["invalid_row_indices"] == []
    assert topk["topk_row_count"] == 1
    assert topk["row_owner_count"] == 2
    assert topk["unowned_topk_row_indices"] == []
    assert topk["decode_valid_rows_all"] is False
    summary = topk["row_summaries"][0]
    assert summary["row_index"] == 0
    assert summary["value_count"] == 3
    assert len(summary["content_hash"]) == 32
    assert summary["min"] == 1
    assert summary["max"] == 7
    assert summary["unique_count"] == 3
    assert summary["identity_prefix"] is False
    assert summary["out_of_range_count"] == 2


def test_cache_tail_fingerprint_reports_scatter_integrity_and_prefix_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)
    cache = torch.zeros((2, 4, 1, 2), dtype=torch.float32)
    cache.reshape(8, 1, 2)[5] = torch.tensor([[3.0, 4.0]])
    produced = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    diagnostics.queue_cache_tail_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.17.self_attn.indexer.k_cache",
        kv_group=1,
        stage="group1_after_current_scatter",
        cache_parts={"index": cache},
        produced_parts={"index": produced},
        slot_mapping=torch.tensor([2, 5]),
        query_start_loc_cpu=[0, 2],
        seq_lens_cpu=[6],
        num_actual_tokens=2,
        attn_state="DecodeOnly",
    )
    assert [event for event, _ in events] == [
        "content_diagnostics_enabled"
    ]

    diagnostics.flush_deferred_diagnostics()

    tail = next(
        fields
        for event, fields in events
        if event == "cache_tail_fingerprint"
    )
    assert tail["layer_id"] == 17
    assert tail["kv_group"] == 1
    assert tail["logical_tokens"] == [4]
    assert tail["physical_slots"] == [2]
    assert tail["query_length"] == 2
    assert tail["cached_prefix_tokens"] == 4
    assert tail["prefix_hit"] is True
    assert tail["selection_reason"] == (
        "first_query_row_at_cached_prefix_boundary"
    )
    assert tail["all_components_match_produced"] is False
    summary = tail["component_summaries"]["index"]
    assert summary["matches_produced"] is False
    assert summary["max_abs_delta"] == 2.0
    assert summary["differing_element_count"] == 2


def test_first_consume_reports_incomplete_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)

    diagnostics.queue_group1_first_consume(
        req_ids=["request-0", "request-1"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        indexer_cache=torch.zeros(2, 4, 1),
        indexer_block_table=torch.tensor([[0]]),
        seq_lens_cpu=torch.tensor([4]),
        block_size=4,
        row_request_indices=torch.tensor([0, 1]),
    )

    skipped = events[-1]
    assert skipped[0] == "content_diagnostic_snapshot_skipped"
    assert skipped[1]["requested_event"] == "group1_first_decode_consume"
    assert skipped[1]["reason"] == "incomplete_request_index_metadata"
    assert skipped[1]["request_count"] == 2
    assert skipped[1]["seq_len_count"] == 1
    assert skipped[1]["block_table_rows"] == 1


def test_attention_probe_observes_persistent_request_without_wire_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)

    diagnostics.queue_group1_first_consume(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        indexer_cache=torch.zeros((1, 4, 1, 2)),
        indexer_block_table=torch.tensor([[0]]),
        seq_lens_cpu=[3],
        block_size=4,
        row_request_indices=[0],
        num_hidden_layers=1,
    )
    diagnostics.queue_selected_topk_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        topk_indices=torch.tensor([[0]]),
        row_request_indices=[0],
        seq_lens_cpu=[3],
        num_hidden_layers=1,
    )

    diagnostics.flush_deferred_diagnostics()

    by_event = {event: fields for event, fields in events}
    consume = by_event["group1_first_decode_consume"]
    assert consume["logical_tokens"] == [0, 1]
    assert consume["source_fingerprint_registered"] is False
    assert consume["comparison_mode"] == "persistent_observed_only"
    assert consume["all_expected_rows_match"] is None
    topk = by_event["group1_selected_topk_fingerprint"]
    assert topk["source_fingerprint_registered"] is False
    assert topk["comparison_mode"] == "persistent_observed_only"


def test_staged_graph_stage_snapshot_is_deferred_and_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    diagnostics.queue_staged_graph_stage_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.attn",
        stage="after_graph_pre_before_retrieve",
        components={"row_req_indices": values},
        row_request_indices=[0, 0],
        seq_lens_cpu=[3],
        num_decode_tokens=2,
        num_actual_tokens=2,
        attn_state="SpecDecoding",
        num_hidden_layers=1,
        graph_key="spec-2",
    )
    values.fill_(9)
    assert [event for event, _ in events] == [
        "content_diagnostics_enabled"
    ]

    diagnostics.flush_deferred_diagnostics()

    event, fields = events[-1]
    assert event == "staged_sfa_graph_fingerprint"
    assert fields["stage"] == "after_graph_pre_before_retrieve"
    assert fields["row_owners"] == [0, 0]
    assert fields["graph_key"] == "spec-2"
    assert fields["device_snapshot_may_perturb_timing"] is True
    component = fields["component_summaries"]["row_req_indices"]
    assert component["value_preview"] == [1, 2, 3, 4]
    assert fields["readback_mode"] == "deferred_after_sampling_workflow"


def test_staged_graph_snapshot_compares_group0_local_cpu_source_after_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)
    # Two plane-stacked chunks with two tokens each: nope width=2, pe width=1.
    source_chunks = (
        torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.bfloat16),
        torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.bfloat16),
    )
    diagnostics.register_group0_source_probe(
        req_id="request",
        layer_id=0,
        num_layers=2,
        source_chunks=source_chunks,
        selected_tokens=torch.tensor([0, 2], dtype=torch.int32),
        selected_count=torch.tensor([2], dtype=torch.int32),
        chunk_size=2,
        total_tokens=4,
        token_major=False,
        layer_cache=(
            torch.zeros((4, 1, 2), dtype=torch.bfloat16),
            torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        ),
    )
    destination_nope = torch.tensor(
        [[[1, 2]], [[7, 8]]], dtype=torch.bfloat16
    )
    destination_pe = torch.tensor([[[5]], [[11]]], dtype=torch.bfloat16)
    diagnostics.queue_staged_graph_stage_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.attn",
        stage="after_selective_retrieve_before_graph_post",
        components={
            "retrieved_nope_sample": destination_nope,
            "retrieved_pe_sample": destination_pe,
        },
        row_request_indices=[0, 0],
        seq_lens_cpu=[4],
        num_decode_tokens=2,
        num_actual_tokens=2,
        attn_state="SpecDecoding",
        num_hidden_layers=2,
        graph_key="spec-2",
    )

    assert [event for event, _ in events] == [
        "content_diagnostics_enabled",
        "group0_source_probe_registered",
    ]
    registered = events[-1][1]
    assert registered["req_id"] == "request"
    assert registered["layer_id"] == 0
    assert registered["total_tokens"] == 4
    diagnostics.flush_deferred_diagnostics()

    fields = next(
        fields
        for event, fields in events
        if event == "group0_source_destination_fingerprint"
    )
    assert fields["all_components_match"] is True
    assert fields["source_details"][0]["selected_token_preview"] == [0, 2]
    source_summaries = fields["component_summaries"]
    assert source_summaries["retrieved_nope_sample"][
        "matches_destination"
    ] is True
    assert source_summaries["retrieved_pe_sample"][
        "matches_destination"
    ] is True


def test_group0_source_probe_reports_missing_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)

    diagnostics.register_group0_source_probe(
        req_id=None,
        layer_id=0,
        num_layers=2,
        source_chunks=(torch.zeros(4, dtype=torch.bfloat16),),
        selected_tokens=torch.tensor([0], dtype=torch.int32),
        selected_count=torch.tensor([1], dtype=torch.int32),
        chunk_size=1,
        total_tokens=1,
        token_major=False,
        layer_cache=(torch.zeros((1, 1, 4), dtype=torch.bfloat16),),
    )

    event, fields = events[-1]
    assert event == "group0_source_probe_error"
    assert fields == {
        "req_id": None,
        "layer_id": 0,
        "reason": "missing_req_id",
    }


def test_attention_diagnostic_metadata_error_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        diagnostics,
        "_content_log",
        lambda event, **fields: events.append((event, fields)),
    )
    diagnostics.configure_npu_content_diagnostics(True)
    diagnostics.register_group1_source_fingerprint(
        "request",
        {
            "sample_layer_ids": [0],
            "sample_logical_tokens": [0],
            "row_hashes": {"0:0": "0" * 32},
        },
    )

    diagnostics.queue_group1_first_consume(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        indexer_cache=object(),
        indexer_block_table=torch.tensor([[0]]),
        seq_lens_cpu=[1],
        block_size=4,
        row_request_indices=[0],
    )

    error = next(
        fields for event, fields in events if event == "group1_fingerprint_error"
    )
    assert error["requested_event"] == "group1_first_decode_consume"
