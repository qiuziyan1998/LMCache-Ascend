# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in Group-1 NPU content fingerprints."""

# Standard
from collections.abc import Iterator

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
        row_request_indices=torch.tensor([0]),
    )
    diagnostics.queue_selected_topk_fingerprint(
        req_ids=["request"],
        layer_name="model.layers.0.self_attn.indexer.k_cache",
        topk_indices=torch.tensor([[7, 3, 1]]),
        row_request_indices=torch.tensor([0]),
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
        "after_group1_wait_before_current_token_scatter"
    )
    topk = by_event["group1_selected_topk_fingerprint"]
    assert topk["value_preview"] == [7, 3, 1]
    assert topk["readback_mode"] == "deferred_after_model_forward"
    assert topk["capture_order"] == ("after_topk_selection_before_compact_remap")


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
        indexer_cache=torch.zeros((1, 4, 1, 2)),
        indexer_block_table=torch.tensor([[0]]),
        seq_lens_cpu=[],
        block_size=4,
        row_request_indices=[0],
    )

    error = next(
        fields for event, fields in events if event == "group1_fingerprint_error"
    )
    assert error["requested_event"] == "group1_first_decode_consume"
