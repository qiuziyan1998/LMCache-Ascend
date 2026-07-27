# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache_ascend.v1.npu_connector import utils


def test_experimental_sparse_k_transfer_forwards_runtime_configuration(
    monkeypatch,
) -> None:
    calls = []

    def fake_op(*args):
        calls.append(args)

    monkeypatch.setattr(
        utils.lmc_ops,
        "sparse_k_batched_direct_kv_transfer_experimental",
        fake_op,
    )
    destination_state = object()
    slot_mapping = object()
    selected_indices = object()
    chunk_ptrs = object()
    selected_counts = object()

    utils.sparse_k_batched_direct_kv_transfer_experimental(
        destination_state,
        slot_mapping,
        selected_indices,
        chunk_ptrs,
        256,
        20000,
        kernel_variant=1,
        aiv_num=12,
        tile_tokens=8,
        pipeline=True,
        addressing_mode=2,
        work_assignment=1,
        selected_token_counts=selected_counts,
    )

    assert calls == [
        (
            destination_state,
            slot_mapping,
            selected_indices,
            chunk_ptrs,
            256,
            20000,
            1,
            12,
            8,
            True,
            2,
            1,
            selected_counts,
        )
    ]


def test_sparse_transfer_hardware_aiv_num_returns_int(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.lmc_ops, "sparse_transfer_hardware_aiv_num", lambda: 48
    )

    assert utils.sparse_transfer_hardware_aiv_num() == 48
