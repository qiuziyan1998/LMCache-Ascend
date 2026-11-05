# SPDX-License-Identifier: Apache-2.0
"""Fixed destination registration and warm-plan reuse, with CPU-only tensors."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.npu_connector import npu_connectors
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


def _destination_setup(monkeypatch, layers=2):
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.num_layers = layers
    connector.dsa_two_groups = False
    connector.runtime_kv_group_layer_counts = None
    connector._group_layouts = {}
    connector.enable_npu_transfer_validation = True
    connector._sparse_destination_plans = {}
    caches = [[torch.zeros((2, 4))] for _ in range(layers)]
    prepare = MagicMock(side_effect=lambda *args: object())
    monkeypatch.setattr(
        npu_connectors, "prepare_sparse_direct_destination_state", prepare
    )
    kwargs = dict(
        kvcaches_ref=caches,
        kv_group=0,
        slot_mapping_ref=torch.arange(4),
        sparse_kv_format=0,
        sparse_k_hidden_dims=1,
        sparse_v_hidden_dims=1,
        sparse_dsa_hidden_dims=0,
        expected_device=torch.device("cpu"),
    )
    return connector, caches, kwargs, prepare


@pytest.mark.parametrize("layers,indexer_layers", [(2, 2), (79, 22)])
@pytest.mark.parametrize("preflight_indexer", [False, True])
def test_decoder_seals_before_first_latent_transfer(
    monkeypatch, layers, indexer_layers, preflight_indexer
):
    connector, caches, _, prepare = _destination_setup(monkeypatch, layers)
    connector.dsa_two_groups = True
    connector.runtime_kv_group_layer_counts = (layers, indexer_layers)
    if preflight_indexer:
        connector._group_layouts[1] = SimpleNamespace(num_layers=indexer_layers)
        connector.num_layers = indexer_layers  # Last transfer/preflight was Group 1.
    initialize = MagicMock(
        side_effect=AssertionError("seal initialized transfer state")
    )
    monkeypatch.setattr(connector, "_lazy_initialize_buffer", initialize)
    monkeypatch.setattr(connector, "initialize_kvcaches_ptr", initialize)
    for name in ("to", "copy_", "cpu", "tolist", "item", "sum"):
        monkeypatch.setattr(torch.Tensor, name, initialize)

    binding = connector.seal_sparse_destination_layout(caches)
    assert binding.kvcaches_ref is caches
    assert connector.seal_sparse_destination_layout(list(caches)) is binding
    assert 0 not in connector._group_layouts
    with pytest.raises(ValueError, match="wrong layer count"):
        connector.seal_sparse_destination_layout(caches[:-1])
    initialize.assert_not_called()
    prepare.assert_not_called()


def test_seal_rejects_initialized_layout_disagreeing_with_runtime(monkeypatch):
    connector, caches, _, _ = _destination_setup(monkeypatch)
    connector.dsa_two_groups = True
    connector.runtime_kv_group_layer_counts = (2, 1)
    connector._group_layouts[0] = SimpleNamespace(num_layers=1)
    with pytest.raises(ValueError):
        connector.seal_sparse_destination_layout(caches)
    assert getattr(connector, "_sealed_sparse_destination_layout", None) is None


def test_seal_requires_runtime_counts_or_an_initialized_dsa_layout(monkeypatch):
    connector, caches, _, _ = _destination_setup(monkeypatch)
    connector.dsa_two_groups = True
    with pytest.raises(RuntimeError, match="before the kv_group=0 layout"):
        connector.seal_sparse_destination_layout(caches)
    connector._group_layouts[0] = SimpleNamespace(num_layers=2)
    connector.num_layers = 1  # A mirrored Group-1 count is not authoritative.
    assert connector.seal_sparse_destination_layout(caches).kvcaches_ref is caches


def test_factory_carries_runtime_cardinality_into_destination_seal(monkeypatch):
    from lmcache.utils import EngineType
    from lmcache_ascend.v1 import npu_connector as factory

    connector, caches, _, _ = _destination_setup(monkeypatch, 3)
    metadata = SimpleNamespace(worker_id=0, use_mla=True,
                               runtime_kv_group_layer_counts=(3, 1))
    config = SimpleNamespace(use_layerwise=True, enable_blending=False,
                             dsa_two_groups=True, enable_npu_transfer_validation=True)
    cpu_device = torch.device("cpu")
    monkeypatch.setattr(factory, "need_gpu_interm_buffer", lambda _: False)
    monkeypatch.setattr(factory, "configure_npu_content_diagnostics", lambda _: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        device_count=lambda: 1, set_device=lambda _: None), raising=False)
    monkeypatch.setattr(torch, "device", lambda _: cpu_device)
    monkeypatch.setattr(VLLMPagedMemLayerwiseNPUConnector, "from_metadata",
                        lambda *a, **kw: connector)
    actual = factory.CreateNPUConnector(config, metadata, EngineType.VLLM)
    assert actual is connector
    assert actual.runtime_kv_group_layer_counts == (3, 1)
    assert actual.seal_sparse_destination_layout(caches).kvcaches_ref is caches


@pytest.mark.parametrize("layers,indexer_layers", [(2, 2), (79, 22)])
def test_first_prepared_load_after_early_seal_initializes_only_latent_metadata(
    monkeypatch, layers, indexer_layers
):
    connector, _, kwargs, prepare = _destination_setup(monkeypatch, layers)
    caches = [
        (torch.zeros(2, 4, 1, 512), torch.zeros(2, 4, 1, 64))
        for _ in range(layers)
    ]
    connector.dsa_two_groups = connector.use_mla = True
    connector.use_gpu = False
    connector.lmcache_chunk_size = 4
    connector.runtime_kv_group_layer_counts = (layers, indexer_layers)
    indexer_layout = npu_connectors._GroupLayout()
    indexer_layout.num_layers = indexer_layers
    connector._group_layouts[1] = indexer_layout
    connector._current_kv_group = 1
    connector.num_layers = indexer_layers
    binding = connector.seal_sparse_destination_layout(caches)
    assert 0 not in connector._group_layouts
    source = SimpleNamespace(
        layers=[SimpleNamespace(chunk_ptrs_npu=torch.ones(1)) for _ in caches],
        total_tokens=4, chunk_token_counts=(4,), validated_chunk_size=4,
    )
    signature = MagicMock(side_effect=AssertionError("sealed metadata was rebuilt"))
    monkeypatch.setattr(connector, "_vllm_layer_cache_identity_signature", signature)
    plans = []
    for _ in range(2):
        loader = connector.batched_to_gpu_head_token_wise(
            prepared_sparse_source=source, kvcaches=caches,
            slot_mapping=kwargs["slot_mapping_ref"],
            registered_destination_layout=binding, kv_group=0, sync=False,
        )
        try:
            next(loader)  # Prepare metadata; no layer payload is submitted.
            plans.append(connector._sparse_destination_plans[0])
        finally:
            loader.close()
    assert connector.get_num_layers(0) == layers
    assert connector._group_layouts[0].gpu_buffer_allocator is None
    assert connector._group_layouts[1] is indexer_layout
    assert plans[0] is plans[1] and plans[0].binding is binding
    assert prepare.call_count == layers
    signature.assert_not_called()


def test_sealed_destination_hit_does_not_walk_or_compare_layers(monkeypatch):
    connector, caches, kwargs, prepare = _destination_setup(monkeypatch, 79)
    comparisons = []

    class Signature(tuple):
        def __eq__(self, other):
            comparisons.append(1)
            return super().__eq__(other)

    original = connector._vllm_layer_cache_identity_signature
    signature = MagicMock(side_effect=lambda layer: Signature(original(layer)))
    monkeypatch.setattr(connector, "_vllm_layer_cache_identity_signature", signature)
    resolve = connector._get_or_create_sparse_destination_plan
    old_plan = resolve(**kwargs)
    signature.reset_mock()
    assert resolve(**kwargs) is old_plan
    assert signature.call_count == 79

    binding = connector.seal_sparse_destination_layout(caches)
    kwargs["registered_destination_layout"] = binding
    plan = resolve(**kwargs)
    assert plan.states is old_plan.states
    assert plan.binding is binding
    signature.reset_mock()
    comparisons.clear()
    prepare.reset_mock()
    for request in range(20):
        kwargs["slot_mapping_ref"] = torch.arange(request + 1)
        caches[0][0].fill_(request)  # Data writes do not change the destination.
        assert resolve(**kwargs) is plan
    signature.assert_not_called()
    prepare.assert_not_called()
    assert comparisons == []


@pytest.mark.parametrize("change", ["tensor", "storage", "shape", "stride", "order"])
def test_sealed_destination_refresh_rejects_structural_change(monkeypatch, change):
    connector, caches, kwargs, _ = _destination_setup(monkeypatch)
    binding = connector.seal_sparse_destination_layout(caches)
    assert connector.seal_sparse_destination_layout(list(caches)) is binding
    old_tensor = caches[0][0]
    if change == "tensor":
        caches[0][0] = torch.zeros_like(old_tensor)
    elif change == "storage":
        old_tensor.set_(torch.zeros_like(old_tensor))
    elif change == "shape":
        old_tensor.resize_(4, 2)
    elif change == "stride":
        old_tensor.as_strided_((2, 4), (1, 2))
    else:
        caches.reverse()
    with pytest.raises(RuntimeError, match="Sealed sparse destinations changed"):
        connector.seal_sparse_destination_layout(caches)
    assert binding.tensor_refs[0][0] is old_tensor


@pytest.mark.parametrize("change", ["binding", "collection", "group", "validation"])
def test_sealed_destination_rejects_incompatible_binding(monkeypatch, change):
    connector, caches, kwargs, prepare = _destination_setup(monkeypatch)
    kwargs["registered_destination_layout"] = connector.seal_sparse_destination_layout(
        caches
    )
    if change == "binding":
        kwargs["registered_destination_layout"] = object()
    elif change == "collection":
        kwargs["kvcaches_ref"] = list(caches)
    elif change == "group":
        kwargs["kv_group"] = 1
    else:
        connector.enable_npu_transfer_validation = False
    with pytest.raises(RuntimeError, match="binding"):
        connector._get_or_create_sparse_destination_plan(**kwargs)
    prepare.assert_not_called()


@pytest.mark.parametrize(
    "change", ["dtype", "format", "dimensions", "device", "layers"]
)
def test_sealed_destination_preserves_runtime_validation(monkeypatch, change):
    connector, caches, kwargs, prepare = _destination_setup(monkeypatch)
    kwargs["registered_destination_layout"] = connector.seal_sparse_destination_layout(
        caches
    )
    first = connector._get_or_create_sparse_destination_plan(**kwargs)
    prepare.reset_mock()
    if change == "dtype":
        kwargs["slot_mapping_ref"] = torch.arange(4, dtype=torch.int32)
    elif change == "format":
        kwargs["sparse_kv_format"] = 1
    elif change == "dimensions":
        kwargs["sparse_k_hidden_dims"] = 2
    elif change == "device":
        kwargs["expected_device"] = torch.device("meta")
    else:
        connector.num_layers += 1
    if change in ("device", "layers"):
        with pytest.raises(ValueError):
            connector._get_or_create_sparse_destination_plan(**kwargs)
        prepare.assert_not_called()
    else:
        second = connector._get_or_create_sparse_destination_plan(**kwargs)
        assert second is not first
        assert prepare.call_count == 2
        assert connector._get_or_create_sparse_destination_plan(**kwargs) is second


def test_sealed_destination_no_transfers_and_stale_miss_not_published(monkeypatch):
    connector, caches, kwargs, prepare = _destination_setup(monkeypatch)
    fail = MagicMock(side_effect=AssertionError("Unexpected tensor transfer/readback"))
    for name in ("to", "copy_", "cpu", "tolist", "item", "sum"):
        monkeypatch.setattr(torch.Tensor, name, fail)
    binding = connector.seal_sparse_destination_layout(caches)
    kwargs["registered_destination_layout"] = binding

    def invalidated_during_native_build(*args):
        connector._sealed_sparse_destination_layout = None
        return object()

    prepare.side_effect = invalidated_during_native_build
    with pytest.raises(RuntimeError, match="changed during preparation"):
        connector._get_or_create_sparse_destination_plan(**kwargs)
    assert not connector._sparse_destination_plans
    connector._sealed_sparse_destination_layout = binding
    prepare.side_effect = lambda *args: object()
    plan = connector._get_or_create_sparse_destination_plan(**kwargs)
    assert connector._get_or_create_sparse_destination_plan(**kwargs) is plan
    fail.assert_not_called()


def test_disabled_validation_does_not_seal_destination(monkeypatch):
    connector, caches, kwargs, _ = _destination_setup(monkeypatch)
    connector.enable_npu_transfer_validation = False
    assert connector.seal_sparse_destination_layout(caches) is None
    assert connector._get_or_create_sparse_destination_plan(**kwargs).binding is None


def test_sealed_destination_generators_forward_current_request_payloads(monkeypatch):
    connector, caches, kwargs, _ = _destination_setup(monkeypatch)
    binding = connector.seal_sparse_destination_layout(caches)
    connector.lmcache_chunk_size = 4
    connector._group_layouts = {
        0: SimpleNamespace(
            num_layers=2,
            k_hidden_dims=1,
            v_hidden_dims=1,
            dsa_hidden_dims=0,
            kv_format=SimpleNamespace(value=0),
            kv_device=torch.device("cpu"),
        )
    }
    connector._sparse_lmc_host_interleaved = lambda group: False
    connector._normalize_sparse_selection = lambda selected, slots: (selected, True)
    connector._pack_sparse_explicit_slot_inputs = lambda selected, slots, counts: (
        slots,
        selected,
        counts,
    )
    connector._maybe_limit_sparse_transfer_inputs = lambda slots, selected, **kw: (
        slots,
        selected,
    )
    transfer = MagicMock()
    monkeypatch.setattr(
        connector, "_run_prepared_sparse_direct_kv_transfer_layer", transfer
    )
    monkeypatch.setattr(
        npu_connectors, "serving_perf_detailed_enabled", lambda: False
    )
    monkeypatch.setattr(
        npu_connectors, "npu_content_diagnostics_enabled", lambda: False
    )
    monkeypatch.setattr(npu_connectors, "_mtp_dw_deep_diag_enabled", lambda: False)
    source = SimpleNamespace(
        layers=[SimpleNamespace(chunk_ptrs_npu=torch.arange(1)) for _ in caches],
        total_tokens=4,
        chunk_token_counts=(4,),
        validated_chunk_size=4,
    )
    generators = [
        connector.batched_to_gpu_head_token_wise(
            prepared_sparse_source=source,
            kvcaches=caches,
            slot_mapping=kwargs["slot_mapping_ref"],
            registered_destination_layout=binding,
            kv_group=0,
            sync=False,
        )
        for _ in range(2)
    ]
    try:
        for generator in generators:
            next(generator)
        for layer_id in range(2):
            for request, generator in enumerate(generators):
                selected = torch.tensor([[request + layer_id]], dtype=torch.int32)
                slots = torch.tensor([[request + 10 * layer_id]])
                counts = torch.ones(1, dtype=torch.int32)
                generator.send(
                    {
                        "selected_token_ids": selected,
                        "target_slot_mapping": slots,
                        "selected_token_counts": counts,
                    }
                )
                passed = transfer.call_args.kwargs
                assert passed["slot_mapping_packed"] is slots
                assert passed["selected_token_idx"] is selected
                assert passed["selected_token_counts"] is counts
                assert (
                    passed["chunk_ptrs_npu"] is source.layers[layer_id].chunk_ptrs_npu
                )
                assert passed["plan"].binding is binding
        assert transfer.call_count == 4
        assert len({id(call.kwargs["plan"]) for call in transfer.call_args_list}) == 1
    finally:
        for generator in generators:
            generator.close()
