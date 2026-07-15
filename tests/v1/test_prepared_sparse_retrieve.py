# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
from lmcache.v1.gpu_connector.sparse import (
    PreparedSparseSource,
    PreparedSparseSourceLayer,
)
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


class _RecordingConnector:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.kwargs = None
        self.transfers = []

    def batched_to_gpu_head_token_wise(self, **kwargs):
        self.kwargs = kwargs
        for _ in range(self.num_layers):
            self.transfers.append((yield))
        yield
        yield


def _prepared_source(num_layers: int = 2) -> PreparedSparseSource:
    layers = tuple(
        PreparedSparseSourceLayer(
            tensors=(torch.zeros(4),),
            chunk_ptrs_npu=torch.tensor([100 + layer_id], dtype=torch.int64),
        )
        for layer_id in range(num_layers)
    )
    return PreparedSparseSource(
        layers=layers,
        total_tokens=4,
    )


def test_prepared_sparse_retrieve_bypasses_bootstrap() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    engine.num_layers = 2
    engine.gpu_connector = _RecordingConnector(num_layers=2)
    engine.is_healthy = lambda: True

    def fail_bootstrap(*args, **kwargs):
        raise AssertionError("prepared retrieve entered the bootstrap path")
        yield

    engine._retrieve_layer_head_token_wise_bootstrap = fail_bootstrap

    source = _prepared_source()
    ret_mask = torch.zeros(4, dtype=torch.bool)
    retriever = engine.retrieve_layer_head_token_wise(
        [1, 2, 3, 4],
        prepared_sparse_source=source,
        kvcaches=[object(), object()],
        slot_mapping=torch.arange(4),
        sync=False,
        kv_group=0,
        ret_mask=ret_mask,
    )

    assert next(retriever) is ret_mask
    first_payload = (torch.tensor([0], dtype=torch.int32), 0)
    second_payload = (torch.tensor([1], dtype=torch.int32), 0)
    assert retriever.send(first_payload) is ret_mask
    assert retriever.send(second_payload) is ret_mask

    connector = engine.gpu_connector
    assert connector.kwargs["prepared_sparse_source"] is source
    assert connector.transfers == [first_payload, second_payload]


def test_dense_store_prepares_sparse_pointer_cache_without_shared_cpu() -> None:
    engine = object.__new__(AscendLMCacheEngine)
    calls = []
    engine.gpu_connector = SimpleNamespace(
        append_sparse_chunk_ptr_cache_for_layer=(lambda *args: calls.append(args))
    )
    tensor = torch.zeros(4)
    cached_tensors = []
    cached_chunk_dev_ptrs = []
    cached_chunk_ptrs_npu = []

    engine._append_layer_store_tensors(
        0,
        [[SimpleNamespace(tensor=tensor)]],
        cached_tensors,
        cached_chunk_dev_ptrs,
        cached_chunk_ptrs_npu,
    )

    assert len(calls) == 1
    assert calls[0][0] == 0
    assert calls[0][1][0] is tensor
    assert calls[0][2] is cached_chunk_dev_ptrs
    assert calls[0][3] is cached_chunk_ptrs_npu
