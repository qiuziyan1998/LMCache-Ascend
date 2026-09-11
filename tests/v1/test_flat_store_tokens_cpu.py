# SPDX-License-Identifier: Apache-2.0
"""CPU regression: real flat allocations, store methods and LocalCPU teardown.

Only accelerator transfer/control dependencies are fixtures. Production store
method ASTs are loaded without importing the NPU extension.
"""

# Standard
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import __future__
import ast
import logging
import threading
import time

# Third Party
import pytest
import torch


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    sibling = Path(__file__).resolve().parents[3] / "LMCache"
    if sibling.is_dir():
        monkeypatch.syspath_prepend(str(sibling))
    # First Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.cache_engine import LayerwiseStoreResult
    from lmcache.v1.memory_management import (
        LayerPageMemoryObj,
        MemoryFormat,
        TensorMemoryAllocator,
    )
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    return SimpleNamespace(
        CacheEngineKey=CacheEngineKey,
        LayerwiseStoreResult=LayerwiseStoreResult,
        LayerPageMemoryObj=LayerPageMemoryObj,
        MemoryFormat=MemoryFormat,
        TensorMemoryAllocator=TensorMemoryAllocator,
        LocalCPUBackend=LocalCPUBackend,
    )


def load_store(name: str, modules: Any, page_store: bool) -> Any:
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "AscendLMCacheEngine"
    )
    method = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {
        **vars(modules),
        "torch": torch,
        "time": time,
        "logger": logging.getLogger(__name__),
        "_lmcache_nvtx_annotate": lambda function: function,
        "_mtp_dw_diag_enabled": lambda: False,
        "mooncake_layer_pages_enabled": lambda config: page_store,
        "mooncake_page_layout_enabled": lambda config: page_store,
        "serving_perf_enabled": lambda: False,
        "assert_layerwise_gpu_connector": lambda connector: None,
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("kv_group", [0, 1])
@pytest.mark.parametrize(
    "layers,page_store", [(1, False), (8, False), (9, False), (8, True)]
)
@pytest.mark.parametrize("store_method", ["store_layer", "_run_store_pipeline"])
def test_flat_store_preserves_full_and_tail_counts_through_backend_close(
    modules: Any, kv_group: int, layers: int, page_store: bool, store_method: str
) -> None:
    fmt = (
        modules.MemoryFormat.KV_MLA_LATENT_FMT
        if kv_group == 0
        else modules.MemoryFormat.KV_DSA_INDEX_FMT
    )
    width = 6 if kv_group == 0 else 4
    allocator = modules.TensorMemoryAllocator(torch.zeros(1 << 20, dtype=torch.uint8))
    backend = object.__new__(modules.LocalCPUBackend)
    backend.cpu_lock = threading.RLock()
    backend.use_hot = True
    backend.hot_cache = OrderedDict()
    backend.batched_msg_sender = None
    backend.memory_allocator = allocator
    backend.cache_policy = MagicMock()
    backend._record_external_retention_mutation_locked = MagicMock()
    engine = MagicMock()
    engine.num_layers = layers
    engine.is_healthy.return_value = True
    engine.is_frozen.return_value = False
    engine._is_passive.return_value = False
    engine._get_req_id.return_value = "r"
    engine.config.chunk_size = 256
    engine.config.get_extra_config_value.side_effect = lambda key, default=None: default
    engine.kv_events_enabled = False
    engine._engine_state_lock = threading.Lock()
    engine._memory_format_for_kv_group.return_value = fmt
    engine._shared_cpu_dtype_for_kv_group.return_value = torch.float16
    engine._layerwise_chunk_fully_stored.return_value = False
    engine.gpu_connector.get_shape.side_effect = lambda count, **kw: torch.Size(
        [count * width]
    )
    engine._metadata_shapes_dtypes_for_kv_group.side_effect = (
        lambda *, num_tokens, **kw: (
            [torch.Size([num_tokens * width])],
            [torch.float16],
        )
    )
    engine.storage_manager.allocate.side_effect = lambda shape, dtype, **kw: (
        allocator.allocate(shape, dtype, fmt=kw["fmt"])
    )
    engine.storage_manager.batched_allocate.side_effect = (
        lambda shape, dtype, batch_size, **kw: allocator.batched_allocate(
            shape, dtype, batch_size, fmt=kw["fmt"]
        )
    )
    engine.storage_manager.supports_batched_put_layer_pages.return_value = True
    local = engine._shared_local_cpu_backend.return_value
    local.batched_allocate_layer_pages.return_value = None
    chunks = [(0, 256), (256, 263), (263, 264)]
    engine.token_database.process_tokens.return_value = [
        (
            start,
            end,
            modules.CacheEngineKey(
                "model", 8, 0, index, torch.float16, kv_group=kv_group
            ),
        )
        for index, (start, end) in enumerate(chunks)
    ]
    published = []

    def put(keys: list, objects: list, **kwargs: Any) -> list:
        for key, obj in zip(keys, objects, strict=True):
            backend.hot_cache[key] = obj  # transfer the allocation reference to cache
            published.append(obj)
        return []

    engine.storage_manager.batched_put.side_effect = put
    engine.gpu_connector.batched_from_gpu.side_effect = lambda *args, **kw: iter(
        [None] * (layers + 1)
    )
    method = load_store(store_method, modules, page_store)
    try:
        if store_method == "store_layer":
            result = list(method(engine, [0] * 264, req_id="r", kv_group=kv_group))[-1]
            assert result.committed_end == 264
        else:
            method(
                engine, "r", [0] * 264, None, None, None, 264, {"kv_group": kv_group}
            )
        # Before the fix this real close raises exactly the reported ValueError.
        backend.close()
        expected = [256, 7, 1] * (layers if store_method == "store_layer" else 1)
        assert sorted(obj.get_num_tokens() for obj in published) == sorted(expected)
        assert all(not obj.is_valid() for obj in published)
        assert not backend.hot_cache
        assert allocator.get_capacity_bytes() == (1 << 20, 1 << 20)
    finally:
        # Only clean up this test's CPU allocations when a regression fails.
        for obj in published:
            if obj.is_valid():
                obj.ref_count_down()


def test_truly_unknown_flat_token_count_still_raises(modules: Any) -> None:
    allocator = modules.TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    obj = allocator.allocate(
        torch.Size([128]), torch.float16, modules.MemoryFormat.KV_DSA_INDEX_FMT
    )
    assert obj is not None
    try:
        with pytest.raises(ValueError, match="valid_tokens metadata"):
            obj.get_num_tokens()
    finally:
        obj.ref_count_down()
