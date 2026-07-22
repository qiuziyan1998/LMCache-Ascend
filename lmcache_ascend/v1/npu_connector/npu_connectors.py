# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
import os
import time
from typing import Any, List, Optional, Set, Union

# Third Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.gpu_connectors import (
    SGLangGPUConnector,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.sparse import (
    PreparedSparseSource,
    PreparedSparseSourceLayer,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.utils import (
    batched_fused_single_layer_kv_transfer,
    dense_mla_dsa_batched_direct_kv_transfer,
    dense_mla_dsa_batched_direct_kv_transfer_fast,
    prepare_sparse_direct_destination_state,
    prepare_sparse_direct_layer_state,
    sparse_mla_dsa_batched_direct_kv_transfer,
    sparse_mla_dsa_batched_direct_kv_transfer_fast,
    sparse_mla_dsa_batched_direct_kv_transfer_prepared,
)
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj

from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_DENSE_DIRECT_DISABLE = os.getenv(
    "LMCACHE_ASCEND_DENSE_DIRECT_DISABLE", "0"
).lower() in ("1", "true", "yes", "on")
_DENSE_DIRECT_LOAD_DISABLE = _DENSE_DIRECT_DISABLE or os.getenv(
    "LMCACHE_ASCEND_DENSE_DIRECT_LOAD_DISABLE", "0"
).lower() in ("1", "true", "yes", "on")
_DENSE_DIRECT_STORE_DISABLE = _DENSE_DIRECT_DISABLE or os.getenv(
    "LMCACHE_ASCEND_DENSE_DIRECT_STORE_DISABLE", "0"
).lower() in ("1", "true", "yes", "on")

_SPARSE_DESTINATION_PLAN_CACHE_SIZE = 2


def _wait_payload_events(stream: Any, payload_event: Any) -> None:
    if isinstance(payload_event, (list, tuple)):
        for event in payload_event:
            if event is not None:
                stream.wait_event(event)
    elif payload_event is not None:
        stream.wait_event(payload_event)


_IS_310P = None


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        # First Party
        from lmcache_ascend import _build_info

        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P


class VLLMBufferLayerwiseNPUConnector(VLLMBufferLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim_size, num_layers, use_gpu, use_double_buffer, **kwargs
        )
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.fused_rotary_emb: Any = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.kv_format = KVCacheFormat.detect(kv_caches)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError("Could not detect KV cache format.")

            ref_tensor = (
                kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]
            )
            self.kv_device = ref_tensor.device

            first_layer_cache = kv_caches[0]

            # flash attention: [num_layers, 2, num_blocks,
            # block_size, num_heads, head_size]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                k_cache_shape_per_layer = first_layer_cache[0].shape
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            self.vllm_two_major = True

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def _prepare_transfer_context(self, kwargs) -> torch.Tensor:
        """
        Initialize context for KV cache transfer, validate required
        parameters and lazy init buffer.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("kvcaches should be provided in kwargs or initialized.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        self._lazy_initialize_buffer(self.kvcaches)
        return kwargs["slot_mapping"]

    def _get_full_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        starts: List[int],
        ends: List[int],
        mode: str = "slice",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate full continuous slot mapping tensor and calculate total token count.
        Supports two modes for different transfer directions (to/from GPU).
        """
        if mode == "slice":
            slot_mapping_full = slot_mapping[starts[0] : ends[-1]]
        elif mode == "concat":
            slot_mapping_chunks = [
                slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
            ]
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        else:
            raise ValueError(
                f"Unsupported slot mapping mode: {mode}, only 'slice'/'concat' allowed"
            )

        num_tokens = len(slot_mapping_full)
        return slot_mapping_full, num_tokens

    def _allocate_gpu_buffers(
        self, num_tokens: int, count: int = 1
    ) -> Union[object, list[object]]:
        """
        Allocate specified number of GPU buffers for KV cache with shape
        calculated by token count. Performs strict assertion checks for
        valid buffer allocation.
        """
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None, (
            "GPU buffer allocator not initialized"
        )
        buffers = []
        for _ in range(count):
            buf_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_2TD
            )
            assert buf_obj is not None, "Failed to allocate GPU buffer in GPUConnector"
            assert buf_obj.tensor is not None, "GPU buffer object has no valid tensor"
            buffers.append(buf_obj)
        return buffers[0] if count == 1 else buffers

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping_full, num_all_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="slice"
        )

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        load_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj, load_gpu_buffer_obj = self._allocate_gpu_buffers(
            num_all_tokens, count=2
        )

        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kv_device
            )
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kv_device
            )

        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(
                    "Finished loading layer %d into paged memory",
                    layer_id - 2,
                )

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(
                    "Finished loading layer %d into buffer",
                    layer_id - 1,
                )

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        buf_start = 0
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(start, end, device=self.kv_device, dtype=torch.int64)
                )

        slot_mapping_full, num_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="concat"
        )

        tmp_gpu_buffer_obj = self._allocate_gpu_buffers(num_tokens, count=1)

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )

                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug("Finished offloading layer %d", layer_id)

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield


class VLLMPagedMemNPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        # Initialize kv_format before calling super().__init__
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        # Initialize MLA/DSA parameters
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.dsa_two_groups: bool = kwargs.get("dsa_two_groups", False)

        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        if is_310p():
            assert "num_kv_head" in kwargs, ("num_kv_head should be provided in 310p",)
            assert "head_size" in kwargs, ("head_size should be provided in 310p",)
            self.num_kv_head = kwargs["num_kv_head"]
            self.head_size = kwargs["head_size"]
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional KV layout hints from the serving engine.

        Returns:
            A new instance of VLLMPagedMemNPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            num_kv_head=num_kv_head,
            head_size=head_size,
            layout_hints=layout_hints,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_format = KVCacheFormat.detect(
            kv_caches,
            use_mla=self.use_mla,
            dsa_two_groups=getattr(self, "dsa_two_groups", False),
        )

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError(
                "Undefined KV cache format detected. "
                "Unable to determine the format of input kv_caches."
            )

        if self.kv_format.is_tuple_format():
            self.kvcaches_device = kv_caches[0][0].device
        else:
            self.kvcaches_device = kv_caches[0].device

        assert self.kvcaches_device.type == "npu", "The device should be Ascend NPU."
        idx = self.kvcaches_device.index

        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        self.kv_size = self.kv_format.get_kv_size()
        pointers_list = []

        if self.kv_format == KVCacheFormat.DSA_KV:
            for cache_tuple in kv_caches:
                k_cache, v_cache, dsa_k_cache = cache_tuple
                pointers_list.append(k_cache.data_ptr())
                pointers_list.append(v_cache.data_ptr())
                pointers_list.append(dsa_k_cache.data_ptr())

            self.kv_cache_pointers = torch.empty(
                self.num_layers * self.kv_size, dtype=torch.int64, device="cpu"
            )
        elif self.kv_format == KVCacheFormat.MLA_KV:
            for k_cache, v_cache in kv_caches:
                pointers_list.append(k_cache.data_ptr())
                pointers_list.append(v_cache.data_ptr())

            self.kv_cache_pointers = torch.empty(
                self.num_layers * self.kv_size, dtype=torch.int64, device="cpu"
            )
        elif self.kv_format == KVCacheFormat.SEPARATE_KV:
            self.kv_size = 2
            pointers_list = []
            for k, v in kv_caches:
                pointers_list.append(k.data_ptr())
                pointers_list.append(v.data_ptr())

            self.kv_cache_pointers = torch.empty(
                self.num_layers * self.kv_size, dtype=torch.int64, device="cpu"
            )
        else:
            self.kv_size = 1
            pointers_list = [t.data_ptr() for t in kv_caches]

            self.kv_cache_pointers = torch.empty(
                self.num_layers, dtype=torch.int64, device="cpu"
            )

        self.kv_cache_pointers.numpy()[:] = pointers_list

        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.kv_cache_pointers.shape, dtype=torch.int64, device=self.kvcaches_device
        )

        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        first_tensor = (
            kv_caches[0][0] if self.kv_format.is_tuple_format() else kv_caches[0]
        )

        if self.use_mla or self.kv_format in (
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
        ):
            if self.kv_format == KVCacheFormat.MLA_KV:
                k_cache, v_cache = kv_caches[0]
                self.page_buffer_size = k_cache.shape[0] * k_cache.shape[1]
                self.kv_lora_rank = k_cache.shape[-1]
                self.qk_rope_head_dim = v_cache.shape[-1]
            elif self.kv_format == KVCacheFormat.DSA_KV:
                k_cache, v_cache, dsa_k_cache = kv_caches[0]
                self.page_buffer_size = k_cache.shape[0] * k_cache.shape[1]
                self.kv_lora_rank = k_cache.shape[-1]
                self.qk_rope_head_dim = v_cache.shape[-1]
                self.dsa_head_dim = dsa_k_cache.shape[-1]
        else:
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                # kv_caches[0]: [tuple(k,v)]
                # 310P: [num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() >= 2
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[0] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[0] * first_tensor.shape[1]
                    )

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
                # 310P: [2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [2, num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() == 5
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[1] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[1] * first_tensor.shape[2]
                    )

        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format "
                    "in order to be processed by VLLMPagedMemNPUConnector."
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        tmp_gpu_buffer.copy_(memory_obj.tensor)

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        assert self.gpu_buffer.device == self.kvcaches_device

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            True,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

        memory_obj.tensor.copy_(tmp_gpu_buffer)
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in "
                    " order to be processed by VLLMPagedMemNPUConnector."
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.kv_format.value,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.dsa_head_dim,
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        with torch.npu.stream(self.store_stream):
            self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        with torch.npu.stream(self.store_stream):
            self._initialize_pointers(self.kvcaches)

        if "slot_mapping_npu" in kwargs:
            slot_mapping: torch.Tensor = kwargs["slot_mapping_npu"]
        elif "slot_mapping" in kwargs:
            slot_mapping = kwargs["slot_mapping"]
            if not isinstance(slot_mapping, torch.Tensor):
                raise ValueError("'slot_mapping' should be a torch.Tensor.")
            # for Ascend kernels to keep test inputs backward compatible.
            if slot_mapping.device.type != "npu":
                with torch.npu.stream(self.store_stream):
                    slot_mapping = slot_mapping.to(
                        self.kvcaches_device,
                        non_blocking=True,
                    )
        else:
            raise ValueError(
                "'slot_mapping_npu' should be provided in kwargs "
                "(or 'slot_mapping' for compatibility)."
            )

        with torch.npu.stream(self.store_stream):
            kv_cache_pointers = self.kv_cache_pointers_on_gpu[
                self.kvcaches_device.index
            ]

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("KV cache format is not initialized!")

        with torch.npu.stream(self.store_stream):
            # No staging buffer or token count mismatch
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                    self.kv_format.value,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    self.dsa_head_dim,
                )
            else:
                assert self.gpu_buffer.device == self.kvcaches_device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.fused_multi_layer_kv_transfer(
                    memory_obj.tensor,  # dst: CPU buffer
                    tmp_gpu_buffer,  # staging cache
                    kv_cache_pointers,  # src: paged KV cache
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,  # from_gpu
                    self.use_mla,
                    self.kv_format.value,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    self.dsa_head_dim,
                )
        no_sync = kwargs.get("no_sync", False)
        if not no_sync and not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        # Check if any memory objects are ProxyMemoryObjs (deferred P2P fetch)
        has_proxy = any(isinstance(m, ProxyMemoryObj) for m in memory_objs)

        if has_proxy:
            assert not is_310p(), "Batched P2P transfer is not supported on 310P."

            self._remote_batched_to_gpu(memory_objs, starts, ends, **kwargs)

            # NOTE (gingfung): Ensure the compute stream waits for
            # load_stream's KV scatter to complete before attention
            # reads the same pages.
            # load_stream.synchronize() in _remote_batched_to_gpu is
            # host-side only, the compute stream has no knowledge of it
            # and can race ahead.
            torch.npu.current_stream().wait_stream(self.load_stream)
        else:
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in zip(
                    memory_objs, starts, ends, strict=False
                ):
                    if is_310p():
                        self.to_gpu_310p(memory_obj, start, end, **kwargs)
                    else:
                        self.to_gpu(memory_obj, start, end, **kwargs)
            self.load_stream.synchronize()

    def _clear_proxy_batch(self, batch) -> None:
        """Clear the backing objects of the proxy batch."""
        for proxy, _, _ in batch:
            proxy.clear_backing_obj()
        return None

    def _scatter_proxy_batch(self, batch, event, **kwargs):
        """Wait for a read event, scatter proxies to KV cache.

        Enqueues work on ``load_stream``.  The caller is responsible for
        recording a scatter-done event afterwards if needed for
        cross-stream synchronization.
        """
        if event is not None:
            self.load_stream.wait_event(event)
        with torch.cuda.stream(self.load_stream):
            for proxy, start, end in batch:
                self.to_gpu(proxy.backing_obj, start, end, **kwargs)

    def _remote_batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """Handle batched_to_gpu when ProxyMemoryObjs are present.

        Uses a ping-pong pipeline with **event-based** cross-stream
        synchronization to overlap remote data fetching (on the HCCL
        transport stream) with KV cache scatter (on the load stream).


        Two pools of PIPELINE_DEPTH buffers are allocated from the
        transfer context's registered memory and alternated (ping-pong).
        This limits peak memory to 2 x PIPELINE_DEPTH chunks regardless
        of the total number of proxy objects.

        After all proxy objects are processed, sends the Done signal
        to release the remote peer's pinned resources.
        """
        transfer_contexts: Set[AscendBaseTransferContext] = set()

        # Separate proxy and non-proxy items
        proxy_items = []
        non_proxy_items = []
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if isinstance(memory_obj, ProxyMemoryObj):
                transfer_contexts.add(memory_obj.transfer_context)
                proxy_items.append((memory_obj, start, end))
            else:
                non_proxy_items.append((memory_obj, start, end))

        if proxy_items:
            # Get the transfer context for buffer allocation
            first_ctx = proxy_items[0][0].transfer_context

            # Derive pipeline depth from NPU buffer capacity so that
            # two full ping-pong pools fit in registered memory.
            pipeline_depth = first_ctx.max_pipeline_depth
            logger.debug(
                "P2P pipeline depth = %d (proxy_items=%d)",
                pipeline_depth,
                len(proxy_items),
            )

            # Allocate ping-pong buffer pools.
            # Initialized to None so the finally block can safely skip
            # release if allocation itself fails.
            pool_size = min(pipeline_depth, len(proxy_items))
            pool_a = None
            pool_b = None

            try:
                pool_a = first_ctx.allocate_buffers(pool_size)
                pool_b = first_ctx.allocate_buffers(pool_size)

                pools = [pool_a, pool_b]
                current_pool = 0

                # Group proxy items into micro-batches
                micro_batches = [
                    proxy_items[i : i + pipeline_depth]
                    for i in range(0, len(proxy_items), pipeline_depth)
                ]

                prev_read_event = None
                prev_batch = None

                # Per-pool scatter-done events: prevent the next RDMA
                # write into a pool from racing with a scatter that is
                # still reading from the same pool on load_stream.
                # Events are pre-allocated and re-recorded each iteration.
                channel = proxy_items[0][0]._transfer_channel
                transport_stream = getattr(channel, "transport_stream", None)
                pool_scatter_events = [
                    torch.npu.Event(),
                    torch.npu.Event(),
                ]
                pool_scatter_recorded = [False, False]

                for batch_idx, batch in enumerate(micro_batches):
                    pool = pools[current_pool]

                    # Ensure the previous scatter from this pool has
                    # finished before RDMA overwrites the pool buffers.
                    if (
                        pool_scatter_recorded[current_pool]
                        and transport_stream is not None
                    ):
                        transport_stream.wait_event(pool_scatter_events[current_pool])

                    # Assign backing buffers from current pool to proxies
                    for i, (proxy, _, _) in enumerate(batch):
                        proxy.set_backing_obj(pool[i])

                    proxies = [item[0] for item in batch]

                    # Submit RDMA read for current batch -> transport_stream.
                    cur_read_event = ProxyMemoryObj.submit_resolve_batch(proxies)

                    # While the current batch is being read on
                    # transport_stream, scatter the previous batch on
                    # load_stream (waits for its RDMA read event).
                    if prev_batch is not None:
                        self._scatter_proxy_batch(
                            prev_batch,
                            prev_read_event,
                            **kwargs,
                        )
                        pool_scatter_events[1 - current_pool].record(self.load_stream)
                        pool_scatter_recorded[1 - current_pool] = True
                        self._clear_proxy_batch(prev_batch)

                    prev_read_event = cur_read_event
                    prev_batch = batch
                    current_pool = 1 - current_pool  # toggle ping-pong

                # Drain: scatter the last micro-batch.
                if prev_batch is not None:
                    self._scatter_proxy_batch(
                        prev_batch,
                        prev_read_event,
                        **kwargs,
                    )
                    self._clear_proxy_batch(prev_batch)
            finally:
                # Guarantee ping-pong buffers are returned and the Done
                # signal is sent even if the pipeline raises or
                # allocate_buffers itself fails.  Without this, an
                # exception would leak NPU pages and leave the sender's
                # pinned resources stuck until its TTL expires.
                self.load_stream.synchronize()
                if pool_a is not None:
                    first_ctx.release_buffers(pool_a)
                if pool_b is not None:
                    first_ctx.release_buffers(pool_b)

                for proxy, _, _ in proxy_items:
                    proxy.mark_consumed()

                for ctx in transfer_contexts:
                    ctx.send_done_now()

        # Process non-proxy items on load_stream (no pipelining needed)
        if non_proxy_items:
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in non_proxy_items:
                    self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        # NOTE (gingfung):
        # Since no_sync is only consumed by us, for now we modify the kwargs directly.
        # We avoid per-object synchronization during batch transfers.
        # A single synchronization is performed at the end of the batch.
        kwargs["no_sync"] = True

        ordering_event = kwargs.pop("ordering_event", None)
        current_stream = torch.npu.current_stream()
        if ordering_event is not None:
            self.store_stream.wait_event(ordering_event)
        else:
            self.store_stream.wait_stream(current_stream)

        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if is_310p():
                self.from_gpu_310p(memory_obj, start, end, **kwargs)
            else:
                self.from_gpu(memory_obj, start, end, **kwargs)
        self.store_stream.synchronize()

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.kv_format == KVCacheFormat.MLA_KV:
            total_hidden_dims = self.kv_lora_rank + self.qk_rope_head_dim
            return torch.Size([1, self.num_layers, num_tokens, total_hidden_dims])
        elif self.kv_format == KVCacheFormat.DSA_KV:
            total_hidden_dims = (
                self.kv_lora_rank + self.qk_rope_head_dim + self.dsa_head_dim
            )
            return torch.Size([1, self.num_layers, num_tokens, total_hidden_dims])
        else:
            kv_size = 2
            return torch.Size(
                [kv_size, self.num_layers, num_tokens, self.hidden_dim_size]
            )


class _GroupLayout:
    """Per-kv_group KV layout state for the layerwise NPU connector.

    In two-group MLA+DSA mode (``dsa_two_groups``) a single connector
    instance serves both ``kv_group=0`` (latent / MLA_LATENT) and
    ``kv_group=1`` (indexer / DSA_INDEX). The two groups have different
    formats, hidden dims and staging-buffer sizes, so each group keeps its
    own layout entry instead of sharing one set of fields on the connector.
    """

    __slots__ = (
        "kv_format",
        "k_hidden_dims",
        "v_hidden_dims",
        "dsa_hidden_dims",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "dsa_head_dim",
        "vllm_two_major",
        "kv_device",
        "gpu_buffer_allocator",
        "staging_bytes_per_slot",
    )

    def __init__(self) -> None:
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.k_hidden_dims: int = 0
        self.v_hidden_dims: int = 0
        self.dsa_hidden_dims: int = 0
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.vllm_two_major: bool = False
        self.kv_device: Optional[torch.device] = None
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None
        self.staging_bytes_per_slot: int = 0


class _SparseDestinationPlan:
    """Process-owned native states for one paged-KV destination group."""

    __slots__ = ("kvcaches_ref", "signature", "states")

    def __init__(
        self,
        kvcaches_ref: list,
        signature: tuple,
        states: tuple[Any, ...],
    ) -> None:
        self.kvcaches_ref = kvcaches_ref
        self.signature = signature
        self.states = states


class VLLMPagedMemLayerwiseNPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.load_stream_num = 4
        self.load_stream_list = [
            torch.cuda.Stream() for __ in range(self.load_stream_num)
        ]
        self.load_stream_idx = 0

        self.lmcache_chunk_size = int(kwargs.get("chunk_size", 0))
        self.dsa_two_groups = kwargs.get("dsa_two_groups", False)
        self.max_staging_tokens = int(kwargs.get("max_staging_tokens", 0) or 0)
        # Concurrent layerwise staging buffers per kv_group (retrieve batch +
        # overlapping store). Default 2 covers retrieve+store for one request.
        self._layerwise_staging_concurrency = 2 if self.dsa_two_groups else 1

        # Per-kv_group layout state. Detection and the GPU staging buffer
        # are initialized lazily per group by _lazy_initialize_buffer, so
        # both kv_group=0 (MLA_LATENT) and kv_group=1 (DSA_INDEX) get their
        # own format/dims instead of the first group pinning a single
        # self.kv_format. _current_kv_group selects the group being served;
        # the mirrored instance attributes below are kept in sync for
        # backward-compatible readers (benchmarks, logs, external callers).
        self._group_layouts: dict[int, _GroupLayout] = {}
        self._current_kv_group: int = 0

        # Mirrored attributes for the current group (updated by
        # _lazy_initialize_buffer). The layerwise hot paths snapshot values
        # from the returned _GroupLayout into locals to remain safe even if
        # per-group generators interleave.
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.k_hidden_dims: int = 0
        self.v_hidden_dims: int = 0
        self.dsa_hidden_dims: int = 0
        self.vllm_two_major: bool = False
        self.kv_device: Optional[torch.device] = None

        self._layerwise_sparse_idx_cache: Optional[torch.Tensor] = None
        # Legacy direct/dense state remains source-layout specialized. The
        # prepared sparse warm path uses destination-only plans below.
        self._sparse_direct_layer_states: Optional[dict] = None
        self._sparse_direct_validated_layers: set = set()
        # One process-owned destination plan per latent/indexer KV group.
        self._sparse_destination_plans: dict[int, _SparseDestinationPlan] = {}

    def _reset_sparse_direct_layer_states(self) -> None:
        self._sparse_direct_layer_states = None
        self._sparse_direct_validated_layers = set()

    def synchronize_shared_cpu_store_publication(self) -> None:
        """Complete this rank's store work before publishing shared handles."""
        self.store_stream.synchronize()

    def _mirror_layout(self, layout: _GroupLayout) -> None:
        """Mirror a group's layout into the instance attributes."""
        self.kv_format = layout.kv_format
        self.k_hidden_dims = layout.k_hidden_dims
        self.v_hidden_dims = layout.v_hidden_dims
        self.dsa_hidden_dims = layout.dsa_hidden_dims
        self.kv_lora_rank = layout.kv_lora_rank
        self.qk_rope_head_dim = layout.qk_rope_head_dim
        self.dsa_head_dim = layout.dsa_head_dim
        self.vllm_two_major = layout.vllm_two_major
        self.kv_device = layout.kv_device
        self.gpu_buffer_allocator = layout.gpu_buffer_allocator

    def _lazy_initialize_buffer_with_staging(
        self,
        kv_caches,
        *,
        kv_group: int,
        init_staging: bool,
    ) -> _GroupLayout:
        return self._lazy_initialize_buffer(
            kv_caches,
            kv_group=kv_group,
            init_staging=init_staging,
        )

    def _sparse_total_tokens_from_layer_chunks(
        self,
        layer_tensors: List[torch.Tensor],
        kv_group: Optional[int] = None,
    ) -> int:
        num_chunks = len(layer_tensors)
        if num_chunks == 0:
            return 0
        if num_chunks == 1:
            return self._lmc_plane_num_tokens(layer_tensors[0], kv_group)
        last_tokens = self._lmc_plane_num_tokens(layer_tensors[-1], kv_group)
        return (num_chunks - 1) * self.lmcache_chunk_size + last_tokens

    def append_sparse_chunk_ptr_cache_for_layer(
        self,
        layer_id: int,
        new_tensors: List[torch.Tensor],
        cached_chunk_dev_ptrs: List[List[int]],
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]],
    ) -> dict[str, float]:
        """Resolve and append NPU device ptrs for newly retrieved chunks only.

        Return non-overlapping diagnostic timings without changing the cache
        installation order or its all-or-nothing behavior.
        """
        timings_ms = {
            "ptr_resolve": 0.0,
            "ptr_tensor_build": 0.0,
            "ptr_tensor_concat": 0.0,
            "ptr_cache_commit": 0.0,
        }
        if not new_tensors:
            return timings_ms

        phase_started = time.perf_counter()
        new_dev_ptrs = [
            self._resolve_registered_cpu_tensor_device_ptr(
                tensor,
                layer_id=layer_id,
                chunk_index=chunk_index,
                source="append_sparse_chunk_ptr_cache_for_layer",
            )
            for chunk_index, tensor in enumerate(new_tensors)
        ]
        timings_ms["ptr_resolve"] = (
            time.perf_counter() - phase_started
        ) * 1000

        updated_ptrs_npu = None
        if cached_chunk_ptrs_npu is not None:
            phase_started = time.perf_counter()
            new_ptrs_npu = torch.tensor(
                new_dev_ptrs, dtype=torch.long, device=self.kv_device
            )
            timings_ms["ptr_tensor_build"] = (
                time.perf_counter() - phase_started
            ) * 1000
            existing = (
                cached_chunk_ptrs_npu[layer_id]
                if layer_id < len(cached_chunk_ptrs_npu)
                else None
            )
            phase_started = time.perf_counter()
            updated_ptrs_npu = (
                new_ptrs_npu
                if existing is None
                else torch.cat((existing, new_ptrs_npu), dim=0)
            )
            timings_ms["ptr_tensor_concat"] = (
                time.perf_counter() - phase_started
            ) * 1000

        phase_started = time.perf_counter()
        num_layers = self.num_layers
        if not cached_chunk_dev_ptrs:
            cached_chunk_dev_ptrs.extend([] for _ in range(num_layers))
        while len(cached_chunk_dev_ptrs) <= layer_id:
            cached_chunk_dev_ptrs.append([])

        if cached_chunk_ptrs_npu is not None and not cached_chunk_ptrs_npu:
            cached_chunk_ptrs_npu.extend(None for _ in range(num_layers))
        while (
            cached_chunk_ptrs_npu is not None and len(cached_chunk_ptrs_npu) <= layer_id
        ):
            cached_chunk_ptrs_npu.append(None)

        cached_chunk_dev_ptrs[layer_id].extend(new_dev_ptrs)

        if cached_chunk_ptrs_npu is not None:
            cached_chunk_ptrs_npu[layer_id] = updated_ptrs_npu
        timings_ms["ptr_cache_commit"] = (
            time.perf_counter() - phase_started
        ) * 1000
        return timings_ms

    def _resolve_registered_cpu_tensor_device_ptr(
        self,
        tensor: torch.Tensor,
        *,
        layer_id: int,
        chunk_index: int,
        source: str,
    ) -> int:
        host_ptr = int(tensor.data_ptr())
        dev_ptr = lmc_ops.get_device_ptr(host_ptr)
        if dev_ptr is None or int(dev_ptr) == 0:
            raise RuntimeError(
                "Ascend sparse pointer-cache install failed: CPU tensor is not "
                "registered or get_device_ptr returned null. "
                f"source={source}, layer_id={layer_id}, "
                f"chunk_index={chunk_index}, host_ptr={host_ptr}"
            )
        return int(dev_ptr)

    @staticmethod
    def _sparse_direct_source_signature(
        *,
        layer_tensors: List[torch.Tensor],
        slot_mapping_ref: torch.Tensor,
        total_tokens: int,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
    ) -> tuple:
        """Metadata that must match to reuse SparseDirectLayerState."""
        return (
            tuple(
                VLLMPagedMemLayerwiseNPUConnector._tensor_layout_signature(tensor)
                for tensor in layer_tensors
            ),
            int(slot_mapping_ref.numel()),
            slot_mapping_ref.dtype,
            str(slot_mapping_ref.device),
            int(total_tokens),
            int(sparse_kv_format),
            bool(sparse_token_major),
            bool(sparse_vllm_two_major),
            int(sparse_k_hidden_dims),
            int(sparse_v_hidden_dims),
            int(sparse_dsa_hidden_dims),
        )

    @staticmethod
    def _stream_context_or_null(stream):
        if stream is None or not hasattr(stream, "device"):
            return nullcontext()
        stream_device = getattr(stream, "device", None)
        if getattr(stream_device, "type", None) == "npu" and hasattr(torch, "npu"):
            return torch.npu.stream(stream)
        return torch.cuda.stream(stream)

    @staticmethod
    def _sparse_direct_pointer_cache_signature(
        *,
        chunk_ptrs_npu: torch.Tensor,
        slot_mapping_ref: torch.Tensor,
        source_layout_ref: Optional[torch.Tensor],
        total_tokens: int,
        chunk_size: int,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
    ) -> tuple:
        """Cheap runtime source identity for the decode hot path.

        The native fast path overwrites slot_mapping_ptr and consumes
        chunk_ptrs_npu at launch time, so the cached layer state only needs the
        stable source layout and scalar metadata. Avoid keying on per-request
        tensor addresses that would force prepare_sparse_direct_layer_state()
        to run again on every decode step.
        """
        return (
            2,
            VLLMPagedMemLayerwiseNPUConnector._tensor_layout_signature(
                source_layout_ref
            ),
            chunk_ptrs_npu.dtype,
            str(chunk_ptrs_npu.device),
            int(chunk_ptrs_npu.numel()),
            slot_mapping_ref.dtype,
            str(slot_mapping_ref.device),
            int(slot_mapping_ref.numel()),
            int(total_tokens),
            int(chunk_size),
            int(sparse_kv_format),
            int(sparse_token_major),
            int(sparse_vllm_two_major),
            int(sparse_k_hidden_dims),
            int(sparse_v_hidden_dims),
            int(sparse_dsa_hidden_dims),
        )

    @staticmethod
    def _tensor_layout_signature(tensor) -> tuple:
        if isinstance(tensor, torch.Tensor):
            return (
                tuple(int(dim) for dim in tensor.shape),
                tuple(int(stride) for stride in tensor.stride()),
                tensor.dtype,
                str(tensor.device),
                int(tensor.element_size()),
                int(tensor.numel() * tensor.element_size()),
            )
        return (type(tensor).__name__,)

    @staticmethod
    def _vllm_layer_cache_identity_signature(value) -> tuple:
        if isinstance(value, (tuple, list)):
            return (id(value), len(value))
        return (id(value),)

    def _get_or_create_sparse_destination_plan(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        slot_mapping_ref: torch.Tensor,
        sparse_kv_format: int,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
        expected_device: Optional[torch.device],
    ) -> _SparseDestinationPlan:
        """Resolve process-invariant native states for one destination group."""
        if expected_device is None:
            raise RuntimeError(f"kv_group={kv_group} has no initialized NPU device")
        if len(kvcaches_ref) != self.num_layers:
            raise ValueError(
                "Prepared sparse destination has the wrong layer count: "
                f"kvcaches={len(kvcaches_ref)}, connector={self.num_layers}"
            )
        if slot_mapping_ref.device != expected_device:
            raise ValueError(
                "Prepared sparse slot mapping is on the wrong device: "
                f"device={slot_mapping_ref.device}, expected={expected_device}"
            )

        signature = (
            slot_mapping_ref.dtype,
            str(slot_mapping_ref.device),
            int(sparse_kv_format),
            int(sparse_k_hidden_dims),
            int(sparse_v_hidden_dims),
            int(sparse_dsa_hidden_dims),
        )
        plans = getattr(self, "_sparse_destination_plans", None)
        if plans is None:
            plans = {}
            self._sparse_destination_plans = plans
        plan = plans.pop(kv_group, None)
        if (
            plan is not None
            and plan.kvcaches_ref is kvcaches_ref
            and plan.signature == signature
        ):
            plans[kv_group] = plan
            return plan

        states = tuple(
            prepare_sparse_direct_destination_state(
                kvcaches_ref[layer_id],
                slot_mapping_ref,
                sparse_kv_format,
                sparse_k_hidden_dims,
                sparse_v_hidden_dims,
                sparse_dsa_hidden_dims,
            )
            for layer_id in range(self.num_layers)
        )
        plan = _SparseDestinationPlan(kvcaches_ref, signature, states)
        plans[kv_group] = plan
        while len(plans) > _SPARSE_DESTINATION_PLAN_CACHE_SIZE:
            del plans[next(iter(plans))]
        return plan

    def _sparse_direct_state_key(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        layer_tensors: List[torch.Tensor],
        slot_mapping_ref: torch.Tensor,
        total_tokens: int,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
        source_signature: Optional[tuple] = None,
    ) -> Optional[tuple]:
        if kvcaches_ref is None or not layer_tensors:
            return None
        if total_tokens <= 0:
            total_tokens = self._sparse_total_tokens_from_layer_chunks(
                layer_tensors, kv_group
            )
        if source_signature is None:
            source_signature = self._sparse_direct_source_signature(
                layer_tensors=layer_tensors,
                slot_mapping_ref=slot_mapping_ref,
                total_tokens=total_tokens,
                sparse_kv_format=sparse_kv_format,
                sparse_token_major=sparse_token_major,
                sparse_vllm_two_major=sparse_vllm_two_major,
                sparse_k_hidden_dims=sparse_k_hidden_dims,
                sparse_v_hidden_dims=sparse_v_hidden_dims,
                sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
            )
        vllm_signature = self._vllm_layer_cache_identity_signature(
            kvcaches_ref[layer_id]
        )
        return (
            id(kvcaches_ref),
            kv_group,
            layer_id,
            vllm_signature,
            source_signature,
        )

    def _get_or_create_sparse_direct_layer_state(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        layer_tensors: List[torch.Tensor],
        slot_mapping_ref: torch.Tensor,
        total_tokens: int,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
        source_signature: Optional[tuple] = None,
        return_key: bool = False,
    ):
        if kvcaches_ref is None:
            return (None, None) if return_key else None

        if self._sparse_direct_layer_states is None:
            self._sparse_direct_layer_states = {}

        if not layer_tensors:
            return (None, None) if return_key else None

        if total_tokens <= 0:
            total_tokens = self._sparse_total_tokens_from_layer_chunks(
                layer_tensors, kv_group
            )

        state_key = self._sparse_direct_state_key(
            kvcaches_ref=kvcaches_ref,
            kv_group=kv_group,
            layer_id=layer_id,
            layer_tensors=layer_tensors,
            slot_mapping_ref=slot_mapping_ref,
            total_tokens=total_tokens,
            sparse_kv_format=sparse_kv_format,
            sparse_token_major=sparse_token_major,
            sparse_vllm_two_major=sparse_vllm_two_major,
            sparse_k_hidden_dims=sparse_k_hidden_dims,
            sparse_v_hidden_dims=sparse_v_hidden_dims,
            sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
            source_signature=source_signature,
        )
        assert state_key is not None
        state = self._sparse_direct_layer_states.get(state_key)
        if state is not None:
            return (state, state_key) if return_key else state

        state = prepare_sparse_direct_layer_state(
            layer_tensors[0],
            kvcaches_ref[layer_id],
            slot_mapping_ref,
            sparse_token_major,
            sparse_vllm_two_major,
            sparse_kv_format,
            sparse_k_hidden_dims,
            sparse_v_hidden_dims,
            sparse_dsa_hidden_dims,
            total_tokens,
        )
        self._sparse_direct_layer_states[state_key] = state
        return (state, state_key) if return_key else state

    @staticmethod
    def _unpack_sparse_dynamic_request(
        sparse_request: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Normalize the vLLM-owned portion of one sparse layer payload."""
        target_slot_mapping = None
        payload_event = None
        if isinstance(sparse_request, dict):
            selected_token_idx = sparse_request.get("selected_token_ids")
            token_start_index = sparse_request.get("token_start_index", 0)
            target_slot_mapping = sparse_request.get("target_slot_mapping")
            payload_event = sparse_request.get(
                "payload_events", sparse_request.get("payload_event")
            )
        elif isinstance(sparse_request, tuple):
            if len(sparse_request) == 3:
                (
                    selected_token_idx,
                    token_start_index,
                    target_slot_mapping,
                ) = sparse_request
            elif len(sparse_request) == 2:
                selected_token_idx, token_start_index = sparse_request
            else:
                raise ValueError("Sparse payload tuple must have 2 or 3 items")
        else:
            selected_token_idx = sparse_request
            token_start_index = 0
        return (
            selected_token_idx,
            token_start_index,
            target_slot_mapping,
            payload_event,
        )

    def _pack_sparse_layer_inputs(
        self,
        slot_mapping: torch.Tensor,
        selected_token_idx: Optional[Union[torch.Tensor, list]],
        token_start_index: int,
        target_slot_mapping: Optional[Union[torch.Tensor, list]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build parallel destination/source arrays for the sparse copy kernel."""
        if selected_token_idx is not None and not isinstance(
            selected_token_idx, torch.Tensor
        ):
            selected_token_idx = torch.tensor(
                selected_token_idx, dtype=torch.int32, device=self.kv_device
            )

        if target_slot_mapping is not None:
            return self._pack_sparse_explicit_slot_inputs(
                selected_token_idx,
                target_slot_mapping,
            )

        if selected_token_idx is not None and selected_token_idx.numel() > 0:
            if selected_token_idx.dim() > 1:
                rows = selected_token_idx.reshape(selected_token_idx.shape[0], -1)
                starts = token_start_index
                start_values = None
                if isinstance(starts, int):
                    start_values = [int(starts)] * int(rows.shape[0])
                elif isinstance(starts, (list, tuple)):
                    if len(starts) == 1 and rows.shape[0] != 1:
                        start_values = [int(starts[0])] * int(rows.shape[0])
                    else:
                        start_values = [int(start) for start in starts]
                elif isinstance(starts, torch.Tensor) and starts.device.type == "cpu":
                    starts = starts.reshape(-1).to(dtype=torch.long)
                    if starts.numel() == 1 and rows.shape[0] != 1:
                        starts = starts.expand(rows.shape[0])
                    start_values = [int(start) for start in starts.tolist()]
                else:
                    if not isinstance(starts, torch.Tensor):
                        starts = torch.tensor(
                            starts,
                            dtype=torch.long,
                            device=slot_mapping.device,
                        )
                    starts = starts.reshape(-1).to(
                        device=slot_mapping.device, dtype=torch.long
                    )
                    if starts.numel() == 1 and rows.shape[0] != 1:
                        starts = starts.expand(rows.shape[0])
                num_starts = (
                    len(start_values)
                    if start_values is not None
                    else int(starts.numel())
                )
                if num_starts != int(rows.shape[0]):
                    raise ValueError(
                        "token_start_index rows must match selected_token_idx rows: "
                        f"{num_starts} vs {rows.shape[0]}"
                    )
                if start_values is None:
                    row_width = int(rows.shape[1])
                    row_offsets = torch.arange(
                        row_width,
                        dtype=torch.long,
                        device=slot_mapping.device,
                    )
                    gather_indices = (
                        starts.reshape(-1, 1) + row_offsets.reshape(1, -1)
                    ).reshape(-1)
                    slot_mapping_packed = slot_mapping[gather_indices]
                    selected_token_idx = rows.reshape(-1)
                    selected_token_idx = self._sparse_selected_token_idx(
                        selected_token_idx, slot_mapping_packed.shape[0]
                    )
                    return slot_mapping_packed, selected_token_idx

                slot_chunks = []
                selected_chunks = []
                for row_idx in range(rows.shape[0]):
                    row = rows[row_idx]
                    start = start_values[row_idx]
                    end = start + int(row.numel())
                    if end > int(slot_mapping.numel()):
                        raise ValueError(
                            "sparse slot_mapping too short for multi-row selected "
                            f"tokens: start={start} end={end} "
                            f"slot_mapping={slot_mapping.numel()}"
                        )
                    slot_chunks.append(slot_mapping[start:end])
                    selected_chunks.append(row)
                slot_mapping_packed = (
                    torch.cat(slot_chunks, dim=0) if slot_chunks else slot_mapping[:0]
                )
                selected_token_idx = (
                    torch.cat(selected_chunks, dim=0)
                    if selected_chunks
                    else selected_token_idx.reshape(-1)[:0]
                )
                selected_token_idx = self._sparse_selected_token_idx(
                    selected_token_idx, slot_mapping_packed.shape[0]
                )
                return slot_mapping_packed, selected_token_idx

            num_sparse = int(selected_token_idx.numel())
            start = int(token_start_index)
            end = start + num_sparse
            if end <= slot_mapping.numel():
                slot_mapping_packed = slot_mapping[start:end]
            elif start < slot_mapping.numel():
                slot_mapping_packed = slot_mapping[start:]
                selected_token_idx = selected_token_idx[: slot_mapping_packed.numel()]
            else:
                slot_mapping_packed = slot_mapping[:0]
                selected_token_idx = selected_token_idx[:0]
            selected_token_idx = self._sparse_selected_token_idx(
                selected_token_idx, slot_mapping_packed.shape[0]
            )
            return slot_mapping_packed, selected_token_idx

        slot_mapping_packed = (
            slot_mapping
            if token_start_index == 0
            else slot_mapping[int(token_start_index) :]
        )
        selected_token_idx = self._sparse_selected_token_idx(
            None, slot_mapping_packed.shape[0]
        )
        return slot_mapping_packed, selected_token_idx

    def _pack_sparse_explicit_slot_inputs(
        self,
        selected_token_idx: Optional[Union[torch.Tensor, list]],
        target_slot_mapping: Union[torch.Tensor, list],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use caller-provided target slots for row-wise MTP sparse loads."""
        if selected_token_idx is None:
            selected_token_idx = []
        if not isinstance(selected_token_idx, torch.Tensor):
            selected_token_idx = torch.tensor(
                selected_token_idx, dtype=torch.int32, device=self.kv_device
            )
        selected_token_idx = selected_token_idx.reshape(-1)

        if not isinstance(target_slot_mapping, torch.Tensor):
            target_slot_mapping = torch.tensor(
                target_slot_mapping, dtype=torch.long, device=self.kv_device
            )
        target_slot_mapping = target_slot_mapping.reshape(-1)
        if (
            target_slot_mapping.dtype != torch.long
            or target_slot_mapping.device != self.kv_device
        ):
            target_slot_mapping = target_slot_mapping.to(
                device=self.kv_device, dtype=torch.long
            )
        if int(target_slot_mapping.numel()) != int(selected_token_idx.numel()):
            raise ValueError(
                "target_slot_mapping and selected_token_idx must have the same "
                f"length: {target_slot_mapping.numel()} vs {selected_token_idx.numel()}"
            )
        selected_token_idx = self._sparse_selected_token_idx(
            selected_token_idx, target_slot_mapping.shape[0]
        )
        return target_slot_mapping, selected_token_idx

    def _run_sparse_direct_kv_transfer_layer(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        load_stream: torch.cuda.Stream,
        current_stream: torch.cuda.Stream,
        slot_mapping_packed: torch.Tensor,
        selected_token_idx: torch.Tensor,
        chunk_size: int,
        total_tokens: int,
        chunk_ptrs_npu: torch.Tensor,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
        sparse_host_interleaved: bool,
        layer_tensors: Optional[List[torch.Tensor]] = None,
        slot_mapping_ref: Optional[torch.Tensor] = None,
        cpu_tensors: Optional[List[torch.Tensor]] = None,
    ) -> None:
        num_sparse = int(selected_token_idx.numel())
        if num_sparse == 0 or total_tokens <= 0 or chunk_ptrs_npu.numel() == 0:
            return
        chunk_count = int(chunk_ptrs_npu.numel())
        chunk_size_int = int(chunk_size)
        covered_tokens = chunk_count * chunk_size_int
        if covered_tokens < int(total_tokens):
            message = (
                "Sparse direct retrieve has insufficient chunk pointers: "
                f"kv_group={kv_group} layer_id={layer_id} "
                f"num_sparse={num_sparse} chunk_count={chunk_count} "
                f"chunk_size={chunk_size_int} covered_tokens={covered_tokens} "
                f"total_tokens={int(total_tokens)}"
            )
            logger.error(message)
            raise ValueError(message)
        resolve_tensors = (
            layer_tensors
            if layer_tensors is not None
            else (cpu_tensors if cpu_tensors is not None else [])
        )
        resolve_slot_mapping = (
            slot_mapping_ref if slot_mapping_ref is not None else slot_mapping_packed
        )
        runtime_source_signature = self._sparse_direct_pointer_cache_signature(
            chunk_ptrs_npu=chunk_ptrs_npu,
            slot_mapping_ref=resolve_slot_mapping,
            source_layout_ref=resolve_tensors[0] if resolve_tensors else None,
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            sparse_kv_format=sparse_kv_format,
            sparse_token_major=sparse_token_major,
            sparse_vllm_two_major=sparse_vllm_two_major,
            sparse_k_hidden_dims=sparse_k_hidden_dims,
            sparse_v_hidden_dims=sparse_v_hidden_dims,
            sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
        )

        layer_state, validate_key = self._get_or_create_sparse_direct_layer_state(
            kvcaches_ref=kvcaches_ref,
            kv_group=kv_group,
            layer_id=layer_id,
            layer_tensors=resolve_tensors,
            slot_mapping_ref=resolve_slot_mapping,
            total_tokens=total_tokens,
            sparse_kv_format=sparse_kv_format,
            sparse_token_major=sparse_token_major,
            sparse_vllm_two_major=sparse_vllm_two_major,
            sparse_k_hidden_dims=sparse_k_hidden_dims,
            sparse_v_hidden_dims=sparse_v_hidden_dims,
            sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
            source_signature=runtime_source_signature,
            return_key=True,
        )
        if validate_key is None:
            validate_key = (kv_group, layer_id)
        with torch.cuda.stream(load_stream):
            load_stream.wait_stream(current_stream)
            if layer_state is not None:
                validate_inputs = (
                    validate_key not in self._sparse_direct_validated_layers
                )
                sparse_mla_dsa_batched_direct_kv_transfer_fast(
                    layer_state,
                    slot_mapping_packed,
                    selected_token_idx,
                    chunk_ptrs_npu,
                    chunk_size,
                    total_tokens,
                    sparse_host_interleaved,
                    validate_inputs,
                )
                if validate_inputs:
                    self._sparse_direct_validated_layers.add(validate_key)
            else:
                assert cpu_tensors is not None and len(cpu_tensors) > 0
                sparse_mla_dsa_batched_direct_kv_transfer(
                    cpu_tensors,
                    kvcaches_ref[layer_id],
                    slot_mapping_packed,
                    selected_token_idx,
                    chunk_size,
                    total_tokens,
                    sparse_kv_format,
                    sparse_token_major,
                    sparse_vllm_two_major,
                    sparse_k_hidden_dims,
                    sparse_v_hidden_dims,
                    sparse_dsa_hidden_dims,
                    sparse_host_interleaved,
                    chunk_ptrs_npu,
                )

        current_stream.wait_stream(load_stream)

    def _run_prepared_sparse_direct_kv_transfer_layer(
        self,
        *,
        plan: _SparseDestinationPlan,
        source_layer: PreparedSparseSourceLayer,
        layer_id: int,
        load_stream: torch.cuda.Stream,
        current_stream: torch.cuda.Stream,
        slot_mapping_packed: torch.Tensor,
        selected_token_idx: torch.Tensor,
        chunk_size: int,
        total_tokens: int,
        sparse_host_interleaved: bool,
    ) -> None:
        """Launch one prepared layer without source or native-state lookup."""
        if selected_token_idx.numel() == 0:
            return
        chunk_ptrs_npu = source_layer.chunk_ptrs_npu

        with torch.cuda.stream(load_stream):
            load_stream.wait_stream(current_stream)
            sparse_mla_dsa_batched_direct_kv_transfer_prepared(
                plan.states[layer_id],
                slot_mapping_packed,
                selected_token_idx,
                chunk_ptrs_npu,
                chunk_size,
                total_tokens,
                sparse_host_interleaved,
            )

        current_stream.wait_stream(load_stream)

    def _sparse_selected_token_idx(
        self,
        selected_token_idx: Optional[torch.Tensor],
        num_sparse: int,
    ) -> torch.Tensor:
        if selected_token_idx is None:
            cached = self._layerwise_sparse_idx_cache
            if cached is None or cached.shape[0] != num_sparse:
                cached = torch.arange(
                    num_sparse, dtype=torch.int32, device=self.kv_device
                )
                self._layerwise_sparse_idx_cache = cached
            return cached
        if (
            selected_token_idx.dtype == torch.int32
            and selected_token_idx.device == self.kv_device
        ):
            return selected_token_idx
        return selected_token_idx.to(device=self.kv_device, dtype=torch.int32)

    def _is_mla_dsa_format(self, kv_group: Optional[int] = None) -> bool:
        fmt = self._fmt_for(kv_group)
        return fmt in (
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.MLA_LATENT,
            KVCacheFormat.DSA_INDEX,
        )

    def _layerwise_token_major(self, kv_group: Optional[int] = None) -> bool:
        # GQA uses token-interleaved CPU chunks; MLA/DSA use stacked K|V|DSA planes.
        return not self._is_mla_dsa_format(kv_group)

    def _sparse_lmc_host_interleaved(self, kv_group: Optional[int] = None) -> bool:
        # Must match batched_fused CPU layout (_layerwise_token_major).
        return self._layerwise_token_major(kv_group)

    def notify_sparse_memory_objs_updated(self) -> None:
        """Reset fast-path state after sparse source MemoryObjs change.

        The legacy path caches layout derived from a sample source tensor. The
        prepared path is not reset here because its native plan contains only
        process-owned destination state and consumes request pointers dynamically.
        """
        self._reset_sparse_direct_layer_states()

    def _resolve_sparse_chunk_ptrs_npu(
        self,
        layer_id: int,
        cpu_tensors: List[torch.Tensor],
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_chunks = len(cpu_tensors)
        if cached_chunk_ptrs_npu is not None and layer_id < len(cached_chunk_ptrs_npu):
            cached = cached_chunk_ptrs_npu[layer_id]
            if cached is not None and cached.numel() == num_chunks:
                if cached.dtype != torch.long:
                    raise RuntimeError(
                        "Ascend sparse pointer-cache reuse failed: cached NPU "
                        "pointer tensor has invalid dtype before direct kernel "
                        f"launch at layer {layer_id}: dtype={cached.dtype}, "
                        "expected=torch.int64."
                    )
                expected_device = torch.device(self.kv_device)
                if cached.device != expected_device:
                    raise RuntimeError(
                        "Ascend sparse pointer-cache reuse failed: cached NPU "
                        "pointer tensor is on the wrong device before direct "
                        f"kernel launch at layer {layer_id}: "
                        f"device={cached.device}, expected={expected_device}."
                    )
                return cached

        dev_ptrs = [
            self._resolve_registered_cpu_tensor_device_ptr(
                tensor,
                layer_id=layer_id,
                chunk_index=chunk_index,
                source="_resolve_sparse_chunk_ptrs_npu",
            )
            for chunk_index, tensor in enumerate(cpu_tensors)
        ]
        chunk_ptrs_npu = torch.tensor(dev_ptrs, dtype=torch.long, device=self.kv_device)
        if cached_chunk_ptrs_npu is not None:
            while len(cached_chunk_ptrs_npu) <= layer_id:
                cached_chunk_ptrs_npu.append(None)
            cached_chunk_ptrs_npu[layer_id] = chunk_ptrs_npu
        return chunk_ptrs_npu

    def _prepare_dense_direct_chunk_metadata(
        self,
        chunk_offsets: List[int],
        chunk_sizes: List[int],
        *,
        total_tokens: int,
        kv_group: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if not chunk_offsets or not chunk_sizes:
            raise ValueError(
                "Dense direct layerwise transfer requires at least one CPU chunk."
            )
        if len(chunk_offsets) != len(chunk_sizes):
            raise ValueError(
                "Dense direct layerwise transfer chunk metadata mismatch: "
                f"offsets={len(chunk_offsets)} sizes={len(chunk_sizes)}"
            )
        if chunk_offsets[0] != 0:
            raise ValueError(
                "Dense direct layerwise transfer chunk offsets must start at 0: "
                f"kv_group={kv_group} first_offset={chunk_offsets[0]}"
            )
        expected_offset = 0
        fixed_chunk_size = int(chunk_sizes[0])
        fixed_layout = fixed_chunk_size > 0
        last_chunk_idx = len(chunk_sizes) - 1
        for chunk_idx, (offset, size) in enumerate(
            zip(chunk_offsets, chunk_sizes, strict=False)
        ):
            offset = int(offset)
            size = int(size)
            if size <= 0:
                raise ValueError(
                    "Dense direct layerwise transfer chunk size must be positive: "
                    f"kv_group={kv_group} chunk_idx={chunk_idx} size={size}"
                )
            if offset != expected_offset:
                raise ValueError(
                    "Dense direct layerwise transfer chunks must be contiguous: "
                    f"kv_group={kv_group} chunk_idx={chunk_idx} "
                    f"offset={offset} expected={expected_offset}"
                )
            expected_offset += size
            if chunk_idx < last_chunk_idx:
                fixed_layout = fixed_layout and size == fixed_chunk_size
            else:
                fixed_layout = fixed_layout and size <= fixed_chunk_size
        if expected_offset != int(total_tokens):
            raise ValueError(
                "Dense direct layerwise transfer chunk metadata does not cover "
                f"the slot mapping: kv_group={kv_group} "
                f"covered_tokens={expected_offset} total_tokens={total_tokens}"
            )
        if fixed_layout:
            dummy = self._dense_direct_dummy_metadata_tensor()
            return fixed_chunk_size, dummy, dummy
        return (
            0,
            torch.tensor(chunk_offsets, dtype=torch.int32, device=self.kv_device),
            torch.tensor(chunk_sizes, dtype=torch.int32, device=self.kv_device),
        )

    def _dense_direct_dummy_metadata_tensor(self) -> torch.Tensor:
        dummy = getattr(self, "_dense_direct_dummy_metadata_npu", None)
        if (
            dummy is None
            or dummy.dtype != torch.int32
            or dummy.device != torch.device(self.kv_device)
        ):
            dummy = torch.empty(1, dtype=torch.int32, device=self.kv_device)
            self._dense_direct_dummy_metadata_npu = dummy
        return dummy

    @staticmethod
    def _dense_direct_pointer_cache_signature(
        *,
        chunk_ptrs_npu: torch.Tensor,
        chunk_offsets_npu: torch.Tensor,
        chunk_sizes_npu: torch.Tensor,
        slot_mapping_ref: torch.Tensor,
        source_layout_ref: Optional[torch.Tensor],
        total_tokens: int,
        fixed_chunk_size: int,
        dense_kv_format: int,
        dense_token_major: bool,
        dense_vllm_two_major: bool,
        dense_k_hidden_dims: int,
        dense_v_hidden_dims: int,
        dense_dsa_hidden_dims: int,
        direction: bool,
    ) -> tuple:
        return (
            4,
            VLLMPagedMemLayerwiseNPUConnector._tensor_layout_signature(
                source_layout_ref
            ),
            chunk_ptrs_npu.dtype,
            str(chunk_ptrs_npu.device),
            int(chunk_ptrs_npu.numel()),
            chunk_offsets_npu.dtype,
            str(chunk_offsets_npu.device),
            int(chunk_offsets_npu.numel()),
            chunk_sizes_npu.dtype,
            str(chunk_sizes_npu.device),
            int(chunk_sizes_npu.numel()),
            slot_mapping_ref.dtype,
            str(slot_mapping_ref.device),
            int(slot_mapping_ref.numel()),
            int(total_tokens),
            int(fixed_chunk_size),
            int(dense_kv_format),
            int(dense_token_major),
            int(dense_vllm_two_major),
            int(dense_k_hidden_dims),
            int(dense_v_hidden_dims),
            int(dense_dsa_hidden_dims),
            int(direction),
        )

    def _run_dense_direct_kv_transfer_layer(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        transfer_stream,
        current_stream,
        slot_mapping_full: torch.Tensor,
        chunk_ptrs_npu: torch.Tensor,
        chunk_offsets_npu: torch.Tensor,
        chunk_sizes_npu: torch.Tensor,
        total_tokens: int,
        fixed_chunk_size: int,
        dense_kv_format: int,
        dense_token_major: bool,
        dense_vllm_two_major: bool,
        dense_k_hidden_dims: int,
        dense_v_hidden_dims: int,
        dense_dsa_hidden_dims: int,
        dense_host_interleaved: bool,
        layer_tensors: List[torch.Tensor],
        direction: bool,
    ) -> None:
        num_tokens = int(slot_mapping_full.numel())
        if num_tokens == 0 or total_tokens <= 0 or chunk_ptrs_npu.numel() == 0:
            return

        source_signature = self._dense_direct_pointer_cache_signature(
            chunk_ptrs_npu=chunk_ptrs_npu,
            chunk_offsets_npu=chunk_offsets_npu,
            chunk_sizes_npu=chunk_sizes_npu,
            slot_mapping_ref=slot_mapping_full,
            source_layout_ref=layer_tensors[0] if layer_tensors else None,
            total_tokens=total_tokens,
            fixed_chunk_size=fixed_chunk_size,
            dense_kv_format=dense_kv_format,
            dense_token_major=dense_token_major,
            dense_vllm_two_major=dense_vllm_two_major,
            dense_k_hidden_dims=dense_k_hidden_dims,
            dense_v_hidden_dims=dense_v_hidden_dims,
            dense_dsa_hidden_dims=dense_dsa_hidden_dims,
            direction=direction,
        )
        layer_state, validate_key = self._get_or_create_sparse_direct_layer_state(
            kvcaches_ref=kvcaches_ref,
            kv_group=kv_group,
            layer_id=layer_id,
            layer_tensors=layer_tensors,
            slot_mapping_ref=slot_mapping_full,
            total_tokens=total_tokens,
            sparse_kv_format=dense_kv_format,
            sparse_token_major=dense_token_major,
            sparse_vllm_two_major=dense_vllm_two_major,
            sparse_k_hidden_dims=dense_k_hidden_dims,
            sparse_v_hidden_dims=dense_v_hidden_dims,
            sparse_dsa_hidden_dims=dense_dsa_hidden_dims,
            source_signature=source_signature,
            return_key=True,
        )
        if validate_key is None:
            validate_key = ("dense", kv_group, layer_id)

        with self._stream_context_or_null(transfer_stream):
            transfer_stream.wait_stream(current_stream)
            if layer_state is not None:
                validate_inputs = (
                    validate_key not in self._sparse_direct_validated_layers
                )
                dense_mla_dsa_batched_direct_kv_transfer_fast(
                    layer_state,
                    slot_mapping_full,
                    chunk_ptrs_npu,
                    chunk_offsets_npu,
                    chunk_sizes_npu,
                    total_tokens,
                    dense_host_interleaved,
                    direction,
                    validate_inputs=validate_inputs,
                    fixed_chunk_size=fixed_chunk_size,
                )
                if validate_inputs:
                    self._sparse_direct_validated_layers.add(validate_key)
            else:
                dense_mla_dsa_batched_direct_kv_transfer(
                    layer_tensors,
                    kvcaches_ref[layer_id],
                    slot_mapping_full,
                    chunk_offsets_npu,
                    chunk_sizes_npu,
                    total_tokens,
                    dense_kv_format,
                    dense_token_major,
                    dense_vllm_two_major,
                    dense_k_hidden_dims,
                    dense_v_hidden_dims,
                    dense_dsa_hidden_dims,
                    dense_host_interleaved,
                    direction,
                    chunk_ptrs_npu=chunk_ptrs_npu,
                    fixed_chunk_size=fixed_chunk_size,
                )
        current_stream.wait_stream(transfer_stream)

    def _lmc_plane_num_tokens(
        self, lmc_tensor: torch.Tensor, kv_group: Optional[int] = None
    ) -> int:
        fmt = self._fmt_for(kv_group)
        layout = self._layout_for(kv_group)
        k_hidden = layout.k_hidden_dims if layout else self.k_hidden_dims
        v_hidden = layout.v_hidden_dims if layout else self.v_hidden_dims
        dsa_hidden = layout.dsa_hidden_dims if layout else self.dsa_hidden_dims
        if fmt in (
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.MLA_LATENT,
            KVCacheFormat.DSA_INDEX,
        ):
            plane = k_hidden + v_hidden
            if fmt == KVCacheFormat.DSA_KV:
                plane += dsa_hidden
            assert plane > 0, (
                "MLA/DSA hidden dims must be initialized before sparse retrieve."
            )
            return lmc_tensor.numel() // plane
        if lmc_tensor.ndim >= 3:
            return int(lmc_tensor.shape[0])
        per_token = 2 * self.hidden_dim_size
        assert per_token > 0, (
            "hidden_dim_size must be positive for GQA sparse retrieve."
        )
        return lmc_tensor.numel() // per_token

    def _expected_memory_format(self, kv_group: Optional[int] = None) -> MemoryFormat:
        fmt = self._fmt_for(kv_group)
        if fmt in (
            KVCacheFormat.MLA_KV,
            KVCacheFormat.MLA_LATENT,
        ):
            return MemoryFormat.KV_MLA_LATENT_FMT
        if fmt == KVCacheFormat.DSA_INDEX:
            return MemoryFormat.KV_DSA_INDEX_FMT
        if fmt == KVCacheFormat.DSA_KV:
            return MemoryFormat.KV_MLA_LATENT_FMT
        return MemoryFormat.KV_T2D

    def _fmt_for(self, kv_group: Optional[int]) -> KVCacheFormat:
        """Resolve the KV format for a group (defaults to current/mirrored)."""
        if kv_group is None:
            kv_group = self._current_kv_group
        layout = self._group_layouts.get(kv_group)
        if layout is not None:
            return layout.kv_format
        return self.kv_format

    def _layout_for(self, kv_group: Optional[int]) -> Optional[_GroupLayout]:
        """Return the layout for a group if initialized, else None."""
        if kv_group is None:
            kv_group = self._current_kv_group
        return self._group_layouts.get(kv_group)

    def set_layerwise_staging_concurrency(self, n: int) -> None:
        """Size per-group staging pools for concurrent layerwise transfers."""
        n = max(1, int(n))
        if n <= self._layerwise_staging_concurrency:
            return
        self._layerwise_staging_concurrency = n
        for kv_group, layout in self._group_layouts.items():
            if layout.gpu_buffer_allocator is None:
                continue
            if layout.gpu_buffer_allocator.allocator.num_active_allocations > 0:
                logger.warning(
                    "Cannot grow staging pool for kv_group=%s while %d "
                    "buffers are still in use",
                    kv_group,
                    layout.gpu_buffer_allocator.allocator.num_active_allocations,
                )
                continue
            per_slot = layout.staging_bytes_per_slot
            if per_slot <= 0:
                continue
            new_size = per_slot * self._layerwise_staging_pool_slots()
            self._assign_group_gpu_allocator(new_size, layout)

    def _layerwise_staging_pool_slots(self) -> int:
        if not self.dsa_two_groups:
            return 1
        return max(1, self._layerwise_staging_concurrency)

    def _layerwise_staging_num_tokens(self, cache_max_tokens: int) -> int:
        """Token count for sizing the layerwise GPU staging pool.

        ``cache_max_tokens`` comes from the paged KV tensor shape
        (num_blocks * block_size), i.e. the vLLM block-pool capacity. That
        capacity is shared across MLA latent and DSA indexer groups in
        two-group mode, and in any case exceeds what a single layerwise
        store/retrieve transfer holds (LMCache chunks). Size staging for
        transfer headroom, not full pool capacity.
        """
        if not self.dsa_two_groups or self.lmcache_chunk_size <= 0:
            return cache_max_tokens

        max_seq_tokens = self.max_staging_tokens
        if max_seq_tokens <= 0:
            # Non-vLLM paths (unit tests, benchmarks) may omit max_model_len.
            max_seq_tokens = self.lmcache_chunk_size * 512
        staging_tokens = min(cache_max_tokens, max_seq_tokens)
        return max(staging_tokens, self.lmcache_chunk_size)

    def _staging_token_cap(self) -> int:
        """Max tokens the layerwise GPU staging pool must cover per transfer."""
        if not self.dsa_two_groups or self.lmcache_chunk_size <= 0:
            return 0
        max_seq_tokens = self.max_staging_tokens
        if max_seq_tokens <= 0:
            max_seq_tokens = self.lmcache_chunk_size * 512
        return max(max_seq_tokens, self.lmcache_chunk_size)

    def _check_staging_transfer_tokens(self, num_tokens: int, kv_group: int) -> None:
        cap = self._staging_token_cap()
        if cap <= 0 or num_tokens <= cap:
            return
        raise ValueError(
            f"Layerwise transfer needs {num_tokens} staging tokens for "
            f"kv_group={kv_group}, but the staging cap is {cap} "
            f"(max_model_len={self.max_staging_tokens}). "
            "Increase vLLM max_model_len or reduce tokens stored per forward."
        )

    def _check_layerwise_transfer_invariants(
        self,
        *,
        operation: str,
        kv_group: int,
        slot_mapping_full: torch.Tensor,
        kvcaches_ref: list,
    ) -> None:
        """Fail before the NPU kernel if per-group transfer metadata is invalid."""
        if not self.dsa_two_groups or kv_group not in (0, 1):
            return
        kvcaches_len = len(kvcaches_ref) if kvcaches_ref is not None else 0
        if kvcaches_len != self.num_layers:
            message = (
                f"{operation} layerwise transfer has mismatched layer counts: "
                f"kv_group={kv_group} connector_num_layers={self.num_layers} "
                f"kvcaches_len={kvcaches_len}"
            )
            logger.error(
                "%s. Refusing to continue before memory_objs[layer_id] access.",
                message,
            )
            raise RuntimeError(message)
        if slot_mapping_full is None or slot_mapping_full.numel() == 0:
            return
        if slot_mapping_full.device.type != "cpu":
            # Avoid a host/device sync in the layerwise hot path. Device-side
            # slot bounds are still protected by the transfer kernels.
            return
        if kvcaches_len == 0:
            return

        first_layer = kvcaches_ref[0]
        first_tensor = (
            first_layer[0]
            if isinstance(first_layer, (list, tuple)) and first_layer
            else first_layer
        )
        shape = list(first_tensor.shape) if first_tensor is not None else []
        capacity = shape[0] * shape[1] if len(shape) >= 2 else 0
        if capacity <= 0:
            return

        slot_min = int(slot_mapping_full.min().item())
        slot_max = int(slot_mapping_full.max().item())
        if slot_min < 0 or slot_max >= capacity:
            message = (
                f"{operation} layerwise transfer slot mapping is out of range: "
                f"kv_group={kv_group} slot_min={slot_min} slot_max={slot_max} "
                f"kvcaches_capacity={capacity} kvcaches_shape={shape} "
                f"connector_num_layers={self.num_layers} "
                f"kvcaches_len={kvcaches_len}"
            )
            logger.error(message)
            raise ValueError(message)

    def _allocate_layerwise_staging_buffer(
        self,
        *,
        num_tokens: int,
        kv_group: int,
        layout: _GroupLayout,
        expected_fmt: MemoryFormat,
    ) -> tuple[Optional[MemoryObj], torch.Tensor]:
        """Return (pool_obj_or_none, staging_tensor) for a layerwise transfer."""
        self._check_staging_transfer_tokens(num_tokens, kv_group)

        gpu_buffer_allocator = layout.gpu_buffer_allocator
        assert gpu_buffer_allocator is not None, (
            f"GPU staging pool for kv_group={kv_group} is not initialized."
        )
        buffer_shape = self.get_shape(num_tokens, kv_group)
        tmp_gpu_buffer_obj = gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, expected_fmt
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate NPU buffer in NPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None
        return tmp_gpu_buffer_obj, tmp_gpu_buffer_obj.tensor

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemLayerwiseNPUConnector":
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        max_staging_tokens = int(getattr(metadata, "max_model_len", 0) or 0)
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
            max_staging_tokens=max_staging_tokens,
        )

    def _assign_group_gpu_allocator(
        self, gpu_buffer_size: int, layout: _GroupLayout
    ) -> None:
        """Create a per-kv_group GPU staging allocator.

        Two-group forwards may keep latent and indexer store generators alive
        at the same time, each holding a staging allocation for the whole
        layer loop. Do not share one pool across groups.
        """
        layout.gpu_buffer_allocator = GPUMemoryAllocator(
            gpu_buffer_size, device=self.device
        )
        self.gpu_buffer_allocator = layout.gpu_buffer_allocator

    def _lazy_initialize_buffer(
        self,
        kv_caches,
        kv_group: int = 0,
        init_staging: Optional[bool] = None,
    ) -> _GroupLayout:
        """
        Lazily initialize per-kv_group format metadata and GPU buffer allocator.

        In two-group MLA+DSA mode the same connector instance handles both
        kv_group=0 (latent) and kv_group=1 (indexer); each group is detected
        and allocated independently on its first call instead of the first
        group pinning a single ``self.kv_format`` for both.
        """
        if init_staging is None:
            init_staging = self.use_gpu

        self._current_kv_group = kv_group
        layout = self._group_layouts.get(kv_group)
        if layout is None:
            layout = _GroupLayout()
            # Both groups need the two-group hint: kv_group=0 is MLA_LATENT
            # and kv_group=1 is DSA_INDEX. Without it, equal latent/PE widths
            # can be mistaken for ordinary SEPARATE_KV at TP=8-like shapes.
            detect_two_groups = getattr(self, "dsa_two_groups", False)
            layout.kv_format = KVCacheFormat.detect(
                kv_caches,
                use_mla=self.use_mla,
                dsa_two_groups=detect_two_groups,
            )
            if layout.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    "Undefined KV cache format detected. "
                    "Unable to determine the format of input kv_caches."
                )

            logger.info(
                f"Detected KV cache format (kv_group={kv_group}): "
                f"{layout.kv_format.name}"
            )
            self._reset_sparse_direct_layer_states()
            first_layer_cache = kv_caches[0]

            if layout.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]
                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )
                layout.kv_device = key_tensor.device
                layout.vllm_two_major = False
                layout.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                layout.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
            elif layout.kv_format == KVCacheFormat.MLA_KV:
                key_tensor, value_tensor = first_layer_cache
                layout.kv_device = key_tensor.device
                layout.vllm_two_major = False
                layout.kv_lora_rank = key_tensor.shape[-1]
                layout.qk_rope_head_dim = value_tensor.shape[-1]
                layout.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                layout.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
            elif layout.kv_format == KVCacheFormat.MLA_LATENT:
                # Two-group latent: (k_nope, k_pe) -> same dim structure as MLA_KV
                key_tensor, value_tensor = first_layer_cache
                layout.kv_device = key_tensor.device
                layout.vllm_two_major = False
                layout.kv_lora_rank = key_tensor.shape[-1]
                layout.qk_rope_head_dim = value_tensor.shape[-1]
                layout.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                layout.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
            elif layout.kv_format == KVCacheFormat.DSA_INDEX:
                # Two-group indexer: (indexer_k,) -> single tensor, single plane
                indexer_tensor = first_layer_cache[0]
                layout.kv_device = indexer_tensor.device
                layout.vllm_two_major = False
                layout.dsa_head_dim = indexer_tensor.shape[-1]
                layout.dsa_hidden_dims = (
                    indexer_tensor.shape[-2] * indexer_tensor.shape[-1]
                )
                # Map onto MLA_KV kernel: k=dsa, v=0
                layout.k_hidden_dims = layout.dsa_hidden_dims
                layout.v_hidden_dims = 0
            elif layout.kv_format == KVCacheFormat.DSA_KV:
                key_tensor, value_tensor, dsa_tensor = first_layer_cache
                layout.kv_device = key_tensor.device
                layout.vllm_two_major = False
                layout.kv_lora_rank = key_tensor.shape[-1]
                layout.qk_rope_head_dim = value_tensor.shape[-1]
                layout.dsa_head_dim = dsa_tensor.shape[-1]
                layout.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                layout.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
                layout.dsa_hidden_dims = dsa_tensor.shape[-2] * dsa_tensor.shape[-1]
            elif layout.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )
                layout.kv_device = first_layer_cache.device
                layout.vllm_two_major = first_layer_cache.shape[0] == 2
                if layout.vllm_two_major:
                    head_tensor = first_layer_cache[0]
                else:
                    head_tensor = first_layer_cache[:, 0]
                head_elems = head_tensor.shape[-2] * head_tensor.shape[-1]
                layout.k_hidden_dims = head_elems
                layout.v_hidden_dims = head_elems
            else:
                raise ValueError(f"Unsupported KV cache format: {layout.kv_format}")

            self._group_layouts[kv_group] = layout

        # Mirror into instance attributes for backward-compatible readers.
        self._mirror_layout(layout)

        if init_staging and layout.gpu_buffer_allocator is None:
            logger.info(f"Lazily initializing GPU buffer (kv_group={kv_group}).")
            first_layer_cache = kv_caches[0]

            if layout.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                k_cache_shape_per_layer = key_tensor.shape
                v_cache_shape_per_layer = first_layer_cache[1].shape
                num_elements = key_tensor.numel() * 2
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            elif layout.kv_format == KVCacheFormat.MLA_KV:
                key_tensor, value_tensor = first_layer_cache
                k_cache_shape_per_layer = key_tensor.shape
                v_cache_shape_per_layer = value_tensor.shape
                max_tokens = key_tensor.shape[0] * key_tensor.shape[1]
                num_elements = max_tokens * (
                    layout.k_hidden_dims + layout.v_hidden_dims
                )
            elif layout.kv_format == KVCacheFormat.MLA_LATENT:
                key_tensor, value_tensor = first_layer_cache
                k_cache_shape_per_layer = key_tensor.shape
                v_cache_shape_per_layer = value_tensor.shape
                max_tokens = key_tensor.shape[0] * key_tensor.shape[1]
                num_elements = max_tokens * (
                    layout.k_hidden_dims + layout.v_hidden_dims
                )
            elif layout.kv_format == KVCacheFormat.DSA_INDEX:
                indexer_tensor = first_layer_cache[0]
                k_cache_shape_per_layer = indexer_tensor.shape
                v_cache_shape_per_layer = indexer_tensor.shape
                max_tokens = indexer_tensor.shape[0] * indexer_tensor.shape[1]
                num_elements = max_tokens * layout.dsa_hidden_dims
            elif layout.kv_format == KVCacheFormat.DSA_KV:
                key_tensor, value_tensor, dsa_tensor = first_layer_cache
                k_cache_shape_per_layer = key_tensor.shape
                v_cache_shape_per_layer = value_tensor.shape
                max_tokens = key_tensor.shape[0] * key_tensor.shape[1]
                num_elements = max_tokens * (
                    layout.k_hidden_dims + layout.v_hidden_dims + layout.dsa_hidden_dims
                )
            else:
                if layout.vllm_two_major:
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                    v_cache_shape_per_layer = first_layer_cache[1].shape
                else:
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
                    v_cache_shape_per_layer = first_layer_cache[:, 1].shape
                num_elements = k_cache_shape_per_layer.numel() * 2
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            staging_tokens = self._layerwise_staging_num_tokens(max_tokens)
            if staging_tokens != max_tokens:
                plane_elems = num_elements // max_tokens
                num_elements = staging_tokens * plane_elems

            pool_slots = self._layerwise_staging_pool_slots()
            per_slot_bytes = num_elements * self.element_size
            layout.staging_bytes_per_slot = per_slot_bytes
            gpu_buffer_size = per_slot_bytes * pool_slots

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {layout.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Value cache shape per layer: {v_cache_shape_per_layer}\n"
                f"  - Pool max tokens: {max_tokens}\n"
                f"  - Staging tokens: {staging_tokens}\n"
                f"  - Staging pool slots: {pool_slots}\n"
                f"  - k_hidden_dims={layout.k_hidden_dims} "
                f"v_hidden_dims={layout.v_hidden_dims} "
                f"dsa_hidden_dims={layout.dsa_hidden_dims}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024):.2f} MB"
            )

            self._assign_group_gpu_allocator(gpu_buffer_size, layout)
            if self.dsa_two_groups:
                logger.info(
                    "dsa_two_groups: per-group NPU staging pool "
                    f"(kv_group={kv_group}, cap={staging_tokens} tokens, "
                    f"slots={pool_slots}, "
                    f"{gpu_buffer_size / (1024 * 1024):.2f} MB)"
                )

        return layout

    def get_shape(self, num_tokens: int, kv_group: Optional[int] = None) -> torch.Size:
        if kv_group is None:
            kv_group = self._current_kv_group
        layout = self._group_layouts.get(kv_group)
        fmt = layout.kv_format if layout is not None else self.kv_format
        k_hidden = layout.k_hidden_dims if layout is not None else self.k_hidden_dims
        v_hidden = layout.v_hidden_dims if layout is not None else self.v_hidden_dims
        dsa_hidden = (
            layout.dsa_hidden_dims if layout is not None else self.dsa_hidden_dims
        )
        if fmt == KVCacheFormat.MLA_KV:
            plane_elems = k_hidden + v_hidden
            return torch.Size([num_tokens * plane_elems])
        if fmt == KVCacheFormat.MLA_LATENT:
            plane_elems = k_hidden + v_hidden
            return torch.Size([num_tokens * plane_elems])
        if fmt == KVCacheFormat.DSA_INDEX:
            plane_elems = dsa_hidden
            return torch.Size([num_tokens * plane_elems])
        if fmt == KVCacheFormat.DSA_KV:
            plane_elems = k_hidden + v_hidden + dsa_hidden
            return torch.Size([num_tokens * plane_elems])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])

    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        kv_group = kwargs.get("kv_group", 0)
        layout = self._lazy_initialize_buffer_with_staging(
            self.kvcaches,
            kv_group=kv_group,
            init_staging=False,
        )
        is_mla_dsa = self._is_mla_dsa_format(kv_group)
        dense_direct = is_mla_dsa and not _DENSE_DIRECT_LOAD_DISABLE

        if not dense_direct:
            if is_mla_dsa and not self.use_gpu:
                raise ValueError(
                    "MLA/DSA layerwise transfer requires use_gpu=True with a "
                    "staging buffer when dense direct load is disabled."
                )
            if self.use_gpu and layout.gpu_buffer_allocator is None:
                layout = self._lazy_initialize_buffer_with_staging(
                    self.kvcaches,
                    kv_group=kv_group,
                    init_staging=True,
                )

        slot_mapping_chunks = []
        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])
            chunk_size = end - start
            chunk_offsets.append(current_offset)
            chunk_sizes.append(chunk_size)
            current_offset += chunk_size

        slot_mapping_full = (
            slot_mapping_chunks[0]
            if len(slot_mapping_chunks) == 1
            else torch.cat(slot_mapping_chunks, dim=0)
        )

        num_tokens = len(slot_mapping_full)
        self._check_layerwise_transfer_invariants(
            operation="retrieve",
            kv_group=kv_group,
            slot_mapping_full=slot_mapping_full,
            kvcaches_ref=self.kvcaches,
        )

        # Snapshot per-group values so interleaved per-group generators
        # do not race on the mirrored instance attributes.
        kv_format_value = layout.kv_format.value
        vllm_two_major = layout.vllm_two_major
        k_hidden_dims, v_hidden_dims, dsa_hidden_dims = (
            layout.k_hidden_dims,
            layout.v_hidden_dims,
            layout.dsa_hidden_dims,
        )
        token_major = self._layerwise_token_major(kv_group)
        expected_fmt = self._expected_memory_format(kv_group)
        kvcaches_snapshot = self.kvcaches
        dense_host_interleaved = self._sparse_lmc_host_interleaved(kv_group)
        chunk_offsets_npu: Optional[torch.Tensor] = None
        chunk_sizes_npu: Optional[torch.Tensor] = None
        dense_fixed_chunk_size = 0
        if dense_direct:
            (
                dense_fixed_chunk_size,
                chunk_offsets_npu,
                chunk_sizes_npu,
            ) = self._prepare_dense_direct_chunk_metadata(
                chunk_offsets,
                chunk_sizes,
                total_tokens=num_tokens,
                kv_group=kv_group,
            )

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        staging_tensor: Optional[torch.Tensor] = None
        if self.use_gpu and not dense_direct:
            tmp_gpu_buffer_obj, staging_tensor = (
                self._allocate_layerwise_staging_buffer(
                    num_tokens=num_tokens,
                    kv_group=kv_group,
                    layout=layout,
                    expected_fmt=expected_fmt,
                )
            )

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            # The generator is resumed from vLLM's attention path; refresh the
            # active compute stream per layer before ordering load -> compute.
            current_stream = torch.cuda.current_stream()
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0 and logger.isEnabledFor(10):
                logger.debug("Finished loading layer %d", layer_id - 1)
            # memobj -> gpu_buffer -> kvcaches
            if dense_direct:
                cpu_tensors = []
                for memory_obj in memory_objs_layer:
                    assert memory_obj.tensor is not None
                    if memory_obj.metadata.fmt != expected_fmt:
                        raise ValueError(
                            f"Expected memory format {expected_fmt}, "
                            f"got {memory_obj.metadata.fmt}."
                        )
                    cpu_tensors.append(memory_obj.tensor)
                chunk_ptrs_npu = self._resolve_sparse_chunk_ptrs_npu(
                    layer_id,
                    cpu_tensors,
                )
                assert chunk_offsets_npu is not None
                assert chunk_sizes_npu is not None
                self._run_dense_direct_kv_transfer_layer(
                    kvcaches_ref=kvcaches_snapshot,
                    kv_group=kv_group,
                    layer_id=layer_id,
                    transfer_stream=self.load_stream,
                    current_stream=current_stream,
                    slot_mapping_full=slot_mapping_full,
                    chunk_ptrs_npu=chunk_ptrs_npu,
                    chunk_offsets_npu=chunk_offsets_npu,
                    chunk_sizes_npu=chunk_sizes_npu,
                    total_tokens=num_tokens,
                    fixed_chunk_size=dense_fixed_chunk_size,
                    dense_kv_format=kv_format_value,
                    dense_token_major=token_major,
                    dense_vllm_two_major=vllm_two_major,
                    dense_k_hidden_dims=k_hidden_dims,
                    dense_v_hidden_dims=v_hidden_dims,
                    dense_dsa_hidden_dims=dsa_hidden_dims,
                    dense_host_interleaved=dense_host_interleaved,
                    layer_tensors=cpu_tensors,
                    direction=False,
                )
            else:
                with torch.cuda.stream(self.load_stream):
                    if self.use_gpu:
                        cpu_tensors = []
                        for memory_obj in memory_objs_layer:
                            assert memory_obj.tensor is not None
                            if memory_obj.metadata.fmt != expected_fmt:
                                raise ValueError(
                                    f"Expected memory format {expected_fmt}, "
                                    f"got {memory_obj.metadata.fmt}."
                                )
                            cpu_tensors.append(memory_obj.tensor)

                        # Fused transfer: N H2D memcpy + 1 scatter kernel
                        batched_fused_single_layer_kv_transfer(
                            cpu_tensors,  # CPU memory objects
                            staging_tensor,  # GPU staging buffer
                            kvcaches_snapshot[layer_id],
                            slot_mapping_full,
                            chunk_offsets,  # offset for each chunk
                            chunk_sizes,  # size for each chunk
                            False,  # to_gpu
                            kv_format_value,
                            token_major,
                            vllm_two_major,
                            k_hidden_dims,
                            v_hidden_dims,
                            dsa_hidden_dims,
                        )

                    else:
                        for start, end, memory_obj in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert memory_obj.tensor is not None

                            lmc_ops.single_layer_kv_transfer(
                                memory_obj.tensor,
                                kvcaches_snapshot[layer_id],
                                slot_mapping[start:end],
                                False,
                                kv_format_value,
                                token_major,
                                vllm_two_major,
                                k_hidden_dims,
                                v_hidden_dims,
                                dsa_hidden_dims,
                            )
                    if logger.isEnabledFor(10):
                        logger.debug("Finished loading layer %d", layer_id)
                continue
            if logger.isEnabledFor(10):
                logger.debug("Finished loading layer %d", layer_id)
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def _batched_to_gpu_head_token_wise_prepared(
        self,
        transfer_kwargs: dict[str, Any],
    ):
        """Sparse warm path with request source and destination state resolved."""
        source: PreparedSparseSource = transfer_kwargs["prepared_sparse_source"]
        kvcaches_snapshot = transfer_kwargs["kvcaches"]
        slot_mapping = transfer_kwargs["slot_mapping"]

        kv_group = int(transfer_kwargs.get("kv_group", 0))
        layout = self._group_layouts.get(kv_group)
        if layout is None:
            layout = self._lazy_initialize_buffer_with_staging(
                kvcaches_snapshot,
                kv_group=kv_group,
                init_staging=False,
            )

        load_stream_idx = self.load_stream_idx
        self.load_stream_idx = (self.load_stream_idx + 1) % self.load_stream_num
        load_stream = self.load_stream_list[load_stream_idx]

        sparse_k_hidden_dims = layout.k_hidden_dims
        sparse_v_hidden_dims = layout.v_hidden_dims
        sparse_dsa_hidden_dims = layout.dsa_hidden_dims
        sparse_host_interleaved = self._sparse_lmc_host_interleaved(kv_group)
        sparse_kv_format = layout.kv_format.value
        chunk_size = self.lmcache_chunk_size

        destination_plan = self._get_or_create_sparse_destination_plan(
            kvcaches_ref=kvcaches_snapshot,
            kv_group=kv_group,
            slot_mapping_ref=slot_mapping,
            sparse_kv_format=sparse_kv_format,
            sparse_k_hidden_dims=sparse_k_hidden_dims,
            sparse_v_hidden_dims=sparse_v_hidden_dims,
            sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
            expected_device=layout.kv_device,
        )

        for layer_id, source_layer in enumerate(source.layers):
            sparse_request = yield
            current_stream = (
                torch.npu.current_stream()
                if hasattr(torch, "npu") and hasattr(torch.npu, "current_stream")
                else torch.cuda.current_stream()
            )

            (
                selected_token_idx,
                token_start_index,
                target_slot_mapping,
                payload_event,
            ) = self._unpack_sparse_dynamic_request(sparse_request)

            if payload_event is not None:
                _wait_payload_events(current_stream, payload_event)
            if target_slot_mapping is not None:
                slot_mapping_packed, selected_token_idx = (
                    self._pack_sparse_explicit_slot_inputs(
                        selected_token_idx,
                        target_slot_mapping,
                    )
                )
            else:
                slot_mapping_packed, selected_token_idx = (
                    self._pack_sparse_layer_inputs(
                        slot_mapping,
                        selected_token_idx,
                        token_start_index,
                    )
                )

            self._run_prepared_sparse_direct_kv_transfer_layer(
                plan=destination_plan,
                source_layer=source_layer,
                layer_id=layer_id,
                load_stream=load_stream,
                current_stream=current_stream,
                slot_mapping_packed=slot_mapping_packed,
                selected_token_idx=selected_token_idx,
                chunk_size=chunk_size,
                total_tokens=source.total_tokens,
                sparse_host_interleaved=sparse_host_interleaved,
            )

        yield
        yield

    def batched_to_gpu_head_token_wise(self, **kwargs):
        """
        Sparse layerwise retrieve: scatter selected KV tokens from CPU pinned
        memory objects into paged NPU KV via direct NPU read (no staging).
        """
        if kwargs.get("prepared_sparse_source") is not None:
            yield from self._batched_to_gpu_head_token_wise_prepared(kwargs)
            return
        connector_started = time.perf_counter()
        phase_started = time.perf_counter()
        self.initialize_kvcaches_ptr(**kwargs)
        initialize_kvcaches_ms = (time.perf_counter() - phase_started) * 1000
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        kv_group = kwargs.get("kv_group", 0)
        phase_started = time.perf_counter()
        layout = self._lazy_initialize_buffer_with_staging(
            self.kvcaches,
            kv_group=kv_group,
            init_staging=False,
        )
        layout_init_ms = (time.perf_counter() - phase_started) * 1000

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        cached_tensors_by_layer: Optional[List[List[torch.Tensor]]] = kwargs.get(
            "cached_tensors"
        )
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]] = kwargs.get(
            "cached_chunk_ptrs_npu"
        )
        lmcache_cached_tokens: int = int(kwargs.get("lmcache_cached_tokens", 0) or 0)

        load_stream_idx = self.load_stream_idx
        self.load_stream_idx = (self.load_stream_idx + 1) % self.load_stream_num

        chunk_size = self.lmcache_chunk_size

        # Snapshot per-group values for the kernel so interleaved per-group
        # generators do not race on the mirrored instance attributes.
        sparse_k_hidden_dims, sparse_v_hidden_dims, sparse_dsa_hidden_dims = (
            layout.k_hidden_dims,
            layout.v_hidden_dims,
            layout.dsa_hidden_dims,
        )
        sparse_token_major = self._layerwise_token_major(kv_group)
        sparse_host_interleaved = self._sparse_lmc_host_interleaved(kv_group)
        sparse_kv_format = layout.kv_format.value
        sparse_vllm_two_major = layout.vllm_two_major
        # Snapshot so interleaved latent/indexer sparse generators do not
        # race on the shared connector self.kvcaches pointer.
        kvcaches_snapshot = self.kvcaches
        phase_totals_ms = {
            "request_unpack": 0.0,
            "payload_wait": 0.0,
            "input_pack": 0.0,
            "tensor_resolve": 0.0,
            "chunk_ptr_resolve": 0.0,
            "transfer_submit": 0.0,
        }
        phase_max_ms = {phase: 0.0 for phase in phase_totals_ms}
        phase_max_layer = {phase: -1 for phase in phase_totals_ms}
        layer_total_ms = 0.0
        max_layer_total_ms = 0.0
        max_layer_total_id = -1
        layers_with_data = 0
        chunk_refs = 0
        selected_tokens_total = 0

        def record_transfer_phase(
            phase_name: str,
            layer_id: int,
            started: float,
        ) -> None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            phase_totals_ms[phase_name] += elapsed_ms
            if elapsed_ms > phase_max_ms[phase_name]:
                phase_max_ms[phase_name] = elapsed_ms
                phase_max_layer[phase_name] = layer_id

        def record_layer_total(layer_id: int, started: float) -> None:
            nonlocal layer_total_ms, max_layer_total_ms, max_layer_total_id
            elapsed_ms = (time.perf_counter() - started) * 1000
            layer_total_ms += elapsed_ms
            if elapsed_ms > max_layer_total_ms:
                max_layer_total_ms = elapsed_ms
                max_layer_total_id = layer_id

        for layer_id in range(self.num_layers):
            sparse_request = yield
            layer_started = time.perf_counter()
            phase_started = time.perf_counter()
            # The generator is resumed from vLLM's attention path; refresh the
            # active compute stream per layer before ordering load -> compute.
            current_stream = (
                torch.npu.current_stream()
                if hasattr(torch, "npu") and hasattr(torch.npu, "current_stream")
                else torch.cuda.current_stream()
            )
            load_stream = self.load_stream_list[load_stream_idx]
            if isinstance(sparse_request, dict):
                memory_objs_layer = sparse_request["memory_objs_layer"]
                dynamic_request = sparse_request
            elif isinstance(sparse_request, tuple):
                if len(sparse_request) == 4:
                    (
                        memory_objs_layer,
                        selected_token_idx,
                        token_start_index,
                        target_slot_mapping,
                    ) = sparse_request
                    dynamic_request = (
                        selected_token_idx,
                        token_start_index,
                        target_slot_mapping,
                    )
                elif len(sparse_request) == 3:
                    (
                        memory_objs_layer,
                        selected_token_idx,
                        token_start_index,
                    ) = sparse_request
                    dynamic_request = (selected_token_idx, token_start_index)
                else:
                    raise ValueError(
                        "Sparse connector payload tuple must have 3 or 4 items"
                    )
            else:
                memory_objs_layer = sparse_request
                dynamic_request = None

            (
                selected_token_idx,
                token_start_index,
                target_slot_mapping,
                payload_event,
            ) = self._unpack_sparse_dynamic_request(dynamic_request)
            record_transfer_phase("request_unpack", layer_id, phase_started)
            explicit_sparse_payload = target_slot_mapping is not None

            # selected_token_idx/target_slot_mapping may be device tensors
            # produced by vLLM's remap path. Packing below is their first
            # connector-side consumer, so wait before packing.
            phase_started = time.perf_counter()
            if payload_event is not None:
                _wait_payload_events(current_stream, payload_event)
            record_transfer_phase("payload_wait", layer_id, phase_started)

            phase_started = time.perf_counter()
            if explicit_sparse_payload:
                slot_mapping_packed, selected_token_idx = (
                    self._pack_sparse_explicit_slot_inputs(
                        selected_token_idx,
                        target_slot_mapping,
                    )
                )
            else:
                slot_mapping_packed, selected_token_idx = (
                    self._pack_sparse_layer_inputs(
                        slot_mapping,
                        selected_token_idx,
                        token_start_index,
                    )
                )
            record_transfer_phase("input_pack", layer_id, phase_started)

            phase_started = time.perf_counter()
            layer_cached_tensors = (
                cached_tensors_by_layer[layer_id]
                if cached_tensors_by_layer is not None
                and layer_id < len(cached_tensors_by_layer)
                and cached_tensors_by_layer[layer_id]
                else None
            )
            if layer_cached_tensors is not None:
                cpu_tensors = layer_cached_tensors
            else:
                cpu_tensors = [
                    memory_obj.tensor
                    for memory_obj in memory_objs_layer
                    if memory_obj.tensor is not None
                ]
            record_transfer_phase("tensor_resolve", layer_id, phase_started)

            if not cpu_tensors:
                record_layer_total(layer_id, layer_started)
                continue

            layers_with_data += 1
            chunk_refs += len(cpu_tensors)
            selected_tokens_total += int(selected_token_idx.numel())
            phase_started = time.perf_counter()
            chunk_ptrs_npu = self._resolve_sparse_chunk_ptrs_npu(
                layer_id,
                cpu_tensors,
                cached_chunk_ptrs_npu,
            )
            record_transfer_phase("chunk_ptr_resolve", layer_id, phase_started)
            total_tokens = (
                lmcache_cached_tokens
                if lmcache_cached_tokens > 0
                else self._sparse_total_tokens_from_layer_chunks(cpu_tensors, kv_group)
            )
            phase_started = time.perf_counter()
            self._run_sparse_direct_kv_transfer_layer(
                kvcaches_ref=kvcaches_snapshot,
                kv_group=kv_group,
                layer_id=layer_id,
                load_stream=load_stream,
                current_stream=current_stream,
                slot_mapping_packed=slot_mapping_packed,
                selected_token_idx=selected_token_idx,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                chunk_ptrs_npu=chunk_ptrs_npu,
                sparse_kv_format=sparse_kv_format,
                sparse_token_major=sparse_token_major,
                sparse_vllm_two_major=sparse_vllm_two_major,
                sparse_k_hidden_dims=sparse_k_hidden_dims,
                sparse_v_hidden_dims=sparse_v_hidden_dims,
                sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
                sparse_host_interleaved=sparse_host_interleaved,
                layer_tensors=cpu_tensors,
                slot_mapping_ref=(
                    slot_mapping_packed if explicit_sparse_payload else slot_mapping
                ),
                cpu_tensors=cpu_tensors,
            )
            record_transfer_phase("transfer_submit", layer_id, phase_started)
            record_layer_total(layer_id, layer_started)

        logger.info(
            "[BOOTSTRAP_ASCEND_NPU_TRANSFER_SUMMARY] "
            "req=%s kv_group=%s layers=%d layers_with_data=%d "
            "chunk_refs=%d selected_tokens_total=%d "
            "initialize_kvcaches_ms=%.3f layout_init_ms=%.3f "
            "layer_total_ms=%.3f max_layer_total_ms=%.3f "
            "max_layer_total_id=%d request_unpack_ms=%.3f "
            "payload_wait_ms=%.3f payload_wait_max_ms=%.3f "
            "payload_wait_max_layer=%d input_pack_ms=%.3f "
            "input_pack_max_ms=%.3f input_pack_max_layer=%d "
            "tensor_resolve_ms=%.3f chunk_ptr_resolve_ms=%.3f "
            "chunk_ptr_resolve_max_ms=%.3f chunk_ptr_resolve_max_layer=%d "
            "transfer_submit_host_ms=%.3f transfer_submit_host_max_ms=%.3f "
            "transfer_submit_host_max_layer=%d function_wall_ms=%.3f",
            kwargs.get("req_id", "unspecified"),
            kv_group,
            self.num_layers,
            layers_with_data,
            chunk_refs,
            selected_tokens_total,
            initialize_kvcaches_ms,
            layout_init_ms,
            layer_total_ms,
            max_layer_total_ms,
            max_layer_total_id,
            phase_totals_ms["request_unpack"],
            phase_totals_ms["payload_wait"],
            phase_max_ms["payload_wait"],
            phase_max_layer["payload_wait"],
            phase_totals_ms["input_pack"],
            phase_max_ms["input_pack"],
            phase_max_layer["input_pack"],
            phase_totals_ms["tensor_resolve"],
            phase_totals_ms["chunk_ptr_resolve"],
            phase_max_ms["chunk_ptr_resolve"],
            phase_max_layer["chunk_ptr_resolve"],
            phase_totals_ms["transfer_submit"],
            phase_max_ms["transfer_submit"],
            phase_max_layer["transfer_submit"],
            (time.perf_counter() - connector_started) * 1000,
        )

        yield

        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slot_mapping_base = int(kwargs.get("slot_mapping_base", 0))
        if slot_mapping_base < 0:
            raise ValueError(
                f"slot_mapping_base must be non-negative, got {slot_mapping_base}"
            )

        kv_group = kwargs.get("kv_group", 0)
        layout = self._lazy_initialize_buffer_with_staging(
            self.kvcaches,
            kv_group=kv_group,
            init_staging=False,
        )
        is_mla_dsa = self._is_mla_dsa_format(kv_group)
        dense_direct = is_mla_dsa and not _DENSE_DIRECT_STORE_DISABLE

        if not dense_direct:
            if is_mla_dsa and not self.use_gpu:
                raise ValueError(
                    "MLA/DSA layerwise transfer requires use_gpu=True with a "
                    "staging buffer when dense direct store is disabled."
                )
            if self.use_gpu and layout.gpu_buffer_allocator is None:
                layout = self._lazy_initialize_buffer_with_staging(
                    self.kvcaches,
                    kv_group=kv_group,
                    init_staging=True,
                )

        slot_mapping_chunks = []
        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0
        for start, end in zip(starts, ends, strict=False):
            local_start = start - slot_mapping_base
            local_end = end - slot_mapping_base
            if (
                local_start < 0
                or local_end < local_start
                or local_end > len(slot_mapping)
            ):
                raise ValueError(
                    "Layerwise store chunk is outside the provided slot-mapping "
                    "window: "
                    f"chunk=[{start}, {end}), base={slot_mapping_base}, "
                    f"mapping_tokens={len(slot_mapping)}, "
                    f"local_chunk=[{local_start}, {local_end})"
                )
            slot_mapping_chunks.append(slot_mapping[local_start:local_end])
            chunk_size = end - start
            chunk_offsets.append(current_offset)
            chunk_sizes.append(chunk_size)
            current_offset += chunk_size

        slot_mapping_full = (
            slot_mapping_chunks[0]
            if len(slot_mapping_chunks) == 1
            else torch.cat(slot_mapping_chunks, dim=0)
        )

        num_tokens = len(slot_mapping_full)
        self._check_layerwise_transfer_invariants(
            operation="store",
            kv_group=kv_group,
            slot_mapping_full=slot_mapping_full,
            kvcaches_ref=self.kvcaches,
        )

        # Snapshot per-group values so interleaved per-group generators
        # do not race on the mirrored instance attributes.
        kv_format_value = layout.kv_format.value
        vllm_two_major = layout.vllm_two_major
        k_hidden_dims, v_hidden_dims, dsa_hidden_dims = (
            layout.k_hidden_dims,
            layout.v_hidden_dims,
            layout.dsa_hidden_dims,
        )
        token_major = self._layerwise_token_major(kv_group)
        expected_fmt = self._expected_memory_format(kv_group)
        kvcaches_snapshot = self.kvcaches
        dense_host_interleaved = self._sparse_lmc_host_interleaved(kv_group)
        chunk_offsets_npu: Optional[torch.Tensor] = None
        chunk_sizes_npu: Optional[torch.Tensor] = None
        dense_fixed_chunk_size = 0
        if dense_direct:
            (
                dense_fixed_chunk_size,
                chunk_offsets_npu,
                chunk_sizes_npu,
            ) = self._prepare_dense_direct_chunk_metadata(
                chunk_offsets,
                chunk_sizes,
                total_tokens=num_tokens,
                kv_group=kv_group,
            )
        if len(memory_objs) != self.num_layers:
            logger.error(
                "NPU layerwise store received wrong memory object layer count: "
                "kv_group=%s memory_layers=%d expected=%d chunk_count=%d "
                "starts=%s ends=%s fmt=%s kvcaches_layers=%d",
                kv_group,
                len(memory_objs),
                self.num_layers,
                len(starts),
                starts,
                ends,
                expected_fmt,
                len(kvcaches_snapshot),
            )
            raise RuntimeError(
                "NPU layerwise store memory object layer count mismatch: "
                f"got {len(memory_objs)}, expected {self.num_layers}"
            )
        if len(kvcaches_snapshot) < self.num_layers:
            logger.error(
                "NPU layerwise store has fewer kv cache layers than expected: "
                "kv_group=%s kvcaches_layers=%d expected=%d fmt=%s",
                kv_group,
                len(kvcaches_snapshot),
                self.num_layers,
                expected_fmt,
            )
            raise RuntimeError(
                "NPU layerwise store kv cache layer count mismatch: "
                f"got {len(kvcaches_snapshot)}, expected {self.num_layers}"
            )

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        staging_tensor: Optional[torch.Tensor] = None
        if self.use_gpu and not dense_direct:
            tmp_gpu_buffer_obj, staging_tensor = (
                self._allocate_layerwise_staging_buffer(
                    num_tokens=num_tokens,
                    kv_group=kv_group,
                    layout=layout,
                    expected_fmt=expected_fmt,
                )
            )

        current_stream = torch.npu.current_stream()

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = memory_objs[layer_id]
                # kvcaches -> gpu_buffer -> memobj
                if dense_direct:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        if memory_obj.metadata.fmt != expected_fmt:
                            raise ValueError(
                                f"Expected memory format {expected_fmt}, "
                                f"got {memory_obj.metadata.fmt}."
                            )
                        cpu_tensors.append(memory_obj.tensor)
                    chunk_ptrs_npu = self._resolve_sparse_chunk_ptrs_npu(
                        layer_id,
                        cpu_tensors,
                    )
                    assert chunk_offsets_npu is not None
                    assert chunk_sizes_npu is not None
                    self._run_dense_direct_kv_transfer_layer(
                        kvcaches_ref=kvcaches_snapshot,
                        kv_group=kv_group,
                        layer_id=layer_id,
                        transfer_stream=self.store_stream,
                        current_stream=current_stream,
                        slot_mapping_full=slot_mapping_full,
                        chunk_ptrs_npu=chunk_ptrs_npu,
                        chunk_offsets_npu=chunk_offsets_npu,
                        chunk_sizes_npu=chunk_sizes_npu,
                        total_tokens=num_tokens,
                        fixed_chunk_size=dense_fixed_chunk_size,
                        dense_kv_format=kv_format_value,
                        dense_token_major=token_major,
                        dense_vllm_two_major=vllm_two_major,
                        dense_k_hidden_dims=k_hidden_dims,
                        dense_v_hidden_dims=v_hidden_dims,
                        dense_dsa_hidden_dims=dsa_hidden_dims,
                        dense_host_interleaved=dense_host_interleaved,
                        layer_tensors=cpu_tensors,
                        direction=True,
                    )
                    logger.debug("Finished offloading layer %d", layer_id)
                else:
                    with torch.npu.stream(self.store_stream):
                        self.store_stream.wait_stream(current_stream)
                        if self.use_gpu:
                            cpu_tensors = []
                            for memory_obj in memory_objs_layer:
                                assert memory_obj.tensor is not None
                                cpu_tensors.append(memory_obj.tensor)

                            # Fused transfer: 1 scatter kernel + N D2H memcpy
                            batched_fused_single_layer_kv_transfer(
                                cpu_tensors,
                                staging_tensor,
                                kvcaches_snapshot[layer_id],
                                slot_mapping_full,
                                chunk_offsets,
                                chunk_sizes,
                                True,  # from_gpu
                                kv_format_value,
                                token_major,
                                vllm_two_major,
                                k_hidden_dims,
                                v_hidden_dims,
                                dsa_hidden_dims,
                            )
                        else:
                            for start, end, memory_obj in zip(
                                starts, ends, memory_objs_layer, strict=False
                            ):
                                assert memory_obj.tensor is not None

                                lmc_ops.single_layer_kv_transfer(
                                    memory_obj.tensor,
                                    kvcaches_snapshot[layer_id],
                                    slot_mapping[start:end],
                                    True,
                                    kv_format_value,
                                    token_major,
                                    vllm_two_major,
                                    k_hidden_dims,
                                    v_hidden_dims,
                                    dsa_hidden_dims,
                                )
                        logger.debug("Finished offloading layer %d", layer_id)
                yield

                # store_layer publishes the CPU MemoryObjs immediately after the
                # generator advances, so the layer's D2H copy must be complete
                # before returning control regardless of the caller's sync hint.
                self.store_stream.synchronize()

            # free the buffer memory
            if self.use_gpu and tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()
                tmp_gpu_buffer_obj = None
            yield
        finally:
            if (
                self.use_gpu
                and tmp_gpu_buffer_obj is not None
                and tmp_gpu_buffer_obj.is_valid()
            ):
                self.store_stream.synchronize()
                tmp_gpu_buffer_obj.ref_count_down()


class SGLangNPUConnector(SGLangGPUConnector):
    pass


class SGLangLayerwiseNPUConnector(SGLangLayerwiseGPUConnector):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [num_blocks, block_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

    def _expected_memory_format(self) -> MemoryFormat:
        return MemoryFormat.KV_T2D

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        # [2, self.layer_num, self.size // self.page_size + 1,
        # self.page_size, self.head_num, self.head_dim,]
        self.kv_format = KVCacheFormat.detect(kv_caches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("Could not detect KV cache format.")

        if self.use_gpu and self.gpu_buffer_allocator is None:
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - num_elements: {num_elements}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, self._expected_memory_format()
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0 and logger.isEnabledFor(10):
                logger.debug("Finished loading layer %d", layer_id - 1)

            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        False,
                        self.kv_format.value,
                        True,
                        True,
                    )

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    True,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        if logger.isEnabledFor(10):
            logger.debug("Finished loading layer %d", layer_id)
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, self._expected_memory_format()
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None

                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        True,
                        self.kv_format.value,
                        True,
                        True,
                    )

            yield
            logger.debug("Finished offloading layer %d", layer_id)

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
