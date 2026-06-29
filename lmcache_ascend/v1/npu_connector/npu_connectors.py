# SPDX-License-Identifier: Apache-2.0
# Standard
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
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.kv_checksum_diag import log_sparse_scatter_entry
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.utils import (
    batched_fused_single_layer_kv_transfer,
    prepare_sparse_direct_layer_state,
    sparse_mla_dsa_batched_direct_kv_transfer,
    sparse_mla_dsa_batched_direct_kv_transfer_fast,
)
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

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

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

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

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

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
            logger.debug(f"Finished offloading layer {layer_id}")

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
        self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

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

                    # Submit RDMA read for current batch → transport_stream.
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
        self.load_stream_list = [torch.cuda.Stream() for __ in range(self.load_stream_num)]
        self.load_stream_idx = 0

        self.lmcache_chunk_size = int(kwargs.get("chunk_size", 0))
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.use_mla = kwargs.get("use_mla", False)

        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.k_hidden_dims: int = 0
        self.v_hidden_dims: int = 0
        self.dsa_hidden_dims: int = 0
        self._layerwise_sparse_idx_cache: Optional[torch.Tensor] = None
        self._sparse_direct_layer_states: Optional[list] = None
        self._sparse_direct_kvcaches_id: Optional[int] = None
        self._sparse_direct_validated_layers: set[int] = set()

    def _reset_sparse_direct_layer_states(self) -> None:
        self._sparse_direct_layer_states = None
        self._sparse_direct_kvcaches_id = None
        self._sparse_direct_validated_layers = set()

    def _sparse_total_tokens_from_layer_chunks(
        self,
        layer_tensors: List[torch.Tensor],
    ) -> int:
        num_chunks = len(layer_tensors)
        if num_chunks == 0:
            return 0
        if num_chunks == 1:
            return self._lmc_plane_num_tokens(layer_tensors[0])
        last_tokens = self._lmc_plane_num_tokens(layer_tensors[-1])
        return (num_chunks - 1) * self.lmcache_chunk_size + last_tokens

    def append_sparse_chunk_ptr_cache_for_layer(
        self,
        layer_id: int,
        new_tensors: List[torch.Tensor],
        cached_chunk_dev_ptrs: List[List[int]],
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]],
    ) -> None:
        """Resolve and append NPU device ptrs for newly retrieved chunks only."""
        if not new_tensors:
            return

        num_layers = self.num_layers
        if not cached_chunk_dev_ptrs:
            cached_chunk_dev_ptrs.extend([] for _ in range(num_layers))
        while len(cached_chunk_dev_ptrs) <= layer_id:
            cached_chunk_dev_ptrs.append([])

        if cached_chunk_ptrs_npu is not None and not cached_chunk_ptrs_npu:
            cached_chunk_ptrs_npu.extend(None for _ in range(num_layers))
        while (
            cached_chunk_ptrs_npu is not None
            and len(cached_chunk_ptrs_npu) <= layer_id
        ):
            cached_chunk_ptrs_npu.append(None)

        new_dev_ptrs = [
            int(lmc_ops.get_device_ptr(int(tensor.data_ptr())))
            for tensor in new_tensors
        ]
        cached_chunk_dev_ptrs[layer_id].extend(new_dev_ptrs)

        if cached_chunk_ptrs_npu is None:
            return

        new_ptrs_npu = torch.tensor(
            new_dev_ptrs, dtype=torch.long, device=self.kv_device
        )
        existing = cached_chunk_ptrs_npu[layer_id]
        if existing is None:
            cached_chunk_ptrs_npu[layer_id] = new_ptrs_npu
        else:
            cached_chunk_ptrs_npu[layer_id] = torch.cat(
                (existing, new_ptrs_npu), dim=0
            )

    def _get_or_create_sparse_direct_layer_state(
        self,
        *,
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
    ):
        if self.kvcaches is None:
            return None

        kvcaches_id = id(self.kvcaches)
        if self._sparse_direct_kvcaches_id != kvcaches_id:
            self._reset_sparse_direct_layer_states()
            self._sparse_direct_kvcaches_id = kvcaches_id

        if self._sparse_direct_layer_states is None:
            self._sparse_direct_layer_states = [None] * self.num_layers

        state = self._sparse_direct_layer_states[layer_id]
        if state is not None:
            return state

        if not layer_tensors:
            return None

        if total_tokens <= 0:
            total_tokens = self._sparse_total_tokens_from_layer_chunks(layer_tensors)

        state = prepare_sparse_direct_layer_state(
            layer_tensors[0],
            self.kvcaches[layer_id],
            slot_mapping_ref,
            sparse_token_major,
            sparse_vllm_two_major,
            sparse_kv_format,
            sparse_k_hidden_dims,
            sparse_v_hidden_dims,
            sparse_dsa_hidden_dims,
            total_tokens,
        )
        self._sparse_direct_layer_states[layer_id] = state
        return state

    def _pack_sparse_layer_inputs(
        self,
        slot_mapping: torch.Tensor,
        selected_token_idx: Optional[Union[torch.Tensor, list]],
        token_start_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align slot_mapping and selected_token_idx to equal length for kernel."""
        if selected_token_idx is not None and not isinstance(
            selected_token_idx, torch.Tensor
        ):
            selected_token_idx = torch.tensor(
                selected_token_idx, dtype=torch.int32, device=self.kv_device
            )

        if selected_token_idx is not None and selected_token_idx.numel() > 0:
            num_sparse = int(selected_token_idx.numel())
            start = int(token_start_index)
            end = start + num_sparse
            if end <= slot_mapping.numel():
                slot_mapping_packed = slot_mapping[start:end]
            elif start < slot_mapping.numel():
                slot_mapping_packed = slot_mapping[start:]
                selected_token_idx = selected_token_idx[
                    : slot_mapping_packed.numel()
                ]
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

    def _run_sparse_direct_kv_transfer_layer(
        self,
        *,
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

        resolve_tensors = (
            layer_tensors
            if layer_tensors is not None
            else (cpu_tensors if cpu_tensors is not None else [])
        )
        resolve_slot_mapping = (
            slot_mapping_ref
            if slot_mapping_ref is not None
            else slot_mapping_packed
        )

        layer_state = self._get_or_create_sparse_direct_layer_state(
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
        )

        log_sparse_scatter_entry(
            req_id=getattr(self, "_sparse_transfer_req_id", None),
            layer_id=layer_id,
            worker_id=0,
            num_sparse=num_sparse,
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            chunk_ptrs=int(chunk_ptrs_npu.numel()),
            use_fast_path=layer_state is not None,
        )

        with torch.cuda.stream(load_stream):
            load_stream.wait_stream(current_stream)
            if layer_state is not None:
                validate_inputs = layer_id not in self._sparse_direct_validated_layers
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
                    self._sparse_direct_validated_layers.add(layer_id)
            else:
                assert cpu_tensors is not None and len(cpu_tensors) > 0
                sparse_mla_dsa_batched_direct_kv_transfer(
                    cpu_tensors,
                    self.kvcaches[layer_id],
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

    def _is_mla_dsa_format(self) -> bool:
        return self.kv_format in (KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV)

    def _layerwise_token_major(self) -> bool:
        # GQA uses token-interleaved CPU chunks; MLA/DSA use stacked K|V|DSA planes.
        return not self._is_mla_dsa_format()

    def _sparse_lmc_host_interleaved(self) -> bool:
        # Must match batched_fused CPU layout (_layerwise_token_major).
        return self._layerwise_token_major()

    def notify_sparse_memory_objs_updated(self) -> None:
        """No-op: sparse chunk device ptrs live in ReqMeta and update incrementally."""

    def invalidate_sparse_chunk_ptr_cache(self) -> None:
        """No-op: sparse chunk device ptrs live in ReqMeta and update incrementally."""

    def _resolve_sparse_chunk_ptrs_npu(
        self,
        layer_id: int,
        cpu_tensors: List[torch.Tensor],
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_chunks = len(cpu_tensors)
        if (
            cached_chunk_ptrs_npu is not None
            and layer_id < len(cached_chunk_ptrs_npu)
        ):
            cached = cached_chunk_ptrs_npu[layer_id]
            if cached is not None and cached.numel() == num_chunks:
                return cached

        dev_ptrs = [
            int(lmc_ops.get_device_ptr(int(tensor.data_ptr())))
            for tensor in cpu_tensors
        ]
        chunk_ptrs_npu = torch.tensor(dev_ptrs, dtype=torch.long, device=self.kv_device)
        if cached_chunk_ptrs_npu is not None:
            while len(cached_chunk_ptrs_npu) <= layer_id:
                cached_chunk_ptrs_npu.append(None)
            cached_chunk_ptrs_npu[layer_id] = chunk_ptrs_npu
        return chunk_ptrs_npu

    def _single_layer_hidden_dim_args(self) -> tuple[int, int, int]:
        return (self.k_hidden_dims, self.v_hidden_dims, self.dsa_hidden_dims)

    def _lmc_plane_num_tokens(self, lmc_tensor: torch.Tensor) -> int:
        if self._is_mla_dsa_format():
            plane = self.k_hidden_dims + self.v_hidden_dims
            if self.kv_format == KVCacheFormat.DSA_KV:
                plane += self.dsa_hidden_dims
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

    def _sparse_retrieve_total_tokens(self, lmc_tensors: List[torch.Tensor]) -> int:
        num_chunks = len(lmc_tensors)
        if num_chunks == 0:
            return 0
        assert self.lmcache_chunk_size > 0, (
            "chunk_size must be configured for sparse layerwise retrieve."
        )
        if num_chunks == 1:
            return self._lmc_plane_num_tokens(lmc_tensors[0])
        last_tokens = self._lmc_plane_num_tokens(lmc_tensors[-1])
        return (num_chunks - 1) * self.lmcache_chunk_size + last_tokens

    def _expected_memory_format(self) -> MemoryFormat:
        if self._is_mla_dsa_format():
            return MemoryFormat.KV_MLA_FMT
        return MemoryFormat.KV_T2D

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize format metadata and the GPU buffer allocator.
        """
        if self.kv_format == KVCacheFormat.UNDEFINED:
            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    "Undefined KV cache format detected. "
                    "Unable to determine the format of input kv_caches."
                )

            logger.info(f"Detected KV cache format: {self.kv_format.name}")
            self._reset_sparse_direct_layer_states()
            first_layer_cache = kv_caches[0]

            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]
                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )
                self.kv_device = key_tensor.device
                self.vllm_two_major = False
                self.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                self.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
            elif self.kv_format == KVCacheFormat.MLA_KV:
                key_tensor, value_tensor = first_layer_cache
                self.kv_device = key_tensor.device
                self.vllm_two_major = False
                self.kv_lora_rank = key_tensor.shape[-1]
                self.qk_rope_head_dim = value_tensor.shape[-1]
                self.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                self.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
            elif self.kv_format == KVCacheFormat.DSA_KV:
                key_tensor, value_tensor, dsa_tensor = first_layer_cache
                self.kv_device = key_tensor.device
                self.vllm_two_major = False
                self.kv_lora_rank = key_tensor.shape[-1]
                self.qk_rope_head_dim = value_tensor.shape[-1]
                self.dsa_head_dim = dsa_tensor.shape[-1]
                self.k_hidden_dims = key_tensor.shape[-2] * key_tensor.shape[-1]
                self.v_hidden_dims = value_tensor.shape[-2] * value_tensor.shape[-1]
                self.dsa_hidden_dims = dsa_tensor.shape[-2] * dsa_tensor.shape[-1]
            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )
                self.kv_device = first_layer_cache.device
                self.vllm_two_major = first_layer_cache.shape[0] == 2
                if self.vllm_two_major:
                    head_tensor = first_layer_cache[0]
                else:
                    head_tensor = first_layer_cache[:, 0]
                head_elems = head_tensor.shape[-2] * head_tensor.shape[-1]
                self.k_hidden_dims = head_elems
                self.v_hidden_dims = head_elems
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            first_layer_cache = kv_caches[0]

            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                k_cache_shape_per_layer = key_tensor.shape
                num_elements = key_tensor.numel() * 2
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            elif self.kv_format == KVCacheFormat.MLA_KV:
                key_tensor, value_tensor = first_layer_cache
                k_cache_shape_per_layer = key_tensor.shape
                max_tokens = key_tensor.shape[0] * key_tensor.shape[1]
                num_elements = max_tokens * (self.k_hidden_dims + self.v_hidden_dims)
            elif self.kv_format == KVCacheFormat.DSA_KV:
                key_tensor, _, _ = first_layer_cache
                k_cache_shape_per_layer = key_tensor.shape
                max_tokens = key_tensor.shape[0] * key_tensor.shape[1]
                num_elements = max_tokens * (
                    self.k_hidden_dims + self.v_hidden_dims + self.dsa_hidden_dims
                )
            else:
                if self.vllm_two_major:
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                else:
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
                num_elements = k_cache_shape_per_layer.numel() * 2
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}"
            )

            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.kv_format == KVCacheFormat.MLA_KV:
            plane_elems = self.k_hidden_dims + self.v_hidden_dims
            return torch.Size([num_tokens * plane_elems])
        if self.kv_format == KVCacheFormat.DSA_KV:
            plane_elems = (
                self.k_hidden_dims + self.v_hidden_dims + self.dsa_hidden_dims
            )
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

        self._lazy_initialize_buffer(self.kvcaches)

        if self._is_mla_dsa_format() and not self.use_gpu:
            raise ValueError(
                "MLA/DSA layerwise transfer requires use_gpu=True with a staging buffer."
            )

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, self._expected_memory_format()
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")
            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                k_hidden_dims, v_hidden_dims, dsa_hidden_dims = (
                    self._single_layer_hidden_dim_args()
                )
                token_major = self._layerwise_token_major()
                if self.use_gpu:
                    cpu_tensors = []
                    expected_fmt = self._expected_memory_format()
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
                        tmp_gpu_buffer_obj.tensor,  # GPU staging buffer
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,  # offset for each chunk
                        chunk_sizes,  # size for each chunk
                        False,  # to_gpu
                        self.kv_format.value,
                        token_major,
                        self.vllm_two_major,
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
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            False,
                            self.kv_format.value,
                            token_major,
                            self.vllm_two_major,
                            k_hidden_dims,
                            v_hidden_dims,
                            dsa_hidden_dims,
                        )
                logger.debug(f"Finished loading layer {layer_id}")
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield


    def batched_to_gpu_head_token_wise(self, **kwargs):
        """
        Sparse layerwise retrieve: scatter selected KV tokens from CPU pinned
        memory objects into paged NPU KV via direct NPU read (no staging).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        self._lazy_initialize_buffer(self.kvcaches)

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]
        self._sparse_transfer_req_id = kwargs.get("req_id")
        cached_tensors_by_layer: Optional[List[List[torch.Tensor]]] = kwargs.get(
            "cached_tensors"
        )
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]] = kwargs.get(
            "cached_chunk_ptrs_npu"
        )
        lmcache_cached_tokens: int = int(kwargs.get("lmcache_cached_tokens", 0) or 0)
        use_cached_retrieve = (
            cached_tensors_by_layer is not None
            and len(cached_tensors_by_layer) == self.num_layers
        )

        current_stream = torch.cuda.current_stream()

        load_stream_idx = self.load_stream_idx
        self.load_stream_idx = (self.load_stream_idx + 1) % self.load_stream_num

        chunk_size = self.lmcache_chunk_size

        sparse_k_hidden_dims, sparse_v_hidden_dims, sparse_dsa_hidden_dims = (
            self._single_layer_hidden_dim_args()
        )
        sparse_token_major = self._layerwise_token_major()
        sparse_host_interleaved = self._sparse_lmc_host_interleaved()
        sparse_kv_format = self.kv_format.value
        sparse_vllm_two_major = self.vllm_two_major

        for layer_id in range(self.num_layers):
            memory_objs_layer, selected_token_idx, token_start_index = yield
            slot_mapping_packed, selected_token_idx = self._pack_sparse_layer_inputs(
                slot_mapping,
                selected_token_idx,
                token_start_index,
            )

            layer_cached_tensors = (
                cached_tensors_by_layer[layer_id]
                if use_cached_retrieve
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

            if not cpu_tensors:
                continue

            chunk_ptrs_npu = self._resolve_sparse_chunk_ptrs_npu(
                layer_id,
                cpu_tensors,
                cached_chunk_ptrs_npu,
            )
            if (
                use_cached_retrieve
                and lmcache_cached_tokens > 0
            ):
                total_tokens = lmcache_cached_tokens
            else:
                total_tokens = self._sparse_total_tokens_from_layer_chunks(cpu_tensors)

            self._run_sparse_direct_kv_transfer_layer(
                layer_id=layer_id,
                load_stream=self.load_stream_list[load_stream_idx],
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
                slot_mapping_ref=slot_mapping,
                cpu_tensors=cpu_tensors,
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
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        if self._is_mla_dsa_format() and not self.use_gpu:
            raise ValueError(
                "MLA/DSA layerwise transfer requires use_gpu=True with a staging buffer."
            )

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, self._expected_memory_format()
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                if self.use_gpu:
                    cpu_tensors = []
                    k_hidden_dims, v_hidden_dims, dsa_hidden_dims = (
                        self._single_layer_hidden_dim_args()
                    )
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: 1 scatter kernel + N D2H memcpy
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,
                        chunk_sizes,
                        True,  # from_gpu
                        self.kv_format.value,
                        self._layerwise_token_major(),
                        self.vllm_two_major,
                        k_hidden_dims,
                        v_hidden_dims,
                        dsa_hidden_dims,
                    )
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        k_hidden_dims, v_hidden_dims, dsa_hidden_dims = (
                            self._single_layer_hidden_dim_args()
                        )
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            self.kv_format.value,
                            self._layerwise_token_major(),
                            self.vllm_two_major,
                            k_hidden_dims,
                            v_hidden_dims,
                            dsa_hidden_dims,
                        )
                logger.debug(f"Finished offloading layer {layer_id}")
            yield

            if sync:
                self.store_stream.synchronize()

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()
        yield


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
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

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

        logger.debug(f"Finished loading layer {layer_id}")
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
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
