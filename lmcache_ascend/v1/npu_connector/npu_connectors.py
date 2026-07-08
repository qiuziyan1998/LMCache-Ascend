# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
import json
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
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.utils import (
    batched_fused_sparse_single_layer_kv_transfer,
    batched_fused_single_layer_kv_transfer,
    prepare_sparse_direct_layer_state,
    sparse_mla_dsa_batched_direct_kv_transfer,
    sparse_mla_dsa_batched_direct_kv_transfer_fast,
)
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj

from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_SPARSE_DIRECT_GUARD = os.getenv("LMCACHE_ASCEND_SPARSE_DIRECT_GUARD", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_SPARSE_DIRECT_RECORD_STREAM = os.getenv(
    "LMCACHE_ASCEND_SPARSE_DIRECT_RECORD_STREAM", "0"
).lower() in ("1", "true", "yes", "on")
_SPARSE_POINTER_CACHE_REUSE_VALIDATE_NULLS = os.getenv(
    "LMCACHE_ASCEND_SPARSE_PTR_REUSE_VALIDATE_NULLS", "0"
).lower() in ("1", "true", "yes", "on")
_SPARSE_POINTER_CACHE_REUSE_VALIDATE_PTRS = os.getenv(
    "LMCACHE_ASCEND_SPARSE_PTR_REUSE_VALIDATE_PTRS", "0"
).lower() in ("1", "true", "yes", "on")
_SPARSE_DIRECT_DISABLE = os.getenv(
    "LMCACHE_ASCEND_SPARSE_DIRECT_DISABLE", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_DIAG = os.getenv("LMCACHE_DSA_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_DSA_STREAM_DIAG = os.getenv("LMCACHE_DSA_STREAM_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
) or os.getenv("LMCACHE_ASCEND_DSA_STREAM_DIAG", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_DSA_SYNC_PROBES = {
    probe.strip().lower()
    for probe in os.getenv("LMCACHE_DSA_SYNC_PROBE", "").split(",")
    if probe.strip()
}
_DSA_WAIT_STREAM_PROBES = {
    probe.strip().lower()
    for probe in os.getenv("LMCACHE_DSA_WAIT_STREAM_PROBE", "").split(",")
    if probe.strip()
}
_DSA_SYNC_PROBE_LOGGED: Set[str] = set()
_DSA_WAIT_STREAM_PROBE_LOGGED: Set[str] = set()
_DSA_DIAG_TENSOR_STATS = os.getenv(
    "LMCACHE_DSA_DIAG_TENSOR_STATS", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_DIAG_FIRST_TOKEN_COMPARE = _DSA_DIAG and os.getenv(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_COMPARE", "0"
).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(
            "Invalid integer env %s=%r; using %s",
            name,
            os.getenv(name),
            default,
        )
        return default


_DSA_DIAG_FIRST_TOKEN_COMPARE_LAYERS = _env_int(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_LAYERS", 8
)
_DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED = _env_int(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_SELECTED", 16
)
_DSA_DIAG_FIRST_TOKEN_COMPARE_HASH_SELECTED = _env_int(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_HASH_SELECTED", 2048
)
_DSA_DIAG_FIRST_TOKEN_COMPARE_VALUES = _env_int(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_VALUES", 8
)
_DSA_DIAG_FIRST_TOKEN_DUMP = _DSA_DIAG and os.getenv(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_DUMP", "0"
).lower() in ("1", "true", "yes", "on")
_DSA_DIAG_FIRST_TOKEN_DUMP_DIR = os.getenv(
    "LMCACHE_DSA_DIAG_FIRST_TOKEN_DUMP_DIR",
    "/tmp/lmcache_dsa_diag",
)
_DSA_TORCH_NPU_MODULE: Any = None
_DSA_PUBLISH_STREAM_WARNING_LOGGED = False


def _describe_stream(stream: Any) -> Any:
    if stream is None:
        return None
    try:
        return {
            "type": type(stream).__name__,
            "npu_stream": getattr(stream, "npu_stream", None),
            "cuda_stream": getattr(stream, "cuda_stream", None),
            "device": str(getattr(stream, "device", None)),
        }
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _current_stream_summary() -> Any:
    summary: dict[str, Any] = {}
    try:
        summary["npu"] = _describe_stream(torch.npu.current_stream())
    except Exception as exc:
        summary["npu"] = f"{type(exc).__name__}: {exc}"
    try:
        summary["cuda"] = _describe_stream(torch.cuda.current_stream())
    except Exception as exc:
        summary["cuda"] = f"{type(exc).__name__}: {exc}"
    return summary


def _stream_diag(label: str, **kwargs) -> None:
    if not _DSA_STREAM_DIAG:
        return
    logger.warning(
        "[DSA_STREAM_DIAG] label=%s current_streams=%s extra=%s",
        label,
        _current_stream_summary(),
        kwargs,
    )


def _payload_event_list(payload_event: Any) -> list[Any]:
    if payload_event is None:
        return []
    if isinstance(payload_event, (list, tuple)):
        return [event for event in payload_event if event is not None]
    return [payload_event]


def _payload_stream_list(payload_stream: Any) -> list[Any]:
    if payload_stream is None:
        return []
    if isinstance(payload_stream, (list, tuple)):
        return [stream for stream in payload_stream if stream is not None]
    return [payload_stream]


def _probe_enabled(probes: Set[str], name: str) -> bool:
    return name in probes or "all" in probes


def _publish_current_npu_stream() -> bool:
    global _DSA_TORCH_NPU_MODULE, _DSA_PUBLISH_STREAM_WARNING_LOGGED
    try:
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "current_device")):
            return False
        if _DSA_TORCH_NPU_MODULE is None:
            _DSA_TORCH_NPU_MODULE = __import__("torch_npu")
        _DSA_TORCH_NPU_MODULE._C._npu_getCurrentRawStream(
            int(torch.npu.current_device())
        )
        return True
    except Exception:
        if not _DSA_PUBLISH_STREAM_WARNING_LOGGED:
            logger.warning(
                "Failed to publish current NPU stream for DSA payload event "
                "ordering.",
                exc_info=True,
            )
            _DSA_PUBLISH_STREAM_WARNING_LOGGED = True
        return False


def _sync_probe(name: str, stream: Any = None) -> None:
    if not _probe_enabled(_DSA_SYNC_PROBES, name):
        return
    try:
        if stream is None:
            stream = torch.npu.current_stream()
        stream.synchronize()
    except Exception:
        logger.warning(
            "[LMCACHE_DSA_SYNC_PROBE_ERROR] probe=%s failed",
            name,
            exc_info=True,
        )
    else:
        if name not in _DSA_SYNC_PROBE_LOGGED:
            _DSA_SYNC_PROBE_LOGGED.add(name)
            logger.warning("[LMCACHE_DSA_SYNC_PROBE] probe=%s", name)


def _wait_stream_probe(
    name: str,
    payload_stream: Any,
    consumer_stream: Any = None,
) -> None:
    payload_streams = _payload_stream_list(payload_stream)
    if not payload_streams or not _probe_enabled(_DSA_WAIT_STREAM_PROBES, name):
        return
    try:
        if consumer_stream is None:
            consumer_stream = torch.npu.current_stream()
        for stream in payload_streams:
            consumer_stream.wait_stream(stream)
        if not _publish_current_npu_stream():
            raise RuntimeError("failed to publish stream after wait_stream")
    except Exception:
        logger.warning(
            "[LMCACHE_DSA_WAIT_STREAM_PROBE_ERROR] probe=%s failed",
            name,
            exc_info=True,
        )
    else:
        if name not in _DSA_WAIT_STREAM_PROBE_LOGGED:
            _DSA_WAIT_STREAM_PROBE_LOGGED.add(name)
            logger.warning(
                "[LMCACHE_DSA_WAIT_STREAM_PROBE] probe=%s stream_count=%d",
                name,
                len(payload_streams),
            )


def _diag_tensor_summary(value: Any, max_items: int = 4) -> Any:
    if isinstance(value, torch.Tensor):
        summary: dict[str, Any] = {
            "type": "Tensor",
            "shape": tuple(int(dim) for dim in value.shape),
            "stride": tuple(int(stride) for stride in value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
            "data_ptr": int(value.data_ptr()),
        }
        if _DSA_DIAG_TENSOR_STATS and value.numel() > 0:
            try:
                sample = value.detach().reshape(-1)
                if sample.device.type != "cpu":
                    sample = sample.to(device="cpu")
                sample = sample[: min(max_items, int(sample.numel()))]
                summary["head"] = sample.tolist()
                if value.dtype.is_floating_point:
                    sample_f = sample.float()
                    summary["sum_head"] = float(sample_f.sum().item())
                    summary["min_head"] = float(sample_f.min().item())
                    summary["max_head"] = float(sample_f.max().item())
            except Exception as exc:
                summary["stats_error"] = str(exc)
        return summary
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": [
                _diag_tensor_summary(item, max_items=max_items)
                for item in list(value)[:2]
            ],
        }
    return value


def _diag_tensor_list_summary(tensors: List[torch.Tensor]) -> dict[str, Any]:
    return {
        "count": len(tensors),
        "head": [
            _diag_tensor_summary(tensor)
            for tensor in tensors[:2]
        ],
    }


def _diag_tensor_digest(tensor: torch.Tensor) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tensor.dtype).encode("utf-8"))
    h.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
    try:
        payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    except Exception:
        payload = repr(tensor.tolist()).encode("utf-8")
    h.update(payload)
    return h.hexdigest()


def _diag_tensor_fingerprint(
    value: Any,
    max_items: int = 16,
) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "value": repr(value)}
    summary: dict[str, Any] = {
        "type": "Tensor",
        "shape": tuple(int(dim) for dim in value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
    }
    if value.numel() == 0:
        summary["digest"] = "empty"
        return summary
    flat = value.detach().reshape(-1)
    if flat.device.type != "cpu":
        flat = flat.to(device="cpu")
    flat = flat.contiguous()
    summary["digest"] = _diag_tensor_digest(flat)
    head = flat[: min(max_items, int(flat.numel()))]
    tail = flat[-min(max_items, int(flat.numel())) :]
    try:
        summary["head"] = head.tolist()
        summary["tail"] = tail.tolist()
    except Exception as exc:
        summary["values_error"] = str(exc)
    if value.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ):
        try:
            summary["min"] = int(flat.min().item())
            summary["max"] = int(flat.max().item())
        except Exception as exc:
            summary["range_error"] = str(exc)
    return summary


def _diag_int_tensor_values(value: Any) -> list[int]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    flat = value.detach().reshape(-1)
    if flat.device.type != "cpu":
        flat = flat.to(device="cpu")
    return [int(item) for item in flat.tolist()]

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
        self.use_mla = kwargs.get("use_mla", False)
        self.dsa_two_groups = kwargs.get("dsa_two_groups", False)
        self.max_staging_tokens = int(kwargs.get("max_staging_tokens", 0) or 0)
        # Concurrent layerwise staging buffers per kv_group (retrieve batch +
        # overlapping store). Default 2 covers retrieve+store for one request.
        self._layerwise_staging_concurrency = (
            2 if self.dsa_two_groups else 1
        )

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
        # Sparse direct state is keyed by kvcaches/group/layer plus source
        # layout metadata so the two groups never collide and a new source
        # layout cannot reuse stale host-side kernel config.
        self._sparse_direct_layer_states: Optional[dict] = None
        self._sparse_direct_kvcaches_id: Optional[int] = None
        self._sparse_direct_validated_layers: set = set()
        self._dsa_diag_first_token_compare_seen: Set[tuple] = set()
        self._dsa_diag_first_token_dump_seen: Set[tuple] = set()

    def _reset_sparse_direct_layer_states(self) -> None:
        self._sparse_direct_layer_states = None
        self._sparse_direct_kvcaches_id = None
        self._sparse_direct_validated_layers = set()

    def _group_layout(self, kv_group: int) -> _GroupLayout:
        """Return the layout for ``kv_group``, raising if not initialized."""
        layout = self._group_layouts.get(kv_group)
        if layout is None:
            raise RuntimeError(
                f"kv_group={kv_group} layout not initialized. "
                f"_lazy_initialize_buffer must be called first."
            )
        return layout

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
    ) -> None:
        """Resolve and append NPU device ptrs for newly retrieved chunks only."""
        if not new_tensors:
            return

        new_dev_ptrs = [
            self._resolve_registered_cpu_tensor_device_ptr(
                tensor,
                layer_id=layer_id,
                chunk_index=chunk_index,
                source="append_sparse_chunk_ptr_cache_for_layer",
            )
            for chunk_index, tensor in enumerate(new_tensors)
        ]

        updated_ptrs_npu = None
        if cached_chunk_ptrs_npu is not None:
            new_ptrs_npu = torch.tensor(
                new_dev_ptrs, dtype=torch.long, device=self.kv_device
            )
            existing = (
                cached_chunk_ptrs_npu[layer_id]
                if layer_id < len(cached_chunk_ptrs_npu)
                else None
            )
            updated_ptrs_npu = (
                new_ptrs_npu
                if existing is None
                else torch.cat((existing, new_ptrs_npu), dim=0)
            )

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

        cached_chunk_dev_ptrs[layer_id].extend(new_dev_ptrs)

        if cached_chunk_ptrs_npu is None:
            return

        cached_chunk_ptrs_npu[layer_id] = updated_ptrs_npu

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
                VLLMPagedMemLayerwiseNPUConnector._tensor_identity_signature(
                    tensor
                )
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
    def _sparse_direct_pointer_cache_signature(
        *,
        chunk_ptrs_npu: torch.Tensor,
        slot_mapping_ref: torch.Tensor,
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

        The expensive per-CPU-chunk signature is still available for preflight
        and tests. During decode, chunk_ptrs_npu has already been constructed
        from the resolved CPU MemoryObjs, so its tensor identity is enough to
        distinguish the installed source pointer table without walking every
        chunk again.
        """
        return (
            1,
            int(chunk_ptrs_npu.data_ptr()),
            int(chunk_ptrs_npu.numel()),
            int(slot_mapping_ref.data_ptr()),
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
    def _tensor_identity_signature(tensor) -> tuple:
        if isinstance(tensor, torch.Tensor):
            return (
                int(tensor.data_ptr()),
                tuple(int(dim) for dim in tensor.shape),
                tuple(int(stride) for stride in tensor.stride()),
                tensor.dtype,
                str(tensor.device),
            )
        return (type(tensor).__name__, id(tensor))

    @staticmethod
    def _tensor_collection_identity_signature(value) -> tuple:
        if isinstance(value, (tuple, list)):
            return tuple(
                VLLMPagedMemLayerwiseNPUConnector._tensor_identity_signature(
                    tensor
                )
                for tensor in value
            )
        return (
            VLLMPagedMemLayerwiseNPUConnector._tensor_identity_signature(value),
        )

    @staticmethod
    def _vllm_layer_cache_identity_signature(value) -> tuple:
        if isinstance(value, (tuple, list)):
            return (id(value), len(value))
        return (id(value),)

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

        vllm_layer_cache = kvcaches_ref[layer_id]
        vllm_tensor_count = (
            len(vllm_layer_cache)
            if isinstance(vllm_layer_cache, (tuple, list))
            else 1
        )

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
            if not isinstance(target_slot_mapping, torch.Tensor):
                target_slot_mapping = torch.tensor(
                    target_slot_mapping, dtype=torch.long, device=self.kv_device
                )
            slot_mapping_packed = target_slot_mapping.reshape(-1).to(
                device=self.kv_device, dtype=torch.long
            )
            selected_token_idx = self._sparse_selected_token_idx(
                selected_token_idx, slot_mapping_packed.shape[0]
            )
            if selected_token_idx.numel() != slot_mapping_packed.numel():
                raise ValueError(
                    "Sparse target_slot_mapping must match selected_token_idx "
                    f"length: target_slots={slot_mapping_packed.numel()} "
                    f"selected_tokens={selected_token_idx.numel()}"
                )
            return slot_mapping_packed, selected_token_idx

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
                slot_chunks = []
                selected_chunks = []
                for row_idx in range(rows.shape[0]):
                    row = rows[row_idx]
                    if start_values is not None:
                        start = start_values[row_idx]
                    else:
                        start = int(
                            starts[row_idx].detach().to(device="cpu").item()
                        )
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
                    torch.cat(slot_chunks, dim=0)
                    if slot_chunks else slot_mapping[:0]
                )
                selected_token_idx = (
                    torch.cat(selected_chunks, dim=0)
                    if selected_chunks else selected_token_idx.reshape(-1)[:0]
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
        target_slot_mapping = target_slot_mapping.reshape(-1).to(
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

    def _kv_cache_token_capacity(
        self,
        layer_id: int,
        kvcaches_ref: Optional[list] = None,
        kv_group: Optional[int] = None,
    ) -> Optional[int]:
        kvcaches_ref = kvcaches_ref if kvcaches_ref is not None else self.kvcaches
        if kvcaches_ref is None or layer_id >= len(kvcaches_ref):
            return None
        layer_cache = kvcaches_ref[layer_id]
        if isinstance(layer_cache, (list, tuple)):
            if not layer_cache:
                return None
            ref_tensor = layer_cache[0]
        else:
            layout = (
                self._group_layouts.get(kv_group)
                if kv_group is not None else None
            )
            vllm_two_major = (
                layout.vllm_two_major if layout is not None else self.vllm_two_major
            )
            ref_tensor = layer_cache[0] if vllm_two_major else layer_cache[:, 0]
        if not isinstance(ref_tensor, torch.Tensor) or ref_tensor.dim() < 2:
            return None
        return int(ref_tensor.shape[0]) * int(ref_tensor.shape[1])

    def _int_tensor_summary(
        self,
        tensor: torch.Tensor,
        name: str,
        sample_size: int = 8,
    ) -> tuple[str, Optional[int], Optional[int], list[int]]:
        flat = tensor.detach().reshape(-1)
        numel = int(flat.numel())
        if numel == 0:
            return f"{name}_shape={tuple(tensor.shape)} {name}_numel=0", None, None, []
        try:
            min_value = int(flat.min().to(device="cpu").item())
            max_value = int(flat.max().to(device="cpu").item())
            sample = [
                int(value)
                for value in flat[: min(sample_size, numel)].to(device="cpu").tolist()
            ]
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to read DSA sparse explicit transfer tensor for "
                f"validation: name={name} shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype} device={tensor.device}"
            ) from exc
        summary = (
            f"{name}_shape={tuple(tensor.shape)} {name}_numel={numel} "
            f"{name}_min={min_value} {name}_max={max_value} "
            f"{name}_sample={sample}"
        )
        return summary, min_value, max_value, sample

    def _dsa_diag_source_planes(
        self,
        kv_group: int,
    ) -> tuple[str, list[tuple[str, int]]]:
        layout = self._layout_for(kv_group)
        if layout is None:
            return "uninitialized", []
        fmt = layout.kv_format
        if fmt == KVCacheFormat.DSA_INDEX:
            return fmt.name, [
                ("index", int(layout.dsa_hidden_dims or layout.k_hidden_dims))
            ]
        if fmt in (KVCacheFormat.MLA_LATENT, KVCacheFormat.MLA_KV):
            return fmt.name, [
                ("latent_k", int(layout.k_hidden_dims)),
                ("latent_v", int(layout.v_hidden_dims)),
            ]
        if fmt == KVCacheFormat.DSA_KV:
            return fmt.name, [
                ("latent_k", int(layout.k_hidden_dims)),
                ("latent_v", int(layout.v_hidden_dims)),
                ("index", int(layout.dsa_hidden_dims)),
            ]
        return fmt.name, []

    def _dsa_diag_selected_source_fingerprint(
        self,
        *,
        cpu_tensors: List[torch.Tensor],
        selected_token_idx: torch.Tensor,
        kv_group: int,
        chunk_size: int,
    ) -> dict[str, Any]:
        selected_values = _diag_int_tensor_values(selected_token_idx)
        kv_format, planes = self._dsa_diag_source_planes(kv_group)
        summary: dict[str, Any] = {
            "kv_format": kv_format,
            "chunk_size": int(chunk_size),
            "tensor_count": len(cpu_tensors),
            "selected_count": len(selected_values),
            "selected_head": selected_values[:_DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED],
            "source_tensors": [
                {
                    "shape": tuple(int(dim) for dim in tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "data_ptr": int(tensor.data_ptr()),
                    "numel": int(tensor.numel()),
                }
                for tensor in cpu_tensors[:4]
            ],
        }
        if not cpu_tensors:
            summary["digest"] = "no_source_tensors"
            return summary
        if not selected_values:
            summary["digest"] = "no_selected_tokens"
            return summary
        if chunk_size <= 0:
            summary["digest"] = "invalid_chunk_size"
            return summary
        if not planes:
            summary["digest"] = "unsupported_format"
            return summary

        digest = hashlib.blake2b(digest_size=16)
        samples: list[dict[str, Any]] = []
        errors: list[str] = []
        max_hash_selected = max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_HASH_SELECTED)
        max_sample_selected = max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED)
        summary["hashed_selected_count"] = min(
            len(selected_values),
            max_hash_selected,
        )
        for token_id in selected_values[:max_hash_selected]:
            chunk_index = int(token_id) // int(chunk_size)
            token_offset = int(token_id) % int(chunk_size)
            record_sample = len(samples) < max_sample_selected
            token_sample: dict[str, Any] = {
                "token": int(token_id),
                "chunk": chunk_index,
                "offset": token_offset,
                "planes": [],
            }
            if chunk_index < 0 or chunk_index >= len(cpu_tensors):
                errors.append(
                    f"token={token_id} chunk={chunk_index} outside "
                    f"tensor_count={len(cpu_tensors)}"
                )
                if record_sample:
                    samples.append(token_sample)
                continue

            tensor = cpu_tensors[chunk_index]
            token_count = self._lmc_plane_num_tokens(tensor, kv_group)
            if token_offset < 0 or token_offset >= token_count:
                errors.append(
                    f"token={token_id} offset={token_offset} outside "
                    f"chunk_tokens={token_count}"
                )
                if record_sample:
                    samples.append(token_sample)
                continue

            flat = tensor.detach().reshape(-1)
            plane_base = 0
            for plane_name, plane_dim in planes:
                if plane_dim <= 0:
                    continue
                start = plane_base + token_offset * plane_dim
                end = start + plane_dim
                plane_base += token_count * plane_dim
                if start < 0 or end > int(flat.numel()):
                    errors.append(
                        f"token={token_id} plane={plane_name} slice=({start},{end}) "
                        f"outside numel={int(flat.numel())}"
                    )
                    continue
                row = flat[start:end]
                if row.device.type != "cpu":
                    row = row.to(device="cpu")
                row = row.contiguous()
                row_digest = _diag_tensor_digest(row)
                digest.update(str(int(token_id)).encode("utf-8"))
                digest.update(plane_name.encode("utf-8"))
                digest.update(row_digest.encode("utf-8"))
                if record_sample:
                    head_values = row[: min(
                        max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_VALUES),
                        int(row.numel()),
                    )]
                    try:
                        head = head_values.tolist()
                    except Exception as exc:
                        head = [f"head_error={exc}"]
                    token_sample["planes"].append(
                        {
                            "plane": plane_name,
                            "dim": plane_dim,
                            "digest": row_digest,
                            "head": head,
                        }
                    )
            if record_sample:
                samples.append(token_sample)

        summary["digest"] = digest.hexdigest()
        summary["sampled_count"] = len(samples)
        summary["samples"] = samples
        if errors:
            summary["errors"] = errors[:8]
        return summary

    def _dsa_diag_rank_info(self) -> tuple[int, int, int, Any]:
        try:
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
            tp_rank = int(getattr(tp_group, "rank_in_group", 0))
            tp_world_size = int(getattr(tp_group, "world_size", 1))
            global_rank = int(getattr(tp_group, "rank", -1))
            return tp_rank, tp_world_size, global_rank, tp_group
        except Exception:
            return -1, 1, -1, None

    def _dsa_diag_gather_rank_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[Any], int, int]:
        try:
            tp_rank, tp_world_size, global_rank, tp_group = (
                self._dsa_diag_rank_info()
            )
            payload["tp_rank"] = tp_rank
            payload["global_rank"] = global_rank
            if (
                tp_world_size <= 1
                or not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or tp_group is None
            ):
                return [payload], tp_rank, tp_world_size
            gathered: list[Any] = [None for _ in range(tp_world_size)]
            torch.distributed.all_gather_object(
                gathered,
                payload,
                group=getattr(tp_group, "cpu_group", None),
            )
            return gathered, tp_rank, tp_world_size
        except Exception as exc:
            payload["tp_gather_error"] = str(exc)
            return [payload], int(payload.get("tp_rank", -1)), 1

    def _dsa_diag_log_first_token_rank_compare(
        self,
        *,
        diag_context: Optional[dict[str, Any]],
        kv_group: int,
        layer_id: int,
        cpu_tensors: List[torch.Tensor],
        selected_token_idx: torch.Tensor,
        slot_mapping_packed: torch.Tensor,
        chunk_size: int,
        total_tokens: int,
        source: str,
        ptr_cache_reused: bool,
    ) -> None:
        if not _DSA_DIAG_FIRST_TOKEN_COMPARE:
            return
        if (
            _DSA_DIAG_FIRST_TOKEN_COMPARE_LAYERS >= 0
            and layer_id >= _DSA_DIAG_FIRST_TOKEN_COMPARE_LAYERS
        ):
            return
        if selected_token_idx is None or int(selected_token_idx.numel()) == 0:
            return
        diag_context = diag_context or {}
        req_id = diag_context.get("req_id", "unspecified")
        diag_session = diag_context.get("diag_session")
        prompt_digest = diag_context.get("prompt_digest")
        prompt_run = diag_context.get("prompt_run")
        seen_key = (diag_session, req_id, int(kv_group), int(layer_id))
        if seen_key in self._dsa_diag_first_token_compare_seen:
            return
        self._dsa_diag_first_token_compare_seen.add(seen_key)

        local_error = None
        try:
            selected_fp = _diag_tensor_fingerprint(
                selected_token_idx,
                max_items=max(1, _DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED),
            )
            slot_fp = _diag_tensor_fingerprint(
                slot_mapping_packed,
                max_items=max(1, _DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED),
            )
            source_fp = self._dsa_diag_selected_source_fingerprint(
                cpu_tensors=cpu_tensors,
                selected_token_idx=selected_token_idx,
                kv_group=kv_group,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            local_error = str(exc)
            selected_fp = {"digest": "local_fingerprint_error"}
            slot_fp = {"digest": "local_fingerprint_error"}
            source_fp = {
                "digest": "local_fingerprint_error",
                "kv_format": "unknown",
                "error": local_error,
            }
        payload = {
            "req_id": req_id,
            "diag_session": diag_session,
            "prompt_digest": prompt_digest,
            "prompt_run": prompt_run,
            "kv_group": int(kv_group),
            "kv_format": source_fp.get("kv_format"),
            "layer": int(layer_id),
            "source": source,
            "ptr_cache_reused": bool(ptr_cache_reused),
            "total_tokens": int(total_tokens),
            "chunk_size": int(chunk_size),
            "selected_digest": selected_fp.get("digest"),
            "slot_digest": slot_fp.get("digest"),
            "source_digest": source_fp.get("digest"),
            "selected": selected_fp,
            "slot": slot_fp,
            "source_rows": source_fp,
        }
        if local_error is not None:
            payload["local_fingerprint_error"] = local_error
        gathered, tp_rank, tp_world_size = self._dsa_diag_gather_rank_payload(payload)
        if tp_rank not in (0, -1):
            return

        fields = ("selected_digest", "slot_digest", "source_digest")
        distinct = {
            field: sorted({repr(item.get(field)) for item in gathered if item})
            for field in fields
        }
        mismatched = [
            field for field, values in distinct.items() if len(values) > 1
        ]
        logger.warning(
            "[DSA_DIAG_RANK_COMPARE] first_token_input req_id=%s "
            "diag_session=%s prompt_digest=%s prompt_run=%s kv_group=%s layer=%s "
            "tp_world_size=%s mismatch=%s mismatch_fields=%s distinct=%s "
            "ranks=%s",
            req_id,
            diag_session,
            prompt_digest,
            prompt_run,
            kv_group,
            layer_id,
            tp_world_size,
            bool(mismatched),
            mismatched,
            distinct,
            gathered,
        )

    def _dsa_diag_safe_name(self, value: Any) -> str:
        raw = str(value if value is not None else "none")
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
            for ch in raw
        )[:160]

    def _dsa_diag_tensor_map_summary(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], str]:
        combined = hashlib.blake2b(digest_size=16)
        summary: dict[str, Any] = {}
        for name in sorted(tensors):
            tensor = tensors[name]
            flat = tensor.detach().reshape(-1).contiguous()
            digest = _diag_tensor_digest(flat)
            combined.update(name.encode("utf-8"))
            combined.update(digest.encode("utf-8"))
            summary[name] = {
                "shape": tuple(int(dim) for dim in tensor.shape),
                "dtype": str(tensor.dtype),
                "digest": digest,
            }
        return summary, combined.hexdigest()

    def _dsa_diag_selected_source_rows(
        self,
        *,
        cpu_tensors: List[torch.Tensor],
        selected_token_idx: torch.Tensor,
        kv_group: int,
        chunk_size: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        selected_values = _diag_int_tensor_values(selected_token_idx)
        kv_format, planes = self._dsa_diag_source_planes(kv_group)
        max_selected = max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_HASH_SELECTED)
        rows_by_plane: dict[str, list[torch.Tensor]] = {
            plane_name: [] for plane_name, plane_dim in planes if plane_dim > 0
        }
        token_records: list[dict[str, int]] = []
        errors: list[str] = []
        if chunk_size <= 0:
            errors.append(f"invalid_chunk_size={chunk_size}")
        for token_id in selected_values[:max_selected]:
            if chunk_size <= 0:
                break
            chunk_index = int(token_id) // int(chunk_size)
            token_offset = int(token_id) % int(chunk_size)
            token_records.append(
                {
                    "token": int(token_id),
                    "chunk": int(chunk_index),
                    "offset": int(token_offset),
                }
            )
            if chunk_index < 0 or chunk_index >= len(cpu_tensors):
                errors.append(
                    f"token={token_id} chunk={chunk_index} outside "
                    f"tensor_count={len(cpu_tensors)}"
                )
                continue
            tensor = cpu_tensors[chunk_index]
            token_count = self._lmc_plane_num_tokens(tensor, kv_group)
            if token_offset < 0 or token_offset >= token_count:
                errors.append(
                    f"token={token_id} offset={token_offset} outside "
                    f"chunk_tokens={token_count}"
                )
                continue
            flat = tensor.detach().reshape(-1)
            plane_base = 0
            for plane_name, plane_dim in planes:
                if plane_dim <= 0:
                    continue
                start = plane_base + token_offset * plane_dim
                end = start + plane_dim
                plane_base += token_count * plane_dim
                if start < 0 or end > int(flat.numel()):
                    errors.append(
                        f"token={token_id} plane={plane_name} "
                        f"slice=({start},{end}) outside numel={int(flat.numel())}"
                    )
                    continue
                row = flat[start:end]
                if row.device.type != "cpu":
                    row = row.to(device="cpu")
                rows_by_plane.setdefault(plane_name, []).append(row.contiguous())

        row_tensors: dict[str, torch.Tensor] = {}
        for plane_name, rows in rows_by_plane.items():
            if rows:
                row_tensors[plane_name] = torch.stack(rows, dim=0).contiguous()
            else:
                row_tensors[plane_name] = torch.empty(0)

        meta = {
            "kv_format": kv_format,
            "selected_count": len(selected_values),
            "dumped_selected_count": min(len(selected_values), max_selected),
            "token_records_head": token_records[
                : max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED)
            ],
        }
        if errors:
            meta["errors"] = errors[:16]
        return row_tensors, meta

    def _dsa_diag_kv_cache_planes(
        self,
        *,
        kvcaches_ref: list,
        layer_id: int,
        kv_group: int,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        if kvcaches_ref is None or layer_id >= len(kvcaches_ref):
            return "missing", {}
        layer_cache = kvcaches_ref[layer_id]
        layout = self._layout_for(kv_group)
        fmt = layout.kv_format if layout is not None else self.kv_format
        if isinstance(layer_cache, (list, tuple)):
            if fmt == KVCacheFormat.DSA_INDEX and len(layer_cache) >= 1:
                return fmt.name, {"index": layer_cache[0]}
            if len(layer_cache) >= 3:
                return fmt.name, {
                    "latent_k": layer_cache[0],
                    "latent_v": layer_cache[1],
                    "index": layer_cache[2],
                }
            if len(layer_cache) >= 2:
                return fmt.name, {
                    "latent_k": layer_cache[0],
                    "latent_v": layer_cache[1],
                }
            if len(layer_cache) == 1:
                return fmt.name, {"value": layer_cache[0]}
            return fmt.name, {}
        if isinstance(layer_cache, torch.Tensor):
            vllm_two_major = (
                layout.vllm_two_major if layout is not None else self.vllm_two_major
            )
            if layer_cache.dim() >= 5 and vllm_two_major:
                return fmt.name, {
                    "latent_k": layer_cache[0],
                    "latent_v": layer_cache[1],
                }
            if layer_cache.dim() >= 5:
                return fmt.name, {
                    "latent_k": layer_cache[:, 0],
                    "latent_v": layer_cache[:, 1],
                }
            return fmt.name, {"value": layer_cache}
        return fmt.name, {}

    def _dsa_diag_destination_rows(
        self,
        *,
        kvcaches_ref: list,
        layer_id: int,
        kv_group: int,
        slot_mapping_packed: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        slot_values = _diag_int_tensor_values(slot_mapping_packed)
        max_selected = max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_HASH_SELECTED)
        slot_values = slot_values[:max_selected]
        kv_format, planes = self._dsa_diag_kv_cache_planes(
            kvcaches_ref=kvcaches_ref,
            layer_id=layer_id,
            kv_group=kv_group,
        )
        rows: dict[str, torch.Tensor] = {}
        errors: list[str] = []
        for plane_name, tensor in planes.items():
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"plane={plane_name} is not tensor: {type(tensor)!r}")
                continue
            if tensor.dim() < 2:
                errors.append(
                    f"plane={plane_name} tensor dim too small: shape={tuple(tensor.shape)}"
                )
                continue
            flat = tensor.reshape(-1, *tensor.shape[2:])
            if not slot_values:
                rows[plane_name] = torch.empty(0)
                continue
            slot_tensor = torch.tensor(
                slot_values,
                dtype=torch.long,
                device=tensor.device,
            )
            selected = flat.index_select(0, slot_tensor)
            rows[plane_name] = selected.detach().to(device="cpu").reshape(
                len(slot_values), -1
            ).contiguous()
        meta = {
            "kv_format": kv_format,
            "slot_count": int(slot_mapping_packed.numel()),
            "dumped_slot_count": len(slot_values),
            "slot_head": slot_values[: max(0, _DSA_DIAG_FIRST_TOKEN_COMPARE_SELECTED)],
        }
        if errors:
            meta["errors"] = errors[:16]
        return rows, meta

    def _dsa_diag_summary_path(
        self,
        *,
        diag_session: Any,
        prompt_digest: Any,
        prompt_run: Any,
        layer_id: int,
        kv_group: int,
    ) -> str:
        return os.path.join(
            _DSA_DIAG_FIRST_TOKEN_DUMP_DIR,
            (
                f"{self._dsa_diag_safe_name(diag_session)}"
                f"_{self._dsa_diag_safe_name(prompt_digest)}"
                f"_run{self._dsa_diag_safe_name(prompt_run)}"
                f"_layer{layer_id}_group{kv_group}_summary.json"
            ),
        )

    @staticmethod
    def _dsa_diag_summary_req_ids(summary: dict[str, Any]) -> list[str]:
        req_ids = {
            str(item.get("req_id"))
            for item in summary.get("ranks", [])
            if item and item.get("req_id") is not None
        }
        if summary.get("req_id") is not None:
            req_ids.add(str(summary["req_id"]))
        return sorted(req_ids)

    @staticmethod
    def _dsa_diag_noncolliding_path(path: str) -> str:
        if not os.path.exists(path):
            return path
        root, ext = os.path.splitext(path)
        return f"{root}_dup{os.getpid()}_{time.time_ns()}{ext}"

    def _dsa_diag_write_json(self, path: str, payload: dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        os.replace(tmp_path, path)

    def _dsa_diag_log_run_file_compare(
        self,
        *,
        current_summary: dict[str, Any],
        diag_session: Any,
        prompt_digest: Any,
        prompt_run: Any,
        layer_id: int,
        kv_group: int,
    ) -> None:
        try:
            run_int = int(prompt_run)
        except (TypeError, ValueError):
            return
        if run_int <= 1:
            return
        prev_path = self._dsa_diag_summary_path(
            diag_session=diag_session,
            prompt_digest=prompt_digest,
            prompt_run=run_int - 1,
            layer_id=layer_id,
            kv_group=kv_group,
        )
        if not os.path.exists(prev_path):
            logger.warning(
                "[DSA_DIAG_RUN_COMPARE] first_token_dump diag_session=%s "
                "prompt_digest=%s "
                "prompt_run=%s prev_run=%s kv_group=%s layer=%s "
                "prev_summary_missing=%s current_summary=%s",
                diag_session,
                prompt_digest,
                prompt_run,
                run_int - 1,
                kv_group,
                layer_id,
                prev_path,
                current_summary.get("summary_path"),
            )
            return
        with open(prev_path, "r", encoding="utf-8") as f:
            previous_summary = json.load(f)

        prev_req_ids = self._dsa_diag_summary_req_ids(previous_summary)
        curr_req_ids = self._dsa_diag_summary_req_ids(current_summary)
        shared_req_ids = sorted(set(prev_req_ids) & set(curr_req_ids))
        if shared_req_ids:
            logger.warning(
                "[DSA_DIAG_RUN_COMPARE] first_token_dump diag_session=%s "
                "prompt_digest=%s "
                "prev_run=%s prompt_run=%s kv_group=%s layer=%s "
                "compare_skipped=same_req_ids same_req_ids=%s "
                "prev_req_ids=%s curr_req_ids=%s prev_summary=%s "
                "current_summary=%s",
                diag_session,
                prompt_digest,
                run_int - 1,
                prompt_run,
                kv_group,
                layer_id,
                shared_req_ids,
                prev_req_ids,
                curr_req_ids,
                prev_path,
                current_summary.get("summary_path"),
            )
            return

        fields = (
            "selected_digest",
            "slot_digest",
            "source_value_digest",
            "dest_value_digest",
        )
        previous_by_rank = {
            str(item.get("tp_rank")): item
            for item in previous_summary.get("ranks", [])
            if item
        }
        current_by_rank = {
            str(item.get("tp_rank")): item
            for item in current_summary.get("ranks", [])
            if item
        }
        mismatches: dict[str, list[str]] = {}
        missing: dict[str, list[str]] = {}
        for rank in sorted(set(previous_by_rank) | set(current_by_rank)):
            prev = previous_by_rank.get(rank)
            curr = current_by_rank.get(rank)
            if prev is None or curr is None:
                missing[rank] = []
                if prev is None:
                    missing[rank].append("previous")
                if curr is None:
                    missing[rank].append("current")
                continue
            changed = [
                field for field in fields if prev.get(field) != curr.get(field)
            ]
            if changed:
                mismatches[rank] = changed

        logger.warning(
            "[DSA_DIAG_RUN_COMPARE] first_token_dump diag_session=%s "
            "prompt_digest=%s "
            "prev_run=%s prompt_run=%s kv_group=%s layer=%s mismatch=%s "
            "mismatches=%s missing=%s prev_req_ids=%s curr_req_ids=%s "
            "prev_summary=%s current_summary=%s",
            diag_session,
            prompt_digest,
            run_int - 1,
            prompt_run,
            kv_group,
            layer_id,
            bool(mismatches or missing),
            mismatches,
            missing,
            prev_req_ids,
            curr_req_ids,
            prev_path,
            current_summary.get("summary_path"),
        )

    def _dsa_diag_dump_first_token_tensors(
        self,
        *,
        diag_context: Optional[dict[str, Any]],
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        cpu_tensors: List[torch.Tensor],
        selected_token_idx: torch.Tensor,
        slot_mapping_packed: torch.Tensor,
        chunk_size: int,
        total_tokens: int,
        source: str,
        ptr_cache_reused: bool,
    ) -> None:
        if not _DSA_DIAG_FIRST_TOKEN_DUMP:
            return
        if (
            _DSA_DIAG_FIRST_TOKEN_COMPARE_LAYERS >= 0
            and layer_id >= _DSA_DIAG_FIRST_TOKEN_COMPARE_LAYERS
        ):
            return
        if selected_token_idx is None or int(selected_token_idx.numel()) == 0:
            return
        diag_context = diag_context or {}
        req_id = diag_context.get("req_id", "unspecified")
        diag_session = diag_context.get("diag_session")
        prompt_digest = diag_context.get("prompt_digest")
        prompt_run = diag_context.get("prompt_run")
        seen_key = (diag_session, req_id, int(kv_group), int(layer_id))
        if seen_key in self._dsa_diag_first_token_dump_seen:
            return
        self._dsa_diag_first_token_dump_seen.add(seen_key)

        local_payload: dict[str, Any] = {}
        dump_path = None
        try:
            os.makedirs(_DSA_DIAG_FIRST_TOKEN_DUMP_DIR, exist_ok=True)
            selected_cpu = selected_token_idx.detach().to(device="cpu").contiguous()
            slot_cpu = slot_mapping_packed.detach().to(device="cpu").contiguous()
            source_rows, source_meta = self._dsa_diag_selected_source_rows(
                cpu_tensors=cpu_tensors,
                selected_token_idx=selected_token_idx,
                kv_group=kv_group,
                chunk_size=chunk_size,
            )
            dest_rows, dest_meta = self._dsa_diag_destination_rows(
                kvcaches_ref=kvcaches_ref,
                layer_id=layer_id,
                kv_group=kv_group,
                slot_mapping_packed=slot_mapping_packed,
            )
            selected_digest = _diag_tensor_digest(selected_cpu.reshape(-1))
            slot_digest = _diag_tensor_digest(slot_cpu.reshape(-1))
            source_summary, source_value_digest = self._dsa_diag_tensor_map_summary(
                source_rows
            )
            dest_summary, dest_value_digest = self._dsa_diag_tensor_map_summary(
                dest_rows
            )

            tp_rank_value, _, global_rank_value, _ = self._dsa_diag_rank_info()
            dump_name = (
                f"{self._dsa_diag_safe_name(diag_session)}"
                f"_{self._dsa_diag_safe_name(prompt_digest)}"
                f"_run{self._dsa_diag_safe_name(prompt_run)}"
                f"_req{self._dsa_diag_safe_name(req_id)}"
                f"_layer{layer_id}_group{kv_group}_rank{tp_rank_value}.pt"
            )
            dump_path = os.path.join(_DSA_DIAG_FIRST_TOKEN_DUMP_DIR, dump_name)
            dump_path = self._dsa_diag_noncolliding_path(dump_path)
            torch.save(
                {
                    "meta": {
                        "req_id": req_id,
                        "diag_session": diag_session,
                        "prompt_digest": prompt_digest,
                        "prompt_run": prompt_run,
                        "kv_group": int(kv_group),
                        "layer": int(layer_id),
                        "source": source,
                        "ptr_cache_reused": bool(ptr_cache_reused),
                        "total_tokens": int(total_tokens),
                        "chunk_size": int(chunk_size),
                        "tp_rank": tp_rank_value,
                        "global_rank": int(global_rank_value),
                        "source_meta": source_meta,
                        "dest_meta": dest_meta,
                    },
                    "selected_token_idx": selected_cpu,
                    "slot_mapping": slot_cpu,
                    "source_rows": source_rows,
                    "dest_rows": dest_rows,
                },
                dump_path,
            )
            local_payload = {
                "req_id": req_id,
                "diag_session": diag_session,
                "prompt_digest": prompt_digest,
                "prompt_run": prompt_run,
                "kv_group": int(kv_group),
                "layer": int(layer_id),
                "source": source,
                "ptr_cache_reused": bool(ptr_cache_reused),
                "total_tokens": int(total_tokens),
                "chunk_size": int(chunk_size),
                "dump_path": dump_path,
                "selected_digest": selected_digest,
                "slot_digest": slot_digest,
                "source_value_digest": source_value_digest,
                "dest_value_digest": dest_value_digest,
                "source_summary": source_summary,
                "dest_summary": dest_summary,
                "source_meta": source_meta,
                "dest_meta": dest_meta,
            }
        except Exception as exc:
            local_payload = {
                "req_id": req_id,
                "diag_session": diag_session,
                "prompt_digest": prompt_digest,
                "prompt_run": prompt_run,
                "kv_group": int(kv_group),
                "layer": int(layer_id),
                "dump_path": dump_path,
                "dump_error": str(exc),
            }

        gathered, tp_rank, tp_world_size = self._dsa_diag_gather_rank_payload(
            local_payload
        )
        if tp_rank not in (0, -1):
            return

        fields = (
            "selected_digest",
            "slot_digest",
            "source_value_digest",
            "dest_value_digest",
        )
        distinct = {
            field: sorted({repr(item.get(field)) for item in gathered if item})
            for field in fields
        }
        mismatched = [
            field for field, values in distinct.items() if len(values) > 1
        ]
        summary_path = self._dsa_diag_summary_path(
            diag_session=diag_session,
            prompt_digest=prompt_digest,
            prompt_run=prompt_run,
            layer_id=layer_id,
            kv_group=kv_group,
        )
        summary = {
            "summary_path": summary_path,
            "req_id": req_id,
            "diag_session": diag_session,
            "prompt_digest": prompt_digest,
            "prompt_run": prompt_run,
            "kv_group": int(kv_group),
            "layer": int(layer_id),
            "tp_world_size": int(tp_world_size),
            "mismatch": bool(mismatched),
            "mismatch_fields": mismatched,
            "distinct": distinct,
            "ranks": gathered,
        }
        current_req_ids = self._dsa_diag_summary_req_ids(summary)
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing_summary = json.load(f)
                existing_req_ids = self._dsa_diag_summary_req_ids(
                    existing_summary
                )
                shared_req_ids = sorted(
                    set(existing_req_ids) & set(current_req_ids)
                )
                if shared_req_ids:
                    logger.warning(
                        "[DSA_DIAG_TENSOR_DUMP] first_token_dump_duplicate_skipped "
                        "req_id=%s diag_session=%s prompt_digest=%s "
                        "prompt_run=%s kv_group=%s layer=%s same_req_ids=%s "
                        "existing_summary=%s existing_req_ids=%s "
                        "current_req_ids=%s",
                        req_id,
                        diag_session,
                        prompt_digest,
                        prompt_run,
                        kv_group,
                        layer_id,
                        shared_req_ids,
                        summary_path,
                        existing_req_ids,
                        current_req_ids,
                    )
                    return
        except Exception as exc:
            summary["summary_collision_check_error"] = str(exc)
        try:
            self._dsa_diag_write_json(summary_path, summary)
        except Exception as exc:
            summary["summary_write_error"] = str(exc)

        logger.warning(
            "[DSA_DIAG_TENSOR_DUMP] first_token_dump req_id=%s "
            "diag_session=%s prompt_digest=%s prompt_run=%s kv_group=%s layer=%s "
            "tp_world_size=%s mismatch=%s mismatch_fields=%s summary=%s "
            "ranks=%s",
            req_id,
            diag_session,
            prompt_digest,
            prompt_run,
            kv_group,
            layer_id,
            tp_world_size,
            bool(mismatched),
            mismatched,
            summary_path,
            gathered,
        )
        self._dsa_diag_log_run_file_compare(
            current_summary=summary,
            diag_session=diag_session,
            prompt_digest=prompt_digest,
            prompt_run=prompt_run,
            layer_id=layer_id,
            kv_group=kv_group,
        )

    def _validate_sparse_direct_explicit_inputs(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        slot_mapping_packed: torch.Tensor,
        selected_token_idx: torch.Tensor,
        chunk_size: int,
        total_tokens: int,
        chunk_ptrs_npu: torch.Tensor,
    ) -> None:
        selected_summary, selected_min, selected_max, _ = self._int_tensor_summary(
            selected_token_idx,
            "selected_token_idx",
        )
        slot_summary, slot_min, slot_max, _ = self._int_tensor_summary(
            slot_mapping_packed,
            "target_slot_mapping",
        )

        selected_numel = int(selected_token_idx.numel())
        slot_numel = int(slot_mapping_packed.numel())
        chunk_count = int(chunk_ptrs_npu.numel())
        kv_capacity = self._kv_cache_token_capacity(
            layer_id,
            kvcaches_ref=kvcaches_ref,
            kv_group=kv_group,
        )
        issues = []

        if selected_numel != slot_numel:
            issues.append(
                f"length_mismatch selected={selected_numel} target={slot_numel}"
            )
        if chunk_size <= 0:
            issues.append(f"invalid_chunk_size={chunk_size}")
        if total_tokens <= 0:
            issues.append(f"invalid_total_tokens={total_tokens}")
        if chunk_count <= 0:
            issues.append(f"invalid_chunk_ptr_count={chunk_count}")

        if selected_min is not None and selected_min < 0:
            issues.append(f"selected_min_negative={selected_min}")
        if (
            selected_max is not None
            and total_tokens > 0
            and selected_max >= total_tokens
        ):
            issues.append(
                f"selected_max_oob selected_max={selected_max} "
                f"total_tokens={total_tokens}"
            )
        if (
            selected_max is not None
            and chunk_size > 0
            and chunk_count > 0
            and selected_max // chunk_size >= chunk_count
        ):
            issues.append(
                f"selected_chunk_oob selected_max={selected_max} "
                f"chunk_size={chunk_size} chunk_ptr_count={chunk_count}"
            )

        if slot_min is not None and slot_min < 0:
            issues.append(f"target_slot_min_negative={slot_min}")
        if (
            slot_max is not None
            and kv_capacity is not None
            and slot_max >= kv_capacity
        ):
            issues.append(
                f"target_slot_max_oob target_slot_max={slot_max} "
                f"kv_capacity={kv_capacity}"
            )

        if issues:
            raise RuntimeError(
                "DSA sparse explicit transfer input invalid: "
                f"kv_group={kv_group} layer_id={layer_id} "
                f"total_tokens={total_tokens} "
                f"chunk_size={chunk_size} chunk_ptr_count={chunk_count} "
                f"kv_capacity={kv_capacity} {'; '.join(issues)} "
                f"{selected_summary} {slot_summary}"
            )

    def _run_sparse_staging_kv_transfer_layer(
        self,
        *,
        kvcaches_ref: list,
        kv_group: int,
        layer_id: int,
        load_stream: torch.cuda.Stream,
        current_stream: torch.cuda.Stream,
        slot_mapping_packed: torch.Tensor,
        selected_token_idx: torch.Tensor,
        total_tokens: int,
        sparse_kv_format: int,
        sparse_token_major: bool,
        sparse_vllm_two_major: bool,
        sparse_k_hidden_dims: int,
        sparse_v_hidden_dims: int,
        sparse_dsa_hidden_dims: int,
        layer_tensors: List[torch.Tensor],
        payload_event: Optional[Any] = None,
        payload_stream: Optional[Any] = None,
        explicit_sparse_payload: bool = False,
    ) -> None:
        """Diagnostic fallback: CPU chunks -> NPU staging -> paged KV.

        This keeps the same sparse source rows and target slots as the direct
        registered-host path, but routes data through the older staging kernel.
        It is intentionally behind LMCACHE_ASCEND_SPARSE_DIRECT_DISABLE because
        it is slower and uses a full retrieved-token staging buffer.
        """
        num_sparse = int(selected_token_idx.numel())
        if num_sparse == 0 or not layer_tensors:
            return

        chunk_offsets: list[int] = []
        chunk_sizes: list[int] = []
        covered_tokens = 0
        for tensor in layer_tensors:
            chunk_tokens = self._lmc_plane_num_tokens(tensor, kv_group)
            chunk_offsets.append(covered_tokens)
            chunk_sizes.append(chunk_tokens)
            covered_tokens += chunk_tokens

        if total_tokens <= 0:
            total_tokens = covered_tokens
        if covered_tokens < int(total_tokens):
            raise ValueError(
                "Sparse staging fallback has insufficient CPU chunk tokens: "
                f"kv_group={kv_group} layer_id={layer_id} "
                f"covered_tokens={covered_tokens} total_tokens={int(total_tokens)}"
            )
        if _SPARSE_DIRECT_GUARD:
            self._validate_sparse_direct_explicit_inputs(
                kvcaches_ref=kvcaches_ref,
                kv_group=kv_group,
                layer_id=layer_id,
                slot_mapping_packed=slot_mapping_packed,
                selected_token_idx=selected_token_idx,
                chunk_size=max(chunk_sizes) if chunk_sizes else 0,
                total_tokens=covered_tokens,
                chunk_ptrs_npu=torch.ones(
                    len(chunk_sizes), dtype=torch.long, device=self.kv_device
                ),
            )

        expected_fmt = self._expected_memory_format(kv_group)
        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        staging_tensor: Optional[torch.Tensor] = None
        try:
            tmp_gpu_buffer_obj, staging_tensor = (
                self._allocate_layerwise_staging_buffer(
                    num_tokens=covered_tokens,
                    kv_group=kv_group,
                    layout=self._group_layout(kv_group),
                    k_hidden_dims=sparse_k_hidden_dims,
                    v_hidden_dims=sparse_v_hidden_dims,
                    dsa_hidden_dims=sparse_dsa_hidden_dims,
                    expected_fmt=expected_fmt,
                )
            )
            with torch.cuda.stream(load_stream):
                load_stream.wait_stream(current_stream)
                _wait_stream_probe("before_transfer", payload_stream, load_stream)
                payload_events = _payload_event_list(payload_event)
                if payload_events:
                    for event in payload_events:
                        load_stream.wait_event(event)
                    if not _publish_current_npu_stream():
                        raise RuntimeError(
                            "Failed to publish load stream after waiting on "
                            "DSA sparse payload event before staging transfer."
                        )
                _sync_probe("before_transfer", load_stream)
                assert staging_tensor is not None
                batched_fused_sparse_single_layer_kv_transfer(
                    layer_tensors,
                    staging_tensor,
                    kvcaches_ref[layer_id],
                    slot_mapping_packed,
                    selected_token_idx,
                    chunk_offsets,
                    chunk_sizes,
                    sparse_kv_format,
                    sparse_token_major,
                    sparse_vllm_two_major,
                    sparse_k_hidden_dims,
                    sparse_v_hidden_dims,
                    sparse_dsa_hidden_dims,
                )
        finally:
            current_stream.wait_stream(load_stream)
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

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
        payload_event: Optional[Any] = None,
        payload_stream: Optional[Any] = None,
        explicit_sparse_payload: bool = False,
        diag_context: Optional[dict[str, Any]] = None,
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
        if _SPARSE_DIRECT_GUARD:
            selected_min = int(selected_token_idx.min().to(device="cpu").item())
            selected_max = int(selected_token_idx.max().to(device="cpu").item())
            if selected_min < 0 or selected_max >= int(total_tokens):
                message = (
                    "Sparse direct retrieve selected token is outside source chunks: "
                    f"kv_group={kv_group} layer_id={layer_id} "
                    f"num_sparse={num_sparse} selected_min={selected_min} "
                    f"selected_max={selected_max} chunk_count={chunk_count} "
                    f"chunk_size={chunk_size_int} covered_tokens={covered_tokens} "
                    f"total_tokens={int(total_tokens)}"
                )
                logger.error(message)
                raise ValueError(message)

        if _SPARSE_DIRECT_RECORD_STREAM:
            for tensor in (slot_mapping_packed, selected_token_idx, chunk_ptrs_npu):
                try:
                    tensor.record_stream(load_stream)
                except RuntimeError:
                    # Some backends/tensor types may not support record_stream.
                    # The transfer still has explicit stream ordering below.
                    pass

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
        runtime_source_signature = self._sparse_direct_pointer_cache_signature(
            chunk_ptrs_npu=chunk_ptrs_npu,
            slot_mapping_ref=resolve_slot_mapping,
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
        if _DSA_DIAG:
            logger.warning(
                "[DSA_DIAG] sparse_direct_temp context=%s layer_state=%s "
                "validate_key=%s runtime_signature=%s",
                diag_context,
                layer_state is not None,
                validate_key,
                runtime_source_signature,
            )
        _stream_diag(
            "connector_sparse_direct_before_launch",
            layer_id=layer_id,
            kv_group=kv_group,
            current_stream=_describe_stream(current_stream),
            load_stream=_describe_stream(load_stream),
            payload_event_count=len(_payload_event_list(payload_event)),
            explicit_sparse_payload=explicit_sparse_payload,
            layer_state=layer_state is not None,
        )
        with torch.cuda.stream(load_stream):
            load_stream.wait_stream(current_stream)
            _wait_stream_probe("before_transfer", payload_stream, load_stream)
            payload_events = _payload_event_list(payload_event)
            if payload_events:
                for event in payload_events:
                    load_stream.wait_event(event)
                if not _publish_current_npu_stream():
                    raise RuntimeError(
                        "Failed to publish load stream after waiting on DSA "
                        "sparse payload event before direct transfer."
                    )
            _sync_probe("before_transfer", load_stream)
            if explicit_sparse_payload and _SPARSE_DIRECT_GUARD:
                self._validate_sparse_direct_explicit_inputs(
                    kvcaches_ref=kvcaches_ref,
                    kv_group=kv_group,
                    layer_id=layer_id,
                    slot_mapping_packed=slot_mapping_packed,
                    selected_token_idx=selected_token_idx,
                    chunk_size=chunk_size,
                    total_tokens=total_tokens,
                    chunk_ptrs_npu=chunk_ptrs_npu,
                )
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
        if _DSA_DIAG:
            logger.warning(
                "[DSA_DIAG] sparse_direct_output context=%s kvcache=%s",
                diag_context,
                _diag_tensor_summary(kvcaches_ref[layer_id]),
            )

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

    def _is_latent_format(self, kv_group: Optional[int] = None) -> bool:
        fmt = self._fmt_for(kv_group)
        return fmt in (KVCacheFormat.MLA_KV, KVCacheFormat.MLA_LATENT)

    def _is_indexer_format(self, kv_group: Optional[int] = None) -> bool:
        return self._fmt_for(kv_group) == KVCacheFormat.DSA_INDEX

    def _layerwise_token_major(self, kv_group: Optional[int] = None) -> bool:
        # GQA uses token-interleaved CPU chunks; MLA/DSA use stacked K|V|DSA planes.
        return not self._is_mla_dsa_format(kv_group)

    def _sparse_lmc_host_interleaved(self, kv_group: Optional[int] = None) -> bool:
        # Must match batched_fused CPU layout (_layerwise_token_major).
        return self._layerwise_token_major(kv_group)

    def notify_sparse_memory_objs_updated(self) -> None:
        """Reset fast-path state after sparse source MemoryObjs change.

        Chunk device pointers live in ReqMeta and update incrementally, but
        SparseDirectLayerState also caches layout derived from a sample source
        tensor. A new retrieve request can switch from Mooncake materialization
        to rank0 hot-cache shared views while keeping the same vLLM KV cache,
        so the state key alone is not enough to prove the source layout is
        still current.
        """
        self._reset_sparse_direct_layer_states()

    def invalidate_sparse_chunk_ptr_cache(self) -> None:
        """Reset fast-path state when sparse pointer cache is invalidated."""
        self._reset_sparse_direct_layer_states()

    def _resolve_sparse_chunk_ptrs_npu(
        self,
        layer_id: int,
        cpu_tensors: List[torch.Tensor],
        cached_chunk_ptrs_npu: Optional[List[Optional[torch.Tensor]]] = None,
        cached_chunk_dev_ptrs: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        num_chunks = len(cpu_tensors)
        if (
            cached_chunk_ptrs_npu is not None
            and layer_id < len(cached_chunk_ptrs_npu)
        ):
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
                if _SPARSE_POINTER_CACHE_REUSE_VALIDATE_NULLS and bool(
                    torch.any(cached == 0).item()
                ):
                    raise RuntimeError(
                        "Ascend sparse pointer-cache reuse failed: cached NPU "
                        f"pointer tensor contains a null pointer at layer {layer_id}."
                    )
                if _SPARSE_POINTER_CACHE_REUSE_VALIDATE_PTRS:
                    expected_dev_ptrs = [
                        self._resolve_registered_cpu_tensor_device_ptr(
                            tensor,
                            layer_id=layer_id,
                            chunk_index=chunk_index,
                            source="_resolve_sparse_chunk_ptrs_npu_validate",
                        )
                        for chunk_index, tensor in enumerate(cpu_tensors)
                    ]
                    cached_dev_ptrs = None
                    if (
                        cached_chunk_dev_ptrs is not None
                        and layer_id < len(cached_chunk_dev_ptrs)
                    ):
                        cached_dev_ptrs = cached_chunk_dev_ptrs[layer_id]
                    if cached_dev_ptrs != expected_dev_ptrs:
                        cached_values = [
                            int(value)
                            for value in cached.detach().to(device="cpu").tolist()
                        ]
                        raise RuntimeError(
                            "Ascend sparse pointer-cache reuse failed: cached "
                            "device pointers do not match current CPU tensors "
                            "before direct kernel launch. "
                            f"layer_id={layer_id}, "
                            f"cached_chunk_dev_ptrs={cached_dev_ptrs}, "
                            f"expected_dev_ptrs={expected_dev_ptrs}, "
                            f"cached_npu_values={cached_values}"
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

    def _single_layer_hidden_dim_args(
        self, kv_group: Optional[int] = None
    ) -> tuple[int, int, int]:
        layout = self._layout_for(kv_group)
        if layout is not None:
            return (
                layout.k_hidden_dims,
                layout.v_hidden_dims,
                layout.dsa_hidden_dims,
            )
        return (self.k_hidden_dims, self.v_hidden_dims, self.dsa_hidden_dims)

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

    def _sparse_retrieve_total_tokens(
        self, lmc_tensors: List[torch.Tensor], kv_group: Optional[int] = None
    ) -> int:
        num_chunks = len(lmc_tensors)
        if num_chunks == 0:
            return 0
        assert self.lmcache_chunk_size > 0, (
            "chunk_size must be configured for sparse layerwise retrieve."
        )
        if num_chunks == 1:
            return self._lmc_plane_num_tokens(lmc_tensors[0], kv_group)
        last_tokens = self._lmc_plane_num_tokens(lmc_tensors[-1], kv_group)
        return (num_chunks - 1) * self.lmcache_chunk_size + last_tokens

    def _expected_memory_format(
        self, kv_group: Optional[int] = None
    ) -> MemoryFormat:
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
            self._assign_group_gpu_allocator(new_size, layout, kv_group)

    def _layerwise_staging_pool_slots(self) -> int:
        if not self.dsa_two_groups:
            return 1
        return max(1, self._layerwise_staging_concurrency)

    def _staging_pool_stats(
        self, layout: _GroupLayout
    ) -> dict[str, int]:
        alloc = layout.gpu_buffer_allocator
        if alloc is None:
            return {}
        inner = alloc.allocator
        return {
            "pool_total_bytes": int(alloc.tensor.numel()),
            "pool_free_bytes": int(inner.address_manager.get_free_size()),
            "pool_allocated_bytes": int(
                inner.address_manager.total_allocated_size
            ),
            "active_allocs": int(inner.num_active_allocations),
        }

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
            logger.warning(
                "%s layerwise transfer has mismatched layer counts: "
                "kv_group=%s connector_num_layers=%s kvcaches_len=%s",
                operation,
                kv_group,
                self.num_layers,
                kvcaches_len,
            )
        if slot_mapping_full is None or slot_mapping_full.numel() == 0:
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
        k_hidden_dims: int,
        v_hidden_dims: int,
        dsa_hidden_dims: int,
        expected_fmt: MemoryFormat,
    ) -> tuple[Optional[MemoryObj], torch.Tensor]:
        """Return (pool_obj_or_none, staging_tensor) for a layerwise transfer."""
        self._check_staging_transfer_tokens(num_tokens, kv_group)

        gpu_buffer_allocator = layout.gpu_buffer_allocator
        assert gpu_buffer_allocator is not None, (
            f"GPU staging pool for kv_group={kv_group} is not initialized."
        )
        buffer_shape = self.get_shape(num_tokens, kv_group)
        request_bytes = int(
            buffer_shape.numel() * self.element_size
            if hasattr(buffer_shape, "numel")
            else buffer_shape[0] * self.element_size
        )
        pool_stats = self._staging_pool_stats(layout)
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
        self, gpu_buffer_size: int, layout: _GroupLayout, kv_group: int
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
        self, kv_caches, kv_group: int = 0
    ) -> _GroupLayout:
        """
        Lazily initialize per-kv_group format metadata and GPU buffer allocator.

        In two-group MLA+DSA mode the same connector instance handles both
        kv_group=0 (latent) and kv_group=1 (indexer); each group is detected
        and allocated independently on its first call instead of the first
        group pinning a single ``self.kv_format`` for both.
        """
        self._current_kv_group = kv_group
        layout = self._group_layouts.get(kv_group)
        if layout is None:
            layout = _GroupLayout()
            # Latent (kv_group=0) uses MLA_KV detection like dev-qzy; only the
            # indexer group (kv_group=1) needs dsa_two_groups detection.
            detect_two_groups = (
                getattr(self, "dsa_two_groups", False) and kv_group != 0
            )
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
                raise ValueError(
                    f"Unsupported KV cache format: {layout.kv_format}"
                )

            self._group_layouts[kv_group] = layout

        # Mirror into instance attributes for backward-compatible readers.
        self._mirror_layout(layout)

        if self.use_gpu and layout.gpu_buffer_allocator is None:
            logger.info(
                f"Lazily initializing GPU buffer (kv_group={kv_group})."
            )
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
                    layout.k_hidden_dims
                    + layout.v_hidden_dims
                    + layout.dsa_hidden_dims
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

            self._assign_group_gpu_allocator(gpu_buffer_size, layout, kv_group)
            if self.dsa_two_groups:
                logger.info(
                    "dsa_two_groups: per-group NPU staging pool "
                    f"(kv_group={kv_group}, cap={staging_tokens} tokens, "
                    f"slots={pool_slots}, "
                    f"{gpu_buffer_size / (1024 * 1024):.2f} MB)"
                )

        return layout

    def get_shape(
        self, num_tokens: int, kv_group: Optional[int] = None
    ) -> torch.Size:
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
        layout = self._lazy_initialize_buffer(self.kvcaches, kv_group=kv_group)

        if self._is_mla_dsa_format(kv_group) and not self.use_gpu:
            raise ValueError(
                "MLA/DSA layerwise transfer requires use_gpu=True with a staging buffer."
            )

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        self._check_staging_transfer_tokens(num_tokens, kv_group)
        self._check_layerwise_transfer_invariants(
            operation="retrieve",
            kv_group=kv_group,
            slot_mapping_full=slot_mapping_full,
            kvcaches_ref=self.kvcaches,
        )


        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

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

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        staging_tensor: Optional[torch.Tensor] = None
        if self.use_gpu:
            tmp_gpu_buffer_obj, staging_tensor = self._allocate_layerwise_staging_buffer(
                num_tokens=num_tokens,
                kv_group=kv_group,
                layout=layout,
                k_hidden_dims=k_hidden_dims,
                v_hidden_dims=v_hidden_dims,
                dsa_hidden_dims=dsa_hidden_dims,
                expected_fmt=expected_fmt,
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
        kv_group = kwargs.get("kv_group", 0)
        layout = self._lazy_initialize_buffer(self.kvcaches, kv_group=kv_group)

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]
        cached_tensors_by_layer: Optional[List[List[torch.Tensor]]] = kwargs.get(
            "cached_tensors"
        )
        cached_chunk_dev_ptrs: Optional[List[List[int]]] = kwargs.get(
            "cached_chunk_dev_ptrs"
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

        for layer_id in range(self.num_layers):
            sparse_request = yield
            # The generator is resumed from vLLM's attention path; refresh the
            # active compute stream per layer before ordering load -> compute.
            current_stream = torch.cuda.current_stream()
            load_stream = self.load_stream_list[load_stream_idx]
            explicit_sparse_payload = isinstance(sparse_request, dict)
            target_slot_mapping = None
            payload_event = None
            payload_stream = None
            if explicit_sparse_payload:
                memory_objs_layer = sparse_request["memory_objs_layer"]
                selected_token_idx = sparse_request.get("selected_token_ids")
                token_start_index = sparse_request.get("token_start_index", 0)
                target_slot_mapping = sparse_request.get("target_slot_mapping")
                payload_event = sparse_request.get(
                    "payload_events", sparse_request.get("payload_event")
                )
                payload_stream = sparse_request.get(
                    "payload_streams", sparse_request.get("payload_stream")
                )
                explicit_sparse_payload = target_slot_mapping is not None
            elif isinstance(sparse_request, tuple):
                if len(sparse_request) == 4:
                    (
                        memory_objs_layer,
                        selected_token_idx,
                        token_start_index,
                        target_slot_mapping,
                    ) = sparse_request
                    explicit_sparse_payload = target_slot_mapping is not None
                else:
                    (
                        memory_objs_layer,
                        selected_token_idx,
                        token_start_index,
                    ) = sparse_request
            else:
                memory_objs_layer = sparse_request
                selected_token_idx = None
                token_start_index = 0

            _stream_diag(
                "connector_before_payload_wait",
                layer_id=layer_id,
                kv_group=kv_group,
                current_stream=_describe_stream(current_stream),
                load_stream=_describe_stream(load_stream),
                payload_event_count=len(_payload_event_list(payload_event)),
                payload_stream_count=len(_payload_stream_list(payload_stream)),
                explicit_sparse_payload=explicit_sparse_payload,
            )
            payload_events = _payload_event_list(payload_event)
            with torch.cuda.stream(current_stream):
                # selected_token_idx/target_slot_mapping may be device tensors
                # produced by vLLM's remap path. Packing below is their first
                # connector-side consumer, so wait before packing, not only
                # later inside the load-stream transfer.
                if payload_events:
                    for event in payload_events:
                        current_stream.wait_event(event)
                    if not _publish_current_npu_stream():
                        raise RuntimeError(
                            "Failed to publish current stream after waiting on DSA "
                            "sparse payload event before packing sparse inputs."
                        )
                    _stream_diag(
                        "connector_after_payload_wait",
                        layer_id=layer_id,
                        kv_group=kv_group,
                        current_stream=_describe_stream(current_stream),
                        load_stream=_describe_stream(load_stream),
                        payload_event_count=len(payload_events),
                    )
                _wait_stream_probe("before_pack", payload_stream, current_stream)
                _sync_probe("before_pack", current_stream)

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
                _sync_probe("after_pack", current_stream)

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

            if not cpu_tensors:
                continue

            ptr_cache_reused = (
                cached_chunk_ptrs_npu is not None
                and layer_id < len(cached_chunk_ptrs_npu)
                and cached_chunk_ptrs_npu[layer_id] is not None
                and cached_chunk_ptrs_npu[layer_id].numel() == len(cpu_tensors)
            )
            chunk_ptrs_npu = self._resolve_sparse_chunk_ptrs_npu(
                layer_id,
                cpu_tensors,
                cached_chunk_ptrs_npu,
                cached_chunk_dev_ptrs,
            )
            if lmcache_cached_tokens > 0:
                total_tokens = lmcache_cached_tokens
            else:
                total_tokens = self._sparse_total_tokens_from_layer_chunks(
                    cpu_tensors, kv_group
                )

            diag_context = None
            if _DSA_DIAG:
                diag_context = {
                    "req_id": kwargs.get("req_id", "unspecified"),
                    "diag_session": kwargs.get("_dsa_diag_session_id"),
                    "prompt_digest": kwargs.get("_dsa_diag_prompt_digest"),
                    "prompt_run": kwargs.get("_dsa_diag_prompt_run"),
                    "kv_group": kv_group,
                    "layer": layer_id,
                    "source": (
                        "cached_tensors"
                        if layer_cached_tensors is not None
                        else "memory_objs"
                    ),
                }
                logger.warning(
                    "[DSA_DIAG] connector_sparse_input context=%s explicit=%s "
                    "ptr_reused=%s total_tokens=%s chunk_size=%s "
                    "cpu_tensors=%s chunk_ptrs=%s selected=%s slot=%s "
                    "layout=%s",
                    diag_context,
                    explicit_sparse_payload,
                    ptr_cache_reused,
                    total_tokens,
                    chunk_size,
                    _diag_tensor_list_summary(cpu_tensors),
                    _diag_tensor_summary(chunk_ptrs_npu),
                    _diag_tensor_summary(selected_token_idx),
                    _diag_tensor_summary(slot_mapping_packed),
                    {
                        "format": sparse_kv_format,
                        "token_major": sparse_token_major,
                        "host_interleaved": sparse_host_interleaved,
                        "k": sparse_k_hidden_dims,
                        "v": sparse_v_hidden_dims,
                        "dsa": sparse_dsa_hidden_dims,
                    },
                )

            self._dsa_diag_log_first_token_rank_compare(
                diag_context=diag_context,
                kv_group=kv_group,
                layer_id=layer_id,
                cpu_tensors=cpu_tensors,
                selected_token_idx=selected_token_idx,
                slot_mapping_packed=slot_mapping_packed,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                source=(
                    "cached_tensors"
                    if layer_cached_tensors is not None
                    else "memory_objs"
                ),
                ptr_cache_reused=ptr_cache_reused,
            )

            if _SPARSE_DIRECT_DISABLE:
                self._run_sparse_staging_kv_transfer_layer(
                    kvcaches_ref=kvcaches_snapshot,
                    kv_group=kv_group,
                    layer_id=layer_id,
                    load_stream=self.load_stream_list[load_stream_idx],
                    current_stream=current_stream,
                    slot_mapping_packed=slot_mapping_packed,
                    selected_token_idx=selected_token_idx,
                    total_tokens=total_tokens,
                    sparse_kv_format=sparse_kv_format,
                    sparse_token_major=sparse_token_major,
                    sparse_vllm_two_major=sparse_vllm_two_major,
                    sparse_k_hidden_dims=sparse_k_hidden_dims,
                    sparse_v_hidden_dims=sparse_v_hidden_dims,
                    sparse_dsa_hidden_dims=sparse_dsa_hidden_dims,
                    layer_tensors=cpu_tensors,
                    payload_event=payload_event,
                    payload_stream=payload_stream,
                    explicit_sparse_payload=explicit_sparse_payload,
                )
            else:
                self._run_sparse_direct_kv_transfer_layer(
                    kvcaches_ref=kvcaches_snapshot,
                    kv_group=kv_group,
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
                    slot_mapping_ref=(
                        slot_mapping_packed
                        if explicit_sparse_payload else slot_mapping
                    ),
                    cpu_tensors=cpu_tensors,
                    payload_event=payload_event,
                    payload_stream=payload_stream,
                    explicit_sparse_payload=explicit_sparse_payload,
                    diag_context=diag_context,
                )

            self._dsa_diag_dump_first_token_tensors(
                diag_context=diag_context,
                kvcaches_ref=kvcaches_snapshot,
                kv_group=kv_group,
                layer_id=layer_id,
                cpu_tensors=cpu_tensors,
                selected_token_idx=selected_token_idx,
                slot_mapping_packed=slot_mapping_packed,
                chunk_size=chunk_size,
                total_tokens=total_tokens,
                source=(
                    "cached_tensors"
                    if layer_cached_tensors is not None
                    else "memory_objs"
                ),
                ptr_cache_reused=ptr_cache_reused,
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

        kv_group = kwargs.get("kv_group", 0)
        layout = self._lazy_initialize_buffer(self.kvcaches, kv_group=kv_group)

        if self._is_mla_dsa_format(kv_group) and not self.use_gpu:
            raise ValueError(
                "MLA/DSA layerwise transfer requires use_gpu=True with a staging buffer."
            )

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        self._check_staging_transfer_tokens(num_tokens, kv_group)
        self._check_layerwise_transfer_invariants(
            operation="store",
            kv_group=kv_group,
            slot_mapping_full=slot_mapping_full,
            kvcaches_ref=self.kvcaches,
        )


        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

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

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        staging_tensor: Optional[torch.Tensor] = None
        if self.use_gpu:
            tmp_gpu_buffer_obj, staging_tensor = self._allocate_layerwise_staging_buffer(
                num_tokens=num_tokens,
                kv_group=kv_group,
                layout=layout,
                k_hidden_dims=k_hidden_dims,
                v_hidden_dims=v_hidden_dims,
                dsa_hidden_dims=dsa_hidden_dims,
                expected_fmt=expected_fmt,
            )

        current_stream = torch.npu.current_stream()

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = memory_objs[layer_id]
                # kvcaches -> gpu_buffer -> memobj
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
