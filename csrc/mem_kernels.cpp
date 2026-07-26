#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>

namespace py = pybind11;

static bool is_mla_dsa_format(kvcache_ops::KVCacheFormat format) {
  return format == kvcache_ops::KVCacheFormat::MLA_KV ||
         format == kvcache_ops::KVCacheFormat::DSA_KV ||
         format == kvcache_ops::KVCacheFormat::MLA_LATENT ||
         format == kvcache_ops::KVCacheFormat::DSA_INDEX;
}

// Map two-group formats to the kernel template they reuse.
// MLA_LATENT has the same 2-plane (k_nope, k_pe) structure as MLA_KV.
// DSA_INDEX is a single plane; mapped to MLA_KV with v_hidden_dims=0
// (V-plane copy becomes a no-op).
static kvcache_ops::KVCacheFormat
kernel_format(kvcache_ops::KVCacheFormat format) {
  if (format == kvcache_ops::KVCacheFormat::DSA_KV)
    return kvcache_ops::KVCacheFormat::DSA_KV;
  // MLA_KV, MLA_LATENT, DSA_INDEX all use the MLA_KV kernel template.
  return kvcache_ops::KVCacheFormat::MLA_KV;
}

static void launch_single_layer_mla_dsa_kernel(const SingleLayerKVConfig &config,
                                               bool page2L) {
  kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa(
      config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
      kernel_format(config.kvcache_format), config.ub_params.aiv_num, config.ub_params.stream,
      config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
      config.ptrs.vllm_dsa_ptr, config.ptrs.slot_mapping_ptr,
      config.strides.lmc_bytes, config.strides.vllm_k_bytes,
      config.strides.vllm_v_bytes, config.strides.vllm_dsa_bytes,
      config.ub_params.max_tokens_per_loop, config.k_hidden_dims,
      config.v_hidden_dims, config.dsa_hidden_dims, config.dims.num_tokens,
      config.dims.lmc_num_tokens, config.dims.block_size, page2L,
      config.token_major);
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each AIV Core processes the copy for a token
 *
 * Therefore:
 *  AIV Core - token
 *
 * The function does:
 * slot_id = slot_mapping[tokenId]
 * ptrs[mem_offset(kv, layer, tokenId, hiddenDims)] = key_value[mem_offset(kv,
 * layer, pages, pageSize, slot_id, hiddenDims)]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);

  // Calculate UB buffer parameters
  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, key_value_ptr]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, key_value_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction,
        config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims);
    return 0;
  });
  cmd.Run();
  return;
};

void fused_multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    torch::Tensor &staging_cache,        // staging buffer
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, // true: from_gpu, false: to_gpu
    const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims) {
  // get host cpu buffer pointer for aclrtMemcpyAsync
  uint8_t *key_value_ptr = static_cast<uint8_t *>(key_value.data_ptr());
  uint8_t *staging_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_cache);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);

  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Calculate and verify the CPU buffer size
  // For MLA_KV and DSA_KV, K/V have different hidden_dims
  // Use staging_cache's actual size for verification
  size_t staging_cache_size =
      static_cast<size_t>(staging_cache.numel()) * staging_cache.element_size();

  size_t required_size = 0;
  switch (config.kvcache_format) {
  case kvcache_ops::KVCacheFormat::MLA_KV:
  case kvcache_ops::KVCacheFormat::MLA_LATENT:
    required_size = static_cast<size_t>(config.num_layers) * config.num_tokens *
                    (config.k_hidden_dims + config.v_hidden_dims) *
                    key_value.element_size();
    break;
  case kvcache_ops::KVCacheFormat::DSA_KV:
    required_size =
        static_cast<size_t>(config.num_layers) * config.num_tokens *
        (config.k_hidden_dims + config.v_hidden_dims + config.dsa_hidden_dims) *
        key_value.element_size();
    break;
  case kvcache_ops::KVCacheFormat::DSA_INDEX:
    required_size = static_cast<size_t>(config.num_layers) * config.num_tokens *
                    config.dsa_hidden_dims * key_value.element_size();
    break;
  default:
    required_size = static_cast<size_t>(config.kv_size) * config.num_layers *
                    config.num_tokens * config.hidden_dims *
                    key_value.element_size();
    break;
  }

  TORCH_CHECK(staging_cache_size >= required_size,
              "staging_cache size insufficient: need ", required_size,
              " bytes, got ", staging_cache_size);

  at_npu::native::OpCommand cmd;
  cmd.Name("fused_multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, staging_cache_ptr, key_value_ptr,
                        required_size]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    aclError ret;
    // direction: false = to_gpu (H2D), true = from_gpu (D2H)
    bool isH2D = !config.direction;

    // Step 1: H2D memcpy (to_gpu) currently not used
    if (isH2D) {
      ret = aclrtMemcpyAsync(staging_cache_ptr, required_size, key_value_ptr,
                             required_size, ACL_MEMCPY_HOST_TO_DEVICE,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "H2D memcpy failed: cpu_buffer -> staging_cache, ret=", ret);
    }

    // Step 2: Kernel (Gather or Scatter)
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, staging_cache_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction,
        config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims);

    // Step 3: D2H memcpy (from_gpu)
    if (!isH2D) {
      ret = aclrtMemcpyAsync(key_value_ptr, required_size, staging_cache_ptr,
                             required_size, ACL_MEMCPY_DEVICE_TO_HOST,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "D2H memcpy failed: staging_cache -> cpu_buffer, ret=", ret);
    }

    return 0;
  });
  cmd.Run();
  return;
}

void multi_layer_kv_transfer_310p(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int num_kv_head,
    const int head_size, const int blockSize, const int kvcache_format_raw) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_310p");
  cmd.SetCustomHandler([config, stream, key_value_ptr, num_kv_head, head_size,
                        blockSize]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::multi_layer_kv_transfer_kernel_310p(
        dtype_num, slot_num, config.kvcache_format, aiv_num, stream,
        config.page_buffer_ptrs, key_value_ptr, config.slot_mapping_ptr,
        config.hidden_dims, config.kv_size, config.num_layers,
        config.page_buffer_size, config.num_tokens, config.direction,
        num_kv_head, head_size, blockSize);
    return 0;
  });
  cmd.Run();
  return;
};

void multi_layer_kv_transfer_unilateral(
    torch::Tensor &key_value, const torch::Tensor &key_ptrs,
    const torch::Tensor &value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction) {
  // TODO:
  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void single_layer_kv_transfer(
    torch::Tensor
        &lmc_key_value_cache, // [num_tokens, 2, num_heads*head_size]
                              // or [2, num_tokens, num_heads*head_size]
    std::vector<torch::Tensor> &vllm_kv_caches,
    // SEPARATE_KV: list[k_tensor, v_tensor]
    // k_tensor/v_tensor = [num_blocks, block_size, num_heads, head_size]
    // MERGED_KV:
    // vllm_two_major=true:  [2, num_blocks, block_size, num_heads, head_size]
    // vllm_two_major=false: [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens]
    const bool direction, // false: LMCache -> Paged, true: Paged -> LMCache
    const int kvcache_format_raw, // 1: MERGED_KV, 2: SEPARATE_KV
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major, // true: [2, blocks, ...], false: [blocks, 2, ...]
                               // (only for MERGED_KV)
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims) {
  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_key_value_cache, vllm_kv_caches, slot_mapping, direction, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims);

  at_npu::native::OpCommand cmd;

  if (is_mla_dsa_format(config.kvcache_format)) {
    cmd.Name("single_layer_kv_transfer_kernel_v2_mla_dsa");
    cmd.SetCustomHandler([config]() -> int {
      launch_single_layer_mla_dsa_kernel(config, config.direction);
      return 0;
    });
  } else if (!is_separate) {
    cmd.Name("single_layer_kv_transfer_kernel_v2");
    cmd.SetCustomHandler([config]() -> int {
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, config.direction, config.token_major);
      return 0;
    });
  } else {
    cmd.Name("single_layer_kv_transfer_kernel_v2_separate");
    cmd.SetCustomHandler([config]() -> int {
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          config.direction, config.token_major);
      return 0;
    });
  }
  cmd.Run();
}

void batched_fused_single_layer_kv_transfer(
    std::vector<torch::Tensor>
        &lmc_tensors, // N CPU pinned memory tensors
                      // token_major=true:  [num_tokens, 2, num_heads*head_size]
                      // token_major=false: [2, num_tokens, num_heads*head_size]
    torch::Tensor &staging_cache, // NPU staging buffer
                                  // token_major=true:  [num_tokens, 2,
                                  // num_heads*head_size] token_major=false: [2,
                                  // num_tokens, num_heads*head_size]
    std::vector<torch::Tensor>    // separate format: list[k_tensor, v_tensor]
        &vllm_kv_caches, // k_tensor/v_tensor = [num_blocks, block_size,
                         // num_heads, head_size]
                         //  Merged format:
                         //  vllm_two_major=true:  [2, num_blocks, block_size,
                         //  num_heads, head_size] vllm_two_major=false:
                         //  [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping_full, // [num_tokens]
    std::vector<int64_t>
        &chunk_offsets,                // token offset in staging for each chunk
    std::vector<int64_t> &chunk_sizes, // token count for each chunk
    const bool direction, // false: CPU -> staging -> paged (to_gpu) true: paged
                          // -> staging -> CPU (from_gpu)
    const int kvcache_format_raw,
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major, // true: [2, blocks, ...], false: [blocks, 2, ...]
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims) {

  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(
      device_of(slot_mapping_full));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_cache, vllm_kv_caches, slot_mapping_full, direction, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims);

  int64_t element_size = staging_cache.element_size();

  if (is_mla_dsa_format(config.kvcache_format)) {
    auto launcher = [config](bool is_gather) {
      launch_single_layer_mla_dsa_kernel(config, is_gather);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);
  } else if (!is_separate) {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);

  } else {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);
  }
}

static void launch_sparse_single_layer_kernel(const SingleLayerKVConfig &config,
                                              uint8_t *selected_token_idx_ptr) {
  if (is_mla_dsa_format(config.kvcache_format)) {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_sparse(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        kernel_format(config.kvcache_format), config.ub_params.aiv_num, config.ub_params.stream,
        config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
        config.ptrs.vllm_dsa_ptr, config.ptrs.slot_mapping_ptr,
        selected_token_idx_ptr, config.strides.lmc_bytes,
        config.strides.vllm_k_bytes, config.strides.vllm_v_bytes,
        config.strides.vllm_dsa_bytes, config.ub_params.max_tokens_per_loop,
        config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims,
        config.dims.num_tokens, config.dims.lmc_num_tokens,
        config.dims.block_size, config.token_major);
    return;
  }

  bool is_separate =
      config.kvcache_format != kvcache_ops::KVCacheFormat::MERGED_KV;
  if (!is_separate) {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_sparse(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, config.ptrs.lmc_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.slot_mapping_ptr,
        selected_token_idx_ptr, config.strides.vllm_k_stride,
        config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
        config.strides.lmc_token_stride, config.strides.lmc_val_offset,
        config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
        config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
        config.dims.block_size, config.token_major);
  } else {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_separate_sparse(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, config.ptrs.lmc_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
        config.ptrs.slot_mapping_ptr, selected_token_idx_ptr,
        config.strides.vllm_k_stride, config.strides.vllm_v_stride,
        config.strides.vllm_k_bytes, config.strides.vllm_v_bytes,
        config.strides.lmc_token_stride, config.strides.lmc_val_offset,
        config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
        config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
        config.dims.block_size, config.token_major);
  }
}

void sparse_single_layer_kv_transfer(
    torch::Tensor &lmc_key_value_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    const int kvcache_format_raw, const bool token_major,
    const bool vllm_two_major, const int64_t k_hidden_dims,
    const int64_t v_hidden_dims, const int64_t dsa_hidden_dims) {
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);
  validate_sparse_single_layer_inputs(slot_mapping_packed, selected_token_idx);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_packed));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_key_value_cache, vllm_kv_caches, slot_mapping_packed, false,
      token_major, vllm_two_major, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);

  uint8_t *selected_token_idx_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(selected_token_idx);

  at_npu::native::OpCommand cmd;
  if (is_mla_dsa_format(config.kvcache_format)) {
    cmd.Name("single_layer_kv_transfer_kernel_v2_mla_dsa_sparse");
  } else {
    bool is_separate =
        config.kvcache_format != kvcache_ops::KVCacheFormat::MERGED_KV;
    cmd.Name(is_separate ? "single_layer_kv_transfer_kernel_v2_separate_sparse"
                         : "single_layer_kv_transfer_kernel_v2_sparse");
  }

  cmd.SetCustomHandler([config, selected_token_idx_ptr]() -> int {
    launch_sparse_single_layer_kernel(config, selected_token_idx_ptr);
    return 0;
  });
  cmd.Run();
}

namespace {

static void launch_sparse_multi_chunk_direct_kernel(
    const SingleLayerKVConfig &config, uint8_t *selected_token_idx_ptr,
    uint8_t *chunk_ptrs_ptr, int32_t num_chunks, int32_t chunk_size,
    int32_t total_tokens, bool lmc_host_interleaved,
    uint8_t *selected_token_counts_ptr = nullptr, int32_t row_width = 0,
    int32_t request_count = 0, int32_t selected_count_stride = 1) {
  if (is_mla_dsa_format(config.kvcache_format)) {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        kernel_format(config.kvcache_format), config.ub_params.aiv_num, config.ub_params.stream,
        chunk_ptrs_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
        config.ptrs.vllm_dsa_ptr, config.ptrs.slot_mapping_ptr,
        selected_token_idx_ptr, selected_token_counts_ptr,
        config.strides.vllm_k_bytes,
        config.strides.vllm_v_bytes, config.strides.vllm_dsa_bytes,
        config.ub_params.max_tokens_per_loop, config.k_hidden_dims,
        config.v_hidden_dims, config.dsa_hidden_dims, config.dims.num_tokens,
        num_chunks, chunk_size, total_tokens, config.dims.block_size,
        lmc_host_interleaved, row_width, request_count,
        selected_count_stride);
    return;
  }

  const bool is_separate =
      config.kvcache_format != kvcache_ops::KVCacheFormat::MERGED_KV;
  if (!is_separate) {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_sparse_multi_chunk(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, chunk_ptrs_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.slot_mapping_ptr,
        selected_token_idx_ptr, config.strides.vllm_k_stride,
        config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
        config.strides.lmc_token_stride, config.strides.lmc_val_offset,
        config.ub_params.max_tokens_per_loop, config.dims.num_heads,
        config.dims.head_dims, config.dims.num_tokens, num_chunks, chunk_size,
        total_tokens, config.dims.block_size, config.token_major);
  } else {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_separate_sparse_multi_chunk(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, chunk_ptrs_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
        config.ptrs.slot_mapping_ptr, selected_token_idx_ptr,
        config.strides.vllm_k_stride, config.strides.vllm_v_stride,
        config.strides.vllm_k_bytes, config.strides.vllm_v_bytes,
        config.strides.lmc_token_stride, config.strides.lmc_val_offset,
        config.ub_params.max_tokens_per_loop, config.dims.num_heads,
        config.dims.head_dims, config.dims.num_tokens, num_chunks, chunk_size,
        total_tokens, config.dims.block_size, config.token_major);
  }
}

static void launch_dense_multi_chunk_direct_kernel(
    const SingleLayerKVConfig &config, uint8_t *chunk_ptrs_ptr,
    uint8_t *chunk_offsets_ptr, uint8_t *chunk_sizes_ptr, int32_t num_chunks,
    int32_t fixed_chunk_size, int32_t total_tokens, bool lmc_host_interleaved,
    bool page2l) {
  TORCH_CHECK(is_mla_dsa_format(config.kvcache_format),
              "Dense direct multi-chunk transfer only supports MLA/DSA formats.");

  kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_dense_multi_chunk(
      config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
      kernel_format(config.kvcache_format), config.ub_params.aiv_num,
      config.ub_params.stream, chunk_ptrs_ptr, chunk_offsets_ptr,
      chunk_sizes_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
      config.ptrs.vllm_dsa_ptr, config.ptrs.slot_mapping_ptr,
      config.strides.vllm_k_bytes, config.strides.vllm_v_bytes,
      config.strides.vllm_dsa_bytes, config.ub_params.max_tokens_per_loop,
      config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims,
      config.dims.num_tokens, num_chunks, fixed_chunk_size, total_tokens,
      config.dims.block_size, page2l, lmc_host_interleaved);
}

static uint32_t direct_aiv_num(int32_t num_tokens) {
  const uint32_t token_cores =
      num_tokens > 0 ? static_cast<uint32_t>(num_tokens) : 1U;
  // The device core count is invariant for the lifetime of a worker process.
  static const uint32_t hardware_cores = [] {
    auto platform =
        platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    return std::max(1U, aiv_num);
  }();
  return std::min(hardware_cores, token_cores);
}

} // namespace

void sparse_mla_dsa_batched_direct_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    const int64_t chunk_size, const int64_t total_tokens,
    const int kvcache_format_raw, const bool token_major,
    const bool vllm_two_major, const int64_t k_hidden_dims,
    const int64_t v_hidden_dims, const int64_t dsa_hidden_dims,
    const bool lmc_host_interleaved,
    const c10::optional<torch::Tensor> &chunk_ptrs_npu,
    const c10::optional<torch::Tensor> &selected_token_counts) {
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);
  validate_sparse_single_layer_inputs(slot_mapping_packed, selected_token_idx);
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive.");
  TORCH_CHECK(total_tokens > 0, "total_tokens must be positive.");
  TORCH_CHECK(!lmc_tensors.empty(), "lmc_tensors must not be empty.");

  for (const auto &chunk : lmc_tensors) {
    TORCH_CHECK(
        chunk.device().is_cpu(),
        "Direct sparse retrieve requires CPU pinned memory objects.");
  }

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_packed));

  const int32_t num_sparse =
      static_cast<int32_t>(selected_token_idx.numel());
  if (num_sparse == 0) {
    return;
  }

  const int32_t num_chunks = static_cast<int32_t>(lmc_tensors.size());
  torch::Tensor chunk_ptrs_tensor;
  if (chunk_ptrs_npu.has_value() && chunk_ptrs_npu->defined() &&
      chunk_ptrs_npu->numel() == num_chunks) {
    chunk_ptrs_tensor = *chunk_ptrs_npu;
  } else {
    std::vector<int64_t> chunk_ptrs_host(num_chunks);
    for (int32_t chunk_i = 0; chunk_i < num_chunks; ++chunk_i) {
      chunk_ptrs_host[chunk_i] = reinterpret_cast<int64_t>(
          get_kernel_ptr<uint8_t, torch::Tensor>(lmc_tensors[chunk_i]));
    }
    auto npu_options = slot_mapping_packed.options();
    chunk_ptrs_tensor =
        torch::tensor(chunk_ptrs_host, npu_options.dtype(at::ScalarType::Long));
  }

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_tensors[0], vllm_kv_caches, slot_mapping_packed, false, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims);
  config.dims.num_tokens = num_sparse;
  const int32_t request_count = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->numel())
      : 0;
  const int32_t row_width =
      request_count > 0 ? num_sparse / request_count : 0;
  const int32_t selected_count_stride = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->stride(0))
      : 1;
  config.ub_params.aiv_num =
      direct_aiv_num(request_count > 0 ? request_count : num_sparse);

  uint8_t *selected_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(selected_token_idx);
  uint8_t *counts_ptr = selected_token_counts.has_value()
      ? get_kernel_ptr<uint8_t, torch::Tensor>(*selected_token_counts)
      : nullptr;
  uint8_t *chunk_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_ptrs_tensor);

  const int32_t chunk_size_i = static_cast<int32_t>(chunk_size);
  const int32_t total_tokens_i = static_cast<int32_t>(total_tokens);

  at_npu::native::OpCommand cmd;
  cmd.Name("sparse_mla_dsa_batched_direct_kv_transfer");
  cmd.SetCustomHandler([config, selected_ptr, counts_ptr, row_width,
                        request_count, selected_count_stride,
                        chunk_ptrs_ptr, num_chunks,
                        chunk_size_i, total_tokens_i,
                        lmc_host_interleaved]() -> int {
    launch_sparse_multi_chunk_direct_kernel(
        config, selected_ptr, chunk_ptrs_ptr, num_chunks, chunk_size_i,
        total_tokens_i, lmc_host_interleaved, counts_ptr, row_width,
        request_count, selected_count_stride);
    return 0;
  });
  cmd.Run();
}

void sparse_mla_dsa_batched_direct_kv_transfer_fast(
    SparseDirectLayerState &layer_state, torch::Tensor &slot_mapping_packed,
    torch::Tensor &selected_token_idx, torch::Tensor &chunk_ptrs_npu,
    const int64_t chunk_size, const int64_t total_tokens,
    const bool lmc_host_interleaved, const bool validate_inputs,
    const c10::optional<torch::Tensor> &selected_token_counts) {
  if (validate_inputs) {
    validate_sparse_single_layer_inputs(slot_mapping_packed, selected_token_idx);
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive.");
    TORCH_CHECK(total_tokens > 0, "total_tokens must be positive.");
    TORCH_CHECK(chunk_ptrs_npu.defined() && chunk_ptrs_npu.numel() > 0,
                "chunk_ptrs_npu must not be empty.");
    TORCH_CHECK(chunk_ptrs_npu.device().is_privateuseone(),
                "chunk_ptrs_npu must be on NPU.");
  }

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_packed));

  const int32_t num_sparse =
      static_cast<int32_t>(selected_token_idx.numel());
  if (num_sparse == 0) {
    return;
  }

  const int32_t num_chunks = static_cast<int32_t>(chunk_ptrs_npu.numel());

  SingleLayerKVConfig config = layer_state.config;
  config.dims.num_tokens = num_sparse;
  const int32_t request_count = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->numel())
      : 0;
  const int32_t row_width =
      request_count > 0 ? num_sparse / request_count : 0;
  const int32_t selected_count_stride = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->stride(0))
      : 1;
  config.ub_params.aiv_num =
      direct_aiv_num(request_count > 0 ? request_count : num_sparse);
  config.ub_params.stream = c10_npu::getCurrentNPUStream().stream();
  config.ptrs.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping_packed);

  uint8_t *selected_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(selected_token_idx);
  uint8_t *counts_ptr = selected_token_counts.has_value()
      ? get_kernel_ptr<uint8_t, torch::Tensor>(*selected_token_counts)
      : nullptr;
  uint8_t *chunk_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_ptrs_npu);

  const int32_t chunk_size_i = static_cast<int32_t>(chunk_size);
  const int32_t total_tokens_i = static_cast<int32_t>(total_tokens);

  at_npu::native::OpCommand cmd;
  cmd.Name("sparse_mla_dsa_batched_direct_kv_transfer");
  cmd.SetCustomHandler([config, selected_ptr, counts_ptr, row_width,
                        request_count, selected_count_stride,
                        chunk_ptrs_ptr, num_chunks,
                        chunk_size_i, total_tokens_i,
                        lmc_host_interleaved]() -> int {
    launch_sparse_multi_chunk_direct_kernel(
        config, selected_ptr, chunk_ptrs_ptr, num_chunks, chunk_size_i,
        total_tokens_i, lmc_host_interleaved, counts_ptr, row_width,
        request_count, selected_count_stride);
    return 0;
  });
  cmd.Run();
}

void sparse_mla_dsa_batched_direct_kv_transfer_prepared(
    const SparseDirectDestinationState &destination_state,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    torch::Tensor &chunk_ptrs_npu, const int64_t chunk_size,
    const int64_t total_tokens, const bool lmc_host_interleaved,
    const c10::optional<torch::Tensor> &selected_token_counts) {
  const c10::OptionalDeviceGuard slot_device_guard(
      device_of(slot_mapping_packed));

  const int32_t num_sparse = static_cast<int32_t>(selected_token_idx.numel());
  if (num_sparse == 0) {
    return;
  }

  const int32_t num_chunks = static_cast<int32_t>(chunk_ptrs_npu.numel());
  const int32_t request_count = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->numel())
      : 0;
  const int32_t row_width =
      request_count > 0 ? num_sparse / request_count : 0;
  const int32_t selected_count_stride = selected_token_counts.has_value()
      ? static_cast<int32_t>(selected_token_counts->stride(0))
      : 1;
  const uint32_t aiv_num =
      direct_aiv_num(request_count > 0 ? request_count : num_sparse);
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping_packed);
  uint8_t *selected_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(selected_token_idx);
  uint8_t *counts_ptr = selected_token_counts.has_value()
      ? get_kernel_ptr<uint8_t, torch::Tensor>(*selected_token_counts)
      : nullptr;
  uint8_t *chunk_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_ptrs_npu);

  const int32_t chunk_size_i = static_cast<int32_t>(chunk_size);
  const int32_t total_tokens_i = static_cast<int32_t>(total_tokens);
  const SparseDirectDestinationState state = destination_state;

  at_npu::native::OpCommand cmd;
  cmd.Name("sparse_mla_dsa_batched_direct_kv_transfer");
  cmd.SetCustomHandler([state, stream, aiv_num, slot_mapping_ptr, selected_ptr,
                        counts_ptr, row_width, request_count,
                        selected_count_stride,
                        chunk_ptrs_ptr, num_sparse, num_chunks, chunk_size_i,
                        total_tokens_i, lmc_host_interleaved]() -> int {
    kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk(
        state.scalar_type_num, state.slot_type_num,
        kernel_format(state.kvcache_format), aiv_num, stream, chunk_ptrs_ptr,
        state.vllm_k_ptr, state.vllm_v_ptr, state.vllm_dsa_ptr,
        slot_mapping_ptr, selected_ptr, counts_ptr, state.vllm_k_bytes,
        state.vllm_v_bytes,
        state.vllm_dsa_bytes, state.max_tokens_per_loop, state.k_hidden_dims,
        state.v_hidden_dims, state.dsa_hidden_dims, num_sparse, num_chunks,
        chunk_size_i, total_tokens_i, state.block_size, lmc_host_interleaved,
        row_width, request_count, selected_count_stride);
    return 0;
  });
  cmd.Run();
}

static void validate_dense_direct_inputs(torch::Tensor &slot_mapping_full,
                                         torch::Tensor &chunk_ptrs_npu,
                                         torch::Tensor &chunk_offsets_npu,
                                         torch::Tensor &chunk_sizes_npu,
                                         int64_t total_tokens,
                                         int64_t fixed_chunk_size) {
  TORCH_CHECK(slot_mapping_full.dim() == 1,
              "slot_mapping_full must be 1D.");
  TORCH_CHECK(chunk_ptrs_npu.dim() == 1, "chunk_ptrs_npu must be 1D.");
  TORCH_CHECK(chunk_offsets_npu.dim() == 1, "chunk_offsets_npu must be 1D.");
  TORCH_CHECK(chunk_sizes_npu.dim() == 1, "chunk_sizes_npu must be 1D.");
  TORCH_CHECK(slot_mapping_full.scalar_type() == at::ScalarType::Int ||
                  slot_mapping_full.scalar_type() == at::ScalarType::Long,
              "slot_mapping_full must be torch.int32 or torch.int64.");
  TORCH_CHECK(chunk_offsets_npu.scalar_type() == at::ScalarType::Int,
              "chunk_offsets_npu must be torch.int32.");
  TORCH_CHECK(chunk_sizes_npu.scalar_type() == at::ScalarType::Int,
              "chunk_sizes_npu must be torch.int32.");
  TORCH_CHECK(chunk_ptrs_npu.scalar_type() == at::ScalarType::Long,
              "chunk_ptrs_npu must be torch.int64.");
  if (fixed_chunk_size > 0) {
    TORCH_CHECK(chunk_offsets_npu.numel() > 0 && chunk_sizes_npu.numel() > 0,
                "fixed dense direct metadata tensors must not be empty.");
  } else {
    TORCH_CHECK(chunk_ptrs_npu.numel() == chunk_offsets_npu.numel() &&
                    chunk_ptrs_npu.numel() == chunk_sizes_npu.numel(),
                "chunk pointer, offset and size tensors must have the same length.");
  }
  TORCH_CHECK(chunk_ptrs_npu.numel() > 0, "chunk_ptrs_npu must not be empty.");
  TORCH_CHECK(total_tokens > 0, "total_tokens must be positive.");
  TORCH_CHECK(fixed_chunk_size >= 0,
              "fixed_chunk_size must be non-negative.");
  TORCH_CHECK(slot_mapping_full.size(0) <= total_tokens,
              "slot_mapping_full length cannot exceed total_tokens.");
  TORCH_CHECK(total_tokens <= std::numeric_limits<int32_t>::max(),
              "total_tokens exceeds dense direct int32 kernel limit.");
  TORCH_CHECK(fixed_chunk_size <= std::numeric_limits<int32_t>::max(),
              "fixed_chunk_size exceeds dense direct int32 kernel limit.");
  TORCH_CHECK(slot_mapping_full.size(0) <= std::numeric_limits<int32_t>::max(),
              "slot_mapping_full length exceeds dense direct int32 kernel limit.");
  TORCH_CHECK(chunk_ptrs_npu.numel() <= std::numeric_limits<int32_t>::max(),
              "chunk count exceeds dense direct int32 kernel limit.");
  if (fixed_chunk_size > 0) {
    const int64_t num_chunks = chunk_ptrs_npu.numel();
    TORCH_CHECK((num_chunks - 1) * fixed_chunk_size < total_tokens &&
                    num_chunks * fixed_chunk_size >= total_tokens,
                "fixed_chunk_size does not cover total_tokens.");
  }
  TORCH_CHECK(slot_mapping_full.device().is_privateuseone(),
              "slot_mapping_full must be on NPU.");
  TORCH_CHECK(chunk_ptrs_npu.device().is_privateuseone(),
              "chunk_ptrs_npu must be on NPU.");
  TORCH_CHECK(chunk_offsets_npu.device().is_privateuseone(),
              "chunk_offsets_npu must be on NPU.");
  TORCH_CHECK(chunk_sizes_npu.device().is_privateuseone(),
              "chunk_sizes_npu must be on NPU.");
}

void dense_mla_dsa_batched_direct_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_full, torch::Tensor &chunk_offsets_npu,
    torch::Tensor &chunk_sizes_npu, const int64_t total_tokens,
    const int kvcache_format_raw, const bool token_major,
    const bool vllm_two_major, const int64_t k_hidden_dims,
    const int64_t v_hidden_dims, const int64_t dsa_hidden_dims,
    const bool lmc_host_interleaved, const bool direction,
    const c10::optional<torch::Tensor> &chunk_ptrs_npu,
    const int64_t fixed_chunk_size) {
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);
  TORCH_CHECK(is_mla_dsa_format(
                  static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw)),
              "Dense direct transfer only supports MLA/DSA formats.");
  TORCH_CHECK(!lmc_tensors.empty(), "lmc_tensors must not be empty.");

  for (const auto &chunk : lmc_tensors) {
    TORCH_CHECK(chunk.device().is_cpu(),
                "Dense direct transfer requires CPU pinned memory objects.");
  }

  const int64_t num_chunks_i64 = static_cast<int64_t>(lmc_tensors.size());
  torch::Tensor chunk_ptrs_tensor;
  if (chunk_ptrs_npu.has_value() && chunk_ptrs_npu->defined() &&
      chunk_ptrs_npu->numel() == num_chunks_i64) {
    chunk_ptrs_tensor = *chunk_ptrs_npu;
  } else {
    std::vector<int64_t> chunk_ptrs_host(num_chunks_i64);
    for (int64_t chunk_i = 0; chunk_i < num_chunks_i64; ++chunk_i) {
      chunk_ptrs_host[chunk_i] = reinterpret_cast<int64_t>(
          get_kernel_ptr<uint8_t, torch::Tensor>(lmc_tensors[chunk_i]));
    }
    auto npu_options = slot_mapping_full.options();
    chunk_ptrs_tensor =
        torch::tensor(chunk_ptrs_host, npu_options.dtype(at::ScalarType::Long));
  }

  validate_dense_direct_inputs(slot_mapping_full, chunk_ptrs_tensor,
                               chunk_offsets_npu, chunk_sizes_npu,
                               total_tokens, fixed_chunk_size);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_full));

  const int32_t num_tokens = static_cast<int32_t>(slot_mapping_full.size(0));
  if (num_tokens == 0) {
    return;
  }

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_tensors[0], vllm_kv_caches, slot_mapping_full, direction,
      token_major, vllm_two_major, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);
  config.dims.num_tokens = num_tokens;
  config.ub_params.aiv_num = direct_aiv_num(num_tokens);

  uint8_t *chunk_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_ptrs_tensor);
  uint8_t *chunk_offsets_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_offsets_npu);
  uint8_t *chunk_sizes_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_sizes_npu);
  const int32_t total_tokens_i = static_cast<int32_t>(total_tokens);
  const int32_t fixed_chunk_size_i = static_cast<int32_t>(fixed_chunk_size);
  const int32_t num_chunks = static_cast<int32_t>(num_chunks_i64);

  at_npu::native::OpCommand cmd;
  cmd.Name("dense_mla_dsa_batched_direct_kv_transfer");
  cmd.SetCustomHandler([config, chunk_ptrs_ptr, chunk_offsets_ptr,
                        chunk_sizes_ptr, num_chunks, fixed_chunk_size_i,
                        total_tokens_i, lmc_host_interleaved, direction]() -> int {
    launch_dense_multi_chunk_direct_kernel(
        config, chunk_ptrs_ptr, chunk_offsets_ptr, chunk_sizes_ptr,
        num_chunks, fixed_chunk_size_i, total_tokens_i, lmc_host_interleaved,
        direction);
    return 0;
  });
  cmd.Run();
}

void dense_mla_dsa_batched_direct_kv_transfer_fast(
    SparseDirectLayerState &layer_state, torch::Tensor &slot_mapping_full,
    torch::Tensor &chunk_ptrs_npu, torch::Tensor &chunk_offsets_npu,
    torch::Tensor &chunk_sizes_npu, const int64_t total_tokens,
    const bool lmc_host_interleaved, const bool direction,
    const bool validate_inputs,
    const int64_t fixed_chunk_size) {
  if (validate_inputs) {
    validate_dense_direct_inputs(slot_mapping_full, chunk_ptrs_npu,
                                 chunk_offsets_npu, chunk_sizes_npu,
                                 total_tokens, fixed_chunk_size);
  }

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_full));

  const int32_t num_tokens = static_cast<int32_t>(slot_mapping_full.size(0));
  if (num_tokens == 0) {
    return;
  }

  const int32_t num_chunks = static_cast<int32_t>(chunk_ptrs_npu.numel());

  SingleLayerKVConfig config = layer_state.config;
  config.dims.num_tokens = num_tokens;
  config.ub_params.aiv_num = direct_aiv_num(num_tokens);
  config.ub_params.stream = c10_npu::getCurrentNPUStream().stream();
  config.ptrs.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping_full);

  uint8_t *chunk_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_ptrs_npu);
  uint8_t *chunk_offsets_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_offsets_npu);
  uint8_t *chunk_sizes_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(chunk_sizes_npu);
  const int32_t total_tokens_i = static_cast<int32_t>(total_tokens);
  const int32_t fixed_chunk_size_i = static_cast<int32_t>(fixed_chunk_size);

  at_npu::native::OpCommand cmd;
  cmd.Name("dense_mla_dsa_batched_direct_kv_transfer");
  cmd.SetCustomHandler([config, chunk_ptrs_ptr, chunk_offsets_ptr,
                        chunk_sizes_ptr, num_chunks, fixed_chunk_size_i,
                        total_tokens_i, lmc_host_interleaved, direction]() -> int {
    launch_dense_multi_chunk_direct_kernel(
        config, chunk_ptrs_ptr, chunk_offsets_ptr, chunk_sizes_ptr,
        num_chunks, fixed_chunk_size_i, total_tokens_i, lmc_host_interleaved,
        direction);
    return 0;
  });
  cmd.Run();
}

void batched_fused_sparse_single_layer_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    const int kvcache_format_raw, const bool token_major,
    const bool vllm_two_major, const int64_t k_hidden_dims,
    const int64_t v_hidden_dims, const int64_t dsa_hidden_dims,
    const c10::optional<torch::Tensor> &sparse_indices_cpu) {
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);
  validate_sparse_single_layer_inputs(slot_mapping_packed, selected_token_idx);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_packed));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_cache, vllm_kv_caches, slot_mapping_packed, false, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims);

  uint8_t *selected_token_idx_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(selected_token_idx);
  int64_t element_size = staging_cache.element_size();

  auto launcher = [config, selected_token_idx_ptr]() {
    launch_sparse_single_layer_kernel(config, selected_token_idx_ptr);
  };

  run_batched_fused_sparse_transfer(config, lmc_tensors, chunk_offsets,
                                    chunk_sizes, element_size,
                                    selected_token_idx, sparse_indices_cpu,
                                    launcher);
}

void load_and_reshape_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();
  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("load_and_reshape_flash_kernel");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, true);
    return 0;
  });
  cmd.Run();
  return;
};

void reshape_and_cache_back_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();

  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("reshape_and_cache_back_flash");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, false);
    return 0;
  });
  cmd.Run();
  return;
};
