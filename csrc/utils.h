/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "kernels/types.h"
#include "managed_mem.h"
#include <c10/core/ScalarType.h>
#include <string>
#include <torch/torch.h>

#include "tiling/platform/platform_ascendc.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType);
} // namespace vllm_ascend

template <typename T, typename TENSOR_TYPE>
T *get_kernel_ptr(TENSOR_TYPE &tensor) {
  torch::Device device = tensor.device();
  // NPU should be using PrivateUse1
  if (device.is_privateuseone() || device.is_cuda()) {
    return static_cast<T *>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    // find device ptr based on the host pinned ptr
    // because acl does not currently support HostGetDevicePointer API
    void *devPtr = get_device_ptr(tensor.data_ptr());
    TORCH_CHECK(
        devPtr != nullptr,
        "Unable to retrieve device ptr, is this a host registered pointer ?");
    return reinterpret_cast<T *>(devPtr);
  } else {
    TORCH_CHECK(
        false,
        "Invalid device. Device must be ascend (PrivateUseOne) or pinned cpu.");
  }
}

struct MultiLayerKVConfig {
  uint8_t *page_buffer_ptrs;
  uint8_t *slot_mapping_ptr;

  int num_layers;
  int num_tokens;
  int hidden_dims;
  int kv_size;

  kvcache_ops::KVCacheFormat kvcache_format;

  aclrtStream stream;
  at::ScalarType scalar_type;
  at::ScalarType slot_type;
  const char *socName;

  uint32_t aiv_num;
  int32_t maxTokensPerLoop;
  int64_t singlePerLoopBuffer;

  int page_buffer_size;
  bool direction;

  int64_t k_hidden_dims;
  int64_t v_hidden_dims;
  int64_t dsa_hidden_dims;
};

MultiLayerKVConfig prepare_multi_layer_kv_config(
    const torch::Tensor &key_value, const torch::Tensor &key_value_ptrs,
    const torch::Tensor &slot_mapping, const torch::Device &paged_memory_device,
    int page_buffer_size, bool direction, bool use_mla, int kvcache_format_raw,
    int64_t k_hidden_dims = 0, int64_t v_hidden_dims = 0,
    int64_t dsa_hidden_dims = 0);

void compute_multi_layer_ub_params(MultiLayerKVConfig &config,
                                   const torch::Tensor &key_value,
                                   const torch::Device &paged_memory_device,
                                   const torch::Tensor &key_value_ptrs);
struct KVTransferDims {
  int32_t num_tokens;
  int32_t lmc_num_tokens; // MLA/DSA stacked-plane capacity (>= num_tokens for sparse)
  int32_t num_heads;
  int32_t head_dims;
  int32_t block_size;
  int32_t kv_size; // 1 (MLA/GQA special) or 2 (Standard K+V)
};

struct KVTransferPointers {
  uint8_t *lmc_ptr;
  uint8_t *vllm_k_ptr;
  uint8_t
      *vllm_v_ptr; // valid only in Separate mode and is nullptr in Merged mode
  uint8_t *vllm_dsa_ptr; // valid only in DSA_KV mode
  uint8_t *slot_mapping_ptr;
};

// Unified the stride representation for both Merged and Separate modes.
// In Merged mode, use vllm_val_offset;
// In Separate mode, use vllm_k_stride and vllm_v_stride.
struct KVTransferStrides {

  int64_t lmc_token_stride;
  int64_t lmc_val_offset; // Offset between K and V (if !token_major)
  int64_t lmc_bytes;

  int64_t vllm_k_stride;

  // valid only in Separate mode
  int64_t vllm_v_stride;

  // valid only in Merged mode
  int64_t vllm_val_offset;

  int64_t vllm_k_bytes;
  int64_t vllm_v_bytes;
  int64_t vllm_dsa_bytes;
  int64_t lmc_v_plane_offset;  // stacked-plane offset to V in elements
  int64_t lmc_dsa_plane_offset; // stacked-plane offset to DSA in elements
};

struct KVTransferUBParams {
  aclrtStream stream;
  kvcache_ops::AscendType scalar_type_num;
  kvcache_ops::AscendType slot_type_num;

  uint32_t aiv_num;
  int32_t max_tokens_per_loop; // Maximum number of tokens processed in each
                               // internal iteration
};

struct SingleLayerKVConfig {
  KVTransferDims dims;
  KVTransferPointers ptrs;
  KVTransferStrides strides;
  KVTransferUBParams ub_params;

  kvcache_ops::KVCacheFormat kvcache_format;
  int64_t k_hidden_dims;
  int64_t v_hidden_dims;
  int64_t dsa_hidden_dims;

  bool direction;   // false: H2D, true: D2H
  bool token_major; // true: [tokens, ...], false: [..., tokens, ...]
};

// Cached per-layer config for sparse direct KV retrieve (host-side hot path).
struct SparseDirectLayerState {
  SingleLayerKVConfig config;
};

// Process-lifetime destination state for prepared MLA/DSA sparse retrieve.
// Request source metadata and per-step launch data are deliberately excluded.
struct SparseDirectDestinationState {
  uint8_t *vllm_k_ptr;
  uint8_t *vllm_v_ptr;
  uint8_t *vllm_dsa_ptr;

  int64_t vllm_k_bytes;
  int64_t vllm_v_bytes;
  int64_t vllm_dsa_bytes;

  int32_t block_size;
  int32_t max_tokens_per_loop;

  kvcache_ops::AscendType scalar_type_num;
  kvcache_ops::AscendType slot_type_num;
  kvcache_ops::KVCacheFormat kvcache_format;

  int64_t k_hidden_dims;
  int64_t v_hidden_dims;
  int64_t dsa_hidden_dims;
};

struct HostChunkMetadata {
  std::vector<uint8_t *> ptrs;
  std::vector<int64_t> copy_sizes; // Bytes to copy per chunk
  std::vector<int64_t> v_offsets;  // Offset to V plane (only for !token_major)
  std::vector<int64_t> dsa_offsets; // Offset to DSA plane in host chunk
  int64_t bytes_per_token;
  int64_t element_size;
};

void compute_single_layer_ub_params(const KVTransferDims &dims,
                                    KVTransferUBParams &ub_params,
                                    const torch::Tensor &vllm_cache,
                                    kvcache_ops::KVCacheFormat format,
                                    int64_t k_hidden_dims, int64_t v_hidden_dims,
                                    int64_t dsa_hidden_dims);

void compute_single_layer_strides(
    const KVTransferDims &dims, KVTransferStrides &strides,
    const torch::Tensor &lmc_cache, const torch::Tensor &vllm_k_cache,
    bool token_major,
    bool vllm_two_major, // valid only in Merged mode
    kvcache_ops::KVCacheFormat format, int64_t k_hidden_dims, int64_t v_hidden_dims,
    int64_t dsa_hidden_dims, const torch::Tensor *vllm_v_cache = nullptr,
    const torch::Tensor *vllm_dsa_cache = nullptr);

SingleLayerKVConfig prepare_single_layer_kv_config(
    torch::Tensor &lmc_dst_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping, bool direction, bool token_major,
    bool vllm_two_major, int kvcache_format_raw, int64_t k_hidden_dims = 0,
    int64_t v_hidden_dims = 0, int64_t dsa_hidden_dims = 0);

SparseDirectLayerState prepare_sparse_direct_layer_state(
    torch::Tensor &lmc_layout_sample,
    std::vector<torch::Tensor> &vllm_kv_caches, torch::Tensor &slot_mapping_ref,
    bool token_major, bool vllm_two_major, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t lmc_num_tokens);

SparseDirectDestinationState prepare_sparse_direct_destination_state(
    std::vector<torch::Tensor> &vllm_kv_caches,
    at::ScalarType slot_mapping_type, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims);

HostChunkMetadata
prepare_host_chunk_metadata(const std::vector<torch::Tensor> &lmc_tensors,
                            const std::vector<int64_t> &chunk_sizes,
                            const KVTransferStrides &strides,
                            int64_t element_size, bool token_major,
                            kvcache_ops::KVCacheFormat format,
                            int64_t k_hidden_dims, int64_t v_hidden_dims,
                            int64_t dsa_hidden_dims);

void execute_batched_memcpy(
    const SingleLayerKVConfig &config, const HostChunkMetadata &meta,
    const std::vector<int64_t> &chunk_offsets,
    bool is_d2h // true: Device to Host, false: Host to Device
);

void execute_batched_sparse_memcpy(
    const SingleLayerKVConfig &config, const HostChunkMetadata &meta,
    const std::vector<int64_t> &chunk_offsets,
    const std::vector<int64_t> &chunk_sizes,
    const std::vector<int32_t> &selected_indices, bool is_d2h);

bool validate_vllm_caches(const std::vector<torch::Tensor> &vllm_kv_caches,
                          int kvcache_format_raw);

void validate_sparse_single_layer_inputs(
    const torch::Tensor &slot_mapping_packed,
    const torch::Tensor &selected_token_idx);

template <typename KernelLauncher>
void run_batched_fused_transfer(const SingleLayerKVConfig &config,
                                const std::vector<torch::Tensor> &lmc_tensors,
                                const std::vector<int64_t> &chunk_offsets,
                                const std::vector<int64_t> &chunk_sizes,
                                int64_t element_size,
                                KernelLauncher kernel_launcher) {

  HostChunkMetadata meta =
      prepare_host_chunk_metadata(lmc_tensors, chunk_sizes, config.strides,
                                  element_size, config.token_major,
                                  config.kvcache_format, config.k_hidden_dims,
                                  config.v_hidden_dims, config.dsa_hidden_dims);

  at_npu::native::OpCommand cmd;
  cmd.Name("batched_fused_single_layer_kv_transfer");

  cmd.SetCustomHandler([config, meta, chunk_offsets, kernel_launcher]() -> int {
    bool is_swap_out = config.direction;

    if (!is_swap_out) {
      // Swap In: CPU -> Staging -> Paged (Kernel)
      execute_batched_memcpy(config, meta, chunk_offsets, false); // H2D
      kernel_launcher(false); // Scatter Kernel
    } else {
      // Swap Out: Paged (Kernel) -> Staging -> CPU
      kernel_launcher(true); // Gather Kernel
      execute_batched_memcpy(config, meta, chunk_offsets, true); // D2H
    }
    return 0;
  });
  cmd.Run();
}

template <typename SparseKernelLauncher>
void run_batched_fused_sparse_transfer(
    const SingleLayerKVConfig &config,
    const std::vector<torch::Tensor> &lmc_tensors,
    const std::vector<int64_t> &chunk_offsets,
    const std::vector<int64_t> &chunk_sizes, int64_t element_size,
    const torch::Tensor &selected_token_idx,
    const c10::optional<torch::Tensor> &sparse_indices_cpu,
    SparseKernelLauncher kernel_launcher) {

  HostChunkMetadata meta =
      prepare_host_chunk_metadata(lmc_tensors, chunk_sizes, config.strides,
                                  element_size, config.token_major,
                                  config.kvcache_format, config.k_hidden_dims,
                                  config.v_hidden_dims, config.dsa_hidden_dims);

  // Prefer caller-provided CPU indices (once per retrieve). Falling back to
  // NPU->CPU here still works but costs one device sync per layer.
  std::vector<int32_t> sparse_indices;
  auto fill_sparse_indices = [&](const torch::Tensor &idx_cpu) {
    TORCH_CHECK(idx_cpu.scalar_type() == at::ScalarType::Int,
                "sparse indices must be int32.");
    TORCH_CHECK(idx_cpu.device().is_cpu(),
                "sparse indices must be on CPU.");
    auto contiguous = idx_cpu.contiguous();
    sparse_indices.assign(contiguous.data_ptr<int32_t>(),
                          contiguous.data_ptr<int32_t>() +
                              contiguous.numel());
  };

  if (sparse_indices_cpu.has_value() && sparse_indices_cpu->defined() &&
      sparse_indices_cpu->numel() > 0) {
    fill_sparse_indices(*sparse_indices_cpu);
  } else if (selected_token_idx.defined() && selected_token_idx.numel() > 0) {
    TORCH_CHECK(selected_token_idx.scalar_type() == at::ScalarType::Int,
                "selected_token_idx must be int32.");
    at::Tensor idx_cpu = selected_token_idx.device().is_cpu()
                             ? selected_token_idx.contiguous()
                             : selected_token_idx.detach().cpu().contiguous();
    fill_sparse_indices(idx_cpu);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("batched_fused_sparse_single_layer_kv_transfer");

  cmd.SetCustomHandler([config, meta, chunk_offsets, chunk_sizes,
                        sparse_indices = std::move(sparse_indices),
                        kernel_launcher]() -> int {
    execute_batched_sparse_memcpy(config, meta, chunk_offsets, chunk_sizes,
                                  sparse_indices, false);
    kernel_launcher();
    return 0;
  });
  cmd.Run();
}
