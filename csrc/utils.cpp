#include "utils.h"
#include "dcmi_management.h"
#include <stdexcept>
#include <string>

#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType) {
  if (scalarType == at::ScalarType::Float) {
    return kvcache_ops::AscendType::FP32;
  } else if (scalarType == at::ScalarType::BFloat16) {
    return kvcache_ops::AscendType::BF16;
  } else if (scalarType == at::ScalarType::Half) {
    return kvcache_ops::AscendType::FP16;
  } else if (scalarType == at::ScalarType::Long) {
    return kvcache_ops::AscendType::INT64;
  } else if (scalarType == at::ScalarType::Int) {
    return kvcache_ops::AscendType::INT32;
  } else {
    TORCH_CHECK(false, "ScalarType not supported.");
  }
}
} // namespace vllm_ascend

MultiLayerKVConfig prepare_multi_layer_kv_config(
    const torch::Tensor &key_value, const torch::Tensor &key_value_ptrs,
    const torch::Tensor &slot_mapping, const torch::Device &paged_memory_device,
    int page_buffer_size, bool direction, bool use_mla, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims) {
  MultiLayerKVConfig config;

  config.page_buffer_ptrs =
      get_kernel_ptr<uint8_t, const torch::Tensor>(key_value_ptrs);
  config.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  config.num_layers = key_value.size(1);
  config.num_tokens = slot_mapping.size(0);

  config.kvcache_format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);

  config.page_buffer_size = page_buffer_size;
  config.direction = direction;

  config.scalar_type = key_value.scalar_type();
  config.slot_type = slot_mapping.scalar_type();

  config.socName = aclrtGetSocName();

  config.k_hidden_dims = k_hidden_dims;
  config.v_hidden_dims = v_hidden_dims;
  config.dsa_hidden_dims = dsa_hidden_dims;

  switch (config.kvcache_format) {
  case kvcache_ops::KVCacheFormat::MERGED_KV:
    config.kv_size = 2;
    config.hidden_dims = key_value.size(-1);
    break;
  case kvcache_ops::KVCacheFormat::SEPARATE_KV:
    config.kv_size = 2;
    config.hidden_dims = key_value.size(-1);
    break;
  case kvcache_ops::KVCacheFormat::MLA_KV:
    config.kv_size = 2;
    config.hidden_dims = config.k_hidden_dims;
    break;
  case kvcache_ops::KVCacheFormat::DSA_KV:
    config.kv_size = 3;
    config.hidden_dims = config.k_hidden_dims;
    break;
  default:
    TORCH_CHECK(false, "Unsupported KVCacheFormat: ", kvcache_format_raw);
  }

  return config;
}

void compute_multi_layer_ub_params(MultiLayerKVConfig &config,
                                   const torch::Tensor &key_value,
                                   const torch::Device &paged_memory_device,
                                   const torch::Tensor &key_value_ptrs) {
  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  config.stream = c10_npu::getCurrentNPUStream().stream();

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  // we only launched with at most 4 aiv
  config.aiv_num = static_cast<uint32_t>(std::min(config.num_layers, 4));

  constexpr int32_t numBuffsOnDev = 2;
  // step 1. use per tokens buff size to derive how many tokens can be allocated
  // per loop
  int64_t max_hidden_dims = config.hidden_dims;
  if (config.kvcache_format == kvcache_ops::KVCacheFormat::MLA_KV) {
    max_hidden_dims = std::max(config.k_hidden_dims, config.v_hidden_dims);
  } else if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
    max_hidden_dims = std::max(
        {config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims});
  }

  int64_t baseBuffSize =
      numBuffsOnDev * max_hidden_dims * key_value.element_size();

  if (ubSize < static_cast<uint64_t>(baseBuffSize)) {
    std::string errStr =
        "Per TokenBuffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact us.").c_str());
    throw py::error_already_set();
  }

  // step 2. work out how many tokens per loop
  config.maxTokensPerLoop =
      static_cast<int32_t>(ubSize / baseBuffSize) -
      1; // Subtract 1 to provide a safety margin and avoid over-allocating the
         // UB buffer, ensuring we do not exceed hardware limits due to possible
         // rounding or small additional allocations.
  config.maxTokensPerLoop = std::min(config.maxTokensPerLoop,
                                     static_cast<int32_t>(config.num_tokens));

  // step 3. double check whether the perloop buffer can accommodate everything
  int64_t totalPerLoopBuffer =
      static_cast<int64_t>(config.maxTokensPerLoop) * baseBuffSize;
  if (ubSize < static_cast<uint64_t>(totalPerLoopBuffer)) {
    std::string errStr =
        "Per Loop Buffer Size: " + std::to_string(totalPerLoopBuffer) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact us.").c_str());
    throw py::error_already_set();
  }

  // using double buffs mean we actually want to allocate half of this per
  // round.
  config.singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;
}

void compute_single_layer_ub_params(const KVTransferDims &dims,
                                    KVTransferUBParams &ub_params,
                                    const torch::Tensor &vllm_cache,
                                    kvcache_ops::KVCacheFormat format,
                                    int64_t k_hidden_dims, int64_t v_hidden_dims,
                                    int64_t dsa_hidden_dims) {

  const c10::OptionalDeviceGuard device_guard(device_of(vllm_cache));

  ub_params.stream = c10_npu::getCurrentNPUStream().stream();
  const char *socName = aclrtGetSocName();

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  ub_params.aiv_num = static_cast<uint32_t>(std::min(4, dims.num_tokens));

  uint32_t numBuffsOnDev = 2;
  int64_t perTokenBufferElems = dims.kv_size * dims.num_heads * dims.head_dims;
  if (format == kvcache_ops::KVCacheFormat::MLA_KV) {
    perTokenBufferElems = std::max(k_hidden_dims, v_hidden_dims);
  } else if (format == kvcache_ops::KVCacheFormat::DSA_KV) {
    perTokenBufferElems =
        std::max({k_hidden_dims, v_hidden_dims, dsa_hidden_dims});
  }

  uint64_t baseBuffSize =
      numBuffsOnDev * perTokenBufferElems * vllm_cache.element_size();

  if (ubSize < baseBuffSize) {
    std::string errStr =
        "Per Token Cache Buffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact LMCache Ascend.").c_str());
    throw py::error_already_set();
  }

  // we are going to work out how many tokens to copy maximally per innerloop
  ub_params.max_tokens_per_loop = static_cast<int32_t>(ubSize / baseBuffSize);
  ub_params.max_tokens_per_loop =
      std::min(ub_params.max_tokens_per_loop, dims.num_tokens);
}

void compute_single_layer_strides(
    const KVTransferDims &dims, KVTransferStrides &strides,
    const torch::Tensor &lmc_cache,
    const torch::Tensor &vllm_k_cache, // for merged
    bool token_major, bool vllm_two_major,
    kvcache_ops::KVCacheFormat format, int64_t k_hidden_dims, int64_t v_hidden_dims,
    int64_t dsa_hidden_dims, const torch::Tensor *vllm_v_cache,
    const torch::Tensor *vllm_dsa_cache) {

  const bool is_mla_dsa =
      format == kvcache_ops::KVCacheFormat::MLA_KV ||
      format == kvcache_ops::KVCacheFormat::DSA_KV;
  const bool is_separate =
      format == kvcache_ops::KVCacheFormat::SEPARATE_KV || is_mla_dsa;

  // LMC strides
  if (is_mla_dsa) {
    strides.lmc_token_stride = k_hidden_dims;
    strides.lmc_val_offset = 0;
  } else if (token_major) {
    // Shape: [tokens, 2, heads*headdim]
    strides.lmc_token_stride = lmc_cache.stride(0);
    strides.lmc_val_offset = lmc_cache.stride(1);
  } else {
    // Shape: [2, tokens, heads*headdim]
    strides.lmc_token_stride = lmc_cache.stride(1);
    strides.lmc_val_offset = lmc_cache.stride(0);
  }
  strides.lmc_bytes = static_cast<int64_t>(lmc_cache.nbytes());
  strides.lmc_v_plane_offset =
      static_cast<int64_t>(dims.lmc_num_tokens) * k_hidden_dims;
  strides.lmc_dsa_plane_offset = static_cast<int64_t>(dims.lmc_num_tokens) *
                                 (k_hidden_dims + v_hidden_dims);

  // vLLM buffer strides
  if (!is_separate) {
    if (vllm_two_major) {
      // Shape: [2, num_blocks, block_size, heads, head_dims]
      strides.vllm_k_stride = vllm_k_cache.stride(1);   // Block stride
      strides.vllm_val_offset = vllm_k_cache.stride(0); // K to V offset
    } else {
      // Shape: [num_blocks, 2, block_size, heads, head_dims]
      strides.vllm_k_stride = vllm_k_cache.stride(0);   // Block stride
      strides.vllm_val_offset = vllm_k_cache.stride(1); // K to V offset
    }
    strides.vllm_k_bytes = static_cast<int64_t>(vllm_k_cache.nbytes());

    // only for separate
    strides.vllm_v_stride = 0;
    strides.vllm_v_bytes = 0;

  } else {
    // SEPARATE
    TORCH_CHECK(vllm_v_cache != nullptr,
                "vllm_v_cache required for SEPARATE format");

    // Shape: [num_blocks, block_size, heads, head_dims]
    strides.vllm_k_stride = vllm_k_cache.stride(0);
    strides.vllm_v_stride = vllm_v_cache->stride(0);

    strides.vllm_k_bytes = static_cast<int64_t>(vllm_k_cache.nbytes());
    strides.vllm_v_bytes = static_cast<int64_t>(vllm_v_cache->nbytes());
    if (vllm_dsa_cache != nullptr) {
      strides.vllm_dsa_bytes = static_cast<int64_t>(vllm_dsa_cache->nbytes());
    } else {
      strides.vllm_dsa_bytes = 0;
    }

    // only for merged
    strides.vllm_val_offset = 0;
  }
}

SingleLayerKVConfig prepare_single_layer_kv_config(
    torch::Tensor &lmc_dst_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping, bool direction, bool token_major,
    bool vllm_two_major, int kvcache_format_raw, int64_t k_hidden_dims,
    int64_t v_hidden_dims, int64_t dsa_hidden_dims) {

  SingleLayerKVConfig config;
  config.kvcache_format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);

  torch::Tensor &vllm_k_cache = vllm_kv_caches[0];
  torch::Tensor *vllm_v_cache =
      (config.kvcache_format == kvcache_ops::KVCacheFormat::MERGED_KV)
          ? nullptr
          : &vllm_kv_caches[1];
  torch::Tensor *vllm_dsa_cache =
      (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV)
          ? &vllm_kv_caches[2]
          : nullptr;

  if (config.kvcache_format == kvcache_ops::KVCacheFormat::MLA_KV ||
      config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
    if (k_hidden_dims <= 0) {
      k_hidden_dims = vllm_k_cache.size(-2) * vllm_k_cache.size(-1);
    }
    if (v_hidden_dims <= 0 && vllm_v_cache != nullptr) {
      v_hidden_dims = vllm_v_cache->size(-2) * vllm_v_cache->size(-1);
    }
    if (dsa_hidden_dims <= 0 && vllm_dsa_cache != nullptr) {
      dsa_hidden_dims = vllm_dsa_cache->size(-2) * vllm_dsa_cache->size(-1);
    }
  }

  config.k_hidden_dims = k_hidden_dims;
  config.v_hidden_dims = v_hidden_dims;
  config.dsa_hidden_dims = dsa_hidden_dims;

  // Dims
  config.dims.num_tokens = slot_mapping.size(0);
  config.dims.lmc_num_tokens = config.dims.num_tokens;
  config.dims.num_heads = vllm_k_cache.size(-2);
  config.dims.head_dims = vllm_k_cache.size(-1);
  config.dims.block_size = vllm_k_cache.size(-3);
  if (config.kvcache_format == kvcache_ops::KVCacheFormat::MLA_KV ||
      config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
    int64_t plane_elems = config.k_hidden_dims + config.v_hidden_dims;
    if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
      plane_elems += config.dsa_hidden_dims;
    }
    TORCH_CHECK(plane_elems > 0, "MLA/DSA plane size must be positive");
    config.dims.lmc_num_tokens =
        static_cast<int32_t>(lmc_dst_cache.numel() / plane_elems);
  }
  if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
    config.dims.kv_size = 3;
  } else if (config.kvcache_format == kvcache_ops::KVCacheFormat::MERGED_KV) {
    config.dims.kv_size = 2;
  } else {
    config.dims.kv_size = 2;
  }

  // ptrs
  config.ptrs.lmc_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(lmc_dst_cache);
  config.ptrs.vllm_k_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(vllm_k_cache);
  config.ptrs.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  if (vllm_v_cache != nullptr) {
    config.ptrs.vllm_v_ptr =
        get_kernel_ptr<uint8_t, const torch::Tensor>(*vllm_v_cache);
  } else {
    config.ptrs.vllm_v_ptr = nullptr;
  }

  if (vllm_dsa_cache != nullptr) {
    config.ptrs.vllm_dsa_ptr =
        get_kernel_ptr<uint8_t, const torch::Tensor>(*vllm_dsa_cache);
  } else {
    config.ptrs.vllm_dsa_ptr = nullptr;
  }

  config.ub_params.scalar_type_num =
      vllm_ascend::get_dtype_from_torch(vllm_k_cache.scalar_type());
  config.ub_params.slot_type_num =
      vllm_ascend::get_dtype_from_torch(slot_mapping.scalar_type());

  config.direction = direction;
  config.token_major = token_major;

  compute_single_layer_ub_params(config.dims, config.ub_params, vllm_k_cache,
                                   config.kvcache_format, config.k_hidden_dims,
                                   config.v_hidden_dims, config.dsa_hidden_dims);

  compute_single_layer_strides(
      config.dims, config.strides, lmc_dst_cache, vllm_k_cache, token_major,
      vllm_two_major, config.kvcache_format, config.k_hidden_dims,
      config.v_hidden_dims, config.dsa_hidden_dims, vllm_v_cache, vllm_dsa_cache);

  return config;
}

SparseDirectLayerState prepare_sparse_direct_layer_state(
    torch::Tensor &lmc_layout_sample,
    std::vector<torch::Tensor> &vllm_kv_caches, torch::Tensor &slot_mapping_ref,
    bool token_major, bool vllm_two_major, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t lmc_num_tokens) {
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  SparseDirectLayerState state;
  state.config = prepare_single_layer_kv_config(
      lmc_layout_sample, vllm_kv_caches, slot_mapping_ref, false, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims);
  state.config.dims.lmc_num_tokens = lmc_num_tokens;
  return state;
}

HostChunkMetadata
prepare_host_chunk_metadata(const std::vector<torch::Tensor> &lmc_tensors,
                            const std::vector<int64_t> &chunk_sizes,
                            const KVTransferStrides &strides,
                            int64_t element_size, bool token_major,
                            kvcache_ops::KVCacheFormat format,
                            int64_t k_hidden_dims, int64_t v_hidden_dims,
                            int64_t dsa_hidden_dims) {

  size_t num_chunks = lmc_tensors.size();
  HostChunkMetadata meta;
  meta.ptrs.resize(num_chunks);
  meta.copy_sizes.resize(num_chunks);
  meta.v_offsets.resize(num_chunks);
  meta.dsa_offsets.resize(num_chunks);
  meta.element_size = element_size;

  const bool is_mla_dsa =
      format == kvcache_ops::KVCacheFormat::MLA_KV ||
      format == kvcache_ops::KVCacheFormat::DSA_KV;

  if (is_mla_dsa) {
    meta.bytes_per_token = k_hidden_dims * element_size;
  } else {
    meta.bytes_per_token = strides.lmc_token_stride * element_size;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    meta.ptrs[i] = static_cast<uint8_t *>(lmc_tensors[i].data_ptr());
    if (is_mla_dsa) {
      meta.copy_sizes[i] = chunk_sizes[i] * k_hidden_dims * element_size;
      meta.v_offsets[i] = chunk_sizes[i] * k_hidden_dims * element_size;
      meta.dsa_offsets[i] =
          chunk_sizes[i] * (k_hidden_dims + v_hidden_dims) * element_size;
    } else if (!token_major) {
      meta.copy_sizes[i] = chunk_sizes[i] * meta.bytes_per_token;
      meta.v_offsets[i] = lmc_tensors[i].stride(0) * element_size;
      meta.dsa_offsets[i] = 0;
    } else {
      meta.copy_sizes[i] = chunk_sizes[i] * meta.bytes_per_token;
      meta.v_offsets[i] = 0;
      meta.dsa_offsets[i] = 0;
    }
  }
  return meta;
}

void execute_batched_memcpy(
    const SingleLayerKVConfig &config, const HostChunkMetadata &meta,
    const std::vector<int64_t> &chunk_offsets,
    bool is_d2h // true: Device to Host, false: Host to Device
) {
  aclError ret;
  size_t num_chunks = meta.ptrs.size();
  aclrtMemcpyKind kind =
      is_d2h ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;

  aclrtStream stream = config.ub_params.stream;
  const bool is_mla_dsa =
      config.kvcache_format == kvcache_ops::KVCacheFormat::MLA_KV ||
      config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV;

  if (is_mla_dsa) {
    int64_t k_bytes_per_token = config.k_hidden_dims * meta.element_size;
    int64_t v_bytes_per_token = config.v_hidden_dims * meta.element_size;
    int64_t dsa_bytes_per_token = config.dsa_hidden_dims * meta.element_size;
    int64_t staging_v_plane_offset =
        config.strides.lmc_v_plane_offset * meta.element_size;
    int64_t staging_dsa_plane_offset =
        config.strides.lmc_dsa_plane_offset * meta.element_size;
    int64_t host_bytes_per_token =
        k_bytes_per_token + v_bytes_per_token +
        (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV
             ? dsa_bytes_per_token
             : 0);

    for (size_t i = 0; i < num_chunks; ++i) {
      int64_t token_offset = chunk_offsets[i];
      int64_t chunk_tokens = meta.copy_sizes[i] / k_bytes_per_token;
      uint8_t *host_base = meta.ptrs[i];

      if (config.token_major) {
        for (int64_t t = 0; t < chunk_tokens; ++t) {
          int64_t staging_token_idx = token_offset + t;
          uint8_t *host_token = host_base + t * host_bytes_per_token;
          uint8_t *staging_k =
              config.ptrs.lmc_ptr + staging_token_idx * k_bytes_per_token;
          uint8_t *staging_v = config.ptrs.lmc_ptr + staging_v_plane_offset +
                               staging_token_idx * v_bytes_per_token;

          if (!is_d2h) {
            ret = aclrtMemcpyAsync(staging_k, k_bytes_per_token, host_token,
                                   k_bytes_per_token, kind, stream);
            TORCH_CHECK(ret == ACL_ERROR_NONE,
                        "Memcpy (MLA/DSA interleaved K) failed chunk ", i);
            ret = aclrtMemcpyAsync(staging_v, v_bytes_per_token,
                                   host_token + k_bytes_per_token,
                                   v_bytes_per_token, kind, stream);
            TORCH_CHECK(ret == ACL_ERROR_NONE,
                        "Memcpy (MLA/DSA interleaved V) failed chunk ", i);
            if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
              uint8_t *staging_dsa =
                  config.ptrs.lmc_ptr + staging_dsa_plane_offset +
                  staging_token_idx * dsa_bytes_per_token;
              ret = aclrtMemcpyAsync(
                  staging_dsa, dsa_bytes_per_token,
                  host_token + k_bytes_per_token + v_bytes_per_token,
                  dsa_bytes_per_token, kind, stream);
              TORCH_CHECK(ret == ACL_ERROR_NONE,
                          "Memcpy (DSA interleaved) failed chunk ", i);
            }
          } else {
            ret = aclrtMemcpyAsync(host_token, k_bytes_per_token, staging_k,
                                   k_bytes_per_token, kind, stream);
            TORCH_CHECK(ret == ACL_ERROR_NONE,
                        "Memcpy (MLA/DSA interleaved K) failed chunk ", i);
            ret = aclrtMemcpyAsync(host_token + k_bytes_per_token,
                                   v_bytes_per_token, staging_v,
                                   v_bytes_per_token, kind, stream);
            TORCH_CHECK(ret == ACL_ERROR_NONE,
                        "Memcpy (MLA/DSA interleaved V) failed chunk ", i);
            if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
              uint8_t *staging_dsa =
                  config.ptrs.lmc_ptr + staging_dsa_plane_offset +
                  staging_token_idx * dsa_bytes_per_token;
              ret = aclrtMemcpyAsync(
                  host_token + k_bytes_per_token + v_bytes_per_token,
                  dsa_bytes_per_token, staging_dsa, dsa_bytes_per_token, kind,
                  stream);
              TORCH_CHECK(ret == ACL_ERROR_NONE,
                          "Memcpy (DSA interleaved) failed chunk ", i);
            }
          }
        }
        continue;
      }

      int64_t k_size = chunk_tokens * k_bytes_per_token;
      int64_t v_size = chunk_tokens * v_bytes_per_token;
      int64_t dsa_size = chunk_tokens * dsa_bytes_per_token;

      uint8_t *staging_k =
          config.ptrs.lmc_ptr + token_offset * k_bytes_per_token;
      uint8_t *staging_v = config.ptrs.lmc_ptr + staging_v_plane_offset +
                           token_offset * v_bytes_per_token;
      uint8_t *host_k = host_base;
      uint8_t *host_v = host_k + meta.v_offsets[i];
      uint8_t *host_dsa = host_v + meta.dsa_offsets[i];

      if (!is_d2h) {
        ret = aclrtMemcpyAsync(staging_k, k_size, host_k, k_size, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (MLA/DSA K) failed chunk ", i);
        ret = aclrtMemcpyAsync(staging_v, v_size, host_v, v_size, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (MLA/DSA V) failed chunk ", i);
        if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
          uint8_t *staging_dsa = config.ptrs.lmc_ptr + staging_dsa_plane_offset +
                                 token_offset * dsa_bytes_per_token;
          ret = aclrtMemcpyAsync(staging_dsa, dsa_size, host_dsa, dsa_size, kind,
                                 stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (DSA) failed chunk ", i);
        }
      } else {
        ret = aclrtMemcpyAsync(host_k, k_size, staging_k, k_size, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (MLA/DSA K) failed chunk ", i);
        ret = aclrtMemcpyAsync(host_v, v_size, staging_v, v_size, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (MLA/DSA V) failed chunk ", i);
        if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
          uint8_t *staging_dsa = config.ptrs.lmc_ptr + staging_dsa_plane_offset +
                                 token_offset * dsa_bytes_per_token;
          ret = aclrtMemcpyAsync(host_dsa, dsa_size, staging_dsa, dsa_size, kind,
                                 stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (DSA) failed chunk ", i);
        }
      }
    }
    return;
  }

  // only for !token_major
  int64_t staging_v_plane_offset =
      config.strides.lmc_val_offset * meta.element_size;

  for (size_t i = 0; i < num_chunks; ++i) {
    uint8_t *staging_ptr =
        config.ptrs.lmc_ptr + chunk_offsets[i] * meta.bytes_per_token;
    uint8_t *host_ptr = meta.ptrs[i];
    int64_t size = meta.copy_sizes[i];

    if (config.token_major) {
      if (!is_d2h) // H2D
        ret = aclrtMemcpyAsync(staging_ptr, size, host_ptr, size, kind, stream);
      else // D2H
        ret = aclrtMemcpyAsync(host_ptr, size, staging_ptr, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy failed (TokenMajor) chunk ", i,
                  " ret=", ret);
    } else {
      // K Plane
      if (!is_d2h)
        ret = aclrtMemcpyAsync(staging_ptr, size, host_ptr, size, kind, stream);
      else
        ret = aclrtMemcpyAsync(host_ptr, size, staging_ptr, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (K) failed chunk ", i,
                  " ret=", ret);

      // V Plane
      uint8_t *staging_v = staging_ptr + staging_v_plane_offset;
      uint8_t *host_v = host_ptr + meta.v_offsets[i];

      if (!is_d2h)
        ret = aclrtMemcpyAsync(staging_v, size, host_v, size, kind, stream);
      else
        ret = aclrtMemcpyAsync(host_v, size, staging_v, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (V) failed chunk ", i,
                  " ret=", ret);
    }
  }
}

namespace {

int find_chunk_for_staging_token(
    int32_t staging_token_idx, const std::vector<int64_t> &chunk_offsets,
    const std::vector<int64_t> &chunk_sizes) {
  for (size_t i = 0; i < chunk_offsets.size(); ++i) {
    if (staging_token_idx >= chunk_offsets[i] &&
        staging_token_idx < chunk_offsets[i] + chunk_sizes[i]) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

} // namespace

void execute_batched_sparse_memcpy(
    const SingleLayerKVConfig &config, const HostChunkMetadata &meta,
    const std::vector<int64_t> &chunk_offsets,
    const std::vector<int64_t> &chunk_sizes,
    const std::vector<int32_t> &selected_indices, bool is_d2h) {
  TORCH_CHECK(!is_d2h, "Sparse retrieve only supports H2D.");

  const int32_t *indices = selected_indices.data();
  const int32_t num_sparse = static_cast<int32_t>(selected_indices.size());
  const int32_t total_tokens = config.dims.lmc_num_tokens;

  if (num_sparse <= 0) {
    return;
  }
  // Bulk chunk memcpy is faster when most tokens are selected.
  if (num_sparse >= (total_tokens * 3) / 4) {
    execute_batched_memcpy(config, meta, chunk_offsets, false);
    return;
  }

  aclError ret;
  aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
  aclrtStream stream = config.ub_params.stream;
  const bool is_mla_dsa =
      config.kvcache_format == kvcache_ops::KVCacheFormat::MLA_KV ||
      config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV;

  if (is_mla_dsa) {
    int64_t k_bytes_per_token = config.k_hidden_dims * meta.element_size;
    int64_t v_bytes_per_token = config.v_hidden_dims * meta.element_size;
    int64_t dsa_bytes_per_token = config.dsa_hidden_dims * meta.element_size;
    int64_t staging_v_plane_offset =
        config.strides.lmc_v_plane_offset * meta.element_size;
    int64_t staging_dsa_plane_offset =
        config.strides.lmc_dsa_plane_offset * meta.element_size;
    int64_t host_bytes_per_token =
        k_bytes_per_token + v_bytes_per_token +
        (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV
             ? dsa_bytes_per_token
             : 0);

    for (int32_t s = 0; s < num_sparse; ++s) {
      const int32_t staging_token_idx = indices[s];
      const int chunk_i =
          find_chunk_for_staging_token(staging_token_idx, chunk_offsets, chunk_sizes);
      TORCH_CHECK(chunk_i >= 0, "selected_token_idx out of chunk range: ",
                  staging_token_idx);
      const int64_t t = staging_token_idx - chunk_offsets[chunk_i];
      uint8_t *host_base = meta.ptrs[chunk_i];

      if (config.token_major) {
        uint8_t *host_token = host_base + t * host_bytes_per_token;
        uint8_t *staging_k =
            config.ptrs.lmc_ptr + staging_token_idx * k_bytes_per_token;
        uint8_t *staging_v = config.ptrs.lmc_ptr + staging_v_plane_offset +
                             staging_token_idx * v_bytes_per_token;
        ret = aclrtMemcpyAsync(staging_k, k_bytes_per_token, host_token,
                               k_bytes_per_token, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE,
                    "Sparse memcpy (MLA/DSA K) failed at index ", staging_token_idx);
        ret = aclrtMemcpyAsync(staging_v, v_bytes_per_token,
                               host_token + k_bytes_per_token, v_bytes_per_token,
                               kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE,
                    "Sparse memcpy (MLA/DSA V) failed at index ", staging_token_idx);
        if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
          uint8_t *staging_dsa =
              config.ptrs.lmc_ptr + staging_dsa_plane_offset +
              staging_token_idx * dsa_bytes_per_token;
          ret = aclrtMemcpyAsync(
              staging_dsa, dsa_bytes_per_token,
              host_token + k_bytes_per_token + v_bytes_per_token,
              dsa_bytes_per_token, kind, stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE,
                      "Sparse memcpy (DSA) failed at index ", staging_token_idx);
        }
      } else {
        uint8_t *host_k = host_base + t * k_bytes_per_token;
        uint8_t *host_v = host_k + meta.v_offsets[chunk_i];
        uint8_t *host_dsa = host_v + meta.dsa_offsets[chunk_i];
        uint8_t *staging_k =
            config.ptrs.lmc_ptr + staging_token_idx * k_bytes_per_token;
        uint8_t *staging_v = config.ptrs.lmc_ptr + staging_v_plane_offset +
                             staging_token_idx * v_bytes_per_token;
        ret = aclrtMemcpyAsync(staging_k, k_bytes_per_token, host_k,
                               k_bytes_per_token, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE,
                    "Sparse memcpy (MLA/DSA stacked K) failed at index ",
                    staging_token_idx);
        ret = aclrtMemcpyAsync(staging_v, v_bytes_per_token, host_v,
                               v_bytes_per_token, kind, stream);
        TORCH_CHECK(ret == ACL_ERROR_NONE,
                    "Sparse memcpy (MLA/DSA stacked V) failed at index ",
                    staging_token_idx);
        if (config.kvcache_format == kvcache_ops::KVCacheFormat::DSA_KV) {
          uint8_t *staging_dsa =
              config.ptrs.lmc_ptr + staging_dsa_plane_offset +
              staging_token_idx * dsa_bytes_per_token;
          ret = aclrtMemcpyAsync(staging_dsa, dsa_bytes_per_token, host_dsa,
                                 dsa_bytes_per_token, kind, stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE,
                      "Sparse memcpy (DSA stacked) failed at index ",
                      staging_token_idx);
        }
      }
    }
    return;
  }

  int64_t staging_v_plane_offset =
      config.strides.lmc_val_offset * meta.element_size;
  for (int32_t s = 0; s < num_sparse; ++s) {
    const int32_t staging_token_idx = indices[s];
    const int chunk_i =
        find_chunk_for_staging_token(staging_token_idx, chunk_offsets, chunk_sizes);
    TORCH_CHECK(chunk_i >= 0, "selected_token_idx out of chunk range: ",
                staging_token_idx);
    const int64_t t = staging_token_idx - chunk_offsets[chunk_i];
    uint8_t *host_ptr = meta.ptrs[chunk_i];
    uint8_t *staging_ptr =
        config.ptrs.lmc_ptr + staging_token_idx * meta.bytes_per_token;

    if (config.token_major) {
      ret = aclrtMemcpyAsync(staging_ptr, meta.bytes_per_token,
                             host_ptr + t * meta.bytes_per_token,
                             meta.bytes_per_token, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Sparse memcpy failed at index ",
                  staging_token_idx);
    } else {
      ret = aclrtMemcpyAsync(staging_ptr, meta.bytes_per_token,
                             host_ptr + t * meta.bytes_per_token,
                             meta.bytes_per_token, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Sparse memcpy (K) failed at index ",
                  staging_token_idx);
      uint8_t *staging_v = staging_ptr + staging_v_plane_offset;
      uint8_t *host_v = host_ptr + meta.v_offsets[chunk_i] +
                        t * meta.bytes_per_token;
      ret = aclrtMemcpyAsync(staging_v, meta.bytes_per_token, host_v,
                             meta.bytes_per_token, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Sparse memcpy (V) failed at index ",
                  staging_token_idx);
    }
  }
}

bool validate_vllm_caches(const std::vector<torch::Tensor> &vllm_kv_caches,
                          int kvcache_format_raw) {
  kvcache_ops::KVCacheFormat format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);

  if (format != kvcache_ops::KVCacheFormat::MERGED_KV &&
      format != kvcache_ops::KVCacheFormat::SEPARATE_KV &&
      format != kvcache_ops::KVCacheFormat::MLA_KV &&
      format != kvcache_ops::KVCacheFormat::DSA_KV) {
    std::string err =
        "Invalid KV cache format: " + std::to_string(kvcache_format_raw) +
        ". Expected 1 (MERGED_KV), 2 (SEPARATE_KV), 3 (MLA_KV), or 4 (DSA_KV)";
    PyErr_SetString(PyExc_ValueError, err.c_str());
    throw py::error_already_set();
  }

  if (vllm_kv_caches.empty()) {
    PyErr_SetString(PyExc_ValueError, "vllm_kv_caches cannot be empty");
    throw py::error_already_set();
  }

  if (format == kvcache_ops::KVCacheFormat::DSA_KV) {
    if (vllm_kv_caches.size() != 3) {
      PyErr_SetString(PyExc_ValueError,
                      "DSA_KV expects 3 tensors (K, V, and DSA_K).");
      throw py::error_already_set();
    }
  } else if (format == kvcache_ops::KVCacheFormat::MLA_KV) {
    if (vllm_kv_caches.size() != 2) {
      PyErr_SetString(PyExc_ValueError, "MLA_KV expects 2 tensors (K and V).");
      throw py::error_already_set();
    }
    if (vllm_kv_caches[0].sizes() == vllm_kv_caches[1].sizes()) {
      throw py::value_error(
          "MLA_KV expects K and V caches to have different shapes.");
    }
  } else if (format == kvcache_ops::KVCacheFormat::SEPARATE_KV) {
    if (vllm_kv_caches.size() != 2) {
      PyErr_SetString(PyExc_ValueError,
                      "SEPARATE_KV expects 2 tensors (K and V).");
      throw py::error_already_set();
    }
    if (vllm_kv_caches[0].sizes() != vllm_kv_caches[1].sizes()) {
      throw py::value_error("K and V caches must have the same shape.");
    }
  } else {
    if (vllm_kv_caches.size() != 1) {
      PyErr_SetString(PyExc_ValueError, "MERGED_KV expects 1 tensor.");
      throw py::error_already_set();
    }
  }

  return format != kvcache_ops::KVCacheFormat::MERGED_KV;
}

void validate_sparse_single_layer_inputs(
    const torch::Tensor &slot_mapping_packed,
    const torch::Tensor &selected_token_idx) {
  if (slot_mapping_packed.dim() != 1 || selected_token_idx.dim() != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "slot_mapping_packed and selected_token_idx must be 1D.");
    throw py::error_already_set();
  }

  if (slot_mapping_packed.size(0) != selected_token_idx.size(0)) {
    PyErr_SetString(
        PyExc_ValueError,
        "slot_mapping_packed and selected_token_idx must have the same length.");
    throw py::error_already_set();
  }

  if (selected_token_idx.scalar_type() != at::ScalarType::Int) {
    PyErr_SetString(PyExc_ValueError,
                    "selected_token_idx must be torch.int32.");
    throw py::error_already_set();
  }

  if (!selected_token_idx.device().is_privateuseone()) {
    PyErr_SetString(PyExc_ValueError,
                    "selected_token_idx must be on NPU device.");
    throw py::error_already_set();
  }
}