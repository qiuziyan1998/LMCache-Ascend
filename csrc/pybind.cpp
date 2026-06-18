// SPDX-License-Identifier: Apache-2.0

#include "cachegen_kernels.h"
#include "dcmi_management.h"
#include "managed_mem.h"
#include "mem_alloc.h"
#include "mem_kernels.h"
#include "pos_kernels.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

namespace py = pybind11;

std::vector<torch::Tensor> normalize_kv_caches(const py::object &input) {
  if (THPVariable_Check(input.ptr())) {
    return {input.cast<torch::Tensor>()};
  } else if (py::isinstance<py::tuple>(input)) {
    return input.cast<std::vector<torch::Tensor>>();
  } else {
    throw std::runtime_error(
        "vllm_kv_caches must be a Tensor or a tuple of Tensors");
  }
}

void single_layer_kv_transfer_wrapper(
    torch::Tensor &lmc_key_value_cache, const py::object &vllm_kv_caches_obj,
    torch::Tensor &slot_mapping, bool direction, int kvcache_format_raw,
    bool token_major, bool vllm_two_major, int64_t k_hidden_dims = 0,
    int64_t v_hidden_dims = 0, int64_t dsa_hidden_dims = 0) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  single_layer_kv_transfer(lmc_key_value_cache, vllm_kv_caches, slot_mapping,
                           direction, kvcache_format_raw, token_major,
                           vllm_two_major, k_hidden_dims, v_hidden_dims,
                           dsa_hidden_dims);
}

void batched_fused_single_layer_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    const py::object &vllm_kv_caches_obj, torch::Tensor &slot_mapping_full,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    bool direction, int kvcache_format_raw, bool token_major,
    bool vllm_two_major, int64_t k_hidden_dims = 0, int64_t v_hidden_dims = 0,
    int64_t dsa_hidden_dims = 0) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  batched_fused_single_layer_kv_transfer(
      lmc_tensors, staging_cache, vllm_kv_caches, slot_mapping_full,
      chunk_offsets, chunk_sizes, direction, kvcache_format_raw, token_major,
      vllm_two_major, k_hidden_dims, v_hidden_dims, dsa_hidden_dims);
}

void sparse_single_layer_kv_transfer_wrapper(
    torch::Tensor &lmc_key_value_cache, const py::object &vllm_kv_caches_obj,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int kvcache_format_raw, bool token_major, bool vllm_two_major,
    int64_t k_hidden_dims = 0, int64_t v_hidden_dims = 0,
    int64_t dsa_hidden_dims = 0) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  sparse_single_layer_kv_transfer(lmc_key_value_cache, vllm_kv_caches,
                                  slot_mapping_packed, selected_token_idx,
                                  kvcache_format_raw, token_major,
                                  vllm_two_major, k_hidden_dims, v_hidden_dims,
                                  dsa_hidden_dims);
}

void batched_fused_sparse_single_layer_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    const py::object &vllm_kv_caches_obj, torch::Tensor &slot_mapping_packed,
    torch::Tensor &selected_token_idx, std::vector<int64_t> &chunk_offsets,
    std::vector<int64_t> &chunk_sizes, int kvcache_format_raw, bool token_major,
    bool vllm_two_major, int64_t k_hidden_dims = 0, int64_t v_hidden_dims = 0,
    int64_t dsa_hidden_dims = 0,
    c10::optional<torch::Tensor> sparse_indices_cpu = c10::nullopt) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  batched_fused_sparse_single_layer_kv_transfer(
      lmc_tensors, staging_cache, vllm_kv_caches, slot_mapping_packed,
      selected_token_idx, chunk_offsets, chunk_sizes, kvcache_format_raw,
      token_major, vllm_two_major, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims, sparse_indices_cpu);
}

void sparse_mla_dsa_batched_direct_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, const py::object &vllm_kv_caches_obj,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int64_t chunk_size, int64_t total_tokens, int kvcache_format_raw,
    bool token_major, bool vllm_two_major, int64_t k_hidden_dims = 0,
    int64_t v_hidden_dims = 0, int64_t dsa_hidden_dims = 0,
    bool lmc_host_interleaved = false,
    const c10::optional<torch::Tensor> &chunk_ptrs_npu = c10::nullopt) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  sparse_mla_dsa_batched_direct_kv_transfer(
      lmc_tensors, vllm_kv_caches, slot_mapping_packed, selected_token_idx,
      chunk_size, total_tokens, kvcache_format_raw, token_major, vllm_two_major,
      k_hidden_dims, v_hidden_dims, dsa_hidden_dims, lmc_host_interleaved,
      chunk_ptrs_npu);
}

SparseDirectLayerState prepare_sparse_direct_layer_state_wrapper(
    torch::Tensor &lmc_layout_sample, const py::object &vllm_kv_caches_obj,
    torch::Tensor &slot_mapping_ref, bool token_major, bool vllm_two_major,
    int kvcache_format_raw, int64_t k_hidden_dims, int64_t v_hidden_dims,
    int64_t dsa_hidden_dims, int32_t lmc_num_tokens) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  return prepare_sparse_direct_layer_state(
      lmc_layout_sample, vllm_kv_caches, slot_mapping_ref, token_major,
      vllm_two_major, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims, lmc_num_tokens);
}

void sparse_mla_dsa_batched_direct_kv_transfer_fast_wrapper(
    SparseDirectLayerState &layer_state, torch::Tensor &slot_mapping_packed,
    torch::Tensor &selected_token_idx, torch::Tensor &chunk_ptrs_npu,
    int64_t chunk_size, int64_t total_tokens, bool lmc_host_interleaved,
    bool validate_inputs) {
  sparse_mla_dsa_batched_direct_kv_transfer_fast(
      layer_state, slot_mapping_packed, selected_token_idx, chunk_ptrs_npu,
      chunk_size, total_tokens, lmc_host_interleaved, validate_inputs);
}

PYBIND11_MODULE(c_ops, m) {
  m.def("get_device_ptr", [](uintptr_t ptr_addr) {
    return reinterpret_cast<uintptr_t>(
        get_device_ptr(reinterpret_cast<void *>(ptr_addr)));
  });
  m.def("register_mapping",
        [](uintptr_t host_ptr, uintptr_t dev_ptr, size_t size) {
          return reinterpret_cast<uintptr_t>(
              register_mapping(reinterpret_cast<void *>(host_ptr),
                               reinterpret_cast<void *>(dev_ptr), size));
        });
  m.def("unregister_ptr", [](uintptr_t ptr_addr) {
    return unregister_ptr(reinterpret_cast<void *>(ptr_addr));
  });
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("fused_multi_layer_kv_transfer", &fused_multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_310p", &multi_layer_kv_transfer_310p);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer_wrapper,
        py::arg("lmc_key_value_cache"), py::arg("vllm_kv_caches"),
        py::arg("slot_mapping"), py::arg("direction"),
        py::arg("kvcache_format_raw"), py::arg("token_major") = false,
        py::arg("vllm_two_major") = false, py::arg("k_hidden_dims") = 0,
        py::arg("v_hidden_dims") = 0, py::arg("dsa_hidden_dims") = 0);
  m.def("batched_fused_single_layer_kv_transfer",
        &batched_fused_single_layer_kv_transfer_wrapper,
        py::arg("lmc_tensors"), py::arg("staging_cache"),
        py::arg("vllm_kv_caches"), py::arg("slot_mapping_full"),
        py::arg("chunk_offsets"), py::arg("chunk_sizes"), py::arg("direction"),
        py::arg("kvcache_format_raw"), py::arg("token_major") = false,
        py::arg("vllm_two_major") = false, py::arg("k_hidden_dims") = 0,
        py::arg("v_hidden_dims") = 0, py::arg("dsa_hidden_dims") = 0);
  m.def("sparse_single_layer_kv_transfer",
        &sparse_single_layer_kv_transfer_wrapper, py::arg("lmc_key_value_cache"),
        py::arg("vllm_kv_caches"), py::arg("slot_mapping_packed"),
        py::arg("selected_token_idx"), py::arg("kvcache_format_raw"),
        py::arg("token_major") = false, py::arg("vllm_two_major") = false,
        py::arg("k_hidden_dims") = 0, py::arg("v_hidden_dims") = 0,
        py::arg("dsa_hidden_dims") = 0);
  m.def("batched_fused_sparse_single_layer_kv_transfer",
        &batched_fused_sparse_single_layer_kv_transfer_wrapper,
        py::arg("lmc_tensors"), py::arg("staging_cache"),
        py::arg("vllm_kv_caches"), py::arg("slot_mapping_packed"),
        py::arg("selected_token_idx"), py::arg("chunk_offsets"),
        py::arg("chunk_sizes"), py::arg("kvcache_format_raw"),
        py::arg("token_major") = false, py::arg("vllm_two_major") = false,
        py::arg("k_hidden_dims") = 0, py::arg("v_hidden_dims") = 0,
        py::arg("dsa_hidden_dims") = 0, py::arg("sparse_indices_cpu") = py::none());
  m.def("sparse_mla_dsa_batched_direct_kv_transfer",
        &sparse_mla_dsa_batched_direct_kv_transfer_wrapper,
        py::arg("lmc_tensors"), py::arg("vllm_kv_caches"),
        py::arg("slot_mapping_packed"), py::arg("selected_token_idx"),
        py::arg("chunk_size"), py::arg("total_tokens"),
        py::arg("kvcache_format_raw"), py::arg("token_major") = false,
        py::arg("vllm_two_major") = false, py::arg("k_hidden_dims") = 0,
        py::arg("v_hidden_dims") = 0, py::arg("dsa_hidden_dims") = 0,
        py::arg("lmc_host_interleaved") = false,
        py::arg("chunk_ptrs_npu") = py::none());
  py::class_<SparseDirectLayerState>(m, "SparseDirectLayerState");
  m.def("prepare_sparse_direct_layer_state",
        &prepare_sparse_direct_layer_state_wrapper,
        py::arg("lmc_layout_sample"), py::arg("vllm_kv_caches"),
        py::arg("slot_mapping_ref"), py::arg("token_major"),
        py::arg("vllm_two_major"), py::arg("kvcache_format_raw"),
        py::arg("k_hidden_dims"), py::arg("v_hidden_dims"),
        py::arg("dsa_hidden_dims"), py::arg("lmc_num_tokens"));
  m.def("sparse_mla_dsa_batched_direct_kv_transfer_fast",
        &sparse_mla_dsa_batched_direct_kv_transfer_fast_wrapper,
        py::arg("layer_state"), py::arg("slot_mapping_packed"),
        py::arg("selected_token_idx"), py::arg("chunk_ptrs_npu"),
        py::arg("chunk_size"), py::arg("total_tokens"),
        py::arg("lmc_host_interleaved"), py::arg("validate_inputs") = false);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_ascend_new);
  m.def("decode_fast_new", &decode_ascend_new);
  m.def("decode_fast_prefsum", &decode_ascend_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);
}
