#pragma once
#include "kernels/types.h"
#include "managed_mem.h"
#include "utils.h"
#include <torch/extension.h>
#include <torch/torch.h>

namespace kvcache_ops {
void multi_layer_kv_transfer_kernel(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    const kvcache_ops::KVCacheFormat kvcache_format, uint32_t blockDim,
    void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor,
    uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs,
    const int32_t numLayers, const int64_t pageBuffSize,
    const int32_t numTokensChunk, const bool page2L);

void multi_layer_kv_transfer_kernel_310p(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    const kvcache_ops::KVCacheFormat kvcache_format, uint32_t blockDim,
    void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor,
    uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs,
    const int32_t numLayers, const int64_t pageBuffSize,
    const int32_t numTokensChunk, const bool page2L, const int32_t numKVHead,
    const int32_t headSize, const int32_t blockSize);

void multi_layer_kv_transfer_kernel_v2(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    const kvcache_ops::KVCacheFormat kvcache_format, uint32_t blockDim,
    void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor,
    uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs,
    const int32_t numLayers, const int64_t pageBuffSize,
    const int32_t numTokensChunk, const int64_t perLoopBuffer,
    const int32_t maxTokensPerLoop, const bool page2L,
    const int64_t kHiddenDims = 0, const int64_t vHiddenDims = 0,
    const int64_t dsaHiddenDims = 0);

void single_layer_kv_transfer_kernel_v2(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *lmcKeyValueCache,
    uint8_t *vllmKeyValueCache, uint8_t *slotmappings,
    const int64_t vllmBlockStride, const int64_t vllmValueOffset,
    const int64_t vllmBufferSize, const int64_t lmcTokenStride,
    const int64_t lmcValueOffset, const int64_t lmcBufferSize,
    const int32_t maxTokensPerLoop, const int32_t numHeads,
    const int32_t headDims, const int32_t numTokens, const int32_t blockSize,
    const bool page2L, const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_separate(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *lmcKeyValueCachePtr,
    uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr, uint8_t *slotMappingPtr,
    const int64_t keyBlockStride, const int64_t valueBlockStride,
    const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,
    const int64_t lmcTokenStride, const int64_t lmcValueOffset,
    const int64_t lmcBufferSize, const int32_t maxTokensPerLoop,
    const int32_t numHeads, const int32_t headDims, const int32_t numTokens,
    const int32_t blockSize, const bool page2L, const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_sparse(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *lmcKeyValueCachePtr,
    uint8_t *vllmKeyValueCache, uint8_t *slotMappingPtr,
    uint8_t *selectedTokenIdxPtr, const int64_t vllmBlockStride,
    const int64_t vllmValueOffset, const int64_t vllmBufferSize,
    const int64_t lmcTokenStride, const int64_t lmcValueOffset,
    const int64_t lmcBufferSize, const int32_t maxTokensPerLoop,
    const int32_t numHeads, const int32_t headDims, const int32_t numTokens,
    const int32_t blockSize, const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_separate_sparse(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *lmcKeyValueCachePtr,
    uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr, uint8_t *slotMappingPtr,
    uint8_t *selectedTokenIdxPtr, const int64_t keyBlockStride,
    const int64_t valueBlockStride, const int64_t vllmKeyBufferSize,
    const int64_t vllmValueBufferSize, const int64_t lmcTokenStride,
    const int64_t lmcValueOffset, const int64_t lmcBufferSize,
    const int32_t maxTokensPerLoop, const int32_t numHeads,
    const int32_t headDims, const int32_t numTokens, const int32_t blockSize,
    const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_sparse_multi_chunk(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *chunkPtrsPtr,
    uint8_t *vllmKeyValueCache, uint8_t *slotMappingPtr,
    uint8_t *selectedTokenIdxPtr, const int64_t vllmBlockStride,
    const int64_t vllmValueOffset, const int64_t vllmBufferSize,
    const int64_t lmcTokenStride, const int64_t lmcValueOffset,
    const int32_t maxTokensPerLoop, const int32_t numHeads,
    const int32_t headDims, const int32_t numTokens, const int32_t numChunks,
    const int32_t chunkSize, const int32_t totalTokens, const int32_t blockSize,
    const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_separate_sparse_multi_chunk(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *chunkPtrsPtr, uint8_t *vllmKeyPtr,
    uint8_t *vllmValuePtr, uint8_t *slotMappingPtr, uint8_t *selectedTokenIdxPtr,
    const int64_t keyBlockStride, const int64_t valueBlockStride,
    const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,
    const int64_t lmcTokenStride, const int64_t lmcValueOffset,
    const int32_t maxTokensPerLoop, const int32_t numHeads,
    const int32_t headDims, const int32_t numTokens, const int32_t numChunks,
    const int32_t chunkSize, const int32_t totalTokens, const int32_t blockSize,
    const bool lmcTokensMajor);

void single_layer_kv_transfer_kernel_v2_mla_dsa(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    kvcache_ops::KVCacheFormat format, uint32_t blockDim, void *stream,
    uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr,
    uint8_t *vllmDsaPtr, uint8_t *slotMappingPtr, const int64_t lmcBufferSize,
    const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,
    const int64_t vllmDsaBufferSize, const int32_t maxTokensPerLoop,
    const int64_t kHiddenDims, const int64_t vHiddenDims,
    const int64_t dsaHiddenDims, const int32_t numTokens, const int32_t numLmcTokens,
    const int32_t blockSize, const bool page2L, const bool lmcHostInterleaved = false);

void single_layer_kv_transfer_kernel_v2_mla_dsa_sparse(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    kvcache_ops::KVCacheFormat format, uint32_t blockDim, void *stream,
    uint8_t *lmcKeyValueCachePtr, uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr,
    uint8_t *vllmDsaPtr, uint8_t *slotMappingPtr, uint8_t *selectedTokenIdxPtr,
    const int64_t lmcBufferSize, const int64_t vllmKeyBufferSize,
    const int64_t vllmValueBufferSize, const int64_t vllmDsaBufferSize,
    const int32_t maxTokensPerLoop, const int64_t kHiddenDims, const int64_t vHiddenDims,
    const int64_t dsaHiddenDims, const int32_t numTokens, const int32_t numLmcTokens,
    const int32_t blockSize, const bool lmcHostInterleaved = false);

void single_layer_kv_transfer_kernel_v2_mla_dsa_sparse_multi_chunk(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    kvcache_ops::KVCacheFormat format, uint32_t blockDim, void *stream,
    uint8_t *chunkPtrsPtr, uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr,
    uint8_t *vllmDsaPtr, uint8_t *slotMappingPtr, uint8_t *selectedTokenIdxPtr,
    uint8_t *selectedTokenCountsPtr,
    const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,
    const int64_t vllmDsaBufferSize, const int32_t maxTokensPerLoop,
    const int64_t kHiddenDims, const int64_t vHiddenDims, const int64_t dsaHiddenDims,
    const int32_t numTokens, const int32_t numChunks, const int32_t chunkSize,
    const int32_t totalTokens, const int32_t blockSize,
    const bool lmcHostInterleaved = false, const int32_t rowWidth = 0,
    const int32_t requestCount = 0, const int32_t selectedCountStride = 1);

void single_layer_kv_transfer_kernel_v2_mla_dsa_dense_multi_chunk(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    kvcache_ops::KVCacheFormat format, uint32_t blockDim, void *stream,
    uint8_t *chunkPtrsPtr, uint8_t *chunkOffsetsPtr, uint8_t *chunkSizesPtr,
    uint8_t *vllmKeyPtr, uint8_t *vllmValuePtr, uint8_t *vllmDsaPtr,
    uint8_t *slotMappingPtr, const int64_t vllmKeyBufferSize,
    const int64_t vllmValueBufferSize, const int64_t vllmDsaBufferSize,
    const int32_t maxTokensPerLoop, const int64_t kHiddenDims,
    const int64_t vHiddenDims, const int64_t dsaHiddenDims,
    const int32_t numTokens, const int32_t numChunks,
    const int32_t fixedChunkSize, const int32_t totalTokens,
    const int32_t blockSize, const bool page2L,
    const bool lmcHostInterleaved = false);

void load_and_reshape_flash_kernel(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    uint32_t blockDim, void *stream, uint8_t *dstCacheTensor,
    uint8_t *keyCachePtr, uint8_t *valueCachePtr, uint8_t *slotmappings,
    const int64_t hiddenDims, const int64_t numPages, const int32_t pagedSize,
    const int32_t numTokens, const int32_t numLayers, const int32_t layerIdx,
    const bool page2L);
} // namespace kvcache_ops

void multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims = 0, const int64_t v_hidden_dims = 0,
    const int64_t dsa_hidden_dims = 0);

void fused_multi_layer_kv_transfer(
    torch::Tensor &key_value,
    torch::Tensor &staging_cache, // staging buffer
    const torch::Tensor &key_value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims = 0, const int64_t v_hidden_dims = 0,
    const int64_t dsa_hidden_dims = 0);

void multi_layer_kv_transfer_310p(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int num_kv_head,
    const int head_size, const int block_size, const int kvcache_format_raw);

void multi_layer_kv_transfer_unilateral(
    torch::Tensor &key_value, const torch::Tensor &key_ptrs,
    const torch::Tensor &value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction);

void single_layer_kv_transfer(torch::Tensor &lmc_key_value_cache,
                              std::vector<torch::Tensor> &vllm_kv_caches,
                              torch::Tensor &slot_mapping, const bool direction,
                              const int kvcache_format_raw,
                              const bool token_major = false,
                              const bool vllm_two_major = false,
                              const int64_t k_hidden_dims = 0,
                              const int64_t v_hidden_dims = 0,
                              const int64_t dsa_hidden_dims = 0);

void batched_fused_single_layer_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_full, std::vector<int64_t> &chunk_offsets,
    std::vector<int64_t> &chunk_sizes, const bool direction,
    const int kvcache_format_raw, const bool token_major = false,
    const bool vllm_two_major = false, const int64_t k_hidden_dims = 0,
    const int64_t v_hidden_dims = 0, const int64_t dsa_hidden_dims = 0);

// Sparse scatter: staging (LMC layout) -> paged vLLM KV.
// slot_mapping_packed and selected_token_idx are parallel packed arrays with no
// -1 holes. selected_token_idx[i] is the LMC staging token index for entry i.
void sparse_single_layer_kv_transfer(
    torch::Tensor &lmc_key_value_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    const int kvcache_format_raw, const bool token_major = false,
    const bool vllm_two_major = false, const int64_t k_hidden_dims = 0,
    const int64_t v_hidden_dims = 0, const int64_t dsa_hidden_dims = 0);

void batched_fused_sparse_single_layer_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    const int kvcache_format_raw, const bool token_major = false,
    const bool vllm_two_major = false, const int64_t k_hidden_dims = 0,
    const int64_t v_hidden_dims = 0, const int64_t dsa_hidden_dims = 0,
    const c10::optional<torch::Tensor> &sparse_indices_cpu = c10::nullopt);

// Sparse retrieve directly from CPU pinned memory objects to paged KV.
// No NPU staging buffer or aclrtMemcpyAsync H2D is used.
void sparse_mla_dsa_batched_direct_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    const int64_t chunk_size, const int64_t total_tokens,
    const int kvcache_format_raw, const bool token_major = false,
    const bool vllm_two_major = false, const int64_t k_hidden_dims = 0,
    const int64_t v_hidden_dims = 0, const int64_t dsa_hidden_dims = 0,
    const bool lmc_host_interleaved = false,
    const c10::optional<torch::Tensor> &chunk_ptrs_npu = c10::nullopt,
    const c10::optional<torch::Tensor> &selected_token_counts = c10::nullopt);

// Hot path: reuse cached per-layer config; no CPU chunk tensors required.
void sparse_mla_dsa_batched_direct_kv_transfer_fast(
    SparseDirectLayerState &layer_state, torch::Tensor &slot_mapping_packed,
    torch::Tensor &selected_token_idx, torch::Tensor &chunk_ptrs_npu,
    const int64_t chunk_size, const int64_t total_tokens,
    const bool lmc_host_interleaved, const bool validate_inputs = false,
    const c10::optional<torch::Tensor> &selected_token_counts = c10::nullopt);

// Prepared warm path: destination state is process-owned; all request and step
// inputs are supplied by the decode generator at launch time.
void sparse_mla_dsa_batched_direct_kv_transfer_prepared(
    const SparseDirectDestinationState &destination_state,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    torch::Tensor &chunk_ptrs_npu, const int64_t chunk_size,
    const int64_t total_tokens, const bool lmc_host_interleaved,
    const c10::optional<torch::Tensor> &selected_token_counts = c10::nullopt,
    const int64_t diagnostic_layer_id = -1);

// Dense MLA/DSA direct transfer between CPU pinned chunks and paged KV.
// direction=false: host chunks -> paged KV; direction=true: paged KV -> host chunks.
void dense_mla_dsa_batched_direct_kv_transfer(
    std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_full, torch::Tensor &chunk_offsets_npu,
    torch::Tensor &chunk_sizes_npu, const int64_t total_tokens,
    const int kvcache_format_raw, const bool token_major = false,
    const bool vllm_two_major = false, const int64_t k_hidden_dims = 0,
    const int64_t v_hidden_dims = 0, const int64_t dsa_hidden_dims = 0,
    const bool lmc_host_interleaved = false, const bool direction = false,
    const c10::optional<torch::Tensor> &chunk_ptrs_npu = c10::nullopt,
    const int64_t fixed_chunk_size = 0);

// Hot path: reuse cached per-layer config; no CPU chunk tensors required.
void dense_mla_dsa_batched_direct_kv_transfer_fast(
    SparseDirectLayerState &layer_state, torch::Tensor &slot_mapping_full,
    torch::Tensor &chunk_ptrs_npu, torch::Tensor &chunk_offsets_npu,
    torch::Tensor &chunk_sizes_npu, const int64_t total_tokens,
    const bool lmc_host_interleaved, const bool direction,
    const bool validate_inputs = false, const int64_t fixed_chunk_size = 0);

// Prepared H2D path: destination state is process-owned; request metadata is
// supplied dynamically and never becomes part of a prepared-state cache key.
void dense_mla_dsa_batched_direct_kv_transfer_prepared(
    const SparseDirectDestinationState &destination_state,
    torch::Tensor &slot_mapping_full, torch::Tensor &chunk_ptrs_npu,
    torch::Tensor &chunk_offsets_npu, torch::Tensor &chunk_sizes_npu,
    const int64_t total_tokens, const bool lmc_host_interleaved,
    const bool validate_inputs = false, const int64_t fixed_chunk_size = 0);

// Group hot path: one host dispatch runs the existing per-layer direct kernel.
void dense_mla_dsa_group_direct_kv_transfer_fast(
    const std::vector<SparseDirectLayerState> &layer_states,
    torch::Tensor &slot_mapping_full, torch::Tensor &layer_chunk_ptrs_npu,
    torch::Tensor &chunk_offsets_npu, torch::Tensor &chunk_sizes_npu,
    const int64_t total_tokens, const bool lmc_host_interleaved,
    const bool direction, const bool validate_inputs = false,
    const int64_t fixed_chunk_size = 0);

void load_and_reshape_flash(torch::Tensor &key_value, torch::Tensor &key_cache,
                            torch::Tensor &value_cache,
                            torch::Tensor &slot_mapping, const int layer_idx);

void reshape_and_cache_back_flash(torch::Tensor &key_value,
                                  torch::Tensor &key_cache,
                                  torch::Tensor &value_cache,
                                  torch::Tensor &slot_mapping,
                                  const int layer_idx);
