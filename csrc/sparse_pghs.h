#pragma once

#include <acl/acl.h>
#include <torch/torch.h>
#include <vector>

// Ping-pong staging buffers for PGHS (Pipelined Gather + H2D + Scatter).
class StagingBufferPool {
public:
  static constexpr int kNumSlots = 2;

  StagingBufferPool(int64_t max_slot_bytes, int32_t max_tokens,
                    int64_t bytes_per_token, at::ScalarType dtype,
                    const torch::Device &npu_device);

  ~StagingBufferPool();

  StagingBufferPool(const StagingBufferPool &) = delete;
  StagingBufferPool &operator=(const StagingBufferPool &) = delete;

  int32_t max_tokens() const { return max_tokens_; }
  int64_t slot_bytes() const { return slot_bytes_; }
  int64_t bytes_per_token() const { return bytes_per_token_; }

  torch::Tensor cpu_staging(int slot_id) const;
  torch::Tensor npu_staging(int slot_id) const;

  // Returns staging token indices [0, 1, ..., count-1] on NPU (cached by count).
  torch::Tensor staging_token_idx(int32_t count);

  void reset();

private:
  int32_t max_tokens_;
  int64_t slot_bytes_;
  int64_t bytes_per_token_;
  at::ScalarType dtype_;
  torch::Device npu_device_;

  void *cpu_ptrs_[kNumSlots]{};
  torch::Tensor npu_tensors_[kNumSlots];
  std::vector<torch::Tensor> cpu_views_;

  // Cache arange indices keyed by count.
  std::vector<std::pair<int32_t, torch::Tensor>> staging_idx_cache_;
};

// Gather sparse tokens from multi-chunk pinned CPU memory into a contiguous
// interleaved staging buffer (token-major K+V(+DSA) per token).
void sparse_mla_dsa_gather_to_staging(
    uint8_t *dst_staging, const std::vector<torch::Tensor> &lmc_chunks,
    const int32_t *global_token_idx, int32_t batch_tokens, int32_t chunk_size,
    int32_t num_chunks, int32_t total_tokens, int64_t bytes_per_token,
    int32_t gather_thread_num);

// Scatter from NPU staging buffer into paged MLA/DSA KV (no H2D memcpy).
void detail_sparse_mla_dsa_scatter_from_staging(
    torch::Tensor &staging_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &staging_token_idx,
    int kvcache_format_raw, int64_t k_hidden_dims, int64_t v_hidden_dims,
    int64_t dsa_hidden_dims, int32_t scatter_aiv_num, aclrtStream stream);

// Full single-layer PGHS retrieve using internal H2D/scatter streams.
void sparse_mla_dsa_pghs_layer_transfer(
    StagingBufferPool &pool, std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int64_t chunk_size, int64_t total_tokens, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t micro_batch_tokens, int32_t gather_thread_num,
    int32_t scatter_aiv_num, int32_t event_timeout_ms = 30000);

// Variant with caller-provided streams (connector integration).
void sparse_mla_dsa_pghs_layer_transfer_streams(
    StagingBufferPool &pool, std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int64_t chunk_size, int64_t total_tokens, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t micro_batch_tokens, int32_t gather_thread_num,
    int32_t scatter_aiv_num, aclrtStream h2d_stream, aclrtStream scatter_stream,
    int32_t event_timeout_ms = 30000);

int32_t compute_effective_batch_tokens(int32_t micro_batch_tokens,
                                       int64_t max_slot_bytes,
                                       int64_t bytes_per_token);
