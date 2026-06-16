#include "sparse_pghs.h"
#include "mem_kernels.h"
#include "mem_alloc.h"
#include "utils.h"

#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace {

constexpr int kNumSlots = StagingBufferPool::kNumSlots;

static bool is_mla_dsa_format(kvcache_ops::KVCacheFormat format) {
  return format == kvcache_ops::KVCacheFormat::MLA_KV ||
         format == kvcache_ops::KVCacheFormat::DSA_KV;
}

static void launch_sparse_mla_dsa_scatter_from_staging(
    const SingleLayerKVConfig &config, uint8_t *staging_token_idx_ptr,
    bool lmc_host_interleaved) {
  kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_sparse(
      config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
      config.kvcache_format, config.ub_params.aiv_num, config.ub_params.stream,
      config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
      config.ptrs.vllm_dsa_ptr, config.ptrs.slot_mapping_ptr,
      staging_token_idx_ptr, config.strides.lmc_bytes,
      config.strides.vllm_k_bytes, config.strides.vllm_v_bytes,
      config.strides.vllm_dsa_bytes, config.ub_params.max_tokens_per_loop,
      config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims,
      config.dims.num_tokens, config.dims.lmc_num_tokens,
      config.dims.block_size, lmc_host_interleaved);
}

static int32_t resolve_last_chunk_tokens(int32_t num_chunks, int32_t chunk_size,
                                         int32_t total_tokens) {
  if (num_chunks <= 0) {
    return 0;
  }
  if (num_chunks == 1) {
    return total_tokens;
  }
  return total_tokens - (num_chunks - 1) * chunk_size;
}

static void gather_token_range(const uint8_t *const *chunk_bases,
                               const int32_t *last_chunk_tokens, int32_t chunk_size,
                               int32_t num_chunks, int32_t total_tokens,
                               const int32_t *global_token_idx, int32_t begin,
                               int32_t end, uint8_t *dst, int64_t bytes_per_token) {
  for (int32_t t = begin; t < end; ++t) {
    const int32_t global_idx = global_token_idx[t];
    const int32_t chunk_id = global_idx / chunk_size;
    const int32_t local_t = global_idx % chunk_size;
    if (chunk_id >= num_chunks || chunk_id < 0) {
      TORCH_CHECK(false, "global_token_idx out of range: ", global_idx);
    }
    if (chunk_id == num_chunks - 1) {
      const int32_t last_tokens = last_chunk_tokens[chunk_id];
      if (local_t >= last_tokens) {
        TORCH_CHECK(false, "global_token_idx out of last chunk range: ",
                    global_idx);
      }
    }
    const uint8_t *src =
        chunk_bases[chunk_id] + static_cast<int64_t>(local_t) * bytes_per_token;
    uint8_t *dst_token = dst + static_cast<int64_t>(t) * bytes_per_token;
    std::memcpy(dst_token, src, static_cast<size_t>(bytes_per_token));
  }
}

static bool query_event_done(aclrtEvent event) {
  aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
  aclError ret = aclrtQueryEventStatus(event, &status);
  if (ret != ACL_SUCCESS) {
    return false;
  }
  return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

static void wait_event_or_throw(aclrtEvent event, int32_t timeout_ms,
                                const char *label) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (!query_event_done(event)) {
    if (std::chrono::steady_clock::now() > deadline) {
      TORCH_CHECK(false, "PGHS timeout waiting for ", label);
    }
    std::this_thread::yield();
  }
}

struct PersistentSlotEvents {
  aclrtEvent h2d_done[kNumSlots]{nullptr, nullptr};
  aclrtEvent scatter_done[kNumSlots]{nullptr, nullptr};

  PersistentSlotEvents() {
    for (int i = 0; i < kNumSlots; ++i) {
      TORCH_CHECK(aclrtCreateEvent(&h2d_done[i]) == ACL_SUCCESS,
                  "aclrtCreateEvent h2d failed");
      TORCH_CHECK(aclrtCreateEvent(&scatter_done[i]) == ACL_SUCCESS,
                  "aclrtCreateEvent scatter failed");
    }
  }

  ~PersistentSlotEvents() {
    for (int i = 0; i < kNumSlots; ++i) {
      if (h2d_done[i] != nullptr) {
        aclrtDestroyEvent(h2d_done[i]);
      }
      if (scatter_done[i] != nullptr) {
        aclrtDestroyEvent(scatter_done[i]);
      }
    }
  }
};

PersistentSlotEvents &persistent_slot_events() {
  static thread_local PersistentSlotEvents events;
  return events;
}

static SingleLayerKVConfig
build_scatter_config(torch::Tensor &staging_view,
                     std::vector<torch::Tensor> &vllm_kv_caches,
                     torch::Tensor &slot_mapping_mb, int kvcache_format_raw,
                     int64_t k_hidden_dims, int64_t v_hidden_dims,
                     int64_t dsa_hidden_dims, int32_t scatter_aiv_num,
                     aclrtStream stream) {
  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_view, vllm_kv_caches, slot_mapping_mb, false, false, false,
      kvcache_format_raw, k_hidden_dims, v_hidden_dims, dsa_hidden_dims);
  config.ub_params.stream = stream;
  config.ub_params.aiv_num = static_cast<uint32_t>(std::max(
      1, std::min(scatter_aiv_num, config.dims.num_tokens)));
  return config;
}

struct SlotTracker {
  enum class State { FREE, SCATTER_INFLIGHT };
  State state[kNumSlots]{State::FREE, State::FREE};
};

} // namespace

int32_t compute_effective_batch_tokens(int32_t micro_batch_tokens,
                                       int64_t max_slot_bytes,
                                       int64_t bytes_per_token) {
  TORCH_CHECK(bytes_per_token > 0, "bytes_per_token must be positive");
  const int32_t cap =
      static_cast<int32_t>(max_slot_bytes / bytes_per_token);
  TORCH_CHECK(cap > 0, "max_slot_bytes too small for one token");
  return std::max(1, std::min(micro_batch_tokens, cap));
}

int32_t compute_slot_token_capacity(int64_t max_slot_bytes,
                                    int64_t bytes_per_token) {
  return compute_effective_batch_tokens(INT32_MAX, max_slot_bytes,
                                        bytes_per_token);
}

int32_t compute_pghs_step_tokens(int32_t num_remaining, int32_t slot_cap,
                                 int32_t micro_batch_tokens) {
  TORCH_CHECK(num_remaining > 0, "num_remaining must be positive");
  TORCH_CHECK(slot_cap > 0, "slot_cap must be positive");
  if (num_remaining <= slot_cap) {
    return num_remaining;
  }
  const int32_t configured =
      micro_batch_tokens > 0 ? micro_batch_tokens : slot_cap;
  return std::max(1, std::min(configured, slot_cap));
}

StagingBufferPool::StagingBufferPool(int64_t max_slot_bytes, int32_t max_tokens,
                                     int64_t bytes_per_token,
                                     at::ScalarType dtype,
                                     const torch::Device &npu_device)
    : max_tokens_(max_tokens), slot_bytes_(max_tokens * bytes_per_token),
      bytes_per_token_(bytes_per_token), dtype_(dtype),
      npu_device_(npu_device) {
  TORCH_CHECK(max_tokens > 0, "max_tokens must be positive");
  TORCH_CHECK(slot_bytes_ <= max_slot_bytes,
              "slot_bytes exceeds max_slot_bytes cap");

  const int64_t elem_size = static_cast<int64_t>(torch::elementSize(dtype_));
  const int64_t plane_elems = bytes_per_token / elem_size;

  cpu_views_.reserve(kNumSlots);
  for (int i = 0; i < kNumSlots; ++i) {
    cpu_ptrs_[i] = reinterpret_cast<void *>(
        alloc_pinned_ptr(static_cast<size_t>(slot_bytes_), 0));

    npu_tensors_[i] = torch::empty(
        {max_tokens_ * plane_elems},
        torch::TensorOptions().dtype(dtype_).device(npu_device_));
    cpu_views_.push_back(torch::from_blob(
        cpu_ptrs_[i], {max_tokens_ * plane_elems},
        torch::TensorOptions().dtype(dtype_).device(torch::kCPU)));
  }
}

StagingBufferPool::~StagingBufferPool() {
  for (int i = 0; i < kNumSlots; ++i) {
    if (cpu_ptrs_[i] != nullptr) {
      free_pinned_ptr(reinterpret_cast<uintptr_t>(cpu_ptrs_[i]));
      cpu_ptrs_[i] = nullptr;
    }
  }
}

torch::Tensor StagingBufferPool::cpu_staging(int slot_id) const {
  TORCH_CHECK(slot_id >= 0 && slot_id < kNumSlots, "invalid slot_id");
  return cpu_views_[slot_id];
}

torch::Tensor StagingBufferPool::npu_staging(int slot_id) const {
  TORCH_CHECK(slot_id >= 0 && slot_id < kNumSlots, "invalid slot_id");
  return npu_tensors_[slot_id];
}

torch::Tensor StagingBufferPool::staging_token_idx(int32_t count) {
  TORCH_CHECK(count > 0 && count <= max_tokens_, "invalid staging index count");
  for (const auto &entry : staging_idx_cache_) {
    if (entry.first == count) {
      return entry.second;
    }
  }
  auto idx = torch::arange(count, torch::TensorOptions()
                                      .dtype(torch::kInt32)
                                      .device(npu_device_));
  staging_idx_cache_.emplace_back(count, idx);
  return idx;
}

void StagingBufferPool::reset() { staging_idx_cache_.clear(); }

void sparse_mla_dsa_gather_to_staging(
    uint8_t *dst_staging, const std::vector<torch::Tensor> &lmc_chunks,
    const int32_t *global_token_idx, int32_t batch_tokens, int32_t chunk_size,
    int32_t num_chunks, int32_t total_tokens, int64_t bytes_per_token,
    int32_t gather_thread_num) {
  TORCH_CHECK(dst_staging != nullptr, "dst_staging is null");
  TORCH_CHECK(batch_tokens > 0, "batch_tokens must be positive");
  TORCH_CHECK(!lmc_chunks.empty(), "lmc_chunks must not be empty");

  std::vector<const uint8_t *> chunk_bases(lmc_chunks.size());
  std::vector<int32_t> last_chunk_tokens(lmc_chunks.size());
  for (size_t i = 0; i < lmc_chunks.size(); ++i) {
    TORCH_CHECK(lmc_chunks[i].device().is_cpu(), "lmc chunk must be CPU pinned");
    chunk_bases[i] = static_cast<const uint8_t *>(lmc_chunks[i].data_ptr());
    last_chunk_tokens[i] = (static_cast<int32_t>(i) == num_chunks - 1)
                               ? resolve_last_chunk_tokens(num_chunks, chunk_size,
                                                           total_tokens)
                               : chunk_size;
  }

  const int32_t threads = std::max(1, gather_thread_num);
  const int32_t parallel_threshold = std::max(threads * 64, 256);
  if (threads == 1 || batch_tokens < parallel_threshold) {
    gather_token_range(chunk_bases.data(), last_chunk_tokens.data(), chunk_size,
                       num_chunks, total_tokens, global_token_idx, 0,
                       batch_tokens, dst_staging, bytes_per_token);
    return;
  }

  const int32_t per_thread = (batch_tokens + threads - 1) / threads;
  std::vector<std::future<void>> futures;
  futures.reserve(threads);
  for (int32_t t = 0; t < threads; ++t) {
    const int32_t begin = t * per_thread;
    const int32_t end = std::min(batch_tokens, begin + per_thread);
    if (begin >= end) {
      break;
    }
    futures.push_back(std::async(std::launch::async, [=]() {
      gather_token_range(chunk_bases.data(), last_chunk_tokens.data(),
                         chunk_size, num_chunks, total_tokens, global_token_idx,
                         begin, end, dst_staging, bytes_per_token);
    }));
  }
  for (auto &f : futures) {
    f.get();
  }
}

void detail_sparse_mla_dsa_scatter_from_staging(
    torch::Tensor &staging_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &staging_token_idx,
    int kvcache_format_raw, int64_t k_hidden_dims, int64_t v_hidden_dims,
    int64_t dsa_hidden_dims, int32_t scatter_aiv_num, aclrtStream stream) {
  TORCH_CHECK(is_mla_dsa_format(
                  static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw)),
              "scatter_from_staging supports MLA/DSA only");
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping_packed));

  SingleLayerKVConfig config = build_scatter_config(
      staging_cache, vllm_kv_caches, slot_mapping_packed, kvcache_format_raw,
      k_hidden_dims, v_hidden_dims, dsa_hidden_dims, scatter_aiv_num, stream);
  uint8_t *staging_idx_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_token_idx);

  at_npu::native::OpCommand cmd;
  cmd.Name("sparse_mla_dsa_scatter_from_staging");
  cmd.SetCustomHandler([config, staging_idx_ptr]() -> int {
    launch_sparse_mla_dsa_scatter_from_staging(config, staging_idx_ptr, true);
    return 0;
  });
  cmd.Run();
}

void sparse_mla_dsa_pghs_layer_transfer(
    StagingBufferPool &pool, std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int64_t chunk_size, int64_t total_tokens, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t micro_batch_tokens, int32_t gather_thread_num,
    int32_t scatter_aiv_num, int32_t event_timeout_ms) {
  thread_local aclrtStream h2d_stream = nullptr;
  thread_local aclrtStream scatter_stream = nullptr;
  if (h2d_stream == nullptr) {
    TORCH_CHECK(aclrtCreateStream(&h2d_stream) == ACL_SUCCESS,
                "aclrtCreateStream h2d failed");
  }
  if (scatter_stream == nullptr) {
    TORCH_CHECK(aclrtCreateStream(&scatter_stream) == ACL_SUCCESS,
                "aclrtCreateStream scatter failed");
  }
  sparse_mla_dsa_pghs_layer_transfer_streams(
      pool, lmc_tensors, vllm_kv_caches, slot_mapping_packed, selected_token_idx,
      chunk_size, total_tokens, kvcache_format_raw, k_hidden_dims, v_hidden_dims,
      dsa_hidden_dims, micro_batch_tokens, gather_thread_num, scatter_aiv_num,
      h2d_stream, scatter_stream, event_timeout_ms);
}

void sparse_mla_dsa_pghs_layer_transfer_streams(
    StagingBufferPool &pool, std::vector<torch::Tensor> &lmc_tensors,
    std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping_packed, torch::Tensor &selected_token_idx,
    int64_t chunk_size, int64_t total_tokens, int kvcache_format_raw,
    int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
    int32_t micro_batch_tokens, int32_t gather_thread_num,
    int32_t scatter_aiv_num, aclrtStream h2d_stream, aclrtStream scatter_stream,
    int32_t event_timeout_ms) {
  TORCH_CHECK(is_mla_dsa_format(
                  static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw)),
              "PGHS supports MLA/DSA only");
  validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const int32_t num_sparse = static_cast<int32_t>(slot_mapping_packed.size(0));
  if (num_sparse == 0) {
    return;
  }

  at::Tensor selected_cpu =
      selected_token_idx.device().is_cpu()
          ? selected_token_idx.contiguous()
          : selected_token_idx.detach().cpu().contiguous();
  TORCH_CHECK(selected_cpu.scalar_type() == at::ScalarType::Int,
              "selected_token_idx must be int32");
  const int32_t *selected_ptr = selected_cpu.data_ptr<int32_t>();

  const int32_t slot_cap = pool.max_tokens();
  const int32_t num_chunks = static_cast<int32_t>(lmc_tensors.size());
  const int32_t chunk_size_i = static_cast<int32_t>(chunk_size);
  const int64_t bytes_per_token = pool.bytes_per_token();
  const int64_t plane_elems =
      bytes_per_token /
      static_cast<int64_t>(pool.npu_staging(0).element_size());

  auto &events = persistent_slot_events();
  SlotTracker tracker;
  int next_slot = 0;

  for (int32_t mb_start = 0; mb_start < num_sparse;) {
    const int32_t remaining = num_sparse - mb_start;
    const int32_t mb_count =
        compute_pghs_step_tokens(remaining, slot_cap, micro_batch_tokens);
    const int32_t mb_end = mb_start + mb_count;
    const int64_t copy_bytes = static_cast<int64_t>(mb_count) * bytes_per_token;
    const int32_t slot = next_slot;

    if (tracker.state[slot] == SlotTracker::State::SCATTER_INFLIGHT) {
      wait_event_or_throw(events.scatter_done[slot], event_timeout_ms,
                          "scatter_done");
      tracker.state[slot] = SlotTracker::State::FREE;
    }

    sparse_mla_dsa_gather_to_staging(
        static_cast<uint8_t *>(pool.cpu_staging(slot).data_ptr()), lmc_tensors,
        selected_ptr + mb_start, mb_count, chunk_size_i, num_chunks,
        static_cast<int32_t>(total_tokens), bytes_per_token, gather_thread_num);

    TORCH_CHECK(aclrtMemcpyAsync(pool.npu_staging(slot).data_ptr(), copy_bytes,
                                 pool.cpu_staging(slot).data_ptr(), copy_bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 h2d_stream) == ACL_SUCCESS,
                "H2D memcpy failed in PGHS micro-batch");
    TORCH_CHECK(aclrtRecordEvent(events.h2d_done[slot], h2d_stream) ==
                    ACL_SUCCESS,
                "aclrtRecordEvent h2d failed");
    TORCH_CHECK(aclrtStreamWaitEvent(scatter_stream, events.h2d_done[slot]) ==
                    ACL_SUCCESS,
                "aclrtStreamWaitEvent h2d failed");

    auto slot_mapping_mb = slot_mapping_packed.slice(0, mb_start, mb_end);
    auto staging_idx = pool.staging_token_idx(mb_count);
    auto staging_view =
        pool.npu_staging(slot).slice(0, 0, mb_count * plane_elems);

    SingleLayerKVConfig config = build_scatter_config(
        staging_view, vllm_kv_caches, slot_mapping_mb, kvcache_format_raw,
        k_hidden_dims, v_hidden_dims, dsa_hidden_dims, scatter_aiv_num,
        scatter_stream);
    uint8_t *staging_idx_ptr =
        get_kernel_ptr<uint8_t, torch::Tensor>(staging_idx);
    launch_sparse_mla_dsa_scatter_from_staging(config, staging_idx_ptr, true);

    TORCH_CHECK(aclrtRecordEvent(events.scatter_done[slot], scatter_stream) ==
                    ACL_SUCCESS,
                "aclrtRecordEvent scatter failed");
    tracker.state[slot] = SlotTracker::State::SCATTER_INFLIGHT;

    mb_start = mb_end;
    next_slot = 1 - slot;
  }

  for (int i = 0; i < kNumSlots; ++i) {
    if (tracker.state[i] == SlotTracker::State::SCATTER_INFLIGHT) {
      wait_event_or_throw(events.scatter_done[i], event_timeout_ms,
                          "final scatter_done");
      tracker.state[i] = SlotTracker::State::FREE;
    }
  }

  TORCH_CHECK(aclrtSynchronizeStream(h2d_stream) == ACL_SUCCESS,
              "aclrtSynchronizeStream h2d failed");
  TORCH_CHECK(aclrtSynchronizeStream(scatter_stream) == ACL_SUCCESS,
              "aclrtSynchronizeStream scatter failed");
}
