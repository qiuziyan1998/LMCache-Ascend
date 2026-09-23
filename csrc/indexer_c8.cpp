// SPDX-License-Identifier: Apache-2.0
#include "indexer_c8.h"
#include "indexer_c8/kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace lmc {
IndexerC8State::IndexerC8State(torch::Tensor key, torch::Tensor scale)
    : keys(std::move(key)), scales(std::move(scale)) {
  TORCH_CHECK(keys.device().type() == c10::DeviceType::PrivateUse1 &&
                  scales.device() == keys.device(),
              "C8 destinations must share one NPU device");
  TORCH_CHECK(keys.scalar_type() == at::kChar &&
                  scales.scalar_type() == at::kHalf,
              "C8 destinations require int8 keys and float16 scales");
  TORCH_CHECK(keys.dim() == 4 && scales.dim() == 4 &&
                  keys.size(0) > 0 && keys.size(1) == 128 &&
                  keys.size(2) == 1 && keys.size(3) == 128 &&
                  scales.size(0) == keys.size(0) && scales.size(1) == 128 &&
                  scales.size(2) == 1 && scales.size(3) == 1 &&
                  keys.is_contiguous() && scales.is_contiguous(),
              "C8 destinations require paired contiguous PA_BSND storage");
  cache_slots = keys.numel() / 128;
  const c10::OptionalDeviceGuard guard(keys.device());
  auto platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
  cores = std::max(1U, platform->GetCoreNumAiv());
}

IndexerC8GroupState::IndexerC8GroupState(std::vector<IndexerC8State> states)
    : layers(std::move(states)) {
  TORCH_CHECK(!layers.empty(), "C8 group must contain at least one layer");
  std::vector<int64_t> keys, scales;
  for (const auto &state : layers) {
    TORCH_CHECK(state.keys.device() == layers[0].keys.device() &&
                    state.cache_slots == layers[0].cache_slots,
                "C8 group layers must share device and slot capacity");
    keys.push_back(reinterpret_cast<int64_t>(state.keys.data_ptr()));
    scales.push_back(reinterpret_cast<int64_t>(state.scales.data_ptr()));
  }
  if (layers.size() > 1) {
    const c10::OptionalDeviceGuard guard(layers[0].keys.device());
    key_ptrs = torch::tensor(keys, at::kLong).to(layers[0].keys.device());
    scale_ptrs = torch::tensor(scales, at::kLong).to(layers[0].keys.device());
  }
}

static uint32_t validate_transfer(
    const IndexerC8State &state, const torch::Tensor &packet_ptrs,
    const torch::Tensor &chunk_offsets, const torch::Tensor &chunk_counts,
    const torch::Tensor &slot_mapping, int64_t chunk_capacity, bool from_npu,
    bool fixed_chunks) {
  for (const auto *tensor : {&packet_ptrs, &chunk_offsets, &chunk_counts,
                            &slot_mapping}) {
    TORCH_CHECK(tensor->device() == state.keys.device() &&
                    (tensor->scalar_type() == at::kLong || tensor->scalar_type() == at::kInt) && tensor->dim() == 1 &&
                    tensor->is_contiguous(),
                "C8 copy metadata must be contiguous int32/int64 vectors on the cache device");
  }
  TORCH_CHECK(packet_ptrs.scalar_type() == at::kLong &&
                  chunk_offsets.scalar_type() == chunk_counts.scalar_type(),
              "C8 pointers must be int64; offsets and counts must share a dtype");
  const int64_t chunks = packet_ptrs.numel();
  TORCH_CHECK(fixed_chunks || (chunk_offsets.numel() == chunks && chunk_counts.numel() == chunks),
              "C8 pointer, offset and count vectors must have equal lengths");
  TORCH_CHECK(chunk_capacity > 0 &&
                  chunk_capacity <= std::numeric_limits<int32_t>::max(),
              "C8 chunk capacity is out of range");
  if (!chunks) return 0;
  TORCH_CHECK(!fixed_chunks ||
                  (slot_mapping.numel() + chunk_capacity - 1) / chunk_capacity == chunks,
              "C8 fixed chunks do not cover the slot mapping");
  const int64_t tiles = (chunk_capacity + 127) / 128;
  TORCH_CHECK(chunks <= std::numeric_limits<int64_t>::max() / tiles,
              "C8 transfer grid is too large");
  const auto cores = static_cast<uint32_t>(std::min<int64_t>(state.cores, chunks * tiles));
  return cores;
}

void indexer_c8_transfer_prepared(
    const IndexerC8State &state, const torch::Tensor &packet_ptrs,
    const torch::Tensor &chunk_offsets, const torch::Tensor &chunk_counts,
    const torch::Tensor &slot_mapping, int64_t chunk_capacity, bool from_npu,
    bool fixed_chunks) {
  const auto cores = validate_transfer(state, packet_ptrs, chunk_offsets,
      chunk_counts, slot_mapping, chunk_capacity, from_npu, fixed_chunks);
  if (!cores) return;
  const auto chunks = packet_ptrs.numel();
  const c10::OptionalDeviceGuard guard(state.keys.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("indexer_c8_transfer_prepared");
  // OpCommand may defer host submission. Retain all tensor handles until the
  // launch; the connector retains packets/metadata until its completion event.
  cmd.SetCustomHandler([state, packet_ptrs, chunk_offsets, chunk_counts,
                        slot_mapping, chunk_capacity, from_npu, chunks,
                        cores, stream, fixed_chunks]() -> int {
    launch_indexer_c8_transfer(
        cores, stream, state.keys.data_ptr(), state.scales.data_ptr(),
        packet_ptrs.data_ptr(), chunk_offsets.data_ptr(), chunk_counts.data_ptr(),
        slot_mapping.data_ptr(), chunks, chunk_capacity, slot_mapping.numel(),
        state.cache_slots, from_npu, chunk_offsets.scalar_type() == at::kLong,
        slot_mapping.scalar_type() == at::kLong, fixed_chunks);
    return 0;
  });
  cmd.Run();
}

void indexer_c8_group_transfer_prepared(
    const IndexerC8GroupState &state, const torch::Tensor &packet_ptrs,
    const torch::Tensor &chunk_offsets, const torch::Tensor &chunk_counts,
    const torch::Tensor &slot_mapping, int64_t chunk_capacity, bool from_npu,
    bool fixed_chunks) {
  const int64_t layers = state.layers.size();
  TORCH_CHECK(packet_ptrs.dim() == 2 && packet_ptrs.size(0) == layers &&
                  packet_ptrs.is_contiguous(),
              "C8 group pointers must have shape [layers, chunks]");
  const auto &first = state.layers[0];
  const auto cores = validate_transfer(first, packet_ptrs[0], chunk_offsets,
      chunk_counts, slot_mapping, chunk_capacity, from_npu, fixed_chunks);
  if (!cores) return;
  const auto chunks = packet_ptrs.size(1);
  const int64_t tiles = (chunk_capacity + 127) / 128;
  TORCH_CHECK(chunks * tiles <= std::numeric_limits<int64_t>::max() / layers,
              "C8 group transfer grid is too large");
  const auto group_cores = static_cast<uint32_t>(std::min<int64_t>(first.cores, layers * chunks * tiles));
  const c10::OptionalDeviceGuard guard(first.keys.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("indexer_c8_group_transfer_prepared");
  cmd.SetCustomHandler([state, packet_ptrs, chunk_offsets, chunk_counts,
                        slot_mapping, chunk_capacity, from_npu, fixed_chunks,
                        layers, chunks, group_cores, stream]() -> int {
    const auto &first = state.layers[0];
    launch_indexer_c8_transfer(
        group_cores, stream,
        layers > 1 ? state.key_ptrs.data_ptr() : first.keys.data_ptr(),
        layers > 1 ? state.scale_ptrs.data_ptr() : first.scales.data_ptr(),
        packet_ptrs.data_ptr(), chunk_offsets.data_ptr(), chunk_counts.data_ptr(),
        slot_mapping.data_ptr(), chunks, chunk_capacity, slot_mapping.numel(),
        first.cache_slots, from_npu, chunk_offsets.scalar_type() == at::kLong,
        slot_mapping.scalar_type() == at::kLong, fixed_chunks, layers);
    return 0;
  });
  cmd.Run();
}
} // namespace lmc
