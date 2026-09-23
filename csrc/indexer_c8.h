// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/extension.h>

namespace lmc {
// Retain both stable destinations for the same lifetime as the prepared state.
struct IndexerC8State {
  torch::Tensor keys, scales;
  int64_t cache_slots;
  int64_t key_bytes, slot_factor;
  uint32_t cores;
  IndexerC8State(torch::Tensor keys, c10::optional<torch::Tensor> scales,
                 int64_t slot_factor = 1);
};

struct IndexerC8GroupState {
  std::vector<IndexerC8State> layers;
  torch::Tensor planes;
  int64_t max_key_bytes = 128;
  explicit IndexerC8GroupState(std::vector<IndexerC8State> states);
};

void indexer_c8_transfer_prepared(
    const IndexerC8State &state, const torch::Tensor &packet_ptrs,
    const torch::Tensor &chunk_offsets, const torch::Tensor &chunk_counts,
    const torch::Tensor &slot_mapping, int64_t chunk_capacity, bool from_npu,
    bool fixed_chunks = false);
void indexer_c8_group_transfer_prepared(
    const IndexerC8GroupState &state, const torch::Tensor &packet_ptrs,
    const torch::Tensor &chunk_offsets, const torch::Tensor &chunk_counts,
    const torch::Tensor &slot_mapping, int64_t chunk_capacity, bool from_npu,
    bool fixed_chunks = false);
} // namespace lmc
