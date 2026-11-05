// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace lmc {
// All data pointers are fixed at capture. Counts, limits and both plane pointer
// tables are device inputs, including the physical PE offset of a partial tail.
void launch_sparse_graph_transfer(
    uint32_t cores, void *stream, bool selected_int64, bool slots_int64,
    uint8_t *k, uint8_t *pe, uint8_t *slots, uint8_t *selected,
    uint8_t *counts, uint8_t *ptrs, uint8_t *limits,
    int32_t chunk_size, int32_t chunks_per_request, int32_t requests,
    int32_t row_width, int32_t count_stride, int32_t k_bytes,
    int32_t pe_bytes, int64_t destination_slots);
} // namespace lmc
