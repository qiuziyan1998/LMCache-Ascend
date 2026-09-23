// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace lmc {
void launch_indexer_c8_transfer(
    uint32_t cores, void *stream, void *keys, void *scales, void *packets,
    void *offsets, void *counts, void *slots, int64_t chunks,
    int64_t chunk_capacity, int64_t slot_count, int64_t cache_slots,
    bool from_npu, bool metadata_int64, bool slots_int64, bool fixed_chunks,
    int64_t layers = 1);
} // namespace lmc
