// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sparse_graph_kernel.h"

namespace lmc {
// Layout in device memory: request end offsets, then {begin, end, first, last}
// per piece. Offsets index the concatenation of the live request prefixes.
__aicore__ inline int32_t active_sparse_rows(int32_t count, int32_t limit, int32_t width) {
  return limit <= 0 || count <= 0 ? 0 : (count < width ? count : width);
}

__aicore__ inline void publish_sparse_plan(
    const __gm__ int32_t *counts, __gm__ int32_t *dense_counts,
    const __gm__ int32_t *limits, __gm__ int32_t *plan, int32_t requests,
    int32_t stride, int32_t width, int32_t pieces, int32_t cores) {
  int32_t total = 0;
  for (int32_t row = 0; row < requests; ++row) {
    const int32_t count = counts[static_cast<int64_t>(row) * stride];
    dense_counts[row] = count;  // Preserve the producer's original bridge output.
    total += active_sparse_rows(count, limits[row], width);
    plan[row] = total;
  }
  const int32_t waves = total == 0 ? 0 : 1 + (total - 1) / cores;
  const int32_t base = waves / pieces;
  const int32_t extra = waves % pieces;
  int32_t begin = 0;
  int32_t row = 0;
  for (int32_t piece = 0; piece < pieces; ++piece) {
    const int32_t n = base + (piece < extra);
    const int32_t remaining = total - begin;
    const int32_t end = begin + (n > remaining / cores ? remaining : n * cores);
    // Monotone across all pieces, including empty/padded request lanes.
    while (row + 1 < requests && plan[row] <= begin) ++row;
    const int32_t first = row;
    while (row + 1 < requests && plan[row] < end) ++row;
    auto record = plan + requests + static_cast<int64_t>(piece) * kPieceFields;
    record[0] = begin;
    record[1] = end;
    record[2] = first;
    record[3] = row;
    begin = end;
  }
}
} // namespace lmc
