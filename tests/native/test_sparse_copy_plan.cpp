// SPDX-License-Identifier: Apache-2.0
// c++ -std=c++17 -Icsrc tests/native/test_sparse_copy_plan.cpp -o test_sparse_copy_plan
#define __gm__
#define __aicore__
#include "graph/sparse_copy_plan.h"
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

static void check(const std::vector<int32_t> &counts, const std::vector<int32_t> &limits,
                  int width, int pieces, int cores, int stride) {
  const int requests = static_cast<int>(counts.size());
  std::vector<int32_t> input(requests * stride, -888), dense(requests, -777);
  std::vector<int32_t> table(requests + lmc::kPieceFields * pieces, -111);
  std::vector<int> lanes;
  for (int r = 0; r < requests; ++r) {
    input[r * stride] = counts[r];
    const int active = limits[r] > 0 ? std::clamp(counts[r], 0, width) : 0;
    lanes.insert(lanes.end(), active, r);
  }
  lmc::publish_sparse_plan(input.data(), dense.data(), limits.data(), table.data(),
                           requests, stride, width, pieces, cores);
  assert(dense == counts);
  for (int r = 0; r < requests; ++r) {
    assert(table[r] == std::upper_bound(lanes.begin(), lanes.end(), r) - lanes.begin());
  }
  const int total = static_cast<int>(lanes.size());
  int assigned = 0, min_waves = total + 1, max_waves = 0;
  for (int p = 0; p < pieces; ++p) {
    const auto *record = &table[requests + lmc::kPieceFields * p];
    const int start = record[0], end = record[1], rows = end - start;
    assert(start == assigned && end >= start && end <= total);
    assert(end == total || end % cores == 0);
    if (rows) {
      assert(record[2] == lanes[start] && record[3] == lanes[end - 1]);
    }
    const int waves = (rows + cores - 1) / cores;
    min_waves = std::min(min_waves, waves);
    max_waves = std::max(max_waves, waves);
    assigned = end;
  }
  assert(assigned == total && max_waves - min_waves <= 1);
}

int main() {
  std::mt19937 rng(47);
  constexpr int width = 4096;
  int cases = 0;
  for (int requests : {1, 3, 16, 32}) {
    std::vector<int32_t> counts(requests), limits(requests);
    for (int fixture = 0; fixture < 22; ++fixture) {
      for (int r = 0; r < requests; ++r) {
        limits[r] = fixture > 5 && rng() % 5 == 0 ? 0 : 8192;
        switch (fixture) {
          case 0: counts[r] = 0; break;
          case 1: counts[r] = 1; break;
          case 2: counts[r] = 410; break;
          case 3: counts[r] = width; break;
          case 4: counts[r] = r == requests - 1 ? width : 0; break;
          case 5: counts[r] = r == 0 ? width + 7 : -3; break;
          default: counts[r] = static_cast<int>(rng() % (width + 8)) - 3;
        }
      }
      for (int cores : {1, 12, 20, 24, 40}) {
        for (int pieces : {2, 7, 16, 32}) {
          for (int stride : {1, 16}) {
            check(counts, limits, width, pieces, cores, stride);
            ++cases;
          }
        }
      }
    }
  }
  std::printf("%d shared slice plan cases passed\n", cases);
}
