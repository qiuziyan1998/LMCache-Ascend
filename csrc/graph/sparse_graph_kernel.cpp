// SPDX-License-Identifier: Apache-2.0
#include "sparse_graph_kernel.h"
#include "kernel_operator.h"

namespace {
template <typename Index, typename Slot> class SparseGraphCopy {
public:
  __aicore__ inline void Run(
      GM_ADDR k, GM_ADDR pe, GM_ADDR slots, GM_ADDR selected, GM_ADDR counts,
      GM_ADDR ptrs, GM_ADDR limits, int32_t chunk_size,
      int32_t chunks_per_request, int32_t requests, int32_t row_width,
      int32_t count_stride, int32_t k_bytes, int32_t pe_bytes,
      int64_t destination_slots) {
    auto count = reinterpret_cast<__gm__ int32_t *>(counts);
    auto limit = reinterpret_cast<__gm__ int32_t *>(limits);
    auto index = reinterpret_cast<__gm__ Index *>(selected);
    auto slot = reinterpret_cast<__gm__ Slot *>(slots);
    auto table = reinterpret_cast<__gm__ uint64_t *>(ptrs);
    int32_t total = 0;
    for (int32_t row = 0; row < requests; ++row) {
      total += ActiveCount(count[row * count_stride], limit[row], row_width);
    }
    // Balance actual packed tokens, not the (usually mostly padded) capacity.
    const int32_t cores = AscendC::GetBlockNum();
    const int32_t core = AscendC::GetBlockIdx();
    const int32_t quotient = total / cores;
    const int32_t remainder = total % cores;
    const int32_t begin = core * quotient + (core < remainder ? core : remainder);
    const int32_t end = begin + quotient + (core < remainder ? 1 : 0);
    if (begin == end) {
      return;
    }
    pipe_.InitBuffer(queue_, 2, k_bytes > pe_bytes ? k_bytes : pe_bytes);
    int32_t prefix = 0;
    for (int32_t row = 0; row < requests; ++row) {
      const int32_t length = ActiveCount(count[row * count_stride], limit[row], row_width);
      const int32_t first = begin > prefix ? begin - prefix : 0;
      const int32_t last = end < prefix + length ? end - prefix : length;
      for (int32_t column = first; column < last; ++column) {
        const int32_t packed = row * row_width + column;
        const int64_t token = static_cast<int64_t>(index[packed]);
        const int64_t destination = static_cast<int64_t>(slot[packed]);
        // Bounds are decided from this replay's top-k/limits on device. Check
        // BEFORE division or dereferencing any source allocation.
        if (token < 0 || token >= limit[row] || destination < 0 ||
            destination >= destination_slots) {
          continue;
        }
        const int64_t chunk = token / chunk_size;
        if (chunk >= chunks_per_request) {
          continue;
        }
        const int64_t offset = token % chunk_size;
        const int64_t table_index = static_cast<int64_t>(row) * chunks_per_request + chunk;
        const uint64_t k_source = table[table_index];
        const uint64_t pe_source = table[static_cast<int64_t>(requests) * chunks_per_request + table_index];
        if (k_source == 0 || pe_source == 0) {
          continue;
        }
        Copy(k_source, k, offset, destination, k_bytes);
        Copy(pe_source, pe, offset, destination, pe_bytes);
      }
      prefix += length;
      if (prefix >= end) {
        break;
      }
    }
  }

private:
  __aicore__ inline int32_t ActiveCount(int32_t count, int32_t limit, int32_t width) {
    return limit <= 0 || count <= 0 ? 0 : (count < width ? count : width);
  }

  __aicore__ inline void Copy(uint64_t source, GM_ADDR destination,
                              int64_t token, int64_t slot, int32_t bytes) {
    // Raw-byte DMA preserves FP16/BF16 bits; no arithmetic/conversion kernel.
    AscendC::GlobalTensor<int8_t> src;
    AscendC::GlobalTensor<int8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(source) + token * bytes, bytes);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(destination) + slot * bytes, bytes);
    auto local = queue_.AllocTensor<int8_t>();
    AscendC::DataCopy(local, src, bytes);
    queue_.EnQue(local);
    local = queue_.DeQue<int8_t>();
    AscendC::DataCopy(dst, local, bytes);
    queue_.FreeTensor(local);
  }

  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> queue_;
};
} // namespace

#define GRAPH_ARGS \
    GM_ADDR k, GM_ADDR pe, GM_ADDR slots, GM_ADDR selected, GM_ADDR counts, \
    GM_ADDR ptrs, GM_ADDR limits, int32_t chunk_size, int32_t chunks_per_request, \
    int32_t requests, int32_t row_width, int32_t count_stride, int32_t k_bytes, \
    int32_t pe_bytes, int64_t destination_slots
#define GRAPH_VALUES \
    k, pe, slots, selected, counts, ptrs, limits, chunk_size, chunks_per_request, \
    requests, row_width, count_stride, k_bytes, pe_bytes, destination_slots
#define GRAPH_KERNEL(NAME, INDEX, SLOT) \
    extern "C" __global__ __aicore__ void NAME(GRAPH_ARGS) { \
      SparseGraphCopy<INDEX, SLOT> copy; \
      copy.Run(GRAPH_VALUES); \
    }
GRAPH_KERNEL(sfa_copy_i32_s32, int32_t, int32_t)
GRAPH_KERNEL(sfa_copy_i32_s64, int32_t, int64_t)
GRAPH_KERNEL(sfa_copy_i64_s32, int64_t, int32_t)
GRAPH_KERNEL(sfa_copy_i64_s64, int64_t, int64_t)

namespace lmc {
void launch_sparse_graph_transfer(
    uint32_t cores, void *stream, bool selected_int64, bool slots_int64,
    uint8_t *k, uint8_t *pe, uint8_t *slots, uint8_t *selected,
    uint8_t *counts, uint8_t *ptrs, uint8_t *limits,
    int32_t chunk_size, int32_t chunks_per_request, int32_t requests,
    int32_t row_width, int32_t count_stride, int32_t k_bytes,
    int32_t pe_bytes, int64_t destination_slots) {
  if (selected_int64) {
    if (slots_int64) {
      sfa_copy_i64_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES);
    } else {
      sfa_copy_i64_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES);
    }
  } else if (slots_int64) {
    sfa_copy_i32_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES);
  } else {
    sfa_copy_i32_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES);
  }
}
} // namespace lmc
#undef GRAPH_KERNEL
#undef GRAPH_VALUES
#undef GRAPH_ARGS
