// SPDX-License-Identifier: Apache-2.0
#include "sparse_graph_kernel.h"
#include "kernel_operator.h"
#include "sparse_copy_plan.h"

namespace {
// Scalar GM reads on 910B use a private DCache. A graph replay can put new
// counts and descriptors at exactly the same addresses as the previous replay.
__aicore__ inline void clean_invalidate_line(__gm__ int32_t *ptr) {
  AscendC::GlobalTensor<int32_t> word;
  word.SetGlobalBuffer(ptr, 1);
  AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(word);
}

__aicore__ inline void clean_invalidate_range(__gm__ int32_t *ptr, int32_t first, int32_t last) {
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr + first);
  clean_invalidate_line(ptr + first);
  for (uintptr_t line = (start & ~uintptr_t{63}) + 64;
       line <= reinterpret_cast<uintptr_t>(ptr + last); line += 64) {
    clean_invalidate_line(reinterpret_cast<__gm__ int32_t *>(line));
  }
}

template <typename Index, typename Slot> class SparseGraphCopy {
public:
  template <bool Sharded, bool Planned = false>
  __aicore__ inline void Run(
      GM_ADDR k, GM_ADDR pe, GM_ADDR slots, GM_ADDR selected, GM_ADDR counts,
      GM_ADDR ptrs, GM_ADDR limits, int32_t chunk_size,
      int32_t chunks_per_request, int32_t requests, int32_t row_width,
      int32_t count_stride, int32_t k_bytes, int32_t pe_bytes,
      int64_t destination_slots, int32_t piece = 0, int32_t pieces = 1,
      __gm__ int32_t *plan = nullptr) {
    auto count = reinterpret_cast<__gm__ int32_t *>(counts);
    auto limit = reinterpret_cast<__gm__ int32_t *>(limits);
    auto index = reinterpret_cast<__gm__ Index *>(selected);
    auto slot = reinterpret_cast<__gm__ Slot *>(slots);
    auto table = reinterpret_cast<__gm__ uint64_t *>(ptrs);
    const int32_t cores = AscendC::GetBlockNum();
    const int32_t core = AscendC::GetBlockIdx();
    int32_t offset = 0;
    int32_t length = 0;
    int32_t first_row = 0;
    int32_t last_row = requests - 1;
    if constexpr (Planned) {
      auto record = plan + requests + static_cast<int64_t>(piece) * lmc::kPieceFields;
      clean_invalidate_range(record, 0, lmc::kPieceFields - 1);
      offset = record[0];
      const int32_t stop = record[1];
      first_row = record[2];
      last_row = record[3];
      if (offset < 0 || stop <= offset || stop > requests * row_width ||
          first_row < 0 || last_row >= requests || first_row > last_row) return;
      length = stop - offset;
    } else {
      for (int32_t row = 0; row < requests; ++row) {
        length += lmc::active_sparse_rows(count[static_cast<int64_t>(row) * count_stride], limit[row], row_width);
      }
      if (Sharded) {
        const int32_t per_piece = length / pieces;
        const int32_t extra = length % pieces;
        offset = piece * per_piece + (piece < extra ? piece : extra);
        length = per_piece + (piece < extra ? 1 : 0);
      }
    }
    const int32_t quotient = length / cores;
    const int32_t remainder = length % cores;
    const int32_t begin = offset + core * quotient + (core < remainder ? core : remainder);
    const int32_t end = begin + quotient + (core < remainder ? 1 : 0);
    if (begin == end) {
      return;
    }
    pipe_.InitBuffer(queue_, 2, k_bytes > pe_bytes ? k_bytes : pe_bytes);
    if constexpr (Planned) {
      clean_invalidate_range(plan, first_row > 0 ? first_row - 1 : 0, last_row);
      // Usually the whole piece spans one or two requests. Skip leading empty
      // requests logarithmically even when the batch is extremely skewed.
      int32_t hi = last_row;
      while (first_row < hi) {
        const int32_t mid = first_row + (hi - first_row) / 2;
        if (plan[mid] <= begin) first_row = mid + 1;
        else hi = mid;
      }
    }
    int32_t prefix = Planned && first_row ? plan[first_row - 1] : 0;
    for (int32_t row = first_row; row <= last_row; ++row) {
      int32_t row_end;
      if constexpr (Planned) {
        row_end = plan[row];
        if (prefix < 0 || row_end < prefix || row_end - prefix > row_width) return;
      } else {
        row_end = prefix + lmc::active_sparse_rows(count[static_cast<int64_t>(row) * count_stride], limit[row], row_width);
      }
      const int32_t first = begin > prefix ? begin - prefix : 0;
      const int32_t last = (end < row_end ? end : row_end) - prefix;
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
      prefix = row_end;
      if (prefix >= end) {
        break;
      }
    }
  }

private:
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

extern "C" __global__ __aicore__ void sparse_publish_slice_plan(
    GM_ADDR counts, GM_ADDR dense, GM_ADDR limits, GM_ADDR plan, int32_t requests,
    int32_t stride, int32_t width, int32_t pieces, int32_t cores) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  auto output = reinterpret_cast<__gm__ int32_t *>(dense);
  auto descriptors = reinterpret_cast<__gm__ int32_t *>(plan);
  auto source = reinterpret_cast<__gm__ int32_t *>(counts);
  auto valid = reinterpret_cast<__gm__ int32_t *>(limits);
  if (stride == 1) {
    clean_invalidate_range(source, 0, requests - 1);
  } else {
    for (int32_t r = 0; r < requests; ++r) clean_invalidate_line(source + static_cast<int64_t>(r) * stride);
  }
  clean_invalidate_range(valid, 0, requests - 1);
  lmc::publish_sparse_plan(source, output, valid, descriptors,
                           requests, stride, width, pieces, cores);
  clean_invalidate_range(output, 0, requests - 1);
  clean_invalidate_range(descriptors, 0, requests + lmc::kPieceFields * pieces - 1);
}

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
      copy.Run<false>(GRAPH_VALUES); \
    }
GRAPH_KERNEL(sfa_copy_i32_s32, int32_t, int32_t)
GRAPH_KERNEL(sfa_copy_i32_s64, int32_t, int64_t)
GRAPH_KERNEL(sfa_copy_i64_s32, int64_t, int32_t)
GRAPH_KERNEL(sfa_copy_i64_s64, int64_t, int64_t)

#define PIECE_KERNEL(NAME, INDEX, SLOT) \
    extern "C" __global__ __aicore__ void NAME(GRAPH_ARGS, int32_t piece, int32_t pieces) { \
      SparseGraphCopy<INDEX, SLOT> copy; \
      copy.Run<true>(GRAPH_VALUES, piece, pieces); \
    }
PIECE_KERNEL(sfa_copy_piece_i32_s32, int32_t, int32_t)
PIECE_KERNEL(sfa_copy_piece_i32_s64, int32_t, int64_t)
PIECE_KERNEL(sfa_copy_piece_i64_s32, int64_t, int32_t)
PIECE_KERNEL(sfa_copy_piece_i64_s64, int64_t, int64_t)

#define PLANNED_KERNEL(NAME, INDEX, SLOT) \
    extern "C" __global__ __aicore__ void NAME(GRAPH_ARGS, int32_t piece, GM_ADDR plan) { \
      KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
      SparseGraphCopy<INDEX, SLOT> copy; \
      copy.Run<true, true>(GRAPH_VALUES, piece, 1, reinterpret_cast<__gm__ int32_t *>(plan)); \
    }
PLANNED_KERNEL(sfa_copy_planned_piece_i32_s32, int32_t, int32_t)
PLANNED_KERNEL(sfa_copy_planned_piece_i32_s64, int32_t, int64_t)
PLANNED_KERNEL(sfa_copy_planned_piece_i64_s32, int64_t, int32_t)
PLANNED_KERNEL(sfa_copy_planned_piece_i64_s64, int64_t, int64_t)

namespace lmc {
void launch_sparse_copy_plan(void *stream, uint8_t *counts, uint8_t *dense, uint8_t *limits,
                             uint8_t *plan, int32_t requests, int32_t stride, int32_t width,
                             int32_t pieces, int32_t cores) {
  sparse_publish_slice_plan<<<1, nullptr, stream>>>(counts, dense, limits, plan, requests, stride, width, pieces, cores);
}

void launch_sparse_graph_piece(
    uint32_t cores, void *stream, bool selected_int64, bool slots_int64,
    uint8_t *k, uint8_t *pe, uint8_t *slots, uint8_t *selected,
    uint8_t *counts, uint8_t *ptrs, uint8_t *limits,
    int32_t chunk_size, int32_t chunks_per_request, int32_t requests,
    int32_t row_width, int32_t count_stride, int32_t k_bytes,
    int32_t pe_bytes, int64_t destination_slots, int32_t piece, int32_t pieces, uint8_t *plan) {
  if (plan != nullptr) {
    if (selected_int64 && slots_int64) {
      sfa_copy_planned_piece_i64_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, plan);
    } else if (selected_int64) {
      sfa_copy_planned_piece_i64_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, plan);
    } else if (slots_int64) {
      sfa_copy_planned_piece_i32_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, plan);
    } else {
      sfa_copy_planned_piece_i32_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, plan);
    }
    return;
  }
  if (pieces > 1) {
    if (selected_int64 && slots_int64) {
      sfa_copy_piece_i64_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, pieces);
    } else if (selected_int64) {
      sfa_copy_piece_i64_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, pieces);
    } else if (slots_int64) {
      sfa_copy_piece_i32_s64<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, pieces);
    } else {
      sfa_copy_piece_i32_s32<<<cores, nullptr, stream>>>(GRAPH_VALUES, piece, pieces);
    }
    return;
  }
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
#undef PIECE_KERNEL
#undef PLANNED_KERNEL
#undef GRAPH_VALUES
#undef GRAPH_ARGS
