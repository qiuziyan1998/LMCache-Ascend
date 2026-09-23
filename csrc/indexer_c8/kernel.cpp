// SPDX-License-Identifier: Apache-2.0
#include "kernel.h"
#include "kernel_operator.h"

namespace {
// One tile copies at most 128 tokens. Contiguous slots share one DMA per
// plane; short scale tails use DataCopyPad so adjacent slots are untouched.
constexpr int64_t kTileTokens = 128;
constexpr uint32_t kKeyBytes = 128;
constexpr uint32_t kScaleBytes = 2;

class IntVector {
 public:
  __aicore__ inline void Init(GM_ADDR address, bool wide) {
    wide_ = wide;
    i32_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
    i64_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
  }
  __aicore__ inline int64_t GetValue(int64_t index) {
    return wide_ ? i64_.GetValue(index) : i32_.GetValue(index);
  }
 private:
  bool wide_;
  AscendC::GlobalTensor<int32_t> i32_;
  AscendC::GlobalTensor<int64_t> i64_;
};

class IndexerC8Copy {
 public:
  __aicore__ inline void Run(
      GM_ADDR keys, GM_ADDR scales, GM_ADDR packets, GM_ADDR offsets,
      GM_ADDR counts, GM_ADDR slots, int64_t chunks, int64_t capacity,
      int64_t slot_count, int64_t cache_slots, bool from_npu,
      bool metadata_int64, bool slots_int64, bool fixed_chunks, int64_t layers) {
    AscendC::GlobalTensor<uint64_t> pointers;
    AscendC::GlobalTensor<uint64_t> key_planes, scale_planes;
    if (layers > 1) {
      key_planes.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(keys));
      scale_planes.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scales));
    }
    IntVector starts, lengths, mapping;
    pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(packets));
    starts.Init(offsets, metadata_int64);
    lengths.Init(counts, metadata_int64);
    mapping.Init(slots, slots_int64);
    pipe_.InitBuffer(queue_, 2, kTileTokens * kKeyBytes);
    const int64_t tiles = (capacity + kTileTokens - 1) / kTileTokens;
    for (int64_t work = AscendC::GetBlockIdx(); work < layers * chunks * tiles;
         work += AscendC::GetBlockNum()) {
      const int64_t layer = work / (chunks * tiles);
      const int64_t chunk = (work / tiles) % chunks;
      GM_ADDR key_base = layers > 1 ? reinterpret_cast<GM_ADDR>(key_planes.GetValue(layer)) : keys;
      GM_ADDR scale_base = layers > 1 ? reinterpret_cast<GM_ADDR>(scale_planes.GetValue(layer)) : scales;
      const int64_t start = fixed_chunks ? chunk * capacity : starts.GetValue(chunk);
      const int64_t remaining = slot_count - start;
      const int64_t count = fixed_chunks ? (remaining < capacity ? remaining : capacity)
                                        : lengths.GetValue(chunk);
      const uint64_t packet = pointers.GetValue(layer * chunks + chunk);
      // Host planners validate ranges before publishing metadata. Keep bounds
      // checks here too; negative slots denote rows excluded by the caller.
      if (!packet || count <= 0 || count > capacity || start < 0 ||
          start > slot_count || count > slot_count - start) {
        continue;
      }
      int64_t token = (work % tiles) * kTileTokens;
      const int64_t end = token + kTileTokens < count ? token + kTileTokens : count;
      while (token < end) {
        const int64_t slot = mapping.GetValue(start + token);
        if (slot < 0 || slot >= cache_slots) {
          ++token;
          continue;
        }
        int64_t run = 1;
        while (token + run < end && slot + run < cache_slots &&
               mapping.GetValue(start + token + run) == slot + run) {
          ++run;
        }
        Copy(reinterpret_cast<GM_ADDR>(packet) + token * kKeyBytes,
             key_base + slot * kKeyBytes, run * kKeyBytes, from_npu);
        Copy(reinterpret_cast<GM_ADDR>(packet) + count * kKeyBytes + token * kScaleBytes,
             scale_base + slot * kScaleBytes, run * kScaleBytes, from_npu);
        token += run;
      }
    }
  }

 private:
  __aicore__ inline void Copy(GM_ADDR packet, GM_ADDR cache, uint32_t bytes,
                             bool from_npu) {
    AscendC::GlobalTensor<uint8_t> src, dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(from_npu ? cache : packet));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(from_npu ? packet : cache));
    const AscendC::DataCopyExtParams copy{1, bytes, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    auto local = queue_.AllocTensor<uint8_t>();
    AscendC::DataCopyPad(local, src, copy, pad);
    queue_.EnQue(local);
    local = queue_.DeQue<uint8_t>();
    AscendC::DataCopyPad(dst, local, copy);
    queue_.FreeTensor(local);
  }
  AscendC::TPipe pipe_;
  // Pure GM -> UB -> GM transport: synchronize MTE2 with MTE3 and protect
  // buffer reuse after the outgoing DMA, without a vector-compute stage.
  AscendC::TQueBind<AscendC::TPosition::VECIN,
                   AscendC::TPosition::VECOUT, 2> queue_;
};
} // namespace

extern "C" __global__ __aicore__ void indexer_c8_copy_kernel(
    GM_ADDR keys, GM_ADDR scales, GM_ADDR packets, GM_ADDR offsets,
    GM_ADDR counts, GM_ADDR slots, int64_t chunks, int64_t capacity,
    int64_t slot_count, int64_t cache_slots, bool from_npu,
    bool metadata_int64, bool slots_int64, bool fixed_chunks, int64_t layers) {
  IndexerC8Copy op;
  op.Run(keys, scales, packets, offsets, counts, slots, chunks, capacity,
         slot_count, cache_slots, from_npu, metadata_int64, slots_int64, fixed_chunks, layers);
}

namespace lmc {
void launch_indexer_c8_transfer(
    uint32_t cores, void *stream, void *keys, void *scales, void *packets,
    void *offsets, void *counts, void *slots, int64_t chunks,
    int64_t capacity, int64_t slot_count, int64_t cache_slots, bool from_npu,
    bool metadata_int64, bool slots_int64, bool fixed_chunks, int64_t layers) {
  indexer_c8_copy_kernel<<<cores, nullptr, stream>>>(
      static_cast<GM_ADDR>(keys), static_cast<GM_ADDR>(scales),
      static_cast<GM_ADDR>(packets), static_cast<GM_ADDR>(offsets),
      static_cast<GM_ADDR>(counts), static_cast<GM_ADDR>(slots), chunks,
      capacity, slot_count, cache_slots, from_npu, metadata_int64, slots_int64, fixed_chunks, layers);
}
} // namespace lmc
