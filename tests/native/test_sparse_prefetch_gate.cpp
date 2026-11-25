// SPDX-License-Identifier: Apache-2.0
// Host test of the production submission loop, built with the CANN ACL headers.
// Build from the repository root:
// c++ -std=c++17 -DLMC_HAS_SFA_STREAM_VALUES=1 -Icsrc -I"$ASCEND_HOME_PATH/include" tests/native/test_sparse_prefetch_gate.cpp -o gate_test
#include "graph/sparse_prefetch_gate.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <set>
#include <tuple>
#include <vector>

namespace {
int main_queue, side_queue;
uint64_t word;
int fail_wait = -1;
int waits = 0;
std::vector<int> calls;
constexpr int wait_code = -1;
enum Op { Allow, Block, Check, Take, Give, CopyBegin, CopyEnd, Compute, Protected, Join };
struct Task { Op op; int n = 0; };
std::vector<Task> front, back;
int notify;
std::vector<Task> &queue(aclrtStream stream) {
  assert(stream == &main_queue || stream == &side_queue);
  return stream == &main_queue ? front : back;
}
}

extern "C" aclError aclrtValueWait(void *addr, uint64_t value, uint32_t flag, aclrtStream stream) {
  assert(addr == &word && value == 1 && flag == ACL_STREAM_WAIT_VALUE_GEQ && stream == &side_queue);
  calls.push_back(wait_code);
  back.push_back({Check});
  return waits++ == fail_wait ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

extern "C" aclError aclrtValueWrite(void *addr, uint64_t value, uint32_t flag, aclrtStream stream) {
  assert(addr == &word && value <= 1 && flag == 0 && stream == &main_queue);
  word = value;
  front.push_back({value ? Allow : Block});
  return ACL_SUCCESS;
}

extern "C" aclError aclrtRecordNotify(aclrtNotify n, aclrtStream stream) {
  assert(n == &notify);
  queue(stream).push_back({Give});
  return ACL_SUCCESS;
}

extern "C" aclError aclrtWaitAndResetNotify(aclrtNotify n, aclrtStream stream, uint32_t timeout) {
  assert(n == &notify && timeout == 0);
  queue(stream).push_back({Take});
  return ACL_SUCCESS;
}

// Explore every distinct ordering of the *production* submission commands.
// An unfinished kernel keeps the token across CopyBegin/CopyEnd. A foreground
// close may interleave even between Check and Take on the background stream.
enum Owner { Foreground, Background, Free };
struct State {
  size_t f = 0, b = 0;
  Owner owner = Foreground;
  bool open = false;
  int copied = 0, at_close = 0, vup = 0, router = 0;
  auto key() const { return std::make_tuple(f, b, owner, open, copied, at_close, vup, router); }
};
std::set<decltype(State{}.key())> seen;
std::set<std::tuple<int, int, int>> divisions;
constexpr int pieces = 8;

bool advance(State &s, Owner who) {
  auto &pc = who == Foreground ? s.f : s.b;
  const auto &tasks = who == Foreground ? front : back;
  if (pc == tasks.size()) return false;
  const auto task = tasks[pc];
  switch (task.op) {
  case Allow: s.open = true; break;
  case Block: s.open = false; s.at_close = s.copied; break;
  case Check: if (!s.open) return false; break;
  case Take:
    if (s.owner != Free) return false;
    s.owner = who;
    if (who == Foreground) assert(s.copied - s.at_close <= 1);
    break;
  case Give: assert(s.owner == who); s.owner = Free; break;
  case CopyBegin: assert(s.owner == Background && task.n == s.copied); break;
  case CopyEnd: assert(s.owner == Background && task.n == s.copied); ++s.copied; break;
  case Protected:
    assert(s.owner == Foreground && !s.open);
    if (task.n == 1) s.vup = s.copied;
    if (task.n == 2) s.router = s.copied - s.vup;
    break;
  case Compute: break;
  case Join: if (s.b != back.size()) return false; break;
  }
  ++pc;
  return true;
}

void explore(State state) {
  if (!seen.insert(state.key()).second) return;
  if (state.f == front.size() && state.b == back.size()) {
    assert(state.owner == Foreground && !state.open && state.copied == pieces);
    divisions.emplace(state.vup, state.router, pieces - state.vup - state.router);
    return;
  }
  bool progress = false;
  for (Owner who : {Foreground, Background}) {
    State next = state;
    if (advance(next, who)) {
      progress = true;
      explore(next);
    }
  }
  assert(progress);
}

void check_dynamic_windows() {
  front.clear();
  back.clear();
  auto launch = [](int i) { back.push_back({CopyBegin, i}); back.push_back({CopyEnd, i}); };
  assert(lmc::submit_sparse_graph_pieces(&side_queue, &word, pieces, launch, &notify) == ACL_SUCCESS);
  front.push_back({Protected});
  for (int window = 1; window <= 3; ++window) {
    assert(lmc::set_copy_window(&word, &notify, true, &main_queue) == ACL_SUCCESS);
    front.push_back({window == 3 ? Join : Compute});
    assert(lmc::set_copy_window(&word, &notify, false, &main_queue) == ACL_SUCCESS);
    front.push_back({Protected, window});
  }
  explore({});
  // Each integral distribution over the two Cube windows and final drain is reachable.
  for (int vup = 0; vup <= pieces; ++vup)
    for (int router = 0; router <= pieces - vup; ++router)
      assert(divisions.count({vup, router, pieces - vup - router}) == 1);
  std::printf("%zu reachable states, %zu distributions of %d pieces, exclusive close and complete drain verified\n",
              seen.size(), divisions.size(), pieces);
}

int main() {
  const auto launch = [](int32_t piece) { calls.push_back(piece); };
  assert(lmc::write_prefetch_gate(&word, true, &main_queue) == ACL_SUCCESS && word == 1);
  assert(lmc::write_prefetch_gate(&word, false, &main_queue) == ACL_SUCCESS && word == 0);
  assert(lmc::submit_sparse_graph_pieces(&side_queue, &word, 4, launch) == ACL_SUCCESS);
  assert((calls == std::vector<int>{wait_code, 0, wait_code, 1, wait_code, 2, wait_code, 3}));
  calls.clear();
  fail_wait = waits + 1;
  assert(lmc::submit_sparse_graph_pieces(&side_queue, &word, 8, launch) == ACL_ERROR_INVALID_PARAM);
  assert((calls == std::vector<int>{wait_code, 0, wait_code}));
  calls.clear();
  assert(lmc::submit_sparse_graph_pieces(&main_queue, nullptr, 1, launch) == ACL_SUCCESS);
  assert(lmc::submit_sparse_graph_pieces(&side_queue, nullptr, 3, launch) == ACL_SUCCESS);
  assert((calls == std::vector<int>{0, 0, 1, 2}));
  check_dynamic_windows();
  return 0;
}
