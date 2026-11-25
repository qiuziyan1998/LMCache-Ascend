// SPDX-License-Identifier: Apache-2.0
// Host test of the production submission loop, built with the CANN ACL headers.
// Build from the repository root:
// c++ -std=c++17 -DLMC_HAS_SFA_STREAM_VALUES=1 -Icsrc -I"$ASCEND_HOME_PATH/include" tests/native/test_sparse_prefetch_gate.cpp -o gate_test
#include "graph/sparse_prefetch_gate.h"
#include <cassert>
#include <vector>

namespace {
int main_queue, side_queue;
uint64_t word;
int fail_wait = -1;
int waits = 0;
std::vector<int> calls;
constexpr int wait_code = -1;
}

extern "C" aclError aclrtValueWait(void *addr, uint64_t value, uint32_t flag, aclrtStream stream) {
  assert(addr == &word && value == 1 && flag == ACL_STREAM_WAIT_VALUE_GEQ && stream == &side_queue);
  calls.push_back(wait_code);
  return waits++ == fail_wait ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

extern "C" aclError aclrtValueWrite(void *addr, uint64_t value, uint32_t flag, aclrtStream stream) {
  assert(addr == &word && value <= 1 && flag == 0 && stream == &main_queue);
  word = value;
  return ACL_SUCCESS;
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
  return 0;
}
