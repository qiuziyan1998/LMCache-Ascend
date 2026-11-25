// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <acl/acl_rt.h>
#include <cstdint>

namespace lmc {
inline aclError write_prefetch_gate(void *word, bool allowed, aclrtStream stream) {
#if LMC_HAS_SFA_STREAM_VALUES
  return aclrtValueWrite(word, allowed ? 1 : 0, 0, stream);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

// One host submission at capture. The side stream observes the word again
// after each preceding piece finishes, including on every device replay.
template <class Launch>
aclError submit_sparse_graph_pieces(aclrtStream stream, void *word, int32_t pieces, Launch launch) {
  for (int32_t piece = 0; piece < pieces; ++piece) {
    if (word != nullptr) {
#if LMC_HAS_SFA_STREAM_VALUES
      aclError rc = aclrtValueWait(word, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream);
      if (rc != ACL_SUCCESS) {
        return rc;
      }
#else
      return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
    }
    launch(piece);
  }
  return ACL_SUCCESS;
}
} // namespace lmc
