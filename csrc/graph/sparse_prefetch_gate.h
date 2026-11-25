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

// The main stream owns the initially empty, single-consumer Notify. An open
// releases its token. A close first stops new admissions, then takes the
// token back before the next protected operator. A side wait which passed
// the word just before close must still own the token to run its kernel.
inline aclError set_copy_window(void *word, aclrtNotify token, bool open, aclrtStream stream) {
#if LMC_HAS_SFA_STREAM_VALUES
  aclError rc = write_prefetch_gate(word, open, stream);
  if (rc != ACL_SUCCESS) {
    return rc;
  }
  return open ? aclrtRecordNotify(token, stream) : aclrtWaitAndResetNotify(token, stream, 0);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

// One host submission at capture. The side stream observes the word again
// after each preceding piece finishes, including on every device replay.
template <class Launch>
aclError submit_sparse_graph_pieces(aclrtStream stream, void *word, int32_t pieces, Launch launch,
                                    aclrtNotify token = nullptr) {
  for (int32_t piece = 0; piece < pieces; ++piece) {
    if (word != nullptr) {
#if LMC_HAS_SFA_STREAM_VALUES
      aclError rc = aclrtValueWait(word, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream);
      if (rc != ACL_SUCCESS) {
        return rc;
      }
      if (token != nullptr) {
        rc = aclrtWaitAndResetNotify(token, stream, 0);
        if (rc != ACL_SUCCESS) {
          return rc;
        }
      }
#else
      return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
    }
    launch(piece);
#if LMC_HAS_SFA_STREAM_VALUES
    if (token != nullptr) {
      aclError rc = aclrtRecordNotify(token, stream);
      if (rc != ACL_SUCCESS) {
        return rc;
      }
    }
#endif
  }
  return ACL_SUCCESS;
}
} // namespace lmc
