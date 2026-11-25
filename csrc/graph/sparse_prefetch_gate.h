// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <acl/acl_base.h>
#include <cstdint>

namespace lmc {
aclError write_prefetch_gate(void *word, bool allowed, aclrtStream stream);

// The main stream owns the initially empty, single-consumer Notify. An open
// releases its token. A close first stops new admissions, then takes the
// token back before the next protected operator. A side wait which passed
// the word just before close must still own the token to run its kernel.
aclError set_copy_window(void *word, aclrtNotify token, bool open, aclrtStream stream);

#if LMC_HAS_SFA_STREAM_VALUES
aclError create_copy_notify(aclrtNotify *token);
aclError destroy_copy_notify(aclrtNotify token);
aclError wait_for_copy_window(void *word, aclrtNotify token, aclrtStream stream);
aclError record_copy_notify(aclrtNotify token, aclrtStream stream);
#endif

// One host submission at capture. The side stream observes the word again
// after each preceding piece finishes, including on every device replay.
template <class Launch>
aclError submit_sparse_graph_pieces(aclrtStream stream, void *word, int32_t pieces, Launch launch,
                                    aclrtNotify token = nullptr) {
  for (int32_t piece = 0; piece < pieces; ++piece) {
    if (word != nullptr) {
#if LMC_HAS_SFA_STREAM_VALUES
      aclError rc = wait_for_copy_window(word, token, stream);
      if (rc != ACL_SUCCESS) {
        return rc;
      }
#else
      return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
    }
    launch(piece);
#if LMC_HAS_SFA_STREAM_VALUES
    if (token != nullptr) {
      aclError rc = record_copy_notify(token, stream);
      if (rc != ACL_SUCCESS) {
        return rc;
      }
    }
#endif
  }
  return ACL_SUCCESS;
}
} // namespace lmc
