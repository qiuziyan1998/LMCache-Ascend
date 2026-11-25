// SPDX-License-Identifier: Apache-2.0
#include "sparse_prefetch_gate.h"
#include <acl/acl_rt.h>

// The torch_npu wheel also ships ACL headers with the SDK's include guards.
// Compile the runtime calls in this translation unit using the toolkit headers.
namespace lmc {
aclError write_prefetch_gate(void *word, bool allowed, aclrtStream stream) {
#if LMC_HAS_SFA_STREAM_VALUES
  return aclrtValueWrite(word, allowed ? 1 : 0, 0, stream);
#else
  return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
}

aclError set_copy_window(void *word, aclrtNotify token, bool open, aclrtStream stream) {
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

#if LMC_HAS_SFA_STREAM_VALUES
aclError create_copy_notify(aclrtNotify *token) {
  return aclrtCreateNotify(token, ACL_NOTIFY_DEFAULT);
}

aclError destroy_copy_notify(aclrtNotify token) {
  return aclrtDestroyNotify(token);
}

aclError wait_for_copy_window(void *word, aclrtNotify token, aclrtStream stream) {
  aclError rc = aclrtValueWait(word, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream);
  if (rc == ACL_SUCCESS && token != nullptr) {
    rc = aclrtWaitAndResetNotify(token, stream, 0);
  }
  return rc;
}

aclError record_copy_notify(aclrtNotify token, aclrtStream stream) {
  return aclrtRecordNotify(token, stream);
}
#endif
} // namespace lmc
