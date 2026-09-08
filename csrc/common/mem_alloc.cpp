#include "mem_alloc.h"
#include "managed_mem.h"
#include "slow_path_diagnostics.h"
#include <acl/acl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib> // for std::getenv
#include <cstring> // for strerror
#include <errno.h>
#include <fcntl.h>
#include <limits>
#include <numaif.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags) {
  void *ptr = nullptr;
  // no flags
  aclError err = aclrtMallocHost(&ptr, size);
  if (err != ACL_SUCCESS) {
    throw std::runtime_error("aclrtMallocHost failed: " + std::to_string(err));
  }

  const char *socVersion = aclrtGetSocName();

  // nullptr means that the chip version failed to be obtained. We cannot be
  // sure about the version of the device. Unless we are sure that we deal with
  // a 310 device, we try to register.
  if (socVersion == nullptr ||
      std::string(socVersion).find("310") == std::string::npos) {
    // not 310p
    auto devPtr = register_ptr(ptr, size);
    if (devPtr == nullptr) {
      free_pinned_ptr(reinterpret_cast<uintptr_t>(ptr));
      throw std::runtime_error("register ptr failed");
    }
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  unregister_ptr(reinterpret_cast<void *>(ptr));
  aclError err = aclrtFreeHost(reinterpret_cast<void *>(ptr));
  if (err != ACL_SUCCESS) {
    throw std::runtime_error("aclrtFreeHost failed: " + std::to_string(err));
  }
}

/*
 * This function is potentially slow for the mbind
 */
uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node) {
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  int err = mbind(ptr, size, MPOL_BIND, &mask, maxnode,
                  MPOL_MF_MOVE | MPOL_MF_STRICT);
  if (err != 0) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(errno));
  }

  memset(ptr, 0, size);

  // as before we need to actually save the dev ptr for later reuse,
  // because acl APIs do not allow retrieving register dev ptr
  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    aclError err = aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
      throw std::runtime_error(
          std::string("unable to register Pinned Numa HostPtr: ") +
          std::to_string(err));
    } else {
      throw std::runtime_error(
          std::string("unable to register Pinned Numa HostPtr."));
    }
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t p, std::size_t size) {
  void *ptr = reinterpret_cast<void *>(p);

  auto unRegErr = unregister_ptr(ptr);
  auto unMapErr = munmap(ptr, size);
  if (unRegErr) {
    throw std::runtime_error("unregister_ptr failed: " +
                             std::to_string(unRegErr));
  }
  if (unMapErr) {
    throw std::runtime_error("munmap failed: " + std::to_string(unMapErr));
  }
}

static void first_touch(void *p, size_t size) {
#ifdef MADV_POPULATE_WRITE
  // Populate the page tables in the kernel instead of taking one user-space
  // write fault per page. This runs only on a newly reserved, zero-filled slab,
  // while the creator's NUMA policy is still active. Older kernels retain the
  // existing path; resource failures must still abort initialization.
  if (madvise(p, size, MADV_POPULATE_WRITE) == 0) {
    return;
  }
  const int err = errno;
  if (err != EINVAL && err != ENOSYS && err != EOPNOTSUPP) {
    throw std::runtime_error(std::string("shared slab prefault failed: ") +
                             strerror(err));
  }
#endif
  const long ps = sysconf(_SC_PAGESIZE);
  if (ps <= 0) {
    throw std::runtime_error("unable to determine shared slab page size");
  }
  for (size_t off = 0; off < size; off += ps) {
    volatile char *c = reinterpret_cast<volatile char *>(p) + off;
    *c = 0;
  }
}

namespace {

class ShmStartupPhase {
 public:
  ShmStartupPhase(const char *phase, std::size_t size)
      : enabled_(lmc::slow_diag::enabled()), phase_(phase), size_(size),
        started_(enabled_ ? lmc::slow_diag::wall_ns() : 0),
        cpu_started_(enabled_ ? lmc::slow_diag::thread_cpu_ns() : 0) {
    if (enabled_) {
      std::fprintf(stderr,
                   "[LMCACHE_COLD_PERF_NATIVE] {\"schema\":1,\"event\":"
                   "\"shared_slab_phase_start\",\"phase\":\"%s\",\"pid\":%ld,"
                   "\"tid\":%ld,\"bytes\":%zu,\"monotonic_ns\":%lld}\n",
                   phase_, static_cast<long>(getpid()),
                   lmc::slow_diag::thread_id(), size_,
                   static_cast<long long>(started_));
    }
  }

  void complete() const {
    if (enabled_) {
      const auto completed = lmc::slow_diag::wall_ns();
      const auto cpu_completed = lmc::slow_diag::thread_cpu_ns();
      std::fprintf(stderr,
                   "[LMCACHE_COLD_PERF_NATIVE] {\"schema\":1,\"event\":"
                   "\"shared_slab_phase_complete\",\"phase\":\"%s\",\"pid\":%ld,"
                   "\"tid\":%ld,\"bytes\":%zu,\"elapsed_ms\":%.3f,"
                   "\"thread_cpu_ms\":%.3f}\n",
                   phase_, static_cast<long>(getpid()),
                   lmc::slow_diag::thread_id(), size_,
                   lmc::slow_diag::elapsed_ms(started_, completed),
                   lmc::slow_diag::elapsed_ms(cpu_started_, cpu_completed));
    }
  }

 private:
  const bool enabled_;
  const char *phase_;
  const std::size_t size_;
  const int64_t started_;
  const int64_t cpu_started_;
};

class ScopedInterleavePolicy {
 public:
  explicit ScopedInterleavePolicy(const std::vector<int> &nodes) {
    if (nodes.empty()) {
      return;
    }

    int mode;
    if (get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0) {
      throw std::runtime_error(std::string("get_mempolicy failed: ") +
                               strerror(errno));
    }
    if (mode != MPOL_DEFAULT) {
      throw std::runtime_error(
          "shared CPU cache NUMA interleave requires the default thread "
          "memory policy; remove the external numactl policy");
    }

    int max_node = *std::max_element(nodes.begin(), nodes.end());
    if (max_node < 0) {
      throw std::runtime_error("NUMA interleave nodes must be non-negative");
    }
    constexpr std::size_t bits_per_word = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(max_node / bits_per_word + 1);
    for (int node : nodes) {
      if (node < 0) {
        throw std::runtime_error("NUMA interleave nodes must be non-negative");
      }
      mask[node / bits_per_word] |= 1UL << (node % bits_per_word);
    }
    if (set_mempolicy(MPOL_INTERLEAVE, mask.data(), max_node + 1) != 0) {
      throw std::runtime_error(std::string("set_mempolicy failed: ") +
                               strerror(errno));
    }
    active_ = true;
  }

  ~ScopedInterleavePolicy() {
    if (active_) {
      set_mempolicy(MPOL_DEFAULT, nullptr, 0);
    }
  }

  void restore() {
    if (!active_) {
      return;
    }
    if (set_mempolicy(MPOL_DEFAULT, nullptr, 0) != 0) {
      throw std::runtime_error(std::string("restore mempolicy failed: ") +
                               strerror(errno));
    }
    active_ = false;
  }

 private:
  bool active_ = false;
};

}  // namespace

static void reserve_shm_storage(int fd, std::size_t size,
                                const std::string &shm_name) {
  if (size > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    throw std::runtime_error("shm size exceeds off_t max for " + shm_name);
  }
  int err = posix_fallocate(fd, 0, static_cast<off_t>(size));
  if (err != 0) {
    throw std::runtime_error(
        std::string("posix_fallocate failed for ") + shm_name +
        " before first_touch (not enough /dev/shm space or quota for shared "
        "CPU cache slab; reduce max_local_cpu_size/shared_cpu_cache_size_gb "
        "or increase /dev/shm): " +
        strerror(err));
  }
}

uintptr_t alloc_shm_pinned_ptr(
    std::size_t size, const std::string &shm_name,
    const std::vector<int> &interleave_nodes) {
  if (size == 0) {
    throw std::runtime_error("alloc_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  }
  if (shm_name.empty()) {
    throw std::runtime_error("alloc_shm_pinned_ptr requires a shm_name");
  }

  ScopedInterleavePolicy numa_policy(interleave_nodes);
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) {
    throw std::runtime_error(
        std::string("shm_open create failed for ") + shm_name +
        " (shared CPU cache segment already exists or cannot be created; "
        "this usually means a live name collision or stale segment from an "
        "unclean shutdown, so choose a unique shared_cpu_cache_name or unlink "
        "the stale segment before restart): " + strerror(errno));
  }

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed for ") + shm_name +
                             ": " + strerror(err));
  }

  try {
    ShmStartupPhase phase("reserve", size);
    reserve_shm_storage(fd, size, shm_name);
    phase.complete();
  } catch (...) {
    close(fd);
    shm_unlink(shm_name.c_str());
    throw;
  }

  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed for ") + shm_name + ": " +
                             strerror(errno));
  }

  try {
    ShmStartupPhase phase("populate", size);
    first_touch(ptr, size);
    phase.complete();
    numa_policy.restore();
  } catch (...) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw;
  }
  ShmStartupPhase phase("owner_register", size);
  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("register_ptr failed for ") +
                             shm_name);
  }
  phase.complete();

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t attach_shm_pinned_ptr(std::size_t size, const std::string &shm_name,
                                bool writable) {
  if (size == 0) {
    throw std::runtime_error("attach_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  }
  if (shm_name.empty()) {
    throw std::runtime_error("attach_shm_pinned_ptr requires a shm_name");
  }

  int fd = shm_open(shm_name.c_str(), writable ? O_RDWR : O_RDONLY, 0600);
  if (fd < 0) {
    throw std::runtime_error(std::string("shm_open attach failed for ") +
                             shm_name + ": " + strerror(errno));
  }

  int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void *ptr = mmap(nullptr, size, prot, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap attach failed for ") + shm_name +
                             ": " + strerror(errno));
  }

  ShmStartupPhase phase("attach_register", size);
  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("register_ptr attach failed for ") +
                             shm_name);
  }
  phase.complete();

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t p, std::size_t size,
                         const std::string &shm_name) {
  if (p == 0) {
    throw std::runtime_error("free_shm_pinned_ptr requires non-null ptr");
  }
  if (size == 0) {
    throw std::runtime_error("free_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  }

  void *ptr = reinterpret_cast<void *>(p);

  auto unRegErr = unregister_ptr(ptr);
  auto unMapErr = munmap(ptr, size);
  shm_unlink(shm_name.c_str());
  if (unRegErr) {
    throw std::runtime_error("unregister_ptr failed: " +
                             std::to_string(unRegErr));
  }
  if (unMapErr) {
    throw std::runtime_error("munmap failed: " + std::to_string(unMapErr));
  }
}

void detach_shm_pinned_ptr(uintptr_t p, std::size_t size) {
  if (p == 0) {
    throw std::runtime_error("detach_shm_pinned_ptr requires non-null ptr");
  }
  if (size == 0) {
    throw std::runtime_error("detach_shm_pinned_ptr requires size > 0");
  }

  void *ptr = reinterpret_cast<void *>(p);

  auto unRegErr = unregister_ptr(ptr);
  auto unMapErr = munmap(ptr, size);
  if (unRegErr) {
    throw std::runtime_error("unregister_ptr detach failed: " +
                             std::to_string(unRegErr));
  }
  if (unMapErr) {
    throw std::runtime_error("munmap detach failed: " +
                             std::to_string(unMapErr));
  }
}

void unlink_shm(const std::string &shm_name) {
  if (shm_unlink(shm_name.c_str()) != 0 && errno != ENOENT) {
    throw std::runtime_error(std::string("shm_unlink failed for ") + shm_name +
                             ": " + strerror(errno));
  }
}
