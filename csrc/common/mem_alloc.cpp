#include "mem_alloc.h"
#include "managed_mem.h"
#include <acl/acl.h>
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
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char *c = reinterpret_cast<volatile char *>(p) + off;
    *c = 0;
  }
}

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

uintptr_t alloc_shm_pinned_ptr(std::size_t size, const std::string &shm_name) {
  if (size == 0) {
    throw std::runtime_error("alloc_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  }
  if (shm_name.empty()) {
    throw std::runtime_error("alloc_shm_pinned_ptr requires a shm_name");
  }

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
    reserve_shm_storage(fd, size, shm_name);
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

  first_touch(ptr, size);
  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("register_ptr failed for ") +
                             shm_name);
  }

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

  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("register_ptr attach failed for ") +
                             shm_name);
  }

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
