#ifndef LMCACHE_ASCEND_SLOW_PATH_DIAGNOSTICS_H
#define LMCACHE_ASCEND_SLOW_PATH_DIAGNOSTICS_H

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace lmc::slow_diag {

constexpr double kSlowPathMs = 100.0;

inline bool enabled() {
  static const bool value = []() {
    const char *raw = std::getenv("LMCACHE_COLD_START_PERF");
    return raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "0") != 0 &&
           std::strcmp(raw, "false") != 0 && std::strcmp(raw, "False") != 0 &&
           std::strcmp(raw, "no") != 0 && std::strcmp(raw, "off") != 0;
  }();
  return value;
}

inline int64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline int64_t thread_cpu_ns() {
  timespec value{};
  return clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0
             ? static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec
             : 0;
}

inline double elapsed_ms(int64_t started_ns, int64_t completed_ns) {
  return static_cast<double>(completed_ns - started_ns) / 1000000.0;
}

inline long thread_id() { return syscall(SYS_gettid); }

} // namespace lmc::slow_diag

#endif
