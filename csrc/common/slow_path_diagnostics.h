#ifndef LMCACHE_ASCEND_SLOW_PATH_DIAGNOSTICS_H
#define LMCACHE_ASCEND_SLOW_PATH_DIAGNOSTICS_H

#include <chrono>
#include <cstdlib>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace lmc::slow_diag {

constexpr double kSlowPathMs = 100.0;

inline bool enabled() {
  static const bool value = []() {
    const char *raw = std::getenv("LMCACHE_COLD_START_PERF");
    if (raw == nullptr) {
      return false;
    }
    std::string value(raw);
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return false;
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    value = value.substr(first, last - first + 1);
    for (char &ch : value) {
      if (ch >= 'A' && ch <= 'Z') {
        ch = static_cast<char>(ch - 'A' + 'a');
      }
    }
    return value != "0" && value != "false" && value != "no" &&
           value != "off";
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
