# Initialize Ascend C once and extend the existing library without editing
# vendored kernels. Its toolkit setup defines process-wide CMake targets.
set(KVCACHE_OPS_ROOT "${PROJECT_SOURCE_DIR}/third_party/kvcache-ops")
file(GLOB KERNEL_FILES
  "${KVCACHE_OPS_ROOT}/kernels/*.cpp"
  "${KVCACHE_OPS_ROOT}/kernels/cachegen/*.cpp"
  "${KVCACHE_OPS_ROOT}/kernels/fused_rope/*.cpp"
  "${KVCACHE_OPS_ROOT}/kernels/single_layer/*.cpp"
  "${KVCACHE_OPS_ROOT}/kernels/multi_layer/*.cpp"
)
list(APPEND KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp")
include("${KVCACHE_OPS_ROOT}/ascendc_with_def.cmake")
if(DEFINED ASCEND_AICORE_ARCH)
  ascendc_library_with_def(cache_kernels SHARED ${KERNEL_FILES})
  ascendc_compile_definitions(cache_kernels PRIVATE
    -DASCEND_AICORE_ARCH=${ASCEND_AICORE_ARCH})
else()
  ascendc_library(cache_kernels SHARED ${KERNEL_FILES})
endif()
target_include_directories(cache_kernels PUBLIC "${KVCACHE_OPS_ROOT}")
