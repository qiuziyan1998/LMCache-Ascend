# Separate local kernel library; leave the vendored BF16 kernels unchanged.
include("${PROJECT_SOURCE_DIR}/third_party/kvcache-ops/ascendc_with_def.cmake")
if(DEFINED ASCEND_AICORE_ARCH)
  ascendc_library_with_def(indexer_c8_kernels SHARED kernel.cpp)
  ascendc_compile_definitions(indexer_c8_kernels PRIVATE
    -DASCEND_AICORE_ARCH=${ASCEND_AICORE_ARCH})
else()
  ascendc_library(indexer_c8_kernels SHARED kernel.cpp)
endif()
install(TARGETS indexer_c8_kernels DESTINATION ${LMC_INSTALL_PATH})
