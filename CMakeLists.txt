# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)
# project information
project(c_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LMC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
if (NOT GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

# for the device kernels compilation
set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})
# custom compile definition for kernels hostside compilation
set(ASCEND_AICORE_ARCH ${ASCEND_AICORE_ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(DEFINED ENV{USE_MINDSPORE})
  add_subdirectory(third_party/kvcache-ops)
  find_package(Threads REQUIRED)
  add_subdirectory(csrc/mindspore)
  target_link_options(c_ops PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
    "-Wl,-rpath,${LMC_INSTALL_PATH}"
  )
  target_link_directories(
    c_ops
    PRIVATE
    ${MINDSPORE_PATH}/lib
    ${MINDSPORE_PATH}/lib/plugin
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
  )
  target_link_libraries(
    c_ops
    PUBLIC
    mindspore_common
    mindspore_ops
    mindspore_pyboost
    mindspore_frontend
    mindspore_backend_manager
    mindspore_runtime_pipeline
    mindspore_pynative
    pthread
    "${MINDSPORE_PATH}/lib/plugin/libmindspore_ascend.so.2"
    cache_kernels
    ascendcl
    platform
    ascend_hal
    tiling_api
    numa
  )

else()
  add_subdirectory(csrc/graph)
  add_subdirectory(csrc)
  # Use the same ACL runtime as the CANN 8.5.1 capture probe.
  # Builds with earlier SDKs retain the ordinary sparse transfer.
  find_library(LMC_ACL_RT acl_rt PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64"
               "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib" NO_DEFAULT_PATH)
  if(LMC_ACL_RT)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${ASCEND_CANN_PACKAGE_PATH}/include")
    check_cxx_source_compiles([[
      #include <acl/acl_rt.h>
      int main() {
        (void)sizeof(&aclrtValueWrite);
        (void)sizeof(&aclrtValueWait);
        (void)sizeof(&aclrtCreateNotify);
        (void)sizeof(&aclrtDestroyNotify);
        (void)sizeof(&aclrtRecordNotify);
        (void)sizeof(&aclrtWaitAndResetNotify);
        return ACL_STREAM_WAIT_VALUE_GEQ;
      }
    ]] LMC_HAS_SFA_STREAM_VALUES)
    cmake_pop_check_state()
    if(LMC_HAS_SFA_STREAM_VALUES)
      target_compile_definitions(c_ops PRIVATE LMC_HAS_SFA_STREAM_VALUES=1)
      target_link_libraries(c_ops PRIVATE "${LMC_ACL_RT}")
    endif()
  endif()
  set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")

  target_link_options(c_ops PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
    "-Wl,-rpath,${LMC_INSTALL_PATH}" 
  )

  target_link_directories(
    c_ops
    PRIVATE
    ${TORCH_LIBS_DIR}
    ${TORCH_NPU_PATH}/lib/
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
  )

  target_link_libraries(
    c_ops
    PUBLIC
    ${TORCH_LIBRARIES}
    libtorch_npu.so
    cache_kernels
    ascendcl
    platform
    ascend_hal
    tiling_api
    numa
  )
endif()

install(TARGETS c_ops cache_kernels DESTINATION ${LMC_INSTALL_PATH})

