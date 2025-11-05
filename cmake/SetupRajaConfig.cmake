###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")
#set(RAJA_FP "RAJA_USE_FLOAT")
option(RAJA_USE_DOUBLE On)
option(RAJA_USE_FLOAT Off)
option(RAJA_USE_COMPLEX Off)

## Pointer options
if (RAJA_ENABLE_CUDA OR RAJA_ENABLE_HIP)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance options
option(RAJA_ENABLE_FT "Enable fault-tolerance features" OFF)
option(RAJA_REPORT_FT "Report on use of fault-tolerant features" OFF)
option(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG "Enable Overflow checking during Iterator operations" OFF)

## Timer options
set(RAJA_TIMER "chrono" CACHE STRING
    "Select a timer backend")
set_property(CACHE RAJA_TIMER PROPERTY STRINGS "chrono" "gettime" "clock" )

if (RAJA_TIMER STREQUAL "chrono")
    set(RAJA_USE_CHRONO  ON  CACHE BOOL "Use the default std::chrono timer" )
else ()
    set(RAJA_USE_CHRONO  OFF  CACHE BOOL "Use the default std::chrono timer" )
endif ()
if (RAJA_TIMER STREQUAL "gettime")
    set(RAJA_USE_GETTIME ON CACHE BOOL "Use clock_gettime from time.h for timer"      )
else ()
    set(RAJA_USE_GETTIME OFF CACHE BOOL "Use clock_gettime from time.h for timer"       )
endif ()
if (RAJA_TIMER STREQUAL "clock")
    set(RAJA_USE_CLOCK   ON CACHE BOOL "Use clock from time.h for timer"     )
else ()
    set(RAJA_USE_CLOCK   OFF CACHE BOOL "Use clock from time.h for timer"    )
endif ()

include(CheckSymbolExists)
check_symbol_exists(posix_memalign stdlib.h RAJA_HAVE_POSIX_MEMALIGN)
check_symbol_exists(std::aligned_alloc stdlib.h RAJA_HAVE_ALIGNED_ALLOC)
check_symbol_exists(_mm_malloc "" RAJA_HAVE_MM_MALLOC)

# Set up RAJA_ENABLE prefixed options
set(RAJA_ENABLE_OPENMP ${ENABLE_OPENMP})
set(RAJA_ENABLE_TARGET_OPENMP ${ENABLE_TARGET_OPENMP})
set(RAJA_ENABLE_CUDA ${ENABLE_CUDA})
set(RAJA_ENABLE_NVTX ${ENABLE_NVTX})
set(RAJA_ENABLE_ROCTX ${ENABLE_ROCTX})
set(RAJA_ENABLE_CLANG_CUDA ${ENABLE_CLANG_CUDA})
set(RAJA_ENABLE_HIP ${ENABLE_HIP})
set(RAJA_ENABLE_SYCL ${ENABLE_SYCL})
set(RAJA_ENABLE_CUB ${ENABLE_CUB})

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp)

# Configure CMake config
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/share/raja/cmake/RAJA-config.cmake.in
  ${PROJECT_BINARY_DIR}/raja-config.cmake
  PATH_VARS CMAKE_INSTALL_PREFIX
  INSTALL_DESTINATION lib/cmake/raja)

install(FILES
  ${PROJECT_BINARY_DIR}/raja-config.cmake
  DESTINATION lib/cmake/raja)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/raja-config-version.cmake
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/raja-config-version.cmake"
  DESTINATION lib/cmake/raja)

# Setup pkg-config
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # convert lists of link libraries into -lstdc++ -lm etc..
  foreach(LIB ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${PLATFORM_LIBS} ${CUDA_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} -l${LIB}")
  endforeach()
  foreach(INCDIR ${INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    set(PC_C_FLAGS "${PC_C_FLAGS} -I${INCDIR}")
  endforeach()
  if(RAJA_ENABLE_EXTERNAL_ROCPRIM)
    foreach(INCDIR ${ROCPRIM_INCLUDE_DIRS})
      set(PC_C_FLAGS "${PC_C_FLAGS} -I${INCDIR}")
    endforeach()
  endif()
  if(RAJA_ENABLE_EXTERNAL_CUB)
    foreach(INCDIR ${CUB_INCLUDE_DIRS})
      set(PC_C_FLAGS "${PC_C_FLAGS} -I${INCDIR}")
    endforeach()
  endif()
  if(RAJA_ENABLE_CUDA)
    foreach(FLAG ${RAJA_NVCC_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  else()
    foreach(FLAG ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  endif()
  # Produce a pkg-config file for linking against the shared lib
  configure_file("share/raja/pkg-config/RAJA.pc.in" "RAJA.pc" @ONLY)
  install(FILES       "${CMAKE_CURRENT_BINARY_DIR}/RAJA.pc"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")
endif()
