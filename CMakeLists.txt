###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

if (APPLE)
 cmake_policy(SET CMP0025 NEW)
endif()

include(CMakeDependentOption)

# Set version number
set(RAJA_VERSION_MAJOR 2025)
set(RAJA_VERSION_MINOR 09)
set(RAJA_VERSION_PATCHLEVEL 1)

if (RAJA_LOADED AND (NOT RAJA_LOADED STREQUAL "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}"))
  message(FATAL_ERROR "You are mixing RAJA versions. Loaded is ${RAJA_LOADED}, expected ${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")
endif()

if (RAJA_LOADED)
  return() # Stop processing file, avoids nesting the whole file
endif()
set (RAJA_LOADED "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")

# Promote RAJA_LOADED to PARENT_SCOPE if it exists, which is only if we are bringing
# in RAJA as a subproject to a larger CMake project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set (RAJA_LOADED ${RAJA_LOADED} PARENT_SCOPE)
endif()

mark_as_advanced(RAJA_LOADED)

# C is required for googletest to find Threads
project(RAJA LANGUAGES CXX C
  VERSION ${RAJA_LOADED})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})
set(BLT_REQUIRED_CLANGFORMAT_VERSION  "14" CACHE STRING "")
include(cmake/SetupRajaOptions.cmake)

cmake_minimum_required(VERSION 3.23)

# Detect C++ standard and add appropriate flag _before_ loading BLT
set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(NOT DEFINED BLT_CXX_STD)
  if("cxx_std_20" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(BLT_CXX_STD c++20 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  elseif("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(BLT_CXX_STD c++17 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(BLT_CXX_STD c++17 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  else() #cmake has no idea what to do, do it ourselves...
    set(flag_var "c++17")
    CHECK_CXX_COMPILER_FLAG("-std=${flag_var}" COMPILER_SUPPORTS_${flag_var})
    if(COMPILER_SUPPORTS_${flag_var})
      set(BLT_CXX_STD ${flag_var} CACHE STRING "Version of C++ standard")
      message("Using C++ standard: ${BLT_CXX_STD}")
    endif()
    unset(flag_var)
  endif()
else() #check BLT_CXX_STD is high enough by disallowing the only invalid option
  if(("${BLT_CXX_STD}" STREQUAL "c++98") OR
     ("${BLT_CXX_STD}" STREQUAL "c++11") OR
     ("${BLT_CXX_STD}" STREQUAL "c++14"))
    message(FATAL_ERROR "RAJA requires minimum C++ standard of c++17")
  endif()
endif(NOT DEFINED BLT_CXX_STD)

set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

# Setup options that depend on BLT
include(cmake/SetupDependentOptions.cmake)
# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)

# Add workaround flags to host compiler if necessary
if (RAJA_ENABLE_CUDA)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU AND CMAKE_SYSTEM_PROCESSOR MATCHES ppc64le)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mno-float128")
    endif ()
  endif ()
endif ()

# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Macros for building executables and libraries
include (cmake/RAJAMacros.cmake)

# Configure `style` target for enforcing code style
raja_add_code_checks()

set (raja_sources
  src/AlignedRangeIndexSetBuilders.cpp
  src/DepGraphNode.cpp
  src/LockFreeIndexSetBuilders.cpp
  src/MemUtils_CUDA.cpp
  src/MemUtils_HIP.cpp
  src/MemUtils_SYCL.cpp
  src/PluginStrategy.cpp)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  set (raja_sources
    ${raja_sources}
    src/RuntimePluginLoader.cpp
    src/KokkosPluginLoader.cpp)

if (RAJA_ENABLE_CALIPER)
   list(APPEND raja_sources src/plugins/caliper-plugin.cpp)
endif ()

endif ()

set (raja_depends)

if (RAJA_ENABLE_CALIPER)

set (raja_depends
    ${raja_depends}
    caliper)
  find_package(caliper REQUIRED
    NO_DEFAULT_PATH
    PATHS ${caliper_DIR}
  )

endif()

if (RAJA_ENABLE_OPENMP)
  set (raja_depends
    ${raja_depends}
    openmp)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)
  message(WARNING "RAJA::simd_exec support requires Intel-17 or greater")
endif()

if(RAJA_ENABLE_CUDA)
  if("${CUDA_VERSION_STRING}" VERSION_LESS "9.2")
    message(FATAL_ERROR "Trying to use CUDA version ${CUDA_VERSION_STRING}. RAJA dependency Googletest requires CUDA version 9.2.x or newer.")
  endif()
endif()

if(RAJA_ENABLE_HIP)
  message(STATUS "HIP version: ${hip_VERSION}")
  set(RAJA_HIP_WAVESIZE "64" CACHE STRING "Set the wave size for GPU architecture. E.g. MI200/MI300 this is 64.")
  if("${hip_VERSION}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Trying to use HIP/ROCm version ${hip_VERSION}. RAJA requires HIP/ROCm version 3.5 or newer. ")
  endif()
endif()

if (RAJA_ENABLE_CUDA)
  set(raja_depends
    ${raja_depends}
    cuda)
  if(RAJA_ENABLE_NVTX)
    set(raja_depends
      ${raja_depends}
      nvtx3)
  endif ()
endif ()

if (RAJA_ENABLE_EXTERNAL_CUB)
  set(raja_depends
    ${raja_depends}
    cub)
endif ()

if (RAJA_ENABLE_HIP)
  set(raja_depends
    ${raja_depends}
    blt::hip)
  set(raja_depends
    ${raja_depends}
    blt::hip_runtime)
  if(RAJA_ENABLE_ROCTX)
      set(raja_depends
      ${raja_depends}
      roctx)
  endif()
endif ()

if (RAJA_ENABLE_EXTERNAL_ROCPRIM)
  set(raja_depends
    ${raja_depends}
    rocPRIM)
endif ()

if (RAJA_ENABLE_SYCL)
  set (RAJA_ENABLE_DESUL_ATOMICS "On")
  set (ENABLE_SYCL "On") # Enable SYCL atomics in Desul
  set (raja_depends
    ${raja_depends}
    sycl)
endif ()


message(STATUS "Desul Atomics support is ${RAJA_ENABLE_DESUL_ATOMICS}")
if (RAJA_ENABLE_DESUL_ATOMICS)
  # NOTE: ENABLE_OPENMP in Desul implies OpenMP OFFLOAD
  if (NOT RAJA_ENABLE_TARGET_OPENMP)
    set (DESUL_ENABLE_OPENMP Off CACHE BOOL "Disable Desul OpenMP offload atomics to select builtin host atomics")
  endif()
  add_subdirectory(tpl/desul)
  set(raja_depends
    ${raja_depends}
    desul_atomics)
endif ()

if (NOT TARGET camp)
  set(EXTERNAL_CAMP_SOURCE_DIR "" CACHE FILEPATH "build with a specific external
camp source repository")
  if (EXTERNAL_CAMP_SOURCE_DIR)
    message(STATUS "Using external source CAMP from: " ${EXTERNAL_CAMP_SOURCE_DIR})
    add_subdirectory(${EXTERNAL_CAMP_SOURCE_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}/tpl/camp)
  else (EXTERNAL_CAMP_SOURCE_DIR)
    if (DEFINED camp_DIR)
      find_package(camp REQUIRED
        NO_DEFAULT_PATH
        PATHS ${camp_DIR}
        ${camp_DIR}/lib/cmake/camp
      )
      message(STATUS "Using CAMP version ${camp_VERSION} installed at: ${camp_INSTALL_PREFIX}")
    else ()
      message(STATUS "Using RAJA CAMP submodule.")
      # propagate setting for sycl since there's no common enable to use
      set (CAMP_ENABLE_SYCL ${RAJA_ENABLE_SYCL})
      add_subdirectory(tpl/camp)
    endif()
  endif (EXTERNAL_CAMP_SOURCE_DIR)

  if (RAJA_ENABLE_TARGET_OPENMP)
    message(STATUS "Setting CAMP_ENABLE_TARGET_OPENMP to ${RAJA_ENABLE_TARGET_OPENMP}")
    blt_add_target_definitions(
      TO camp
      SCOPE INTERFACE
      TARGET_DEFINITIONS CAMP_ENABLE_TARGET_OPENMP)
  endif()
endif (NOT TARGET camp)

set (raja_defines)

if (COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS)
  set (raja_defines
    ${raja_defines}
    RAJA_WIN_STATIC_BUILD)
endif ()

blt_add_library(
  NAME RAJA
  SOURCES ${raja_sources}
  DEPENDS_ON ${raja_depends} camp ${CMAKE_DL_LIBS}
  DEFINES ${raja_defines})


install(TARGETS RAJA
  EXPORT RAJATargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  )

install(EXPORT RAJATargets DESTINATION lib/cmake/raja)

target_include_directories(RAJA
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if (RAJA_ENABLE_CUDA AND NOT RAJA_ENABLE_EXTERNAL_CUB)
  target_include_directories(RAJA SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl/cub>)
endif()
if (RAJA_ENABLE_HIP AND NOT RAJA_ENABLE_EXTERNAL_ROCPRIM)
  target_include_directories(RAJA SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl/rocPRIM/rocprim/include>)
endif()

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
if (RAJA_ENABLE_CUDA AND NOT RAJA_ENABLE_EXTERNAL_CUB)
  install(DIRECTORY tpl/cub/cub/ DESTINATION include/cub FILES_MATCHING PATTERN *.cuh)
endif()
if (RAJA_ENABLE_HIP AND NOT RAJA_ENABLE_EXTERNAL_ROCPRIM)
  install(DIRECTORY tpl/rocPRIM/rocprim/include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
endif()

install(FILES
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
  DESTINATION "include/RAJA/")


# Setup internal RAJA configuration options
include(cmake/SetupRajaConfig.cmake)

if(RAJA_ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(RAJA_ENABLE_REPRODUCERS)
  add_subdirectory(reproducers)
endif()

if(RAJA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(RAJA_ENABLE_EXERCISES)
  add_subdirectory(exercises)
endif()

if (RAJA_ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif ()

if (RAJA_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
endif ()
