###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_BUILD_TYPE Release CACHE BOOL "")

set(ENABLE_HIP ON CACHE BOOL "")
set(RAJA_ENABLE_OPENMP OFF CACHE BOOL "")
set(RAJA_ENABLE_CUDA Off CACHE BOOL "")

if(DEFINED ROCM_DIR)
  set(HIP_ROOT_DIR "${ROCM_DIR}/hip" CACHE PATH "HIP ROOT directory path")
endif()

set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# set(HIP_COMMON_OPT_FLAGS  "--amdgpu-target=gfx900")
set(HIP_COMMON_OPT_FLAGS )
set(HIP_COMMON_DEBUG_FLAGS)
set(HOST_OPT_FLAGS)

if (RAJA_ENABLE_OPENMP)
	set(HIP_COMMON_OPT_FLAGS "-fopenmp ${HIP_COMMON_OPT_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJA_HIPCC_FLAGS "-O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJA_HIPCC_FLAGS "-g -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJA_HIPCC_FLAGS "-g -O0 ${HIP_COMMON_DEBUG_FLAGS}" CACHE STRING "")
endif()

set(RAJA_RANGE_ALIGN 4 CACHE STRING "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE STRING "")
set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
