###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(BLT_CXX_STD c++20 CACHE STRING "")
set(ENABLE_TESTS ON CACHE BOOL "")
set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "")
set(RAJA_RANGE_ALIGN 4 CACHE STRING "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE STRING "")
set(RAJA_DATA_ALIGN 64 CACHE STRING "")
set(RAJA_HOST_CONFIG_LOADED ON CACHE BOOL "")

set(CMAKE_PREFIX_PATH "/opt/rocm" CACHE PATH "")
set(CMAKE_C_COMPILER hipcc CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER hipcc CACHE FILEPATH "")
set(CMAKE_CXX_FLAGS "-Werror -Wno-deprecated-attributes" CACHE STRING "")
set(ENABLE_HIP ON CACHE BOOL "")
set(ENABLE_OPENMP ON CACHE BOOL "")
set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING "")
set(AMDGPU_TARGETS gfx942 CACHE STRING "")
set(ROCPRIM_DIR "/opt/rocm" CACHE PATH "")
set(RAJA_HIPCC_FLAGS_RELEASE "-O2" CACHE STRING "")
set(RAJA_HIPCC_FLAGS_RELWITHDEBINFO "-g -O2" CACHE STRING "")
set(RAJA_HIPCC_FLAGS_DEBUG "-g -O0" CACHE STRING "")
