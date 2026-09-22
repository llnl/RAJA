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

set(CMAKE_CXX_COMPILER g++ CACHE FILEPATH "")
set(CMAKE_CXX_FLAGS "-Werror" CACHE STRING "")
set(ENABLE_CUDA ON CACHE BOOL "")
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O2" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" CACHE STRING "")
