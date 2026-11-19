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

set(CMAKE_CXX_COMPILER icpx CACHE FILEPATH "")
set(CMAKE_CXX_FLAGS "-fsycl -w -fp-model=precise -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-fsycl" CACHE STRING "")
set(RAJA_ENABLE_SYCL ON CACHE BOOL "")
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "")
