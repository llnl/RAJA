###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(HOST_OPT_FLAGS "-Xcompiler -O3 -Xcompiler -finline-functions ")

if(ENABLE_OPENMP)
  set(HOST_OPT_FLAGS "${HOST_OPT_FLAGS} -Xcompiler -fopenmp")
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 ${HOST_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O3 ${HOST_OPT_FLAGS}" CACHE STRING "")

set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
