###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -haccel=amd_${HIP_ARCH}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -haccel=amd_${HIP_ARCH}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -haccel=amd_${HIP_ARCH}" CACHE STRING "")

# hcpu flag needs more experimentation, can cause runtime vectorization failures.
# -hcpu=x86-genoa
