###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -haccel=amd_${HIP_ARCH}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -haccel=amd_${HIP_ARCH}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -haccel=amd_${HIP_ARCH}" CACHE STRING "")

# hcpu flag needs more experimentation, can cause runtime vectorization failures.
# -hcpu=x86-genoa
