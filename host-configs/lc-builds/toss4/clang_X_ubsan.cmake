###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O3 -march=native -funroll-loops -finline-functions -fsanitize=undefined -fno-omit-frame-pointer -fsanitize=integer" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O3 -g -march=native -funroll-loops -finline-functions -fsanitize=undefined -fno-omit-frame-pointer -fsanitize=integer" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O0 -g -fsanitize=undefined -fno-omit-frame-pointer -fsanitize=integer" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
