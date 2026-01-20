###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

set(COMMON_OPT_FLAGS "-Ofast -march=native")
##set(COMMON_OPT_FLAGS "-Ofast -march=native -finline-functions -finline-limit=20000")

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_OPT_FLAGS} -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
