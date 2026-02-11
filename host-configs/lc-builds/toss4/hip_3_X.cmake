###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(HIP_COMMON_OPT_FLAGS )
set(HIP_COMMON_DEBUG_FLAGS)
set(HOST_OPT_FLAGS)

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJA_HIPCC_FLAGS "-fPIC -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJA_HIPCC_FLAGS "-fPIC -g -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJA_HIPCC_FLAGS "-fPIC -g -O0 ${HIP_COMMON_DEBUG_FLAGS}" CACHE STRING "")
endif()

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
