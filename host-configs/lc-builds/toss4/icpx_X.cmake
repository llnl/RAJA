###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

##set(COMMON_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1")
set(COMMON_OPT_FLAGS "-march=native -finline-functions -fp-model=precise")
#set(COMMON_OPT_FLAGS "-march=native -finline-functions")

##set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 ${COMMON_OPT_FLAGS}" CACHE STRING "")
##set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g ${COMMON_OPT_FLAGS}" CACHE STRING "")
##set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${COMMON_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g ${COMMON_OPT_FLAGS}" "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
