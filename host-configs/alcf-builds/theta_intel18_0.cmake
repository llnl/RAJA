###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries_2018.0.128/linux/bin/intel64/icpc" CACHE PATH "")
set(CMAKE_C_COMPILER "/opt/intel/compilers_and_libraries_2018.0.128/linux/bin/intel64/icc" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -unroll-aggressive -finline-functions -xMIC-AVX512" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -unroll-aggressive -finline-functions -xMIC-AVX512" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
