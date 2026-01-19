###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

#set(CMAKE_CXX_COMPILER "clang++" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CXX_COMPILER "dpcpp" CACHE PATH "")

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fsycl -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend '-device skl' " CACHE STRING "")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -g  -fsycl -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend '-device skl'" CACHE STRING "")
#set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -fsycl -fsycl-unnamed-lambda -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend '-device skl'" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++17 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -g -std=c++17 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -std=c++17 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_LINK_FLAGS "-fsycl -Wl,-rpath,/usr/tce/packages/oneapi/oneapi-2021.2/compiler/2021.2.0/linux/compiler/lib/intel64_lin/"  CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
