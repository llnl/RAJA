#
## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "dpcpp" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++20 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -g -std=c++20 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -std=c++20 -fsycl -fsycl-unnamed-lambda" CACHE STRING "")
set(CMAKE_CXX_LINK_FLAGS "-fsycl -Wl,-rpath,/usr/tce/packages/oneapi/oneapi-2021.2/compiler/2021.2.0/linux/compiler/lib/intel64_lin/"  CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
