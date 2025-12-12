#!/usr/bin/env bash

##
## Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read RAJAPerf/LICENSE.
##

BUILD_SUFFIX=sycl
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../host-configs/alcf-builds/sycl.cmake

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

DATE=$(printf '%(%Y-%m-%d)T\n' -1)

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DRAJA_ENABLE_TARGET_OPENMP=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DRAJA_ENABLE_SYCL=On \
  -DCMAKE_LINKER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DBLT_CXX_STD=c++20 \
  -DENABLE_TESTS=On \
  -DENABLE_EXAMPLES=On \
  "$@" \
  ..

make -j 8

#-DCMAKE_INSTALL_PREFIX=/home/bhomerding/install/raja.${DATE} \
