#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 1 ]]; then
  echo
  echo "You must pass 1 argument to the script: "
  echo "   1) SYCL compiler installation path"
  echo
  echo "For example: "
  echo "    corona_sycl.sh /usr/workspace/raja-dev/clang_sycl_730cd3a5275f_hip_gcc10.3.1_rocm6.0.2"
  exit
fi

SYCL_PATH=$1
shift 1

BUILD_SUFFIX=corona-sycl
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../host-configs/lc-builds/toss4/corona_sycl.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

DATE=$(printf '%(%Y-%m-%d)T\n' -1)

export PATH=${SYCL_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_PATH}/lib:${SYCL_PATH}/lib64:$LD_LIBRARY_PATH

module load cmake/3.23.1

cmake \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DSYCL_LIB_PATH:STRING="${SYCL_PATH}/lib" \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-14.0.6/bin/clang-format \
  -DRAJA_ENABLE_TARGET_OPENMP=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DRAJA_ENABLE_SYCL=On \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_LINKER=clang++ \
  -DBLT_CXX_STD=c++17 \
  -DENABLE_TESTS=On \
  -DENABLE_EXAMPLES=On \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX}_${USER} and run make to build RAJA"
echo
echo "To run RAJA tests, exercises, etc. with the build, please do the following:"
echo
echo "   1) Load the ROCm module version matching the version in the compiler path"
echo "      you passed to this script."
echo
echo "   2) Prefix the LD_LIBRARY_PATH environment variable with "
echo "        ${SYCL_PATH}/lib:${SYCL_PATH}/lib64"
echo
echo "***********************************************************************"
