#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# Default CMake version if not provided
DEFAULT_CMAKE_VER=3.25.2

# Require compiler version
if [ "$1" == "" ]; then
  echo
  echo "You must pass a SYCL compiler path to script. For example,"
  echo "    corona_sycl.sh /usr/workspace/raja-dev/clang_sycl_16b7bcb09915_hip_gcc10.3.1_rocm6.4.2 [3.27.4]"
  echo "An optional second argument can be given to set the CMake version to load."
  echo "If no CMake version is provided, version ${DEFAULT_CMAKE_VER} will be used."
  exit 1
fi

SYCL_PATH=$1

# Detect optional second positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$2
  shift 2
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 1
fi

BUILD_SUFFIX=corona-sycl
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../host-configs/lc-builds/toss4/corona_sycl.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

DATE=$(printf '%(%Y-%m-%d)T\n' -1)

export PATH=${SYCL_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_PATH}/lib:${SYCL_PATH}/lib64:$LD_LIBRARY_PATH

module load cmake/${CMAKE_VER}

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
