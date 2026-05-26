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

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in the following order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3) optional CMake version to load."
  echo
  echo "For example: "
  echo "    toss4_clang.sh 14.0.6-magic asan [3.27.4]"
  echo "If no CMake version is provided, version ${DEFAULT_CMAKE_VER} will be used."
  exit 1
fi

COMP_VER=$1
SAN_VER=$2

# Detect optional third positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$3" ] && [[ "$3" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$3
  shift 3
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 2
fi

if [[ ( ${SAN_VER} != "asan" ) && ( ${SAN_VER} != "ubsan" ) ]] ; then
  echo "Sanitizer version must be \"asan\" or \"ubsan\". Exiting!" ; exit
fi

BUILD_SUFFIX=lc_toss4-clang-${COMP_VER}-${SAN_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/${CMAKE_VER}

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++20 \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-19.1.3/bin/clang-format \
  -C ../host-configs/lc-builds/toss4/clang_X_${SAN_VER}.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_BENCHMARKS=ON \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

if [[ ( ${SAN_VER} = "ubsan" ) ]] ; then
  echo "To view ubsan output, set the following environment variable: "
  echo "    UBSAN_OPTIONS=log_path=/path/to/ubsan_log_prefix"
  echo
  echo "Each test will create a ubsan output file with the prefix \"ubsan_log_prefix\"."
fi
