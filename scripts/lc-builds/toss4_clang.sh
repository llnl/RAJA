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

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to the script. For example,"
  echo "    toss4_clang.sh 14.0.6 [3.27.4]"
  echo "An optional second argument can be given to set the CMake version to load."
  echo "If no CMake version is provided, version ${DEFAULT_CMAKE_VER} will be used."
  exit 1
fi

COMP_VER=$1

# Detect optional second positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$2
  shift 2
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 1
fi

BUILD_SUFFIX=lc_toss4-clang-${COMP_VER}

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
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++20 \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-19.1.3/bin/clang-format \
  -C ../host-configs/lc-builds/toss4/clang_X.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_BENCHMARKS=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
