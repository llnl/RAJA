#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass 2 arguments to the script (in this order): "
  echo "    1) compiler version number for clang"
  echo "    2) sanitizer version (one of 2 options: asan, or ubsan)"
  echo
  echo "For example: "
  echo "    toss4_clang.sh 14.0.6-magic asan"
  exit
fi

COMP_VER=$1
SAN_VER=$2
shift 2

if [[ ( ${SAN_VER} != "asan" ) && ( ${SAN_VER} != "ubsan" ) ]] ; then
  echo "Sanitizer version must be \"asan\" or \"ubsan\". Exiting!" ; exit
fi

BUILD_SUFFIX=lc_toss4-clang-${COMP_VER}-${SAN_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.23.1

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++17 \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-14.0.6/bin/clang-format \
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
