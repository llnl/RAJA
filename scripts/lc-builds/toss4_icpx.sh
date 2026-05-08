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
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss4_icpx.sh 2022.1.0 [3.27.4]"
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

BUILD_SUFFIX=lc_toss4-icpx-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/${CMAKE_VER}

##
# CMake option -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile
# times at a potential cost of slower 'forall' execution.
##

if [[ ${COMP_VER} == 2024.2.1 ]]
then
  source /collab/usr/global/tools/intel/toss_4_x86_64_ib/oneapi-2024.2.1/setvars.sh
else
  source /usr/tce/packages/intel/intel-${COMP_VER}/setvars.sh
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx \
  -DBLT_CXX_STD=c++20 \
  -C ../host-configs/lc-builds/toss4/icpx_X.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-19.1.3/bin/clang-format \
  -DENABLE_OPENMP=On \
  -DENABLE_BENCHMARKS=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
if [[ ${COMP_VER} == 2024.2.1 ]]
then
echo "To successfully build and run all tests, you may need to run the"
echo "command to make sure your environment is set up properly:"
echo
echo "  source /collab/usr/global/tools/intel/toss_4_x86_64_ib/oneapi-2024.2.1/setvars.sh"
echo
fi
echo "***********************************************************************"
