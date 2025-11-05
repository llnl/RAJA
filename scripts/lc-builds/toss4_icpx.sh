#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss4_icpx.sh 2022.1.0"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_toss4-icpx-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.23.1

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
  -DBLT_CXX_STD=c++17 \
  -C ../host-configs/lc-builds/toss4/icpx_X.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-14.0.6/bin/clang-format \
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
