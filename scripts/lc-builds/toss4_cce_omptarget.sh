#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 3 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_cce_omptarget.sh 20.0.0-magic gfx942"
  exit
fi

COMP_VER=$1
HIP_ARCH=$2
shift 2

HOSTCONFIG="cce_omptarget_X"

BUILD_SUFFIX=lc_toss4-cce-${COMP_VER}-${HIP_ARCH}-omptarget

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.24.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_ARCH=${HIP_ARCH} \
  -DCMAKE_C_COMPILER="/usr/tce/packages/cce/cce-${COMP_VER}/bin/craycc" \
  -DCMAKE_CXX_COMPILER="/usr/tce/packages/cce/cce-${COMP_VER}/bin/crayCC" \
  -DBLT_CXX_STD=c++17 \
  -DENABLE_CLANGFORMAT=Off \
  -C "../host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake" \
  -DENABLE_HIP=OFF \
  -DENABLE_OPENMP=ON \
  -DRAJA_ENABLE_TARGET_OPENMP=ON \
  -DENABLE_CUDA=OFF \
  -DENABLE_BENCHMARKS=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "    srun -n1 make"
echo
echo "***********************************************************************"
