#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number for nvcc"
  echo "   2) CUDA compute architecture (number only, e.g., '90' not 'sm_90')"
  echo "   3) compiler version number for clang"
  echo
  echo "For example: "
  echo "    toss4_nvcc_clang.sh 12.6.0 90 14.0.6"
  echo
  echo "    toss4_nvcc_clang.sh 12.9.1 90 19.1.3-magic"
  echo "         (note: a compilation issue with one RAJA benchmark code)"
  exit
fi

COMP_NVCC_VER=$1
COMP_ARCH=$2
COMP_CLANG_VER=$3
shift 3

BUILD_SUFFIX=lc_toss4-nvcc${COMP_NVCC_VER}-${COMP_ARCH}-clang${COMP_CLANG_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.25.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++17 \
  -C ../host-configs/lc-builds/toss4/nvcc_clang_X.cmake \
  -DENABLE_CLANGFORMAT=Off \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-14.0.6/bin/clang-format \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DRAJA_ENABLE_NVTX=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER}/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=${COMP_ARCH} \
  -DENABLE_BENCHMARKS=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "On LC machines, you are likely to hit cudafe++ signal errors due to"
echo "insufficient virtual memory if you try to build in parallel using all"
echo "available cores; i.e., make -j. Try backing off to make -j 16."
echo
echo "***********************************************************************"
