#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_amdclang_asan.sh 5.7.0 gfx90a"
  exit
fi

COMP_VER=$1
COMP_ARCH=$2
shift 2

HOSTCONFIG="hip_3_X"

if [[ ${COMP_VER} == 4.* ]]
then
##HIP_CLANG_FLAGS="-mllvm -amdgpu-fixed-function-abi=1"
  HOSTCONFIG="hip_4_link_X"
elif [[ ${COMP_VER} == 3.* ]]
then
  HOSTCONFIG="hip_3_X"
else
  echo "Unknown hip version, using ${HOSTCONFIG} host-config"
fi

BUILD_SUFFIX=lc_toss4-amdclang-${COMP_VER}-${COMP_ARCH}-asan

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo
echo "To get cmake to work you may have to configure with"
echo "   -DHIP_PLATFORM=amd"
echo
echo "To use fp64 HW atomics you must configure with these options when using gfx90a and hip >= 5.2"
echo "   -DCMAKE_CXX_FLAGS=\"-munsafe-fp-atomics\""
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.24.2

# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
# module unload rocm

if [[ ${COMP_VER} =~ .*magic.* ]]; then
  ROCM_PATH="/usr/tce/packages/rocmcc/rocmcc-${COMP_VER}"
else
  ROCM_PATH="/usr/tce/packages/rocmcc-tce/rocmcc-${COMP_VER}"
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_ROOT_DIR="${ROCM_PATH}" \
  -DHIP_ROOT_DIR="${ROCM_PATH}/hip" \
  -DHIP_PATH=${ROCM_PATH}/llvm/bin \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}:xnack+" \
  -DGPU_TARGETS="${COMP_ARCH}:xnack+" \
  -DAMDGPU_TARGETS="${COMP_ARCH}:xnack+" \
  -DCMAKE_C_FLAGS="-fsanitize=address -shared-libsan" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -shared-libsan" \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/opt/rocm-5.2.3/llvm/bin/clang-format \
  -DBLT_CXX_STD=c++17 \
  -C "../host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake" \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=ON \
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
echo "  Please note that you have to have a consistent build environment"
echo "  when you make RAJA as cmake may reconfigure; load the appropriate"
echo "  rocm and rocmcc modules (${COMP_VER}) when building."
echo
echo "    module load rocm/COMP_VER rocmcc/COMP_VER"
echo "    srun -n1 make"
echo
echo "  Run with these environment options when using asan"
echo "    ASAN_OPTIONS=print_suppressions=0:detect_leaks=0"
echo "    HSA_XNACK=1"
echo
echo "***********************************************************************"
