//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Kernel execution policy lists used throughout plugin tests
//

#ifndef __RAJA_test_plugin_kernelpol_HPP__
#define __RAJA_test_plugin_kernelpol_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/list.hpp"

// Sequential execution policy types
using SequentialPluginKernelExecPols = camp::list<
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0>>>,
      RAJA::KernelPolicy<
        RAJA::statement::Tile<0, RAJA::tile_fixed<2>, RAJA::seq_exec,
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::Lambda<0>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::simd_exec,
          RAJA::statement::Lambda<0>>>,
      RAJA::KernelPolicy<
        RAJA::statement::Tile<0, RAJA::tile_fixed<2>, RAJA::seq_exec,
          RAJA::statement::For<0, RAJA::simd_exec,
            RAJA::statement::Lambda<0>>>>
    >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPPluginKernelExecPols = camp::list<
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
          RAJA::statement::Lambda<0>>>,
      RAJA::KernelPolicy<
        RAJA::statement::Tile<0, RAJA::tile_fixed<2>, RAJA::omp_parallel_for_exec,
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::Lambda<0>>>>
    >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetPluginKernelExecPols = camp::list<
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<64>,
          RAJA::statement::Lambda<0>>>
    >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaPluginKernelExecPols = camp::list<
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
            RAJA::statement::Lambda<0>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
          RAJA::statement::Tile<0, RAJA::tile_fixed<128>, RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixed<128,
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
            RAJA::statement::Lambda<0>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixed<128,
          RAJA::statement::Tile<0, RAJA::tile_fixed<128>, RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>>>>>
    >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipPluginKernelExecPols = camp::list<
      RAJA::KernelPolicy<
        RAJA::statement::HipKernel<
          RAJA::statement::For<0, RAJA::hip_thread_x_loop,
            RAJA::statement::Lambda<0>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::HipKernel<
          RAJA::statement::Tile<0, RAJA::tile_fixed<128>, RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0>>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixed<128,
          RAJA::statement::For<0, RAJA::hip_thread_x_loop,
            RAJA::statement::Lambda<0>>>>,
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixed<128,
          RAJA::statement::Tile<0, RAJA::tile_fixed<128>, RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0>>>>>
    >;
#endif

#endif  // __RAJA_test_plugin_kernelpol_HPP__
