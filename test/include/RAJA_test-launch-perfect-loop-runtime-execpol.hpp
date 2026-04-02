//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_TEST_LAUNCH_PERFECT_LOOP_RUNTIME_EXECPOL_HPP__
#define __RAJA_TEST_LAUNCH_PERFECT_LOOP_RUNTIME_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

#if defined(RAJA_ENABLE_CUDA)
using seq_cuda_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::cuda_launch_t<false>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>>>;

using seq_cuda_perfect_loop_explicit_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t,
                       RAJA::policy::cuda::cuda_launch_explicit_t<true, 0, 0>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>>>;

using Sequential_launch_perfect_loop_runtime_policies =
    camp::list<seq_cuda_perfect_loop_policies,
               seq_cuda_perfect_loop_explicit_policies>;

#elif defined(RAJA_ENABLE_HIP)
using seq_hip_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::hip_launch_t<true>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_x_loop>>>;

using Sequential_launch_perfect_loop_runtime_policies =
    camp::list<seq_hip_perfect_loop_policies>;

#elif defined(RAJA_ENABLE_SYCL)
using seq_sycl_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::sycl_launch_t<true>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_0_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_1_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_2_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_0_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_1_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_2_loop>>>;

using Sequential_launch_perfect_loop_runtime_policies =
    camp::list<seq_sycl_perfect_loop_policies>;

#else
using Sequential_launch_perfect_loop_runtime_policies = camp::list<camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>>>>;
#endif

#if defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_CUDA)
using omp_cuda_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t, RAJA::cuda_launch_t<false>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::cuda_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>>>;

using omp_cuda_perfect_loop_explicit_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t,
                       RAJA::policy::cuda::cuda_launch_explicit_t<false, 0, 0>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::cuda_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>>>;

using OpenMP_launch_perfect_loop_runtime_policies =
    camp::list<omp_cuda_perfect_loop_policies,
               omp_cuda_perfect_loop_explicit_policies>;

#elif defined(RAJA_ENABLE_HIP)
using omp_hip_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t, RAJA::hip_launch_t<true>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::hip_block_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_x_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_z_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_y_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_x_loop>>>;

using OpenMP_launch_perfect_loop_runtime_policies =
    camp::list<omp_hip_perfect_loop_policies>;

#elif defined(RAJA_ENABLE_SYCL)
using omp_sycl_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t, RAJA::sycl_launch_t<true>>,
    RAJA::PerfectLoopPolicy<
        RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::sycl_group_0_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_1_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_2_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_0_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_1_loop>,
        RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_2_loop>>>;

using OpenMP_launch_perfect_loop_runtime_policies =
    camp::list<omp_sycl_perfect_loop_policies>;

#else
using OpenMP_launch_perfect_loop_runtime_policies = camp::list<camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::omp_parallel_for_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>>>>;
#endif

#endif

#if defined(RAJA_ENABLE_CUDA)
using Cuda_launch_perfect_loop_runtime_policies =
    camp::list<seq_cuda_perfect_loop_policies,
               seq_cuda_perfect_loop_explicit_policies
#if defined(RAJA_ENABLE_OPENMP)
               ,
               omp_cuda_perfect_loop_policies,
               omp_cuda_perfect_loop_explicit_policies
#endif
               >;
#endif

#if defined(RAJA_ENABLE_HIP)
using Hip_launch_perfect_loop_runtime_policies =
    camp::list<seq_hip_perfect_loop_policies
#if defined(RAJA_ENABLE_OPENMP)
               ,
               omp_hip_perfect_loop_policies
#endif
               >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using Sycl_launch_perfect_loop_runtime_policies =
    camp::list<seq_sycl_perfect_loop_policies
#if defined(RAJA_ENABLE_OPENMP)
               ,
               omp_sycl_perfect_loop_policies
#endif
               >;
#endif

#endif  // __RAJA_TEST_LAUNCH_PERFECT_LOOP_RUNTIME_EXECPOL_HPP__
