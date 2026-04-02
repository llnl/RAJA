//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout perfect-loop launch tests
//

#ifndef __RAJA_TEST_LAUNCH_PERFECT_LOOP_EXECPOL_HPP__
#define __RAJA_TEST_LAUNCH_PERFECT_LOOP_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

using seq_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::seq_launch_t>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>>>;

using Sequential_launch_perfect_loop_policies =
    camp::list<seq_perfect_loop_policies>;

#if defined(RAJA_ENABLE_OPENMP)
using omp_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::omp_launch_t>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::omp_for_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>,
                            RAJA::LoopPolicy<RAJA::seq_exec>>>;

using OpenMP_launch_perfect_loop_policies =
    camp::list<omp_perfect_loop_policies>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using cuda_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::cuda_block_z_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_block_y_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_block_x_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_z_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>>>;

using cuda_perfect_loop_explicit_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::policy::cuda::cuda_launch_explicit_t<true, 0, 0>>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::cuda_block_z_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_block_y_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_block_x_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_z_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>,
                            RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>>>;

using Cuda_launch_perfect_loop_policies =
    camp::list<cuda_perfect_loop_policies,
               cuda_perfect_loop_explicit_policies>;
#endif

#if defined(RAJA_ENABLE_HIP)
using hip_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::hip_launch_t<true>>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::hip_block_z_loop>,
                            RAJA::LoopPolicy<RAJA::hip_block_y_loop>,
                            RAJA::LoopPolicy<RAJA::hip_block_x_loop>,
                            RAJA::LoopPolicy<RAJA::hip_thread_z_loop>,
                            RAJA::LoopPolicy<RAJA::hip_thread_y_loop>,
                            RAJA::LoopPolicy<RAJA::hip_thread_x_loop>>>;

using Hip_launch_perfect_loop_policies = camp::list<hip_perfect_loop_policies>;
#endif

#if defined(RAJA_ENABLE_SYCL)
using sycl_perfect_loop_policies = camp::list<
    RAJA::LaunchPolicy<RAJA::sycl_launch_t<true>>,
    RAJA::PerfectLoopPolicy<RAJA::LoopPolicy<RAJA::sycl_group_0_loop>,
                            RAJA::LoopPolicy<RAJA::sycl_group_1_loop>,
                            RAJA::LoopPolicy<RAJA::sycl_group_2_loop>,
                            RAJA::LoopPolicy<RAJA::sycl_local_0_loop>,
                            RAJA::LoopPolicy<RAJA::sycl_local_1_loop>,
                            RAJA::LoopPolicy<RAJA::sycl_local_2_loop>>>;

using Sycl_launch_perfect_loop_policies =
    camp::list<sycl_perfect_loop_policies>;
#endif

#endif  // __RAJA_TEST_LAUNCH_PERFECT_LOOP_EXECPOL_HPP__
