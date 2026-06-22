//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Compile-time coverage for RAJA::device_* policy aliases.
///

#include "RAJA/RAJA.hpp"

#include <type_traits>

#include "RAJA_gtest.hpp"

namespace
{

#if defined(RAJA_CUDA_ACTIVE)
static_assert(std::is_same<RAJA::device_exec<128, false>,
                           RAJA::cuda_exec<128, false>>::value,
              "device_exec should map to cuda_exec when RAJA_CUDA_ACTIVE");
static_assert(std::is_same<RAJA::device_exec_with_reduce<128, false>,
                           RAJA::cuda_exec_with_reduce<128, false>>::value,
              "device_exec_with_reduce should map to cuda_exec_with_reduce");
static_assert(std::is_same<RAJA::device_atomic,
                           RAJA::cuda_atomic>::value,
              "device_atomic should map to cuda_atomic when RAJA_CUDA_ACTIVE");
static_assert(std::is_same<RAJA::device_atomic_explicit<RAJA::seq_atomic>,
                           RAJA::cuda_atomic_explicit<RAJA::seq_atomic>>::value,
              "device_atomic_explicit should map to cuda_atomic_explicit");
static_assert(std::is_same<RAJA::device_reduce, RAJA::cuda_reduce>::value,
              "device_reduce should map to cuda_reduce when RAJA_CUDA_ACTIVE");
static_assert(std::is_same<RAJA::device_reduce_atomic,
                           RAJA::cuda_reduce_atomic>::value,
              "device_reduce_atomic should map to cuda_reduce_atomic");
static_assert(std::is_same<RAJA::device_reduce_base<false>,
                           RAJA::cuda_reduce>::value,
              "device_reduce_base<false> should map to cuda_reduce");
static_assert(std::is_same<RAJA::device_reduce_base<true>,
                           RAJA::cuda_reduce_atomic>::value,
              "device_reduce_base<true> should map to cuda_reduce_atomic");
static_assert(std::is_same<RAJA::device_multi_reduce_atomic,
                           RAJA::cuda_multi_reduce_atomic>::value,
              "device_multi_reduce_atomic should map to cuda_multi_reduce_atomic");
static_assert(std::is_same<
                  RAJA::device_multi_reduce_atomic_low_performance_low_overhead,
                  RAJA::cuda_multi_reduce_atomic_low_performance_low_overhead>::value,
              "device_multi_reduce_atomic_low_performance_low_overhead should map to cuda");
static_assert(std::is_same<RAJA::device_launch_t<false>,
                           RAJA::cuda_launch_t<false>>::value,
              "device_launch_t should map to cuda_launch_t when RAJA_CUDA_ACTIVE");
static_assert(std::is_same<RAJA::device_global_size_x_direct<64>,
                           RAJA::cuda_global_size_x_direct<64>>::value,
              "device_global_size_x_direct should map to cuda_global_size_x_direct");
static_assert(std::is_same<RAJA::device_global_size_x_loop<64>,
                           RAJA::cuda_global_size_x_loop<64>>::value,
              "device_global_size_x_loop should map to cuda_global_size_x_loop");
static_assert(std::is_same<RAJA::device_thread_x_direct,
                           RAJA::cuda_thread_x_direct>::value,
              "device_thread_x_direct should map to cuda_thread_x_direct");
static_assert(std::is_same<RAJA::device_thread_size_x_direct<32>,
                           RAJA::cuda_thread_size_x_direct<32>>::value,
              "device_thread_size_x_direct should map to cuda_thread_size_x_direct");
static_assert(std::is_same<RAJA::device_block_x_loop,
                           RAJA::cuda_block_x_loop>::value,
              "device_block_x_loop should map to cuda_block_x_loop");
static_assert(std::is_same<RAJA::device_block_size_x_loop<32>,
                           RAJA::cuda_block_size_x_loop<32>>::value,
              "device_block_size_x_loop should map to cuda_block_size_x_loop");
static_assert(std::is_same<RAJA::device_thread_xy_direct,
                           RAJA::cuda_thread_xy_direct>::value,
              "device_thread_xy_direct should map to cuda_thread_xy_direct");
static_assert(std::is_same<RAJA::device_block_xyz_loop,
                           RAJA::cuda_block_xyz_loop>::value,
              "device_block_xyz_loop should map to cuda_block_xyz_loop");
static_assert(std::is_same<RAJA::device_thread_size_xy_direct<4, 5>,
                           RAJA::cuda_thread_size_xy_direct<4, 5>>::value,
              "device_thread_size_xy_direct should map to cuda_thread_size_xy_direct");
static_assert(std::is_same<RAJA::device_block_syncable_loop<RAJA::named_dim::x,
                                                            RAJA::named_dim::y>,
                           RAJA::cuda_block_xy_syncable_loop>::value,
              "device_block_syncable_loop should map to cuda_block_xy_syncable_loop");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_direct,
                           RAJA::cuda_flatten_block_threads_xy_direct>::value,
              "device_flatten_block_threads_xy_direct should map to cuda flatten");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_loop,
                           RAJA::cuda_flatten_block_threads_xy_loop>::value,
              "device_flatten_block_threads_xy_loop should map to cuda flatten");

#elif defined(RAJA_HIP_ACTIVE)
static_assert(std::is_same<RAJA::device_exec<128, false>,
                           RAJA::hip_exec<128, false>>::value,
              "device_exec should map to hip_exec when RAJA_HIP_ACTIVE");
static_assert(std::is_same<RAJA::device_exec_with_reduce<128, false>,
                           RAJA::hip_exec_with_reduce<128, false>>::value,
              "device_exec_with_reduce should map to hip_exec_with_reduce");
static_assert(std::is_same<RAJA::device_atomic,
                           RAJA::hip_atomic>::value,
              "device_atomic should map to hip_atomic when RAJA_HIP_ACTIVE");
static_assert(std::is_same<RAJA::device_atomic_explicit<RAJA::seq_atomic>,
                           RAJA::hip_atomic_explicit<RAJA::seq_atomic>>::value,
              "device_atomic_explicit should map to hip_atomic_explicit");
static_assert(std::is_same<RAJA::device_reduce, RAJA::hip_reduce>::value,
              "device_reduce should map to hip_reduce when RAJA_HIP_ACTIVE");
static_assert(std::is_same<RAJA::device_reduce_atomic,
                           RAJA::hip_reduce_atomic>::value,
              "device_reduce_atomic should map to hip_reduce_atomic");
static_assert(std::is_same<RAJA::device_reduce_base<false>,
                           RAJA::hip_reduce>::value,
              "device_reduce_base<false> should map to hip_reduce");
static_assert(std::is_same<RAJA::device_reduce_base<true>,
                           RAJA::hip_reduce_atomic>::value,
              "device_reduce_base<true> should map to hip_reduce_atomic");
static_assert(std::is_same<RAJA::device_multi_reduce_atomic,
                           RAJA::hip_multi_reduce_atomic>::value,
              "device_multi_reduce_atomic should map to hip_multi_reduce_atomic");
static_assert(std::is_same<
                  RAJA::device_multi_reduce_atomic_low_performance_low_overhead,
                  RAJA::hip_multi_reduce_atomic_low_performance_low_overhead>::value,
              "device_multi_reduce_atomic_low_performance_low_overhead should map to hip");
static_assert(std::is_same<RAJA::device_launch_t<false>,
                           RAJA::hip_launch_t<false>>::value,
              "device_launch_t should map to hip_launch_t when RAJA_HIP_ACTIVE");
static_assert(std::is_same<RAJA::device_global_size_x_direct<64>,
                           RAJA::hip_global_size_x_direct<64>>::value,
              "device_global_size_x_direct should map to hip_global_size_x_direct");
static_assert(std::is_same<RAJA::device_global_size_x_loop<64>,
                           RAJA::hip_global_size_x_loop<64>>::value,
              "device_global_size_x_loop should map to hip_global_size_x_loop");
static_assert(std::is_same<RAJA::device_thread_x_direct,
                           RAJA::hip_thread_x_direct>::value,
              "device_thread_x_direct should map to hip_thread_x_direct");
static_assert(std::is_same<RAJA::device_thread_size_x_direct<32>,
                           RAJA::hip_thread_size_x_direct<32>>::value,
              "device_thread_size_x_direct should map to hip_thread_size_x_direct");
static_assert(std::is_same<RAJA::device_block_x_loop,
                           RAJA::hip_block_x_loop>::value,
              "device_block_x_loop should map to hip_block_x_loop");
static_assert(std::is_same<RAJA::device_block_size_x_loop<32>,
                           RAJA::hip_block_size_x_loop<32>>::value,
              "device_block_size_x_loop should map to hip_block_size_x_loop");
static_assert(std::is_same<RAJA::device_thread_xy_direct,
                           RAJA::hip_thread_xy_direct>::value,
              "device_thread_xy_direct should map to hip_thread_xy_direct");
static_assert(std::is_same<RAJA::device_block_xyz_loop,
                           RAJA::hip_block_xyz_loop>::value,
              "device_block_xyz_loop should map to hip_block_xyz_loop");
static_assert(std::is_same<RAJA::device_thread_size_xy_direct<4, 5>,
                           RAJA::hip_thread_size_xy_direct<4, 5>>::value,
              "device_thread_size_xy_direct should map to hip_thread_size_xy_direct");
static_assert(std::is_same<RAJA::device_block_syncable_loop<RAJA::named_dim::x,
                                                            RAJA::named_dim::y>,
                           RAJA::hip_block_xy_syncable_loop>::value,
              "device_block_syncable_loop should map to hip_block_xy_syncable_loop");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_direct,
                           RAJA::hip_flatten_block_threads_xy_direct>::value,
              "device_flatten_block_threads_xy_direct should map to hip flatten");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_loop,
                           RAJA::hip_flatten_block_threads_xy_loop>::value,
              "device_flatten_block_threads_xy_loop should map to hip flatten");

#elif defined(RAJA_SYCL_ACTIVE)
static_assert(std::is_same<RAJA::device_exec<128, false>,
                           RAJA::sycl_exec<128, false>>::value,
              "device_exec should map to sycl_exec when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_atomic,
                           RAJA::sycl_atomic>::value,
              "device_atomic should map to sycl_atomic when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_atomic_explicit<RAJA::seq_atomic>,
                           RAJA::sycl_atomic_explicit<RAJA::seq_atomic>>::value,
              "device_atomic_explicit should map to sycl_atomic_explicit");
static_assert(std::is_same<RAJA::device_reduce, RAJA::sycl_reduce>::value,
              "device_reduce should map to sycl_reduce when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_launch_t<false>,
                           RAJA::sycl_launch_t<false>>::value,
              "device_launch_t should map to sycl_launch_t when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_global_size_x_direct<64>,
                           RAJA::sycl_global_2<64>>::value,
              "device_global_size_x_direct should map to sycl_global_2");
static_assert(std::is_same<RAJA::device_global_thread_x,
                           RAJA::sycl_global_item_2>::value,
              "device_global_thread_x should map to sycl_global_item_2");
static_assert(std::is_same<RAJA::device_thread_x_direct,
                           RAJA::sycl_local_2_direct>::value,
              "device_thread_x_direct should map to sycl_local_2_direct");
static_assert(std::is_same<RAJA::device_thread_x_loop,
                           RAJA::sycl_local_2_loop>::value,
              "device_thread_x_loop should map to sycl_local_2_loop");
static_assert(std::is_same<RAJA::device_block_x_loop,
                           RAJA::sycl_group_2_loop>::value,
              "device_block_x_loop should map to sycl_group_2_loop");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_direct,
                           RAJA::sycl_flatten_group_local_21_direct>::value,
              "device_flatten_block_threads_xy_direct should map to sycl flatten");
static_assert(std::is_same<RAJA::device_flatten_block_threads_xy_loop,
                           RAJA::sycl_flatten_group_local_21_loop>::value,
              "device_flatten_block_threads_xy_loop should map to sycl flatten");
#endif

}  // namespace

TEST(DevicePolicyAliases, compile_time_coverage) { SUCCEED(); }
