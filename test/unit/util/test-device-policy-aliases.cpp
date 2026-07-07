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

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)

static_assert(std::is_same<RAJA::device_exec<128, false>,

                           RAJA::RAJA_DEVICE_ALIAS(exec)<128, false>>::value,
              "device_exec should map to the active GPU backend");
static_assert(std::is_same<RAJA::device_launch_t<false>,
                           RAJA::RAJA_DEVICE_ALIAS(launch_t)<false>>::value,
              "device_launch_t should map to the active GPU backend");
static_assert(std::is_same<RAJA::device_global_size_x_direct<64>,
                           RAJA::RAJA_DEVICE_ALIAS(global_size_x_direct)<64>>::value,
              "device_global_size_x_direct should map to the active GPU backend");
static_assert(std::is_same<RAJA::device_thread_x_direct,
                           RAJA::RAJA_DEVICE_ALIAS(thread_x_direct)>::value,
              "device_thread_x_direct should map to the active GPU backend");
static_assert(std::is_same<RAJA::device_block_x_loop,
                           RAJA::RAJA_DEVICE_ALIAS(block_x_loop)>::value,
              "device_block_x_loop should map to the active GPU backend");

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
