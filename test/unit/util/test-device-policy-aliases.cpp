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

template<typename Exec,
         typename Launch,
         typename Atomic,
         typename AtomicExplicit,
         typename Reduce,
         typename GlobalSizeXDirect,
         typename GlobalThreadX,
         typename ThreadXDirect,
         typename ThreadXLoop,
         typename BlockXLoop,
         typename FlattenBlockThreadsXYDirect,
         typename FlattenBlockThreadsXYLoop>
struct DeviceAliasChecks
{
  static_assert(std::is_same_v<RAJA::device_exec<128, false>, Exec>,
                "device_exec should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_launch_t<false>, Launch>,
                "device_launch_t should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_atomic, Atomic>,
                "device_atomic should map to the active GPU backend");
  static_assert(
      std::is_same_v<RAJA::device_atomic_explicit<RAJA::seq_atomic>,
                     AtomicExplicit>,
      "device_atomic_explicit should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_reduce, Reduce>,
                "device_reduce should map to the active GPU backend");
  static_assert(
      std::is_same_v<RAJA::device_global_size_x_direct<64>, GlobalSizeXDirect>,
      "device_global_size_x_direct should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_global_thread_x, GlobalThreadX>,
                "device_global_thread_x should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_thread_x_direct, ThreadXDirect>,
                "device_thread_x_direct should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_thread_x_loop, ThreadXLoop>,
                "device_thread_x_loop should map to the active GPU backend");
  static_assert(std::is_same_v<RAJA::device_block_x_loop, BlockXLoop>,
                "device_block_x_loop should map to the active GPU backend");
  static_assert(
      std::is_same_v<RAJA::device_flatten_block_threads_xy_direct,
                     FlattenBlockThreadsXYDirect>,
      "device_flatten_block_threads_xy_direct should map to the active GPU backend");
  static_assert(
      std::is_same_v<RAJA::device_flatten_block_threads_xy_loop,
                     FlattenBlockThreadsXYLoop>,
      "device_flatten_block_threads_xy_loop should map to the active GPU backend");
};

#if defined(RAJA_CUDA_ACTIVE)
using ActiveDeviceAliasChecks = DeviceAliasChecks<RAJA::cuda_exec<128, false>,
                                                  RAJA::cuda_launch_t<false>,
                                                  RAJA::cuda_atomic,
                                                  RAJA::cuda_atomic_explicit<RAJA::seq_atomic>,
                                                  RAJA::cuda_reduce,
                                                  RAJA::cuda_global_size_x_direct<64>,
                                                  RAJA::cuda_global_thread_x,
                                                  RAJA::cuda_thread_x_direct,
                                                  RAJA::cuda_thread_x_loop,
                                                  RAJA::cuda_block_x_loop,
                                                  RAJA::cuda_flatten_block_threads_xy_direct,
                                                  RAJA::cuda_flatten_block_threads_xy_loop>;
#elif defined(RAJA_HIP_ACTIVE)
using ActiveDeviceAliasChecks = DeviceAliasChecks<RAJA::hip_exec<128, false>,
                                                  RAJA::hip_launch_t<false>,
                                                  RAJA::hip_atomic,
                                                  RAJA::hip_atomic_explicit<RAJA::seq_atomic>,
                                                  RAJA::hip_reduce,
                                                  RAJA::hip_global_size_x_direct<64>,
                                                  RAJA::hip_global_thread_x,
                                                  RAJA::hip_thread_x_direct,
                                                  RAJA::hip_thread_x_loop,
                                                  RAJA::hip_block_x_loop,
                                                  RAJA::hip_flatten_block_threads_xy_direct,
                                                  RAJA::hip_flatten_block_threads_xy_loop>;
#elif defined(RAJA_SYCL_ACTIVE)
using ActiveDeviceAliasChecks = DeviceAliasChecks<RAJA::sycl_exec<128, false>,
                                                  RAJA::sycl_launch_t<false>,
                                                  RAJA::sycl_atomic,
                                                  RAJA::sycl_atomic_explicit<RAJA::seq_atomic>,
                                                  RAJA::sycl_reduce,
                                                  RAJA::sycl_global_2<64>,
                                                  RAJA::sycl_global_item_2,
                                                  RAJA::sycl_local_2_direct,
                                                  RAJA::sycl_local_2_loop,
                                                  RAJA::sycl_group_2_loop,
                                                  RAJA::sycl_flatten_group_local_21_direct,
                                                  RAJA::sycl_flatten_group_local_21_loop>;
#endif

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) || \
    defined(RAJA_SYCL_ACTIVE)
static_assert(sizeof(ActiveDeviceAliasChecks) > 0,
              "instantiate active device alias checks");
#endif

}  // namespace

TEST(DevicePolicyAliases, compile_time_coverage) { SUCCEED(); }
