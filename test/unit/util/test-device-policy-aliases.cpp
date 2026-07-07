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

#if defined(RAJA_CUDA_ACTIVE)
#define RAJA_DEVICE_BACKEND_PREFIX cuda
#elif defined(RAJA_HIP_ACTIVE)
#define RAJA_DEVICE_BACKEND_PREFIX hip
#endif

#define RAJA_DEVICE_CONCAT_IMPL(prefix, suffix) prefix##suffix
#define RAJA_DEVICE_CONCAT(prefix, suffix) RAJA_DEVICE_CONCAT_IMPL(prefix, suffix)
#define RAJA_DEVICE_ALIAS(name) RAJA_DEVICE_CONCAT(RAJA_DEVICE_BACKEND_PREFIX, _##name)

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

#undef RAJA_DEVICE_ALIAS
#undef RAJA_DEVICE_CONCAT
#undef RAJA_DEVICE_CONCAT_IMPL
#undef RAJA_DEVICE_BACKEND_PREFIX

#elif defined(RAJA_SYCL_ACTIVE)
static_assert(std::is_same<RAJA::device_exec<128, false>,
                           RAJA::sycl_exec<128, false>>::value,
              "device_exec should map to sycl_exec when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_launch_t<false>,
                           RAJA::sycl_launch_t<false>>::value,
              "device_launch_t should map to sycl_launch_t when RAJA_SYCL_ACTIVE");
static_assert(std::is_same<RAJA::device_global_size_x_direct<64>,
                           RAJA::sycl_global_2<64>>::value,
              "device_global_size_x_direct should map to sycl_global_2");
static_assert(std::is_same<RAJA::device_thread_x_direct,
                           RAJA::sycl_local_2_direct>::value,
              "device_thread_x_direct should map to sycl_local_2_direct");
static_assert(std::is_same<RAJA::device_block_x_loop,
                           RAJA::sycl_group_2_loop>::value,
              "device_block_x_loop should map to sycl_group_2_loop");
#endif

}  // namespace

TEST(DevicePolicyAliases, compile_time_coverage) { SUCCEED(); }
