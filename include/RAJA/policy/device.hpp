/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing device policy aliases.
 *
 *          These aliases are intended to reduce downstream preprocessor
 *          conditionals when targeting different GPU backends.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_device_HPP
#define RAJA_policy_device_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/policy.hpp"
#elif defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/policy.hpp"
#elif defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/launch.hpp"
#endif

namespace RAJA
{

namespace detail
{

template<auto... Values>
struct sycl_device_alias_unavailable
{
  static_assert(
      sizeof...(Values) < 0,
      "This device alias is not available for the active SYCL backend.");
};

}  // namespace detail

/*!
 * Generic device policy aliases.
 *
 * These aliases select the active GPU backend (CUDA/HIP/SYCL) at compile time.
 *
 * For SYCL, CUDA-like x/y/z naming is used to match CUDA/HIP conventions:
 *   x -> dim2, y -> dim1, z -> dim0
 */

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)

// Internal helper for selecting the active CUDA/HIP backend policy alias in
// this header. Selection is compile-time only, based on RAJA_CUDA_ACTIVE or
// RAJA_HIP_ACTIVE. It is undefined after this block so it is not user API.
#if defined(RAJA_CUDA_ACTIVE)
#define RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(name) cuda_##name
#elif defined(RAJA_HIP_ACTIVE)
#define RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(name) hip_##name
#endif

// forall
template<size_t BLOCK_SIZE, bool Async = false>
using device_exec =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec)<BLOCK_SIZE, Async>;

template<size_t BLOCK_SIZE>
using device_exec_async = device_exec<BLOCK_SIZE, true>;

template<size_t BLOCK_SIZE, bool Async = false>
using device_exec_with_reduce =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_with_reduce)<BLOCK_SIZE, Async>;

template<size_t BLOCK_SIZE>
using device_exec_with_reduce_async = device_exec_with_reduce<BLOCK_SIZE, true>;

template<bool with_reduce, size_t BLOCK_SIZE, bool Async = false>
using device_exec_base = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_base)<with_reduce, BLOCK_SIZE, Async>;

template<bool with_reduce, size_t BLOCK_SIZE>
using device_exec_base_async = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_base_async)<with_reduce, BLOCK_SIZE>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE, bool Async = false>
using device_exec_grid = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_grid)<BLOCK_SIZE, GRID_SIZE, Async>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE>
using device_exec_grid_async =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_grid_async)<BLOCK_SIZE, GRID_SIZE>;

template<size_t BLOCK_SIZE, bool Async = false>
using device_exec_occ_calc =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_occ_calc)<BLOCK_SIZE, Async>;

template<size_t BLOCK_SIZE>
using device_exec_occ_calc_async =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_occ_calc_async)<BLOCK_SIZE>;

template<size_t BLOCK_SIZE, bool Async = false>
using device_exec_occ_max =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_occ_max)<BLOCK_SIZE, Async>;

template<size_t BLOCK_SIZE>
using device_exec_occ_max_async =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(exec_occ_max_async)<BLOCK_SIZE>;

template<size_t BLOCK_SIZE, typename Fraction, bool Async = false>
using device_exec_occ_fraction = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_occ_fraction)<BLOCK_SIZE, Fraction, Async>;

template<size_t BLOCK_SIZE, typename Fraction>
using device_exec_occ_fraction_async = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_occ_fraction_async)<BLOCK_SIZE, Fraction>;

template<size_t BLOCK_SIZE, typename Concretizer, bool Async = false>
using device_exec_occ_custom = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_occ_custom)<BLOCK_SIZE, Concretizer, Async>;

template<size_t BLOCK_SIZE, typename Concretizer>
using device_exec_occ_custom_async = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    exec_occ_custom_async)<BLOCK_SIZE, Concretizer>;

// reducers and atomics
using device_atomic = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(atomic);
template<typename host_policy>
using device_atomic_explicit =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(atomic_explicit)<host_policy>;
using device_reduce        = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(reduce);
using device_reduce_atomic = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(reduce_atomic);
template<bool with_atomic>
using device_reduce_base =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(reduce_base)<with_atomic>;
using device_multi_reduce_atomic =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(multi_reduce_atomic);
using device_multi_reduce_atomic_low_performance_low_overhead =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        multi_reduce_atomic_low_performance_low_overhead);

// launch
template<bool Async, int num_threads = RAJA::named_usage::unspecified>
using device_launch_t =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(launch_t)<Async, num_threads>;

// kernel (For) index mapping
template<int nx_threads>
using device_global_size_x_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_x_direct)<nx_threads>;
template<int ny_threads>
using device_global_size_y_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_y_direct)<ny_threads>;
template<int nz_threads>
using device_global_size_z_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_z_direct)<nz_threads>;

template<int nx_threads>
using device_global_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        global_size_x_direct_unchecked)<nx_threads>;
template<int ny_threads>
using device_global_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        global_size_y_direct_unchecked)<ny_threads>;
template<int nz_threads>
using device_global_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        global_size_z_direct_unchecked)<nz_threads>;

template<int nx_threads>
using device_global_size_x_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_x_loop)<nx_threads>;
template<int ny_threads>
using device_global_size_y_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_y_loop)<ny_threads>;
template<int nz_threads>
using device_global_size_z_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_size_z_loop)<nz_threads>;

// launch (loop) index mapping
using device_global_thread_x =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_thread_x);
using device_global_thread_y =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_thread_y);
using device_global_thread_z =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(global_thread_z);

// kernel (loop) index mapping
using device_thread_x_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_x_direct);
using device_thread_y_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_y_direct);
using device_thread_z_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_z_direct);

using device_thread_x_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_x_loop);
using device_thread_y_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_y_loop);
using device_thread_z_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_z_loop);

using device_block_x_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_x_direct);
using device_block_y_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_y_direct);
using device_block_z_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_z_direct);

using device_block_x_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_x_loop);
using device_block_y_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_y_loop);
using device_block_z_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_z_loop);

template<int X_SIZE>
using device_thread_size_x_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_x_direct)<X_SIZE>;
template<int Y_SIZE>
using device_thread_size_y_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_y_direct)<Y_SIZE>;
template<int Z_SIZE>
using device_thread_size_z_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_z_direct)<Z_SIZE>;

template<int X_SIZE>
using device_thread_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_x_direct_unchecked)<X_SIZE>;
template<int Y_SIZE>
using device_thread_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_y_direct_unchecked)<Y_SIZE>;
template<int Z_SIZE>
using device_thread_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_z_direct_unchecked)<Z_SIZE>;

template<int X_SIZE>
using device_thread_size_x_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_x_loop)<X_SIZE>;
template<int Y_SIZE>
using device_thread_size_y_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_y_loop)<Y_SIZE>;
template<int Z_SIZE>
using device_thread_size_z_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(thread_size_z_loop)<Z_SIZE>;

template<int X_SIZE>
using device_block_size_x_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_x_direct)<X_SIZE>;
template<int Y_SIZE>
using device_block_size_y_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_y_direct)<Y_SIZE>;
template<int Z_SIZE>
using device_block_size_z_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_z_direct)<Z_SIZE>;

template<int X_SIZE>
using device_block_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_x_direct_unchecked)<X_SIZE>;
template<int Y_SIZE>
using device_block_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_y_direct_unchecked)<Y_SIZE>;
template<int Z_SIZE>
using device_block_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_z_direct_unchecked)<Z_SIZE>;

template<int X_SIZE>
using device_block_size_x_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_x_loop)<X_SIZE>;
template<int Y_SIZE>
using device_block_size_y_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_y_loop)<Y_SIZE>;
template<int Z_SIZE>
using device_block_size_z_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(block_size_z_loop)<Z_SIZE>;

using device_flatten_block_threads_xy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xy_direct);
using device_flatten_block_threads_xz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xz_direct);
using device_flatten_block_threads_yx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yx_direct);
using device_flatten_block_threads_yz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yz_direct);
using device_flatten_block_threads_zx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zx_direct);
using device_flatten_block_threads_zy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zy_direct);
using device_flatten_block_threads_xyz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xyz_direct);
using device_flatten_block_threads_xzy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xzy_direct);
using device_flatten_block_threads_yxz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yxz_direct);
using device_flatten_block_threads_yzx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yzx_direct);
using device_flatten_block_threads_zxy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zxy_direct);
using device_flatten_block_threads_zyx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zyx_direct);

using device_flatten_block_threads_xy_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xy_loop);
using device_flatten_block_threads_xz_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xz_loop);
using device_flatten_block_threads_yx_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yx_loop);
using device_flatten_block_threads_yz_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yz_loop);
using device_flatten_block_threads_zx_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zx_loop);
using device_flatten_block_threads_zy_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zy_loop);
using device_flatten_block_threads_xyz_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xyz_loop);
using device_flatten_block_threads_xzy_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_xzy_loop);
using device_flatten_block_threads_yxz_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yxz_loop);
using device_flatten_block_threads_yzx_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_yzx_loop);
using device_flatten_block_threads_zxy_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zxy_loop);
using device_flatten_block_threads_zyx_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_threads_zyx_loop);

template<int nx_threads>
using device_flatten_thread_size_x_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_x_direct)<nx_threads>;
template<int ny_threads>
using device_flatten_thread_size_y_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_y_direct)<ny_threads>;
template<int nz_threads>
using device_flatten_thread_size_z_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_z_direct)<nz_threads>;

template<int nx_threads>
using device_flatten_thread_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_x_direct_unchecked)<nx_threads>;
template<int ny_threads>
using device_flatten_thread_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_y_direct_unchecked)<ny_threads>;
template<int nz_threads>
using device_flatten_thread_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_z_direct_unchecked)<nz_threads>;

template<int nx_threads>
using device_flatten_thread_size_x_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_thread_size_x_loop)<nx_threads>;
template<int ny_threads>
using device_flatten_thread_size_y_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_thread_size_y_loop)<ny_threads>;
template<int nz_threads>
using device_flatten_thread_size_z_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_thread_size_z_loop)<nz_threads>;

template<int X_SIZE, int Y_SIZE>
using device_flatten_thread_size_xy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_xy_direct)<X_SIZE, Y_SIZE>;
template<int X_SIZE, int Z_SIZE>
using device_flatten_thread_size_xz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_xz_direct)<X_SIZE, Z_SIZE>;
template<int Y_SIZE, int X_SIZE>
using device_flatten_thread_size_yx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_yx_direct)<Y_SIZE, X_SIZE>;
template<int Y_SIZE, int Z_SIZE>
using device_flatten_thread_size_yz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_yz_direct)<Y_SIZE, Z_SIZE>;
template<int Z_SIZE, int X_SIZE>
using device_flatten_thread_size_zx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_zx_direct)<Z_SIZE, X_SIZE>;
template<int Z_SIZE, int Y_SIZE>
using device_flatten_thread_size_zy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_zy_direct)<Z_SIZE, Y_SIZE>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using device_flatten_thread_size_xyz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_xyz_direct)<X_SIZE, Y_SIZE, Z_SIZE>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using device_flatten_thread_size_xzy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_xzy_direct)<X_SIZE, Z_SIZE, Y_SIZE>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using device_flatten_thread_size_yxz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_yxz_direct)<Y_SIZE, X_SIZE, Z_SIZE>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using device_flatten_thread_size_yzx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_yzx_direct)<Y_SIZE, Z_SIZE, X_SIZE>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using device_flatten_thread_size_zxy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_zxy_direct)<Z_SIZE, X_SIZE, Y_SIZE>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using device_flatten_thread_size_zyx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_thread_size_zyx_direct)<Z_SIZE, Y_SIZE, X_SIZE>;

template<int X_SIZE, int Y_SIZE>
using device_flatten_thread_size_xy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_xy_loop)<X_SIZE, Y_SIZE>;
template<int X_SIZE, int Z_SIZE>
using device_flatten_thread_size_xz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_xz_loop)<X_SIZE, Z_SIZE>;
template<int Y_SIZE, int X_SIZE>
using device_flatten_thread_size_yx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_yx_loop)<Y_SIZE, X_SIZE>;
template<int Y_SIZE, int Z_SIZE>
using device_flatten_thread_size_yz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_yz_loop)<Y_SIZE, Z_SIZE>;
template<int Z_SIZE, int X_SIZE>
using device_flatten_thread_size_zx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_zx_loop)<Z_SIZE, X_SIZE>;
template<int Z_SIZE, int Y_SIZE>
using device_flatten_thread_size_zy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_zy_loop)<Z_SIZE, Y_SIZE>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using device_flatten_thread_size_xyz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_xyz_loop)<X_SIZE, Y_SIZE, Z_SIZE>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using device_flatten_thread_size_xzy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_xzy_loop)<X_SIZE, Z_SIZE, Y_SIZE>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using device_flatten_thread_size_yxz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_yxz_loop)<Y_SIZE, X_SIZE, Z_SIZE>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using device_flatten_thread_size_yzx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_yzx_loop)<Y_SIZE, Z_SIZE, X_SIZE>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using device_flatten_thread_size_zxy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_zxy_loop)<Z_SIZE, X_SIZE, Y_SIZE>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using device_flatten_thread_size_zyx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_thread_size_zyx_loop)<Z_SIZE, Y_SIZE, X_SIZE>;

template<int nx_threads>
using device_flatten_block_size_x_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_x_direct)<nx_threads>;
template<int ny_threads>
using device_flatten_block_size_y_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_y_direct)<ny_threads>;
template<int nz_threads>
using device_flatten_block_size_z_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_z_direct)<nz_threads>;

template<int nx_threads>
using device_flatten_block_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_x_direct_unchecked)<nx_threads>;
template<int ny_threads>
using device_flatten_block_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_y_direct_unchecked)<ny_threads>;
template<int nz_threads>
using device_flatten_block_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_z_direct_unchecked)<nz_threads>;

template<int nx_threads>
using device_flatten_block_size_x_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_size_x_loop)<nx_threads>;
template<int ny_threads>
using device_flatten_block_size_y_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_size_y_loop)<ny_threads>;
template<int nz_threads>
using device_flatten_block_size_z_loop =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(flatten_block_size_z_loop)<nz_threads>;

template<int X_SIZE, int Y_SIZE>
using device_flatten_block_size_xy_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xy_direct)<X_SIZE, Y_SIZE>;
template<int X_SIZE, int Z_SIZE>
using device_flatten_block_size_xz_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xz_direct)<X_SIZE, Z_SIZE>;
template<int Y_SIZE, int X_SIZE>
using device_flatten_block_size_yx_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yx_direct)<Y_SIZE, X_SIZE>;
template<int Y_SIZE, int Z_SIZE>
using device_flatten_block_size_yz_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yz_direct)<Y_SIZE, Z_SIZE>;
template<int Z_SIZE, int X_SIZE>
using device_flatten_block_size_zx_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zx_direct)<Z_SIZE, X_SIZE>;
template<int Z_SIZE, int Y_SIZE>
using device_flatten_block_size_zy_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zy_direct)<Z_SIZE, Y_SIZE>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using device_flatten_block_size_xyz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_xyz_direct)<X_SIZE, Y_SIZE, Z_SIZE>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using device_flatten_block_size_xzy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_xzy_direct)<X_SIZE, Z_SIZE, Y_SIZE>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using device_flatten_block_size_yxz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_yxz_direct)<Y_SIZE, X_SIZE, Z_SIZE>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using device_flatten_block_size_yzx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_yzx_direct)<Y_SIZE, Z_SIZE, X_SIZE>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using device_flatten_block_size_zxy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_zxy_direct)<Z_SIZE, X_SIZE, Y_SIZE>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using device_flatten_block_size_zyx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_block_size_zyx_direct)<Z_SIZE, Y_SIZE, X_SIZE>;

template<int X_SIZE, int Y_SIZE>
using device_flatten_block_size_xy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xy_loop)<X_SIZE, Y_SIZE>;
template<int X_SIZE, int Z_SIZE>
using device_flatten_block_size_xz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xz_loop)<X_SIZE, Z_SIZE>;
template<int Y_SIZE, int X_SIZE>
using device_flatten_block_size_yx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yx_loop)<Y_SIZE, X_SIZE>;
template<int Y_SIZE, int Z_SIZE>
using device_flatten_block_size_yz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yz_loop)<Y_SIZE, Z_SIZE>;
template<int Z_SIZE, int X_SIZE>
using device_flatten_block_size_zx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zx_loop)<Z_SIZE, X_SIZE>;
template<int Z_SIZE, int Y_SIZE>
using device_flatten_block_size_zy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zy_loop)<Z_SIZE, Y_SIZE>;
template<int X_SIZE, int Y_SIZE, int Z_SIZE>
using device_flatten_block_size_xyz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xyz_loop)<X_SIZE, Y_SIZE, Z_SIZE>;
template<int X_SIZE, int Z_SIZE, int Y_SIZE>
using device_flatten_block_size_xzy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_xzy_loop)<X_SIZE, Z_SIZE, Y_SIZE>;
template<int Y_SIZE, int X_SIZE, int Z_SIZE>
using device_flatten_block_size_yxz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yxz_loop)<Y_SIZE, X_SIZE, Z_SIZE>;
template<int Y_SIZE, int Z_SIZE, int X_SIZE>
using device_flatten_block_size_yzx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_yzx_loop)<Y_SIZE, Z_SIZE, X_SIZE>;
template<int Z_SIZE, int X_SIZE, int Y_SIZE>
using device_flatten_block_size_zxy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zxy_loop)<Z_SIZE, X_SIZE, Y_SIZE>;
template<int Z_SIZE, int Y_SIZE, int X_SIZE>
using device_flatten_block_size_zyx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_block_size_zyx_loop)<Z_SIZE, Y_SIZE, X_SIZE>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_x_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_x_direct)<X_BLOCK_SIZE, X_GRID_SIZE>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_y_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_y_direct)<Y_BLOCK_SIZE, Y_GRID_SIZE>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_z_direct = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_z_direct)<Z_BLOCK_SIZE, Z_GRID_SIZE>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_x_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_x_direct_unchecked)<X_BLOCK_SIZE, X_GRID_SIZE>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_y_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_y_direct_unchecked)<Y_BLOCK_SIZE, Y_GRID_SIZE>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_z_direct_unchecked =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_z_direct_unchecked)<Z_BLOCK_SIZE, Z_GRID_SIZE>;

template<int X_BLOCK_SIZE, int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_x_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_x_loop)<X_BLOCK_SIZE, X_GRID_SIZE>;
template<int Y_BLOCK_SIZE, int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_y_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_y_loop)<Y_BLOCK_SIZE, Y_GRID_SIZE>;
template<int Z_BLOCK_SIZE, int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_z_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_z_loop)<Z_BLOCK_SIZE, Z_GRID_SIZE>;

template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_xy_direct)<X_BLOCK_SIZE,
                                       Y_BLOCK_SIZE,
                                       X_GRID_SIZE,
                                       Y_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_xz_direct)<X_BLOCK_SIZE,
                                       Z_BLOCK_SIZE,
                                       X_GRID_SIZE,
                                       Z_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_yx_direct)<Y_BLOCK_SIZE,
                                       X_BLOCK_SIZE,
                                       Y_GRID_SIZE,
                                       X_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_yz_direct)<Y_BLOCK_SIZE,
                                       Z_BLOCK_SIZE,
                                       Y_GRID_SIZE,
                                       Z_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_zx_direct)<Z_BLOCK_SIZE,
                                       X_BLOCK_SIZE,
                                       Z_GRID_SIZE,
                                       X_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_zy_direct)<Z_BLOCK_SIZE,
                                       Y_BLOCK_SIZE,
                                       Z_GRID_SIZE,
                                       Y_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xyz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_xyz_direct)<X_BLOCK_SIZE,
                                        Y_BLOCK_SIZE,
                                        Z_BLOCK_SIZE,
                                        X_GRID_SIZE,
                                        Y_GRID_SIZE,
                                        Z_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xzy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_xzy_direct)<X_BLOCK_SIZE,
                                        Z_BLOCK_SIZE,
                                        Y_BLOCK_SIZE,
                                        X_GRID_SIZE,
                                        Z_GRID_SIZE,
                                        Y_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yxz_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_yxz_direct)<Y_BLOCK_SIZE,
                                        X_BLOCK_SIZE,
                                        Z_BLOCK_SIZE,
                                        Y_GRID_SIZE,
                                        X_GRID_SIZE,
                                        Z_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yzx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_yzx_direct)<Y_BLOCK_SIZE,
                                        Z_BLOCK_SIZE,
                                        X_BLOCK_SIZE,
                                        Y_GRID_SIZE,
                                        Z_GRID_SIZE,
                                        X_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zxy_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_zxy_direct)<Z_BLOCK_SIZE,
                                        X_BLOCK_SIZE,
                                        Y_BLOCK_SIZE,
                                        Z_GRID_SIZE,
                                        X_GRID_SIZE,
                                        Y_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zyx_direct =
    RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
        flatten_global_size_zyx_direct)<Z_BLOCK_SIZE,
                                        Y_BLOCK_SIZE,
                                        X_BLOCK_SIZE,
                                        Z_GRID_SIZE,
                                        Y_GRID_SIZE,
                                        X_GRID_SIZE>;

template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_xy_loop)<X_BLOCK_SIZE,
                                 Y_BLOCK_SIZE,
                                 X_GRID_SIZE,
                                 Y_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_xz_loop)<X_BLOCK_SIZE,
                                 Z_BLOCK_SIZE,
                                 X_GRID_SIZE,
                                 Z_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_yx_loop)<Y_BLOCK_SIZE,
                                 X_BLOCK_SIZE,
                                 Y_GRID_SIZE,
                                 X_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_yz_loop)<Y_BLOCK_SIZE,
                                 Z_BLOCK_SIZE,
                                 Y_GRID_SIZE,
                                 Z_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_zx_loop)<Z_BLOCK_SIZE,
                                 X_BLOCK_SIZE,
                                 Z_GRID_SIZE,
                                 X_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_zy_loop)<Z_BLOCK_SIZE,
                                 Y_BLOCK_SIZE,
                                 Z_GRID_SIZE,
                                 Y_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xyz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_xyz_loop)<X_BLOCK_SIZE,
                                  Y_BLOCK_SIZE,
                                  Z_BLOCK_SIZE,
                                  X_GRID_SIZE,
                                  Y_GRID_SIZE,
                                  Z_GRID_SIZE>;
template<int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_xzy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_xzy_loop)<X_BLOCK_SIZE,
                                  Z_BLOCK_SIZE,
                                  Y_BLOCK_SIZE,
                                  X_GRID_SIZE,
                                  Z_GRID_SIZE,
                                  Y_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yxz_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_yxz_loop)<Y_BLOCK_SIZE,
                                  X_BLOCK_SIZE,
                                  Z_BLOCK_SIZE,
                                  Y_GRID_SIZE,
                                  X_GRID_SIZE,
                                  Z_GRID_SIZE>;
template<int Y_BLOCK_SIZE,
         int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_yzx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_yzx_loop)<Y_BLOCK_SIZE,
                                  Z_BLOCK_SIZE,
                                  X_BLOCK_SIZE,
                                  Y_GRID_SIZE,
                                  Z_GRID_SIZE,
                                  X_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zxy_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_zxy_loop)<Z_BLOCK_SIZE,
                                  X_BLOCK_SIZE,
                                  Y_BLOCK_SIZE,
                                  Z_GRID_SIZE,
                                  X_GRID_SIZE,
                                  Y_GRID_SIZE>;
template<int Z_BLOCK_SIZE,
         int Y_BLOCK_SIZE,
         int X_BLOCK_SIZE,
         int Z_GRID_SIZE = RAJA::named_usage::unspecified,
         int Y_GRID_SIZE = RAJA::named_usage::unspecified,
         int X_GRID_SIZE = RAJA::named_usage::unspecified>
using device_flatten_global_size_zyx_loop = RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS(
    flatten_global_size_zyx_loop)<Z_BLOCK_SIZE,
                                  Y_BLOCK_SIZE,
                                  X_BLOCK_SIZE,
                                  Z_GRID_SIZE,
                                  Y_GRID_SIZE,
                                  X_GRID_SIZE>;

#undef RAJA_INTERNAL_CUDA_HIP_POLICY_ALIAS

#elif defined(RAJA_SYCL_ACTIVE)

// SYCL currently exposes only the device aliases that have direct backend
// equivalents. Unsupported aliases remain declared as compile-time errors so
// downstream code fails immediately if it relies on a CUDA/HIP-only policy.

// forall
template<size_t WORK_GROUP_SIZE, bool Async = false>
using device_exec = RAJA::sycl_exec<WORK_GROUP_SIZE, Async>;

template<size_t WORK_GROUP_SIZE>
using device_exec_async = device_exec<WORK_GROUP_SIZE, true>;

template<size_t WORK_GROUP_SIZE, bool Async = false>
using device_exec_with_reduce = detail::sycl_device_alias_unavailable<>;

template<size_t WORK_GROUP_SIZE>
using device_exec_with_reduce_async = detail::sycl_device_alias_unavailable<>;

template<bool with_reduce, size_t BLOCK_SIZE, bool Async = false>
using device_exec_base = detail::sycl_device_alias_unavailable<>;

template<bool with_reduce, size_t BLOCK_SIZE>
using device_exec_base_async = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE, bool Async = false>
using device_exec_grid = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, size_t GRID_SIZE>
using device_exec_grid_async = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, bool Async = false>
using device_exec_occ_calc = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE>
using device_exec_occ_calc_async = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, bool Async = false>
using device_exec_occ_max = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE>
using device_exec_occ_max_async = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, typename Fraction, bool Async = false>
using device_exec_occ_fraction = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, typename Fraction>
using device_exec_occ_fraction_async = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, typename Concretizer, bool Async = false>
using device_exec_occ_custom = detail::sycl_device_alias_unavailable<>;

template<size_t BLOCK_SIZE, typename Concretizer>
using device_exec_occ_custom_async = detail::sycl_device_alias_unavailable<>;

// reducers and atomics
using device_atomic = RAJA::sycl_atomic;
template<typename host_policy>
using device_atomic_explicit = RAJA::sycl_atomic_explicit<host_policy>;
using device_reduce          = RAJA::sycl_reduce;
using device_reduce_atomic   = detail::sycl_device_alias_unavailable<>;

template<bool with_atomic>
using device_reduce_base = detail::sycl_device_alias_unavailable<>;

using device_multi_reduce_atomic = detail::sycl_device_alias_unavailable<>;
using device_multi_reduce_atomic_low_performance_low_overhead =
    detail::sycl_device_alias_unavailable<>;

// launch
template<bool Async, int num_threads = RAJA::named_usage::unspecified>
using device_launch_t = RAJA::sycl_launch_t<Async, num_threads>;

// kernel (For) index mapping (x/y/z -> dim2/dim1/dim0)
template<int nx_threads>
using device_global_size_x_direct = RAJA::sycl_global_2<nx_threads>;
template<int ny_threads>
using device_global_size_y_direct = RAJA::sycl_global_1<ny_threads>;
template<int nz_threads>
using device_global_size_z_direct = RAJA::sycl_global_0<nz_threads>;

template<int nx_threads>
using device_global_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_global_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_global_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_global_size_x_loop = detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_global_size_y_loop = detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_global_size_z_loop = detail::sycl_device_alias_unavailable<>;

// launch (loop) index mapping (x/y/z -> dim2/dim1/dim0)
using device_global_thread_x = RAJA::sycl_global_item_2;
using device_global_thread_y = RAJA::sycl_global_item_1;
using device_global_thread_z = RAJA::sycl_global_item_0;

// kernel (loop) index mapping (x/y/z -> dim2/dim1/dim0)
using device_thread_x_direct = RAJA::sycl_local_2_direct;
using device_thread_y_direct = RAJA::sycl_local_1_direct;
using device_thread_z_direct = RAJA::sycl_local_0_direct;

using device_thread_x_loop = RAJA::sycl_local_2_loop;
using device_thread_y_loop = RAJA::sycl_local_1_loop;
using device_thread_z_loop = RAJA::sycl_local_0_loop;

using device_block_x_direct = RAJA::sycl_group_2_direct;
using device_block_y_direct = RAJA::sycl_group_1_direct;
using device_block_z_direct = RAJA::sycl_group_0_direct;

using device_block_x_loop = RAJA::sycl_group_2_loop;
using device_block_y_loop = RAJA::sycl_group_1_loop;
using device_block_z_loop = RAJA::sycl_group_0_loop;

template<int X_SIZE>
using device_thread_size_x_direct = detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_thread_size_y_direct = detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_thread_size_z_direct = detail::sycl_device_alias_unavailable<>;

template<int X_SIZE>
using device_thread_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_thread_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_thread_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int X_SIZE>
using device_thread_size_x_loop = detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_thread_size_y_loop = detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_thread_size_z_loop = detail::sycl_device_alias_unavailable<>;

template<int X_SIZE>
using device_block_size_x_direct = detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_block_size_y_direct = detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_block_size_z_direct = detail::sycl_device_alias_unavailable<>;

template<int X_SIZE>
using device_block_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_block_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_block_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int X_SIZE>
using device_block_size_x_loop = detail::sycl_device_alias_unavailable<>;
template<int Y_SIZE>
using device_block_size_y_loop = detail::sycl_device_alias_unavailable<>;
template<int Z_SIZE>
using device_block_size_z_loop = detail::sycl_device_alias_unavailable<>;

using device_flatten_block_threads_xy_direct =
    RAJA::sycl_flatten_group_local_21_direct;
using device_flatten_block_threads_xz_direct =
    RAJA::sycl_flatten_group_local_20_direct;
using device_flatten_block_threads_yx_direct =
    RAJA::sycl_flatten_group_local_12_direct;
using device_flatten_block_threads_yz_direct =
    RAJA::sycl_flatten_group_local_10_direct;
using device_flatten_block_threads_zx_direct =
    RAJA::sycl_flatten_group_local_02_direct;
using device_flatten_block_threads_zy_direct =
    RAJA::sycl_flatten_group_local_01_direct;
using device_flatten_block_threads_xyz_direct =
    RAJA::sycl_flatten_group_local_210_direct;
using device_flatten_block_threads_xzy_direct =
    RAJA::sycl_flatten_group_local_201_direct;
using device_flatten_block_threads_yxz_direct =
    RAJA::sycl_flatten_group_local_120_direct;
using device_flatten_block_threads_yzx_direct =
    RAJA::sycl_flatten_group_local_102_direct;
using device_flatten_block_threads_zxy_direct =
    RAJA::sycl_flatten_group_local_021_direct;
using device_flatten_block_threads_zyx_direct =
    RAJA::sycl_flatten_group_local_012_direct;

using device_flatten_block_threads_xy_loop =
    RAJA::sycl_flatten_group_local_21_loop;
using device_flatten_block_threads_xz_loop =
    RAJA::sycl_flatten_group_local_20_loop;
using device_flatten_block_threads_yx_loop =
    RAJA::sycl_flatten_group_local_12_loop;
using device_flatten_block_threads_yz_loop =
    RAJA::sycl_flatten_group_local_10_loop;
using device_flatten_block_threads_zx_loop =
    RAJA::sycl_flatten_group_local_02_loop;
using device_flatten_block_threads_zy_loop =
    RAJA::sycl_flatten_group_local_01_loop;
using device_flatten_block_threads_xyz_loop =
    RAJA::sycl_flatten_group_local_210_loop;
using device_flatten_block_threads_xzy_loop =
    RAJA::sycl_flatten_group_local_201_loop;
using device_flatten_block_threads_yxz_loop =
    RAJA::sycl_flatten_group_local_120_loop;
using device_flatten_block_threads_yzx_loop =
    RAJA::sycl_flatten_group_local_102_loop;
using device_flatten_block_threads_zxy_loop =
    RAJA::sycl_flatten_group_local_021_loop;
using device_flatten_block_threads_zyx_loop =
    RAJA::sycl_flatten_group_local_012_loop;

template<int nx_threads>
using device_flatten_thread_size_x_direct =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_thread_size_y_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_thread_size_z_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_thread_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_thread_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_thread_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_thread_size_x_loop =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_thread_size_y_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_thread_size_z_loop =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_thread_size_xy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xyz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xzy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yxz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yzx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zxy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zyx_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_thread_size_xy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xyz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_xzy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yxz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_yzx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zxy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_thread_size_zyx_loop =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_block_size_x_direct =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_block_size_y_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_block_size_z_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_block_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_block_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_block_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_block_size_x_loop =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_block_size_y_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_block_size_z_loop =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_block_size_xy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xyz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xzy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yxz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yzx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zxy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zyx_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_block_size_xy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xyz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_xzy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yxz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_yzx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zxy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_block_size_zyx_loop =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_global_size_x_direct =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_global_size_y_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_global_size_z_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_global_size_x_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_global_size_y_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_global_size_z_direct_unchecked =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_global_size_x_loop =
    detail::sycl_device_alias_unavailable<>;
template<int ny_threads>
using device_flatten_global_size_y_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nz_threads>
using device_flatten_global_size_z_loop =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_global_size_xy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xyz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xzy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yxz_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yzx_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zxy_direct =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zyx_direct =
    detail::sycl_device_alias_unavailable<>;

template<int nx_threads>
using device_flatten_global_size_xy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xyz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_xzy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yxz_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_yzx_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zxy_loop =
    detail::sycl_device_alias_unavailable<>;
template<int nx_threads>
using device_flatten_global_size_zyx_loop =
    detail::sycl_device_alias_unavailable<>;

#endif  // active backend

}  // namespace RAJA

#endif  // RAJA_policy_device_HPP
