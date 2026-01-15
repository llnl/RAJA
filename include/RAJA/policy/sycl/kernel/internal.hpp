/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with SYCL.
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

#ifndef RAJA_policy_sycl_kernel_internal_HPP
#define RAJA_policy_sycl_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{

namespace internal
{

// LaunchDims and Helper functions
struct LaunchDims
{
  sycl_dim_3_t group;
  sycl_dim_3_t local;
  sycl_dim_3_t global;
  sycl_dim_3_t min_groups;
  sycl_dim_3_t min_locals;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  LaunchDims()
      : group {0, 0, 0},
        local {1, 1, 1},
        global {1, 1, 1},
        min_groups {0, 0, 0},
        min_locals {0, 0, 0}
  {}

  RAJA_INLINE

  RAJA_HOST_DEVICE
  LaunchDims(LaunchDims const& c)
      : group(c.group),
        local(c.local),
        global(c.global)
  {}

  RAJA_INLINE
  LaunchDims max(LaunchDims const& c) const
  {
    LaunchDims result;

    result.group.x = std::max(c.group.x, group.x);
    result.group.y = std::max(c.group.y, group.y);
    result.group.z = std::max(c.group.z, group.z);

    result.local.x = std::max(c.local.x, local.x);
    result.local.y = std::max(c.local.y, local.y);
    result.local.z = std::max(c.local.z, local.z);

    result.global.x = std::max(c.global.x, global.x);
    result.global.y = std::max(c.global.y, global.y);
    result.global.z = std::max(c.global.z, global.z);

    return result;
  }

  ::sycl::nd_range<3> fit_nd_range(::sycl::queue* q)
  {

    sycl_dim_3_t launch_global;

    sycl_dim_3_t launch_local {1, 1, 1};
    launch_local.x = std::max(launch_local.x, local.x);
    launch_local.y = std::max(launch_local.y, local.y);
    launch_local.z = std::max(launch_local.z, local.z);

    ::sycl::device dev = q->get_device();

    auto max_work_group_size =
        dev.get_info<::sycl::info::device::max_work_group_size>();

    if (launch_local.x > max_work_group_size)
    {
      launch_local.x = max_work_group_size;
    }
    if (launch_local.y > max_work_group_size)
    {
      launch_local.y = max_work_group_size;
    }
    if (launch_local.z > max_work_group_size)
    {
      launch_local.z = max_work_group_size;
    }


    // Make sure the multiple of locals fits
    // Prefer larger z -> y -> x
    if (launch_local.x * launch_local.y * launch_local.z > max_work_group_size)
    {
      unsigned long remaining = 1;
      // local z cannot be > max_wrk from above
      // if equal then remaining is 1, on handle <
      if (max_work_group_size > launch_local.z)
      {
        // keep local z
        remaining = max_work_group_size / launch_local.z;
      }
      if (remaining >= launch_local.y)
      {
        // keep local y
        remaining = remaining / launch_local.y;
      }
      else
      {
        launch_local.y = remaining;
        remaining      = remaining / launch_local.y;
      }
      if (remaining < launch_local.x)
      {
        launch_local.x = remaining;
      }
    }


    // User gave group policy, use to calculate global space
    if (group.x != 0 || group.y != 0 || group.z != 0)
    {
      sycl_dim_3_t launch_group {1, 1, 1};
      launch_group.x = std::max(launch_group.x, group.x);
      launch_group.y = std::max(launch_group.y, group.y);
      launch_group.z = std::max(launch_group.z, group.z);

      launch_global.x = launch_local.x * launch_group.x;
      launch_global.y = launch_local.y * launch_group.y;
      launch_global.z = launch_local.z * launch_group.z;
    }
    else
    {
      launch_global.x =
          launch_local.x * ((global.x + (launch_local.x - 1)) / launch_local.x);
      launch_global.y =
          launch_local.y * ((global.y + (launch_local.y - 1)) / launch_local.y);
      launch_global.z =
          launch_local.z * ((global.z + (launch_local.z - 1)) / launch_local.z);
    }


    if (launch_global.x % launch_local.x != 0)
    {
      launch_global.x =
          ((launch_global.x / launch_local.x) + 1) * launch_local.x;
    }
    if (launch_global.y % launch_local.y != 0)
    {
      launch_global.y =
          ((launch_global.y / launch_local.y) + 1) * launch_local.y;
    }
    if (launch_global.z % launch_local.z != 0)
    {
      launch_global.z =
          ((launch_global.z / launch_local.z) + 1) * launch_local.z;
    }

    ::sycl::range<3> ret_th = {launch_local.x, launch_local.y, launch_local.z};
    ::sycl::range<3> ret_gl = {launch_global.x, launch_global.y,
                               launch_global.z};

    return ::sycl::nd_range<3>(ret_gl, ret_th);
  }
};

template<camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct SyclStatementListExecutorHelper
{

  using next_helper_t =
      SyclStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;

  template<typename Data>
  inline static RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Execute stmt
    cur_stmt_t::exec(data, item, thread_active);

    // Execute next stmt
    next_helper_t::exec(data, item, thread_active);
  }

  template<typename Data>
  inline static LaunchDims calculateDimensions(Data& data)
  {
    // Compute this statements launch dimensions
    LaunchDims statement_dims = cur_stmt_t::calculateDimensions(data);

    // call the next statement in the list
    LaunchDims next_dims = next_helper_t::calculateDimensions(data);

    // Return the maximum of the two
    return statement_dims.max(next_dims);
  }
};

template<camp::idx_t num_stmts, typename StmtList>
struct SyclStatementListExecutorHelper<num_stmts, num_stmts, StmtList>
{

  template<typename Data>
  inline static RAJA_DEVICE void exec(Data&,
                                      ::sycl::nd_item<3> RAJA_UNUSED_ARG(item),
                                      bool)
  {
    // nop terminator
  }

  template<typename Data>
  inline static LaunchDims calculateDimensions(Data&)
  {
    return LaunchDims();
  }
};

template<typename Data, typename Policy, typename Types>
struct SyclStatementExecutor;

template<typename Data, typename StmtList, typename Types>
struct SyclStatementListExecutor;

template<typename Data, typename... Stmts, typename Types>
struct SyclStatementListExecutor<Data, StatementList<Stmts...>, Types>
{

  using enclosed_stmts_t =
      camp::list<SyclStatementExecutor<Data, Stmts, Types>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static inline RAJA_DEVICE void exec(Data& data,
                                      ::sycl::nd_item<3> item,
                                      bool thread_active)
  {
    // Execute statements in order with helper class
    SyclStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(
        data, item, thread_active);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    // Compute this statements launch dimensions
    return SyclStatementListExecutorHelper<
        0, num_stmts, enclosed_stmts_t>::calculateDimensions(data);
  }
};

template<typename StmtList, typename Data, typename Types>
using sycl_statement_list_executor_t =
    SyclStatementListExecutor<Data, StmtList, Types>;

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
