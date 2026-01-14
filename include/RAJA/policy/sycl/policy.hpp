/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA SYCL policy definitions.
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

#ifndef policy_sycl_HPP
#define policy_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_SYCL_ACTIVE)

#include "RAJA/util/sycl_compat.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

#include <cstddef>

namespace RAJA
{

struct uint3
{
  unsigned long x, y, z;
};

using sycl_dim_t = ::sycl::range<1>;

using sycl_dim_3_t = uint3;

namespace detail
{
template<bool Async>
struct get_launch
{
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template<>
struct get_launch<false>
{
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace sycl
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////

template<size_t BLOCK_SIZE, bool Async = false>
struct sycl_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::sycl,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::sycl>
{};

template<bool Async, int num_threads = 0>
struct sycl_launch_t : public RAJA::make_policy_pattern_launch_platform_t<
                           RAJA::Policy::sycl,
                           RAJA::Pattern::region,
                           detail::get_launch<Async>::value,
                           RAJA::Platform::sycl>
{};

struct sycl_reduce
    : make_policy_pattern_t<RAJA::Policy::sycl, RAJA::Pattern::reduce>
{};

//
// Sycl atomic policy for using sycl atomics on the device and
// the provided Policy on the host
//
template<typename host_policy>
struct sycl_atomic_explicit
{};

//
// Default sycl atomic policy uses sycl atomics on the device and non-atomics
// on the host
//
using sycl_atomic = sycl_atomic_explicit<seq_atomic>;

template<typename Mask>
struct sycl_local_masked_direct
{};

template<typename Mask>
struct sycl_local_masked_loop
{};

}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_exec;
using policy::sycl::sycl_reduce;

using policy::sycl::sycl_atomic;
using policy::sycl::sycl_atomic_explicit;

using policy::sycl::sycl_local_masked_direct;
using policy::sycl::sycl_local_masked_loop;

using policy::sycl::sycl_launch_t;

/*!
 * Maps indices to SYCL global id
 * Optional WORK_GROUP_SIZE to
 */
template<int dim, int WORK_GROUP_SIZE = 1>
struct sycl_global_012
{};

template<int WORK_GROUP_SIZE>
using sycl_global_0 = sycl_global_012<0, WORK_GROUP_SIZE>;
template<int WORK_GROUP_SIZE>
using sycl_global_1 = sycl_global_012<1, WORK_GROUP_SIZE>;
template<int WORK_GROUP_SIZE>
using sycl_global_2 = sycl_global_012<2, WORK_GROUP_SIZE>;

/*!
 * Maps segment indices to SYCL group ids.
 * Loops to allow for any value
 */
template<int... dim>
struct sycl_group_012_loop
{};

using sycl_group_0_loop = sycl_group_012_loop<0>;
using sycl_group_1_loop = sycl_group_012_loop<1>;
using sycl_group_2_loop = sycl_group_012_loop<2>;

/*!
 * Maps segment indices to SYCL local ids.
 * Loops to allow for any value
 */
template<int... dim>
struct sycl_local_012_loop
{};

using sycl_local_0_loop = sycl_local_012_loop<0>;
using sycl_local_1_loop = sycl_local_012_loop<1>;
using sycl_local_2_loop = sycl_local_012_loop<2>;

/*!
 * Maps segment indices to SYCL group ids.
 */
template<int... dim>
struct sycl_group_012_direct
{};

using sycl_group_0_direct = sycl_group_012_direct<0>;
using sycl_group_1_direct = sycl_group_012_direct<1>;
using sycl_group_2_direct = sycl_group_012_direct<2>;

/*!
 * Maps segment indices to SYCL local ids.
 */
template<int... dim>
struct sycl_local_012_direct
{};

using sycl_local_0_direct = sycl_local_012_direct<0>;
using sycl_local_1_direct = sycl_local_012_direct<1>;
using sycl_local_2_direct = sycl_local_012_direct<2>;

namespace internal
{

template<int dim>
struct SyclDimHelper;

template<>
struct SyclDimHelper<0>
{

  template<typename dim_t>
  inline static constexpr auto get(dim_t const& d) -> decltype(d.x)
  {
    return d.x;
  }

  template<typename dim_t>
  inline static void set(dim_t& d, int value)
  {
    d.x = value;
  }
};

template<>
struct SyclDimHelper<1>
{

  template<typename dim_t>
  inline static constexpr auto get(dim_t const& d) -> decltype(d.x)
  {
    return d.y;
  }

  template<typename dim_t>
  inline static void set(dim_t& d, int value)
  {
    d.y = value;
  }
};

template<>
struct SyclDimHelper<2>
{

  template<typename dim_t>
  inline static constexpr auto get(dim_t const& d) -> decltype(d.x)
  {
    return d.z;
  }

  template<typename dim_t>
  inline static void set(dim_t& d, int value)
  {
    d.z = value;
  }
};

template<int dim, typename dim_t>
constexpr auto get_sycl_dim(dim_t const& d) -> decltype(d.x)
{
  return SyclDimHelper<dim>::get(d);
}

template<int dim, typename dim_t>
void set_sycl_dim(dim_t& d, int value)
{
  return SyclDimHelper<dim>::set(d, value);
}
}  // namespace internal

}  // namespace RAJA

#endif  // RAJA_ENABLE_SYCL

#endif
