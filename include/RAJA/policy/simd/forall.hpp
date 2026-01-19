/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 *          Note: Reduction operations should not be used with simd
 *          policies. Limited support.
 *
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

#ifndef RAJA_forall_simd_HPP
#define RAJA_forall_simd_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/simd/policy.hpp"

#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{
namespace policy
{
namespace simd
{


template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(resources::Host host_res,
            const simd_exec& pol,
            Iterable&& iter,
            Func&& body,
            ForallParam f_params)
{
  expt::ParamMultiplexer::parampack_init(pol, f_params);

  RAJA_EXTRACT_BED_IT(iter);

  RAJA_SIMD
  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    expt::invoke_body(f_params, body, *(begin_it + i));
  }

  expt::ParamMultiplexer::parampack_resolve(pol, f_params);

  return resources::EventProxy<resources::Host>(host_res);
}

template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Host host_res,
            const simd_exec&,
            Iterable&& iter,
            Func&& body,
            ForallParam)
{
  RAJA_EXTRACT_BED_IT(iter);

  RAJA_SIMD
  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    body(*(begin_it + i));
  }

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace simd

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
