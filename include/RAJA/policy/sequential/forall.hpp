/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for sequential execution.
 *
 *          These methods should work on any platform.
 *
 *          Note: GNU compiler does not enforce sequential iterations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_sequential_HPP
#define RAJA_forall_sequential_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/detail/forall.hpp"

#include "RAJA/util/resource.hpp"

#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{
namespace policy
{
namespace sequential
{


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    expt::type_traits::is_ForallParamPack<ForallParam>>
forall_impl(resources::Host host_res,
            const seq_exec& pol,
            Iterable&& iter,
            Func&& body,
            ForallParam f_params)
{
  constexpr bool has_reducers =
      !expt::type_traits::is_ForallParamPack_empty<ForallParam>::value;
  expt::ParamMultiplexer::parampack_init(pol, f_params);

  RAJA_EXTRACT_BED_IT(iter);

  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    if constexpr (has_reducers)
    {
      expt::invoke_body(f_params, body, *(begin_it + i));
    }
    else
    {
      body(*(begin_it + i));
    }
  }


  expt::ParamMultiplexer::parampack_resolve(pol, f_params);

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace sequential

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
