/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for statement wrappers and executors.
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

#ifndef RAJA_policy_simd_kernel_ForICount_HPP
#define RAJA_policy_simd_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/simd/policy.hpp"
#include "RAJA/policy/simd/kernel/For.hpp"

namespace RAJA
{

namespace internal
{


/*!
 * RAJA::kernel forall_impl executor specialization for statement::ForICount.
 * Assumptions: RAJA::simd_exec is the inner most policy,
 * only one lambda is used, no reductions are done within the lambda.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template<camp::idx_t ArgumentId,
         typename ParamId,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::
        ForICount<ArgumentId, ParamId, RAJA::simd_exec, EnclosedStmts...>,
    Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {

    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

    auto iter     = get<ArgumentId>(data.segment_tuple);
    auto begin    = std::begin(iter);
    auto end      = std::end(iter);
    auto distance = std::distance(begin, end);

    RAJA_SIMD
    for (decltype(distance) i = 0; i < distance; ++i)
    {

      // Offsets and parameters need to be privatized
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // Privatize data for SIMD correctness reasons
      using RAJA::internal::thread_privatize;
      auto privatizer    = thread_privatize(data);
      auto& private_data = privatizer.get_priv();

      Invoke_all_Lambda<NewTypes, EnclosedStmts...>::lambda_special(
          private_data);
    }
  }
};

}  // namespace internal
}  // end namespace RAJA


#endif
