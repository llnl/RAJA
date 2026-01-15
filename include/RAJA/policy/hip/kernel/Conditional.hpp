/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP kernel conditional methods.
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

#ifndef RAJA_policy_hip_kernel_Conditional_HPP
#define RAJA_policy_hip_kernel_Conditional_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Conditional.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{


template<typename Data,
         typename Conditional,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<Data,
                            statement::If<Conditional, EnclosedStmts...>,
                            Types>
{

  using stmt_list_t      = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    if (Conditional::eval(data))
    {

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
