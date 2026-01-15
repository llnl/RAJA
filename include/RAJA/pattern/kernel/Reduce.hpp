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

#ifndef RAJA_pattern_kernel_Reduce_HPP
#define RAJA_pattern_kernel_Reduce_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that implements a reduction of a Param.
 * This reduces a value down to a "root" thread, and then only executes
 * the enclosed statements on the thread which contains the reduced value.
 *
 */
template<typename ReducePolicy,
         template<typename...>
         class ReduceOperator,
         typename ParamId,
         typename... EnclosedStmts>
struct Reduce : public internal::Statement<camp::nil, EnclosedStmts...>
{

  static_assert(std::is_base_of<RAJA::expt::detail::ParamBase, ParamId>::value,
                "Inappropriate ParamId, ParamId must be of type "
                "RAJA::Statement::Param< # >");

  using execution_policy_t = camp::nil;
};


}  // end namespace statement


}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_Reduce_HPP */
