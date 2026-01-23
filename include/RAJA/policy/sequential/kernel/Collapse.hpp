/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for sequential kernel loop collapse executors.
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

#ifndef RAJA_policy_sequential_kernel_Collapse_HPP
#define RAJA_policy_sequential_kernel_Collapse_HPP

#include "RAJA/pattern/kernel.hpp"

namespace RAJA
{

namespace internal
{


//
// Termination case for seq_exec collapsed loops
//
template<typename... EnclosedStmts, typename Types>
struct StatementExecutor<
    statement::Collapse<seq_exec, ArgList<>, EnclosedStmts...>,
    Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {
    // termination case: no more loops, just execute enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>, Types>(data);
  }
};

//
// Executor that handles collapsing of an arbitrarily deep set of seq_exec
// loops
//
template<camp::idx_t Arg0,
         camp::idx_t... ArgRest,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::Collapse<seq_exec, ArgList<Arg0, ArgRest...>, EnclosedStmts...>,
    Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {

    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, Arg0, Data>;

    // compute next-most inner loop Executor
    using next_loop_t = StatementExecutor<
        statement::Collapse<seq_exec, ArgList<ArgRest...>, EnclosedStmts...>,
        NewTypes>;

    auto len0 = segment_length<Arg0>(data);

    for (auto i0 = 0; i0 < len0; ++i0)
    {
      data.template assign_offset<Arg0>(i0);

      next_loop_t::exec(data);
    }
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
