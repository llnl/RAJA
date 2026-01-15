/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals.
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

#ifndef RAJA_pattern_kernel_internal_StatementList_HPP
#define RAJA_pattern_kernel_internal_StatementList_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "camp/camp.hpp"

#include <type_traits>

namespace RAJA
{
namespace internal
{


// forward decl
template<typename Policy, typename Types>
struct StatementExecutor;


template<typename... Stmts>
using StatementList = camp::list<Stmts...>;


template<camp::idx_t idx, camp::idx_t N, typename StmtList, typename Types>
struct StatementListExecutor;

template<camp::idx_t statement_index,
         camp::idx_t num_statements,
         typename StmtList,
         typename Types>
struct StatementListExecutor
{

  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {

    // Get the statement we're going to execute
    using statement = camp::at_v<StmtList, statement_index>;

    // Execute this statement
    StatementExecutor<statement, Types>::exec(std::forward<Data>(data));

    // call our next statement
    StatementListExecutor<statement_index + 1, num_statements, StmtList,
                          Types>::exec(std::forward<Data>(data));
  }
};

/*
 * termination case, a NOP.
 */

template<camp::idx_t num_statements, typename StmtList, typename Types>
struct StatementListExecutor<num_statements, num_statements, StmtList, Types>
{

  template<typename Data>
  static RAJA_INLINE void exec(Data&&)
  {}
};

template<typename StmtList, typename Types, typename Data>
RAJA_INLINE void execute_statement_list(Data&& data)
{
  StatementListExecutor<0, camp::size<StmtList>::value, StmtList, Types>::exec(
      std::forward<Data>(data));
}


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_HPP */
