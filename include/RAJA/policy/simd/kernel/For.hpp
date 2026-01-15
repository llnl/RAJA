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

#ifndef RAJA_policy_simd_kernel_For_HPP
#define RAJA_policy_simd_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{

namespace internal
{


/*!
 *
 * Helper structs to detect lambdas
 *
 */
template<class T>
struct TypeIsLambda
{
  static const bool value = false;
};

template<camp::idx_t BodyIdx, typename... Args>
struct TypeIsLambda<RAJA::statement::Lambda<BodyIdx, Args...>>
{
  static const bool value = true;
};

/*!
 *
 *  Helper structs to invoke a chain of lambdas
 *
 */

template<typename Types, class... Statements>
struct Invoke_all_Lambda;

template<typename Types>
struct Invoke_all_Lambda<Types>
{

  template<typename Data>
  static RAJA_INLINE void lambda_special(Data&&)
  {
    // NOP terminator
  }
};

template<typename Types, class Statement, class... StatementRest>
struct Invoke_all_Lambda<Types, Statement, StatementRest...>
{

  // Lambda check
  static const bool value = TypeIsLambda<camp::decay<Statement>>::value;
  static_assert(value, "Lambdas are only supported post RAJA::simd_exec");

  // Invoke the chain of lambdas
  template<typename Data>
  static RAJA_INLINE void lambda_special(Data&& data)
  {

    // Execute this Lambda
    StatementExecutor<Statement, Types>::exec(data);

    // Execute next Lambda
    Invoke_all_Lambda<Types, StatementRest...>::lambda_special(data);
  }
};

/*!
 * RAJA::kernel forall_impl executor specialization for statement::For.
 * Assumptions: RAJA::simd_exec is the inner most policy,
 * only one lambda is used, no reductions are done within the lambda.
 * Assigns the loop index to offset ArgumentId
 */
template<camp::idx_t ArgumentId, typename... EnclosedStmts, typename Types>
struct StatementExecutor<
    statement::For<ArgumentId, RAJA::simd_exec, EnclosedStmts...>,
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

      // Privatize data for SIMD correctness reasons
      using RAJA::internal::thread_privatize;
      auto privatizer    = thread_privatize(data);
      auto& private_data = privatizer.get_priv();

      // Assign offset on privatized data
      private_data.template assign_offset<ArgumentId>(i);

      Invoke_all_Lambda<NewTypes, EnclosedStmts...>::lambda_special(
          private_data);
    }
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_HPP */
