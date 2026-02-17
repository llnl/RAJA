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

#ifndef RAJA_pattern_kernel_For_HPP
#define RAJA_pattern_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that implements a single loop.
 * Assigns the loop iterate to argument ArgumentId
 *
 */
template<camp::idx_t ArgumentId,
         typename ExecPolicy = camp::nil,
         typename... EnclosedStmts>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, ExecPolicy>,
             public internal::Statement<ExecPolicy, EnclosedStmts...>
{

  // TODO: add static_assert for valid policy in Pol
  using execution_policy_t = ExecPolicy;
};


}  // end namespace statement

namespace internal
{

/*!
 * A generic RAJA::kernel forall_impl loop wrapper for statement::For
 * Assigns the loop index to offset ArgumentId
 *
 */
template<camp::idx_t ArgumentId,
         typename Data,
         typename Types,
         typename... EnclosedStmts>
struct ForWrapper : public GenericWrapper<Data, Types, EnclosedStmts...>
{

  using Base = GenericWrapper<Data, Types, EnclosedStmts...>;
  using Base::Base;
  using privatizer = NestedPrivatizer<ForWrapper>;

  template<typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
    Base::data.template assign_offset<ArgumentId>(i);
    Base::exec();
  }
};

/*!
 * A generic RAJA::kernel forall_impl executor for statement::For
 *
 *
 */
template<camp::idx_t ArgumentId,
         typename ExecPolicy,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    statement::For<ArgumentId, ExecPolicy, EnclosedStmts...>,
    Types>
{


  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {

    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

    // Create a wrapper, just in case forall_impl needs to thread_privatize
    ForWrapper<ArgumentId, Data, NewTypes, EnclosedStmts...> for_wrapper(data);

    auto len    = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    auto r = data.res;

    forall_impl(r, ExecPolicy {}, TypedRangeSegment<len_t>(0, len), for_wrapper,
                RAJA::expt::get_empty_forall_param_pack());
  }
};

/*!
 * A generic RAJA::kernel forall_impl executor for statement::For
 *
 *
 */
template<camp::idx_t ArgumentId, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
                         Types>
{


  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

    // Create a wrapper, just in case forall_impl needs to thread_privatize
    ForWrapper<ArgumentId, Data, NewTypes, EnclosedStmts...> for_wrapper(data);

    auto len    = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    RAJA_EXTRACT_BED_IT(TypedRangeSegment<len_t>(0, len));
    for (decltype(distance_it) i = 0; i < distance_it; ++i)
    {
      for_wrapper(*(begin_it + i));
    }
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_For_HPP */
