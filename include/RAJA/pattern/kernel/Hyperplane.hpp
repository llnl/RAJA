/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for hyperplane patern executor.
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

#ifndef RAJA_pattern_kernel_Hyperplane_HPP
#define RAJA_pattern_kernel_Hyperplane_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{
namespace statement
{


/*!
 * A RAJA::kernel statement that performs hyperplane iteration over multiple
 * indices.
 *
 * Given segments S0, S1, ...
 * and iterates i0, i1, ... that range from 0 to Ni, where Ni = length(Si),
 * hyperplanes are defined as h = i0 + i1 + i2 + ...
 * For h = 0 ... sum(Ni)
 *
 * The iteration is advanced for
 *
 * -i0 = -h + i1 + i2 + ...
 *
 * Where HpArg is the argument id for i0, and Args define the arguments ids for
 * i1, i2, ...
 *
 *
 *
 *
 * The implemented loop pattern looks like:
 *
 *  RAJA::forall<HpExecPolicy>(RangeSegment(0, Nh), [=](RAJA::Index_type h){
 *
 *     RAJA::kernel::forall<Collapse<ExecPolicy, ArgList<1,2,...>, Lambda<0>>>(
 *        RAJA::make_tuple(S1, S2, ...),
 *        [=](RAJA::Index_type i1, RAJA::Index_type i2, ...){
 *
 *          // Compute i0
 *          RAJA::Index_type i0 = h - sum(i1, i2, ...);
 *
 *          // Check if h is in bounds
 *          if(h >= 0 && h < Nh){
 *
              loop_body(i0, i1, i2, ...);
 *          }
 *
 *        });
 *
 *  });
 *
 */
template<camp::idx_t HpArgumentId,
         typename HpExecPolicy,
         typename ArgList,
         typename ExecPolicy,
         typename... EnclosedStmts>
struct Hyperplane : public internal::Statement<ExecPolicy, EnclosedStmts...>
{};

}  // end namespace statement

namespace internal
{


template<camp::idx_t HpArgumentId, typename ArgList, typename... EnclosedStmts>
struct HyperplaneInner : public internal::Statement<camp::nil, EnclosedStmts...>
{};

template<camp::idx_t HpArgumentId,
         typename HpExecPolicy,
         camp::idx_t... Args,
         typename ExecPolicy,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<statement::Hyperplane<HpArgumentId,
                                               HpExecPolicy,
                                               ArgList<Args...>,
                                               ExecPolicy,
                                               EnclosedStmts...>,
                         Types>
{


  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {

    // get type of Hp arguments index
    using data_t = camp::decay<Data>;
    using idx_t =
        camp::tuple_element_t<HpArgumentId, typename data_t::offset_tuple_t>;

    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, HpArgumentId, Data>;

    // Add a Collapse policy around our enclosed statements that will handle
    // the inner hyperplane loop's execution
    using kernel_policy = statement::Collapse<
        ExecPolicy, ArgList<Args...>,
        HyperplaneInner<HpArgumentId, ArgList<Args...>, EnclosedStmts...>>;

    // Create a For-loop wrapper for the outer loop
    ForWrapper<HpArgumentId, Data, NewTypes, kernel_policy> outer_wrapper(data);

    // compute manhattan distance of iteration space to determine
    // as:  hp_len = l0 + l1 + l2 + ...
    idx_t hp_len =
        segment_length<HpArgumentId>(data) +
        foldl(RAJA::operators::plus<idx_t>(), segment_length<Args>(data)...);

    /* Execute the outer loop over hyperplanes
     *
     * This will store h in the index_tuple as argument HpArgumentId, so that
     * later, the HyperplaneInner executor can pull it out, and calculate that
     * arguments actual value (and restrict to valid hyperplane indices)
     */
    auto r = resources::get_resource<HpExecPolicy>::type::get_default();
    forall_impl(r, HpExecPolicy {}, TypedRangeSegment<idx_t>(0, hp_len),
                outer_wrapper, RAJA::expt::get_empty_forall_param_pack());
  }
};

template<camp::idx_t HpArgumentId,
         camp::idx_t... Args,
         typename... EnclosedStmts,
         typename Types>
struct StatementExecutor<
    HyperplaneInner<HpArgumentId, ArgList<Args...>, EnclosedStmts...>,
    Types>
{


  template<typename Data>
  static RAJA_INLINE void exec(Data& data)
  {

    // get h value
    auto h      = camp::get<HpArgumentId>(data.offset_tuple);
    using idx_t = decltype(h);

    // compute actual iterate for HpArgumentId
    // as:  i0 = h - (i1 + i2 + i3 + ...)
    idx_t i = h - foldl(RAJA::operators::plus<idx_t>(),
                        camp::get<Args>(data.offset_tuple)...);

    // get length of Hp indexed argument
    auto len = segment_length<HpArgumentId>(data);

    // check bounds
    if (i >= 0 && i < len)
    {

      // store in tuple
      data.template assign_offset<HpArgumentId>(i);

      // execute enclosed statements
      execute_statement_list<StatementList<EnclosedStmts...>, Types>(data);

      // reset h for next iteration
      data.template assign_offset<HpArgumentId>(h);
    }
  }
};


}  // end namespace internal

}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
