/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
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

#ifndef RAJA_sequential_reduce_HPP
#define RAJA_sequential_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{
template<typename T, typename Reduce>
class ReduceSeq
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>;

public:
  //! prohibit compiler-generated default ctor
  ReduceSeq() = delete;

  using Base::Base;
};


}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(seq_reduce, detail::ReduceSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
