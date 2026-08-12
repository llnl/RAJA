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
class ReduceSeq  // This is a Combinable and is the first layer of that
                 // implementation detail used in Reducers
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>;

public:
  using Base::Base;
  using Base::reset;

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  ReduceSeq(Policy p, T init_val, T identity_) : Base(init_val, identity_)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    policy_matches_or_throw(
        "SeqReduce", reduction_supported_policies_t<Policy::sequential> {}, p);
#endif
    RAJA_UNUSED_VAR(p);
  }

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  void reset(Policy p, T init_val, T identity_)
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    policy_matches_or_throw(
        "SeqReduce::reset",
        reduction_supported_policies_t<Policy::sequential> {}, p);
#endif
    RAJA_UNUSED_VAR(p);
    Base::reset(init_val, identity_);
  }
};


}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(seq_reduce, detail::ReduceSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
