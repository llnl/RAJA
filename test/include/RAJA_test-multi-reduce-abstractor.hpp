//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Reduction policies used for reduction tests
//

#ifndef __RAJA_test_multi_reduce_abstractor_HPP__
#define __RAJA_test_multi_reduce_abstractor_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//
// Get the identity value for the operation used by the given multi reducer
//
template < typename MultiReducer >
inline auto get_op_identity(MultiReducer const& RAJA_UNUSED_ARG(multi_reduce))
{
  return MultiReducer::MultiReduceOp::identity();
}


struct SumAbstractor
{
  template < typename DATA_TYPE >
  static constexpr bool supports() { return std::is_arithmetic<DATA_TYPE>::value; }

  template < typename Reducer >
  static bool consistent(Reducer const&)
  {
    return RAJA::policy_has_trait<typename Reducer::policy, RAJA::reduce::ordered>::value ||
           !std::is_floating_point<typename Reducer::value_type>::value;
  }

  template < typename policy, typename DATA_TYPE >
  using reducer = RAJA::ReduceSum<policy, DATA_TYPE>;

  template < typename policy, typename DATA_TYPE >
  using multi_reducer = RAJA::MultiReduceSum<policy, DATA_TYPE>;

  template < typename Lhs, typename Rhs >
  RAJA_HOST_DEVICE
  static auto combine(Lhs const& lhs, Rhs const& rhs) { return lhs + rhs; }

  template < typename Reducer, typename Rhs >
  RAJA_HOST_DEVICE
  static decltype(auto) reduce(Reducer&& lhs, Rhs const& rhs) { return std::forward<Reducer>(lhs) += rhs; }

  template < typename Reducer >
  static auto identity(Reducer const&) { return Reducer::MultiReduceOp::identity(); }
};

struct MinAbstractor
{
  template < typename DATA_TYPE >
  static constexpr bool supports() { return std::is_arithmetic<DATA_TYPE>::value; }

  template < typename Reducer >
  static constexpr bool consistent(Reducer const&) { return true; }

  template < typename policy, typename DATA_TYPE >
  using reducer = RAJA::ReduceSum<policy, DATA_TYPE>;

  template < typename policy, typename DATA_TYPE >
  using multi_reducer = RAJA::MultiReduceMin<policy, DATA_TYPE>;

  template < typename Lhs, typename Rhs >
  RAJA_HOST_DEVICE
  static auto combine(Lhs const& lhs, Rhs const& rhs) { return (lhs > rhs) ? rhs : lhs; }

  template < typename Reducer, typename Rhs >
  RAJA_HOST_DEVICE
  static decltype(auto) reduce(Reducer&& lhs, Rhs const& rhs) { return std::forward<Reducer>(lhs).min(rhs); }

  template < typename Reducer >
  static auto identity(Reducer const&) { return Reducer::MultiReduceOp::identity(); }
};

struct MaxAbstractor
{
  template < typename DATA_TYPE >
  static constexpr bool supports() { return std::is_arithmetic<DATA_TYPE>::value; }

  template < typename Reducer >
  static constexpr bool consistent(Reducer const&) { return true; }

  template < typename policy, typename DATA_TYPE >
  using reducer = RAJA::ReduceSum<policy, DATA_TYPE>;

  template < typename policy, typename DATA_TYPE >
  using multi_reducer = RAJA::MultiReduceMax<policy, DATA_TYPE>;

  template < typename Lhs, typename Rhs >
  RAJA_HOST_DEVICE
  static auto combine(Lhs const& lhs, Rhs const& rhs) { return (lhs < rhs) ? rhs : lhs; }

  template < typename Reducer, typename Rhs >
  RAJA_HOST_DEVICE
  static decltype(auto) reduce(Reducer&& lhs, Rhs const& rhs) { return std::forward<Reducer>(lhs).max(rhs); }

  template < typename Reducer >
  static auto identity(Reducer const&) { return Reducer::MultiReduceOp::identity(); }
};

struct BitAndAbstractor
{
  template < typename DATA_TYPE >
  static constexpr bool supports() { return std::is_integral<DATA_TYPE>::value; }

  template < typename Reducer >
  static constexpr bool consistent(Reducer const&) { return true; }

  template < typename policy, typename DATA_TYPE >
  using reducer = RAJA::ReduceSum<policy, DATA_TYPE>;

  template < typename policy, typename DATA_TYPE >
  using multi_reducer = RAJA::MultiReduceBitAnd<policy, DATA_TYPE>;

  template < typename Lhs, typename Rhs >
  RAJA_HOST_DEVICE
  static auto combine(Lhs const& lhs, Rhs const& rhs) { return lhs & rhs; }

  template < typename Reducer, typename Rhs >
  RAJA_HOST_DEVICE
  static decltype(auto) reduce(Reducer&& lhs, Rhs const& rhs) { return std::forward<Reducer>(lhs) &= rhs; }

  template < typename Reducer >
  static auto identity(Reducer const&) { return Reducer::MultiReduceOp::identity(); }
};

struct BitOrAbstractor
{
  template < typename DATA_TYPE >
  static constexpr bool supports() { return std::is_integral<DATA_TYPE>::value; }

  template < typename Reducer >
  static constexpr bool consistent(Reducer const&) { return true; }

  template < typename policy, typename DATA_TYPE >
  using reducer = RAJA::ReduceSum<policy, DATA_TYPE>;

  template < typename policy, typename DATA_TYPE >
  using multi_reducer = RAJA::MultiReduceBitOr<policy, DATA_TYPE>;

  template < typename Lhs, typename Rhs >
  RAJA_HOST_DEVICE
  static auto combine(Lhs const& lhs, Rhs const& rhs) { return lhs | rhs; }

  template < typename Reducer, typename Rhs >
  RAJA_HOST_DEVICE
  static decltype(auto) reduce(Reducer&& lhs, Rhs const& rhs) { return std::forward<Reducer>(lhs) |= rhs; }

  template < typename Reducer >
  static auto identity(Reducer const&) { return Reducer::MultiReduceOp::identity(); }
};


// Sequential reduction policy types
using ReduceSumAbstractors = camp::list< SumAbstractor >;
using ReduceMinAbstractors = camp::list< MinAbstractor >;
using ReduceMaxAbstractors = camp::list< MaxAbstractor >;
using ReduceBitAndAbstractors = camp::list< BitAndAbstractor >;
using ReduceBitOrAbstractors = camp::list< BitOrAbstractor >;

#endif  // __RAJA_test_multi_reduce_abstractor_HPP__
