//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA reducer reset.
///

#ifndef __TEST_REDUCER_RESET__
#define __TEST_REDUCER_RESET__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include <type_traits>

#include "../test-reducer.hpp"

template <typename T>
class ReducerResetUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerResetUnitTest);


template <  typename ReducePolicy,
            typename NumericType,
            typename ForOnePol  >
struct TestReducerReset
{
  const NumericType initVal = (NumericType)5;
  const NumericType resetVal = (NumericType)10;
  const RAJA::Index_type initLoc = 1;
  const RAJA::Index_type resetLoc = -1;
  const RAJA::tuple<RAJA::Index_type, RAJA::Index_type> initLocTup{1, 1};
  const RAJA::tuple<RAJA::Index_type, RAJA::Index_type> resetLocTup{-1, -1};

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy_or_not..., initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy_or_not..., initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy_or_not..., initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_sum += 1;
        reduce_min.min(0);
        reduce_max.max(0);
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_sum += 1;
        reduce_min.min(0);
        reduce_max.max(0);
      });
    }

    reduce_sum.reset(runtime_policy_or_not..., resetVal);
    reduce_min.reset(runtime_policy_or_not..., resetVal);
    reduce_max.reset(runtime_policy_or_not..., resetVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_min.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_max.get(), resetVal);
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loc(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy_or_not..., initVal, initLoc);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy_or_not..., initVal, initLoc);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_minloc.minloc(0,0);
        reduce_maxloc.maxloc(0,0);
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_minloc.minloc(0,0);
        reduce_maxloc.maxloc(0,0);
      });
    }

    reduce_minloc.reset(runtime_policy_or_not..., resetVal, resetLoc);
    reduce_maxloc.reset(runtime_policy_or_not..., resetVal, resetLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloc.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), resetLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), resetLoc);
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loctup(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(runtime_policy_or_not..., initVal, initLocTup);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(runtime_policy_or_not..., initVal, initLocTup);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        RAJA::tuple<RAJA::Index_type, RAJA::Index_type> temploc(0,0);
        reduce_minloctup.minloc(0,temploc);
        reduce_maxloctup.maxloc(0,temploc);
      });
    } else {
      forone<ForOnePol>( [=] () {
        RAJA::tuple<RAJA::Index_type, RAJA::Index_type> temploc(0,0);
        reduce_minloctup.minloc(0,temploc);
        reduce_maxloctup.maxloc(0,temploc);
      });
    }

    reduce_maxloctup.reset(runtime_policy_or_not..., resetVal, resetLocTup);
    reduce_minloctup.reset(runtime_policy_or_not..., resetVal, resetLocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), resetLoc);
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy_or_not..., initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy_or_not..., initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= 4;
        reduce_bitand &= 4;
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_bitor |= 4;
        reduce_bitand &= 4;
      });
    }

    reduce_bitor.reset(runtime_policy_or_not..., resetVal);
    reduce_bitand.reset(runtime_policy_or_not..., resetVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), resetVal);
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ForOnePol1,
          typename ForOnePol2,
          typename ForOnePol3>
struct TestReducerResetTransition
{
  const NumericType initVal = (NumericType)5;
  const NumericType resetVal = (NumericType)10;
  const RAJA::Index_type initLoc = 1;
  const RAJA::Index_type resetLoc = -1;

  template < RAJA::Policy... policy1_or_not,
             RAJA::Policy... policy2_or_not,
             RAJA::Policy... policy3_or_not >
  void test_core(RAJA::PolicyList<policy1_or_not...>,
                 RAJA::PolicyList<policy2_or_not...>,
                 RAJA::PolicyList<policy3_or_not...>)
  {
    RAJA::ReduceSum<ReducePolicy, NumericType> transition_sum(
        policy1_or_not...,
        NumericType(5));
    RAJA::ReduceMin<ReducePolicy, NumericType> transition_min(
        policy1_or_not...,
        NumericType(5));
    RAJA::ReduceMax<ReducePolicy, NumericType> transition_max(
        policy1_or_not...,
        NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol1>) {
      forone<ForOnePol1>( [=] RAJA_HOST_DEVICE () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    } else {
      forone<ForOnePol1>( [=] () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    }

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));

    transition_sum.reset(policy2_or_not..., NumericType(5));
    transition_min.reset(policy2_or_not..., NumericType(5));
    transition_max.reset(policy2_or_not..., NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol2>) {
      forone<ForOnePol2>( [=] RAJA_HOST_DEVICE () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    } else {
      forone<ForOnePol2>( [=] () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    }

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));

    transition_sum.reset(policy3_or_not..., NumericType(5));
    transition_min.reset(policy3_or_not..., NumericType(5));
    transition_max.reset(policy3_or_not..., NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol3>) {
      forone<ForOnePol3>( [=] RAJA_HOST_DEVICE () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    } else {
      forone<ForOnePol3>( [=] () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    }

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));
  }

  template < RAJA::Policy... policy1_or_not,
             RAJA::Policy... policy2_or_not,
             RAJA::Policy... policy3_or_not >
  void test_loc(RAJA::PolicyList<policy1_or_not...>,
                 RAJA::PolicyList<policy2_or_not...>,
                 RAJA::PolicyList<policy3_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> transition_minloc(
        policy1_or_not...,
        NumericType(5),
        RAJA::Index_type(1));
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> transition_maxloc(
        policy1_or_not...,
        NumericType(5),
        RAJA::Index_type(1));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol1>) {
      forone<ForOnePol1>( [=] RAJA_HOST_DEVICE () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    } else {
      forone<ForOnePol1>( [=] () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    }

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));

    transition_minloc.reset(policy2_or_not..., NumericType(5), RAJA::Index_type(1));
    transition_maxloc.reset(policy2_or_not..., NumericType(5), RAJA::Index_type(1));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol2>) {
      forone<ForOnePol2>( [=] RAJA_HOST_DEVICE () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    } else {
      forone<ForOnePol2>( [=] () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    }

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));

    transition_minloc.reset(policy3_or_not...,
                            NumericType(5),
                            RAJA::Index_type(1));
    transition_maxloc.reset(policy3_or_not...,
                            NumericType(5),
                            RAJA::Index_type(1));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol3>) {
      forone<ForOnePol3>( [=] RAJA_HOST_DEVICE () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    } else {
      forone<ForOnePol3>( [=] () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    }

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));
  }

  template < RAJA::Policy... policy1_or_not,
             RAJA::Policy... policy2_or_not,
             RAJA::Policy... policy3_or_not >
  void test_bitwise(RAJA::PolicyList<policy1_or_not...>,
                    RAJA::PolicyList<policy2_or_not...>,
                    RAJA::PolicyList<policy3_or_not...>)
  {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(
        policy1_or_not...,
        NumericType(5));
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(
        policy1_or_not...,
        NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol1>) {
      forone<ForOnePol1>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    } else {
      forone<ForOnePol1>( [=] () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    }

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));

    reduce_bitor.reset(policy2_or_not..., NumericType(5));
    reduce_bitand.reset(policy2_or_not..., NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol2>) {
      forone<ForOnePol2>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    } else {
      forone<ForOnePol2>( [=] () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    }

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));

    reduce_bitor.reset(policy3_or_not..., NumericType(5));
    reduce_bitand.reset(policy3_or_not..., NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol3>) {
      forone<ForOnePol3>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    } else {
      forone<ForOnePol3>( [=] () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    }

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));
  }
};

TYPED_TEST_P(ReducerResetUnitTest, ReducerBasicReset)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value> runtime_policy{};

  TestReducerReset< ReducePolicy, NumericType, ForOnePol > test;
  test.test_core(not_policy);
  test.test_core(runtime_policy);
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    test.test_loc(not_policy);
    test.test_loc(runtime_policy);
    test.test_loctup(not_policy);
    test.test_loctup(runtime_policy);
  }
  if constexpr (std::is_integral_v<NumericType>) {
    test.test_bitwise(not_policy);
    test.test_bitwise(runtime_policy);
  }
}

TYPED_TEST_P(ReducerResetUnitTest, ReducerResetTransition)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::Policy::undefined> undef_policy{};
  static constexpr RAJA::PolicyList<RAJA::Policy::sequential> seq_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value> runtime_policy{};

  TestReducerResetTransition< ReducePolicy, NumericType,
      test_seq, ForOnePol, test_seq > test;
  test.test_core(not_policy, not_policy, not_policy);
  test.test_core(seq_policy, runtime_policy, seq_policy);
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    test.test_loc(not_policy, not_policy, not_policy);
    test.test_loc(seq_policy, runtime_policy, seq_policy);
  }
  if constexpr (std::is_integral_v<NumericType>) {
    test.test_bitwise(not_policy, not_policy, not_policy);
    test.test_bitwise(seq_policy, runtime_policy, seq_policy);
  }
}

REGISTER_TYPED_TEST_SUITE_P(ReducerResetUnitTest,
                            ReducerBasicReset,
                            ReducerResetTransition);

#endif  //__TEST_REDUCER_RESET__
