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
            typename WORKING_RES,
            typename ForOnePol  >
void testReducerReset()
{
  const NumericType initVal = (NumericType)5;
  const NumericType resetVal = (NumericType)10;
  const RAJA::Index_type initLoc = 1;
  const RAJA::Index_type resetLoc = -1;
  const RAJA::tuple<RAJA::Index_type, RAJA::Index_type> initLocTup(initLoc, initLoc);
  const RAJA::tuple<RAJA::Index_type, RAJA::Index_type> resetLocTup(resetLoc, resetLoc);

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_sum += initVal;
        reduce_min.min(0);
        reduce_max.max(0);
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_sum += initVal;
        reduce_min.min(0);
        reduce_max.max(0);
      });
    }

    reduce_sum.reset(resetVal);
    reduce_min.reset(resetVal);
    reduce_max.reset(resetVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_min.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_max.get(), resetVal);
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, initLoc);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, initLoc);

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

    reduce_minloc.reset(resetVal, resetLoc);
    reduce_maxloc.reset(resetVal, resetLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloc.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), resetLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), resetLoc);
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, initLocTup);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, initLocTup);

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

    reduce_maxloctup.reset(resetVal, resetLocTup);
    reduce_minloctup.reset(resetVal, resetLocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), resetLoc);
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= initVal;
        reduce_bitand &= initVal;
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_bitor |= initVal;
        reduce_bitand &= initVal;
      });
    }

    reduce_bitor.reset(resetVal);
    reduce_bitand.reset(resetVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), resetVal);
  }
}

template <typename ReducePolicy,
          typename NumericType,
          typename WORKING_RES,
          typename ForOnePol>
void testRuntimePolicyReducerReset()
{
  RAJA_UNUSED_VAR((WORKING_RES::get_default()));

  const NumericType initVal = (NumericType)5;
  const NumericType resetVal = (NumericType)10;
  const RAJA::Index_type initLoc = 1;
  const RAJA::Index_type resetLoc = -1;
  const RAJA::Policy runtime_policy = RAJA::policy_of<ReducePolicy>::value;

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy, initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy, initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy, initVal);

    reduce_sum.reset(runtime_policy, resetVal);
    reduce_min.reset(runtime_policy, resetVal);
    reduce_max.reset(runtime_policy, resetVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_min.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_max.get(), resetVal);
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy,
                                                                initVal,
                                                                initLoc);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy,
                                                                initVal,
                                                                initLoc);

    reduce_minloc.reset(runtime_policy, resetVal, resetLoc);
    reduce_maxloc.reset(runtime_policy, resetVal, resetLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloc.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), resetLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), resetLoc);
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy,
                                                              initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy,
                                                                initVal);

    reduce_bitor.reset(runtime_policy, resetVal);
    reduce_bitand.reset(runtime_policy, resetVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), resetVal);
  }

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> transition_sum(
        RAJA::Policy::sequential,
        NumericType(5));
    RAJA::ReduceMin<ReducePolicy, NumericType> transition_min(
        RAJA::Policy::sequential,
        NumericType(5));
    RAJA::ReduceMax<ReducePolicy, NumericType> transition_max(
        RAJA::Policy::sequential,
        NumericType(5));

    forone<test_seq>( [=] () {
      transition_sum += NumericType(4);
      transition_min.min(NumericType(1));
      transition_max.max(NumericType(9));
    });

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));

    transition_sum.reset(runtime_policy, NumericType(5));
    transition_min.reset(runtime_policy, NumericType(5));
    transition_max.reset(runtime_policy, NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    } else {
      forone<ForOnePol>( [=] () {
        transition_sum += NumericType(4);
        transition_min.min(NumericType(1));
        transition_max.max(NumericType(9));
      });
    }

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));

    transition_sum.reset(RAJA::Policy::sequential, NumericType(5));
    transition_min.reset(RAJA::Policy::sequential, NumericType(5));
    transition_max.reset(RAJA::Policy::sequential, NumericType(5));

    forone<test_seq>( [=] () {
      transition_sum += NumericType(4);
      transition_min.min(NumericType(1));
      transition_max.max(NumericType(9));
    });

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(9));
    ASSERT_EQ((NumericType)transition_min.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_max.get(), NumericType(9));
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> transition_minloc(
        RAJA::Policy::sequential,
        NumericType(5),
        RAJA::Index_type(1));
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> transition_maxloc(
        RAJA::Policy::sequential,
        NumericType(5),
        RAJA::Index_type(1));

    forone<test_seq>( [=] () {
      transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
      transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
    });

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));

    transition_minloc.reset(runtime_policy, NumericType(5), RAJA::Index_type(1));
    transition_maxloc.reset(runtime_policy, NumericType(5), RAJA::Index_type(1));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    } else {
      forone<ForOnePol>( [=] () {
        transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
        transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
      });
    }

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));

    transition_minloc.reset(RAJA::Policy::sequential,
                            NumericType(5),
                            RAJA::Index_type(1));
    transition_maxloc.reset(RAJA::Policy::sequential,
                            NumericType(5),
                            RAJA::Index_type(1));

    forone<test_seq>( [=] () {
      transition_minloc.minloc(NumericType(1), RAJA::Index_type(7));
      transition_maxloc.maxloc(NumericType(9), RAJA::Index_type(7));
    });

    ASSERT_EQ((NumericType)transition_minloc.get(), NumericType(1));
    ASSERT_EQ((NumericType)transition_maxloc.get(), NumericType(9));
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), RAJA::Index_type(7));
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), RAJA::Index_type(7));
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(
        RAJA::Policy::sequential,
        NumericType(5));
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(
        RAJA::Policy::sequential,
        NumericType(5));

    forone<test_seq>( [=] () {
      reduce_bitor |= NumericType(2);
      reduce_bitand &= NumericType(3);
    });

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));

    reduce_bitor.reset(runtime_policy, NumericType(5));
    reduce_bitand.reset(runtime_policy, NumericType(5));

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    } else {
      forone<ForOnePol>( [=] () {
        reduce_bitor |= NumericType(2);
        reduce_bitand &= NumericType(3);
      });
    }

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));

    reduce_bitor.reset(RAJA::Policy::sequential, NumericType(5));
    reduce_bitand.reset(RAJA::Policy::sequential, NumericType(5));
    forone<test_seq>( [=] () {
      reduce_bitor |= NumericType(2);
      reduce_bitand &= NumericType(3);
    });

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType(7));
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType(1));
  }
}

TYPED_TEST_P(ReducerResetUnitTest, BasicReset)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;
  testReducerReset< ReduceType, NumericType, ResourceType, ForOneType >();
}

TYPED_TEST_P(ReducerResetUnitTest, RuntimePolicyReset)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;
  testRuntimePolicyReducerReset< ReduceType,
                                 NumericType,
                                 ResourceType,
                                 ForOneType >();
}

REGISTER_TYPED_TEST_SUITE_P(ReducerResetUnitTest,
                            BasicReset,
                            RuntimePolicyReset);

#endif  //__TEST_REDUCER_RESET__
