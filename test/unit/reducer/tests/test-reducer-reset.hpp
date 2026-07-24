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
#include "RAJA_test-reducer-api.hpp"

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

  template < typename Api >
  void test_core(Api api)
  {
    auto reduce_sum =
        api.template make<RAJA::ReduceSum<ReducePolicy, NumericType>>(initVal);
    auto reduce_min =
        api.template make<RAJA::ReduceMin<ReducePolicy, NumericType>>(initVal);
    auto reduce_max =
        api.template make<RAJA::ReduceMax<ReducePolicy, NumericType>>(initVal);

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

    api.reset(reduce_sum, resetVal);
    api.reset(reduce_min, resetVal);
    api.reset(reduce_max, resetVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_min.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_max.get(), resetVal);
  }

  template < typename Api >
  void test_loc(Api api)
  {
    auto reduce_minloc =
        api.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType>>(
            initVal, initLoc);
    auto reduce_maxloc =
        api.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType>>(
            initVal, initLoc);

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

    api.reset(reduce_minloc, resetVal, resetLoc);
    api.reset(reduce_maxloc, resetVal, resetLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloc.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), resetLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), resetLoc);
  }

  template < typename Api >
  void test_loctup(Api api)
  {
    auto reduce_minloctup =
        api.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>(
            initVal, initLocTup);
    auto reduce_maxloctup =
        api.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>(
            initVal, initLocTup);

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

    api.reset(reduce_maxloctup, resetVal, resetLocTup);
    api.reset(reduce_minloctup, resetVal, resetLocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), resetVal);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), resetLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), resetLoc);
  }

  template < typename Api >
  void test_bitwise(Api api)
  {
    auto reduce_bitor =
        api.template make<RAJA::ReduceBitOr<ReducePolicy, NumericType>>(
            initVal);
    auto reduce_bitand =
        api.template make<RAJA::ReduceBitAnd<ReducePolicy, NumericType>>(
            initVal);

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

    api.reset(reduce_bitor, resetVal);
    api.reset(reduce_bitand, resetVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), resetVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), resetVal);
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ConstructPhase>
struct CoreTransitionHelper
{
  using ConstructApi = typename ConstructPhase::Api;

  RAJA::ReduceSum<ReducePolicy, NumericType> transition_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> transition_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> transition_max;

  CoreTransitionHelper(NumericType initVal,
                       RAJA::Index_type RAJA_UNUSED_ARG(initLoc))
      : transition_sum(
            ConstructApi::template make<RAJA::ReduceSum<ReducePolicy, NumericType>>(
                initVal)),
        transition_min(
            ConstructApi::template make<RAJA::ReduceMin<ReducePolicy, NumericType>>(
                initVal)),
        transition_max(
            ConstructApi::template make<RAJA::ReduceMax<ReducePolicy, NumericType>>(
                initVal))
  {
  }

  template <typename Phase>
  void phase(Phase,
             NumericType initVal,
             RAJA::Index_type RAJA_UNUSED_ARG(initLoc))
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    Api::reset(transition_sum, initVal);
    Api::reset(transition_min, initVal);
    Api::reset(transition_max, initVal);

    auto& reduce_sum = transition_sum;
    auto& reduce_min = transition_min;
    auto& reduce_max = transition_max;

    const NumericType addend = NumericType(3);
    const NumericType minVal = NumericType(initVal - NumericType(2));
    const NumericType maxVal = NumericType(initVal + NumericType(4));

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_sum += addend;
        reduce_min.min(minVal);
        reduce_max.max(maxVal);
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        reduce_sum += addend;
        reduce_min.min(minVal);
        reduce_max.max(maxVal);
      });
    }

    ASSERT_EQ((NumericType)transition_sum.get(), NumericType(initVal + addend));
    ASSERT_EQ((NumericType)transition_min.get(), minVal);
    ASSERT_EQ((NumericType)transition_max.get(), maxVal);
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ConstructPhase>
struct LocTransitionHelper
{
  using ConstructApi = typename ConstructPhase::Api;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType> transition_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> transition_maxloc;

  LocTransitionHelper(NumericType initVal, RAJA::Index_type initLoc)
      : transition_minloc(
            ConstructApi::template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType>>(
                initVal, initLoc)),
        transition_maxloc(
            ConstructApi::template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType>>(
                initVal, initLoc))
  {
  }

  template <typename Phase>
  void phase(Phase, NumericType initVal, RAJA::Index_type initLoc)
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    Api::reset(transition_minloc, initVal, initLoc);
    Api::reset(transition_maxloc, initVal, initLoc);

    auto& reduce_minloc = transition_minloc;
    auto& reduce_maxloc = transition_maxloc;

    const NumericType minVal = NumericType(initVal - NumericType(2));
    const NumericType maxVal = NumericType(initVal + NumericType(4));
    const RAJA::Index_type minLoc = RAJA::Index_type(initLoc + 11);
    const RAJA::Index_type maxLoc = RAJA::Index_type(initLoc + 13);

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_minloc.minloc(minVal, minLoc);
        reduce_maxloc.maxloc(maxVal, maxLoc);
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        reduce_minloc.minloc(minVal, minLoc);
        reduce_maxloc.maxloc(maxVal, maxLoc);
      });
    }

    ASSERT_EQ((NumericType)transition_minloc.get(), minVal);
    ASSERT_EQ((NumericType)transition_maxloc.get(), maxVal);
    ASSERT_EQ((RAJA::Index_type)transition_minloc.getLoc(), minLoc);
    ASSERT_EQ((RAJA::Index_type)transition_maxloc.getLoc(), maxLoc);
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ConstructPhase>
struct TupleLocTransitionHelper
{
  using LocType = RAJA::tuple<RAJA::Index_type, RAJA::Index_type>;
  using ConstructApi = typename ConstructPhase::Api;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType, LocType> transition_minloctup;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, LocType> transition_maxloctup;

  TupleLocTransitionHelper(NumericType initVal, RAJA::Index_type initLoc)
      : transition_minloctup(
            ConstructApi::template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType, LocType>>(
                initVal, LocType(initLoc, initLoc))),
        transition_maxloctup(
            ConstructApi::template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType, LocType>>(
                initVal, LocType(initLoc, initLoc)))
  {
  }

  template <typename Phase>
  void phase(Phase, NumericType initVal, RAJA::Index_type initLoc)
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    const LocType resetLoc(initLoc, initLoc);
    Api::reset(transition_minloctup, initVal, resetLoc);
    Api::reset(transition_maxloctup, initVal, resetLoc);

    auto& reduce_minloctup = transition_minloctup;
    auto& reduce_maxloctup = transition_maxloctup;

    const NumericType minVal = NumericType(initVal - NumericType(2));
    const NumericType maxVal = NumericType(initVal + NumericType(4));
    const LocType minLoc(RAJA::Index_type(initLoc + 17),
                         RAJA::Index_type(initLoc + 19));
    const LocType maxLoc(RAJA::Index_type(initLoc + 23),
                         RAJA::Index_type(initLoc + 29));

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_minloctup.minloc(minVal, minLoc);
        reduce_maxloctup.maxloc(maxVal, maxLoc);
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        reduce_minloctup.minloc(minVal, minLoc);
        reduce_maxloctup.maxloc(maxVal, maxLoc);
      });
    }

    ASSERT_EQ((NumericType)transition_minloctup.get(), minVal);
    ASSERT_EQ((NumericType)transition_maxloctup.get(), maxVal);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(transition_minloctup.getLoc())),
              RAJA::get<0>(minLoc));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(transition_minloctup.getLoc())),
              RAJA::get<1>(minLoc));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(transition_maxloctup.getLoc())),
              RAJA::get<0>(maxLoc));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(transition_maxloctup.getLoc())),
              RAJA::get<1>(maxLoc));
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ConstructPhase>
struct BitwiseTransitionHelper
{
  using ConstructApi = typename ConstructPhase::Api;

  RAJA::ReduceBitOr<ReducePolicy, NumericType> transition_bitor;
  RAJA::ReduceBitAnd<ReducePolicy, NumericType> transition_bitand;

  BitwiseTransitionHelper(NumericType initVal,
                          RAJA::Index_type RAJA_UNUSED_ARG(initLoc))
      : transition_bitor(
            ConstructApi::template make<RAJA::ReduceBitOr<ReducePolicy, NumericType>>(
                initVal)),
        transition_bitand(
            ConstructApi::template make<RAJA::ReduceBitAnd<ReducePolicy, NumericType>>(
                initVal))
  {
  }

  template <typename Phase>
  void phase(Phase,
             NumericType initVal,
             RAJA::Index_type RAJA_UNUSED_ARG(initLoc))
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    Api::reset(transition_bitor, initVal);
    Api::reset(transition_bitand, initVal);

    auto& reduce_bitor = transition_bitor;
    auto& reduce_bitand = transition_bitand;

    const NumericType orMask = NumericType(2);
    const NumericType andMask = NumericType(7);

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        reduce_bitor |= orMask;
        reduce_bitand &= andMask;
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        reduce_bitor |= orMask;
        reduce_bitand &= andMask;
      });
    }

    ASSERT_EQ((NumericType)transition_bitor.get(),
              NumericType(initVal | orMask));
    ASSERT_EQ((NumericType)transition_bitand.get(),
              NumericType(initVal & andMask));
  }
};

template <typename ReducePolicy,
          typename NumericType,
          typename ForOnePol,
          template <typename, typename, typename> class Helper>
void run_reducer_reset_transition_chains()
{
  using SeqPhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<RAJA::Policy::sequential>>,
                      test_seq>;
  using RuntimePhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value>>,
                      ForOnePol>;
  using LegacySeqPhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<>>,
                      test_seq>;
  using LegacyRuntimePhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<>>,
                      ForOnePol>;

  {
    Helper<ReducePolicy, NumericType, SeqPhase> tester(
                                       NumericType(5),  RAJA::Index_type(1));
    tester.phase(RuntimePhase{},       NumericType(11), RAJA::Index_type(2));
    tester.phase(SeqPhase{},           NumericType(17), RAJA::Index_type(3));
    tester.phase(RuntimePhase{},       NumericType(23), RAJA::Index_type(4));
    tester.phase(LegacyRuntimePhase{}, NumericType(29), RAJA::Index_type(5));
    tester.phase(RuntimePhase{},       NumericType(37), RAJA::Index_type(6));
  }

  {
    Helper<ReducePolicy, NumericType, RuntimePhase> tester(
                                 NumericType(43), RAJA::Index_type(7));
    tester.phase(SeqPhase{},     NumericType(47), RAJA::Index_type(8));
    tester.phase(RuntimePhase{}, NumericType(53), RAJA::Index_type(9));
  }

  {
    Helper<ReducePolicy, NumericType, LegacyRuntimePhase> tester(
                                   NumericType(59), RAJA::Index_type(10));
    tester.phase(RuntimePhase{},   NumericType(67), RAJA::Index_type(11));
    tester.phase(SeqPhase{},       NumericType(73), RAJA::Index_type(12));
    tester.phase(LegacySeqPhase{}, NumericType(79), RAJA::Index_type(13));
  }
}


TYPED_TEST_P(ReducerResetUnitTest, ReducerBasicReset)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value>> runtime_api{};

  TestReducerReset< ReducePolicy, NumericType, ForOnePol > test;
  test.test_core(legacy_api);
  test.test_core(runtime_api);
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    test.test_loc(legacy_api);
    test.test_loc(runtime_api);
    test.test_loctup(legacy_api);
    test.test_loctup(runtime_api);
  }
  if constexpr (std::is_integral_v<NumericType>) {
    test.test_bitwise(legacy_api);
    test.test_bitwise(runtime_api);
  }
}

TYPED_TEST_P(ReducerResetUnitTest, ReducerResetTransition)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  run_reducer_reset_transition_chains<
      ReducePolicy, NumericType, ForOnePol, CoreTransitionHelper>();
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    run_reducer_reset_transition_chains<
        ReducePolicy, NumericType, ForOnePol, LocTransitionHelper>();
    run_reducer_reset_transition_chains<
        ReducePolicy, NumericType, ForOnePol, TupleLocTransitionHelper>();
  }
  if constexpr (std::is_integral_v<NumericType>) {
    run_reducer_reset_transition_chains<
        ReducePolicy, NumericType, ForOnePol, BitwiseTransitionHelper>();
  }
}

REGISTER_TYPED_TEST_SUITE_P(ReducerResetUnitTest,
                            ReducerBasicReset,
                            ReducerResetTransition);

#endif  //__TEST_REDUCER_RESET__
