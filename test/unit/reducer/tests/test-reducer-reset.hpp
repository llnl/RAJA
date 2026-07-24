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
          typename ForOnePol1,
          typename ForOnePol2,
          typename ForOnePol3>
struct TestReducerResetTransition
{
  const NumericType initVal = (NumericType)5;
  const RAJA::Index_type initLoc = 1;

  template < typename Api1, typename Api2, typename Api3 >
  void test_core(Api1 api1, Api2 api2, Api3 api3)
  {
    auto transition_sum =
        api1.template make<RAJA::ReduceSum<ReducePolicy, NumericType>>(
            initVal);
    auto transition_min =
        api1.template make<RAJA::ReduceMin<ReducePolicy, NumericType>>(
            initVal);
    auto transition_max =
        api1.template make<RAJA::ReduceMax<ReducePolicy, NumericType>>(
            initVal);

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

    api2.reset(transition_sum, initVal);
    api2.reset(transition_min, initVal);
    api2.reset(transition_max, initVal);

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

    api3.reset(transition_sum, initVal);
    api3.reset(transition_min, initVal);
    api3.reset(transition_max, initVal);

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

  template < typename Api1, typename Api2, typename Api3 >
  void test_loc(Api1 api1, Api2 api2, Api3 api3)
  {
    auto transition_minloc =
        api1.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType>>(
            initVal,
            initLoc);
    auto transition_maxloc =
        api1.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType>>(
            initVal,
            initLoc);

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

    api2.reset(transition_minloc, initVal, initLoc);
    api2.reset(transition_maxloc, initVal, initLoc);

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

    api3.reset(transition_minloc, initVal, initLoc);
    api3.reset(transition_maxloc, initVal, initLoc);

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

  template < typename Api1, typename Api2, typename Api3 >
  void test_bitwise(Api1 api1, Api2 api2, Api3 api3)
  {
    auto reduce_bitor =
        api1.template make<RAJA::ReduceBitOr<ReducePolicy, NumericType>>(
            initVal);
    auto reduce_bitand =
        api1.template make<RAJA::ReduceBitAnd<ReducePolicy, NumericType>>(
            initVal);

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

    api2.reset(reduce_bitor, initVal);
    api2.reset(reduce_bitand, initVal);

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

    api3.reset(reduce_bitor, initVal);
    api3.reset(reduce_bitand, initVal);

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
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::Policy::sequential>> seq_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value>> runtime_api{};

  TestReducerResetTransition< ReducePolicy, NumericType,
      test_seq, ForOnePol, test_seq > test;
  test.test_core(legacy_api, legacy_api, legacy_api);
  test.test_core(seq_api, runtime_api, seq_api);
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    test.test_loc(legacy_api, legacy_api, legacy_api);
    test.test_loc(seq_api, runtime_api, seq_api);
  }
  if constexpr (std::is_integral_v<NumericType>) {
    test.test_bitwise(legacy_api, legacy_api, legacy_api);
    test.test_bitwise(seq_api, runtime_api, seq_api);
  }
}

REGISTER_TYPED_TEST_SUITE_P(ReducerResetUnitTest,
                            ReducerBasicReset,
                            ReducerResetTransition);

#endif  //__TEST_REDUCER_RESET__
