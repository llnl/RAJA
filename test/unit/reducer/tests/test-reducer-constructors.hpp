//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA reducer constructors and initialization.
///

#ifndef __TEST_REDUCER_CONSTRUCTOR__
#define __TEST_REDUCER_CONSTRUCTOR__

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA_test-reducer-api.hpp"

#include <type_traits>

#include "../test-reducer.hpp"

template <typename T>
class ReducerConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerConstructorUnitTest);


template <typename ReducePolicy,
          typename NumericType>
struct TestBasicReducerConstructor
{

  template < typename Api >
  void test_core(Api api)
  {
    auto reduce_sum =
        api.template make<RAJA::ReduceSum<ReducePolicy, NumericType>>();
    auto reduce_min =
        api.template make<RAJA::ReduceMin<ReducePolicy, NumericType>>();
    auto reduce_max =
        api.template make<RAJA::ReduceMax<ReducePolicy, NumericType>>();

    ASSERT_EQ((NumericType)reduce_sum.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_min.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_max.get(), RAJA::operators::limits<NumericType>::min());
  }

  template < typename Api >
  void test_loc(Api api)
  {
    auto reduce_minloc =
        api.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType>>();
    auto reduce_maxloc =
        api.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType>>();

    ASSERT_EQ((NumericType)reduce_minloc.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloc.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type{-1});
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type{-1});
  }

  template < typename Api >
  void test_loctup(Api api)
  {
    auto reduce_minloctup =
        api.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>();
    auto reduce_maxloctup =
        api.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>();

    ASSERT_EQ((NumericType)reduce_minloctup.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  }

  template < typename Api >
  void test_bitwise(Api api)
  {
    auto reduce_bitor =
        api.template make<RAJA::ReduceBitOr<ReducePolicy, NumericType>>();
    auto reduce_bitand =
        api.template make<RAJA::ReduceBitAnd<ReducePolicy, NumericType>>();

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType{-1});
  }
};

template <typename ReducePolicy,
          typename NumericType>
struct TestInitReducerConstructor
{
  const NumericType initVal = (NumericType)5;
  const RAJA::Index_type initLoc = 1;

  template < typename Api >
  void test_core(Api api)
  {
    auto reduce_sum =
        api.template make<RAJA::ReduceSum<ReducePolicy, NumericType>>(initVal);
    auto reduce_min =
        api.template make<RAJA::ReduceMin<ReducePolicy, NumericType>>(initVal);
    auto reduce_max =
        api.template make<RAJA::ReduceMax<ReducePolicy, NumericType>>(initVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));
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

    ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)initLoc);
  }

  template < typename Api >
  void test_loctup(Api api)
  {
    RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(initLoc, initLoc);
    auto reduce_minloctup =
        api.template make<RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>(
            initVal, LocTup);
    auto reduce_maxloctup =
        api.template make<RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>>>(
            initVal, LocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)initLoc);
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

    ASSERT_EQ((NumericType)reduce_bitor.get(), initVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), initVal);
  }
};


TYPED_TEST_P(ReducerConstructorUnitTest, BasicReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value>> runtime_api{};

  TestBasicReducerConstructor< ReducePolicy, NumericType > test;
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

TYPED_TEST_P(ReducerConstructorUnitTest, InitReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value>> runtime_api{};

  TestInitReducerConstructor< ReducePolicy, NumericType > test;
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


REGISTER_TYPED_TEST_SUITE_P(ReducerConstructorUnitTest,
                            BasicReducerConstructor,
                            InitReducerConstructor);

#endif  //__TEST_REDUCER_CONSTRUCTOR__
