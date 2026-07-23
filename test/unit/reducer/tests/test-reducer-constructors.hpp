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

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy_or_not...);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy_or_not...);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy_or_not...);

    ASSERT_EQ((NumericType)reduce_sum.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_min.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_max.get(), RAJA::operators::limits<NumericType>::min());
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loc(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy_or_not...);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy_or_not...);

    ASSERT_EQ((NumericType)reduce_minloc.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloc.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type{-1});
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type{-1});
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loctup(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(runtime_policy_or_not...);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(runtime_policy_or_not...);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy_or_not...);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy_or_not...);

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

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy_or_not..., initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy_or_not..., initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy_or_not..., initVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loc(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy_or_not..., initVal, initLoc);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy_or_not..., initVal, initLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)initLoc);
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_loctup(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(initLoc, initLoc);
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(runtime_policy_or_not..., initVal, LocTup);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(runtime_policy_or_not..., initVal, LocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)initLoc);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)initLoc);
  }

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy_or_not..., initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy_or_not..., initVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), initVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), initVal);
  }
};


TYPED_TEST_P(ReducerConstructorUnitTest, BasicReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value> runtime_policy{};

  TestBasicReducerConstructor< ReducePolicy, NumericType > test;
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

TYPED_TEST_P(ReducerConstructorUnitTest, InitReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<ReducePolicy>::value> runtime_policy{};

  TestInitReducerConstructor< ReducePolicy, NumericType > test;
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


REGISTER_TYPED_TEST_SUITE_P(ReducerConstructorUnitTest,
                            BasicReducerConstructor,
                            InitReducerConstructor);

#endif  //__TEST_REDUCER_CONSTRUCTOR__
