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
class ReducerBasicConstructorUnitTest : public ::testing::Test
{
};

template <typename T>
class ReducerInitConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest);
TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest);


template <typename ReducePolicy,
          typename NumericType>
void testReducerConstructor()
{
  const RAJA::Policy runtime_policy = RAJA::policy_of<ReducePolicy>::value;

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;

    ASSERT_EQ((NumericType)reduce_sum.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_min.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_max.get(), RAJA::operators::limits<NumericType>::min());
  }
  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy);

    ASSERT_EQ((NumericType)reduce_sum.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_min.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_max.get(), RAJA::operators::limits<NumericType>::min());
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

    ASSERT_EQ((NumericType)reduce_minloc.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloc.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type{-1});
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type{-1});
  }
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy);

    ASSERT_EQ((NumericType)reduce_minloc.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloc.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type{-1});
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type{-1});
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup;
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup;

    ASSERT_EQ((NumericType)reduce_minloctup.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  }
  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(runtime_policy);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(runtime_policy);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), RAJA::operators::limits<NumericType>::max());
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), RAJA::operators::limits<NumericType>::min());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor;
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand;

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType{-1});
  }
  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy);

    ASSERT_EQ((NumericType)reduce_bitor.get(), NumericType{0});
    ASSERT_EQ((NumericType)reduce_bitand.get(), NumericType{-1});
  }
}

TYPED_TEST_P(ReducerBasicConstructorUnitTest, BasicReducerConstructor)
{
  using ReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  testReducerConstructor< ReducePolicy, NumericType >();
}


template <typename ReducePolicy,
          typename NumericType,
          typename WORKING_RES,
          typename ForOnePol>
void testInitReducerConstructor()
{
  NumericType initVal = (NumericType)5;

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

    ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
    RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

    ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
    ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(initVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), initVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), initVal);
  }
}

template <typename ReducePolicy,
          typename NumericType,
          typename WORKING_RES,
          typename ForOnePol>
void testRuntimePolicyReducerConstructor()
{
  RAJA_UNUSED_VAR((WORKING_RES::get_default()));
  RAJA_UNUSED_VAR(sizeof(ForOnePol));

  const NumericType initVal = (NumericType)5;
  const RAJA::Index_type initLoc = 1;
  const RAJA::Policy runtime_policy = RAJA::policy_of<ReducePolicy>::value;

  {
    RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(runtime_policy, initVal);
    RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(runtime_policy, initVal);
    RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(runtime_policy, initVal);

    ASSERT_EQ((NumericType)reduce_sum.get(), initVal);
    ASSERT_EQ((NumericType)reduce_min.get(), initVal);
    ASSERT_EQ((NumericType)reduce_max.get(), initVal);
  }

  if constexpr (!RAJA::policy_is<ReducePolicy, RAJA::Policy::sycl>::value) {
    RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(runtime_policy,
                                                                initVal,
                                                                initLoc);
    RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(runtime_policy,
                                                                initVal,
                                                                initLoc);

    ASSERT_EQ((NumericType)reduce_minloc.get(), initVal);
    ASSERT_EQ((NumericType)reduce_maxloc.get(), initVal);
    ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), initLoc);
    ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), initLoc);
  }

  if constexpr (std::is_integral_v<NumericType>) {
    RAJA::ReduceBitOr<ReducePolicy, NumericType> reduce_bitor(runtime_policy,
                                                              initVal);
    RAJA::ReduceBitAnd<ReducePolicy, NumericType> reduce_bitand(runtime_policy,
                                                                initVal);

    ASSERT_EQ((NumericType)reduce_bitor.get(), initVal);
    ASSERT_EQ((NumericType)reduce_bitand.get(), initVal);
  }
}

TYPED_TEST_P(ReducerInitConstructorUnitTest, InitReducerConstructor)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;

  testInitReducerConstructor< ReduceType, NumericType, ResourceType, ForOneType >();
}

TYPED_TEST_P(ReducerInitConstructorUnitTest, RuntimePolicyInitReducerConstructor)
{
  using ReduceType = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<2>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<3>>::type;

  testRuntimePolicyReducerConstructor< ReduceType,
                                       NumericType,
                                       ResourceType,
                                       ForOneType >();
}


REGISTER_TYPED_TEST_SUITE_P(ReducerBasicConstructorUnitTest,
                            BasicReducerConstructor);

REGISTER_TYPED_TEST_SUITE_P(ReducerInitConstructorUnitTest,
                            InitReducerConstructor,
                            RuntimePolicyInitReducerConstructor);

#endif  //__TEST_REDUCER_CONSTRUCTOR__
