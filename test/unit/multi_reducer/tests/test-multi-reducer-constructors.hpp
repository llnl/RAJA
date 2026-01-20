//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA multi reducer constructors and initialization.
///

#ifndef __TEST_MULTI_REDUCER_CONSTRUCTOR__
#define __TEST_MULTI_REDUCER_CONSTRUCTOR__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-multi-reducer.hpp"

#include <vector>
#include <list>
#include <set>

template <typename T>
class MultiReducerBasicConstructorUnitTest : public ::testing::Test
{
};

template <typename T>
class MultiReducerSingleInitConstructorUnitTest : public ::testing::Test
{
};

template <typename T>
class MultiReducerContainerInitConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(MultiReducerBasicConstructorUnitTest);
TYPED_TEST_SUITE_P(MultiReducerSingleInitConstructorUnitTest);
TYPED_TEST_SUITE_P(MultiReducerContainerInitConstructorUnitTest);


template <typename MultiReducePolicy,
          typename NumericType>
void testBasicMultiReducerConstructorRegular(size_t num_bins)
{
  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(num_bins);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(num_bins);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(num_bins);

  ASSERT_EQ(multi_reduce_sum.size(), num_bins);
  ASSERT_EQ(multi_reduce_min.size(), num_bins);
  ASSERT_EQ(multi_reduce_max.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_sum.get(bin), get_op_identity(multi_reduce_sum));
    ASSERT_EQ(multi_reduce_min.get(bin), get_op_identity(multi_reduce_min));
    ASSERT_EQ(multi_reduce_max.get(bin), get_op_identity(multi_reduce_max));

    ASSERT_EQ((NumericType)multi_reduce_sum[bin].get(), get_op_identity(multi_reduce_sum));
    ASSERT_EQ((NumericType)multi_reduce_min[bin].get(), get_op_identity(multi_reduce_min));
    ASSERT_EQ((NumericType)multi_reduce_max[bin].get(), get_op_identity(multi_reduce_max));
  }
}

template <typename MultiReducePolicy,
          typename NumericType>
void testBasicMultiReducerConstructorBitwise(size_t num_bins)
{
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(num_bins);
  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(num_bins);

  ASSERT_EQ(multi_reduce_or.size(), num_bins);
  ASSERT_EQ(multi_reduce_and.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_or.get(bin), get_op_identity(multi_reduce_or));
    ASSERT_EQ(multi_reduce_and.get(bin), get_op_identity(multi_reduce_and));

    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), get_op_identity(multi_reduce_or));
    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), get_op_identity(multi_reduce_and));
  }
}

template <typename MultiReducePolicy,
          typename NumericType,
          std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr>
void testBasicMultiReducerConstructor(size_t num_bins)
{
  testBasicMultiReducerConstructorRegular< MultiReducePolicy, NumericType >(num_bins);
  testBasicMultiReducerConstructorBitwise< MultiReducePolicy, NumericType >(num_bins);
}
///
template <typename MultiReducePolicy,
          typename NumericType,
          std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr>
void testBasicMultiReducerConstructor(size_t num_bins)
{
  testBasicMultiReducerConstructorRegular< MultiReducePolicy, NumericType >(num_bins);
}

TYPED_TEST_P(MultiReducerBasicConstructorUnitTest, MultiReducerConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  testBasicMultiReducerConstructor< MultiReducePolicy, NumericType >(0);
  testBasicMultiReducerConstructor< MultiReducePolicy, NumericType >(1);
  testBasicMultiReducerConstructor< MultiReducePolicy, NumericType >(2);
  testBasicMultiReducerConstructor< MultiReducePolicy, NumericType >(10);
}


template <typename MultiReducePolicy,
          typename NumericType>
void testMultiReducerSingleInitConstructorRegular(size_t num_bins, NumericType initVal)
{
  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(num_bins, initVal);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(num_bins, initVal);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(num_bins, initVal);

  ASSERT_EQ(multi_reduce_sum.size(), num_bins);
  ASSERT_EQ(multi_reduce_min.size(), num_bins);
  ASSERT_EQ(multi_reduce_max.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_sum.get(bin), initVal);
    ASSERT_EQ(multi_reduce_min.get(bin), initVal);
    ASSERT_EQ(multi_reduce_max.get(bin), initVal);

    ASSERT_EQ((NumericType)multi_reduce_sum[bin].get(), initVal);
    ASSERT_EQ((NumericType)multi_reduce_min[bin].get(), initVal);
    ASSERT_EQ((NumericType)multi_reduce_max[bin].get(), initVal);
  }
}

template <typename MultiReducePolicy,
          typename NumericType>
void testMultiReducerSingleInitConstructorBitwise(size_t num_bins, NumericType initVal)
{
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(num_bins, initVal);
  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(num_bins, initVal);

  ASSERT_EQ(multi_reduce_or.size(), num_bins);
  ASSERT_EQ(multi_reduce_and.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_or.get(bin), initVal);
    ASSERT_EQ(multi_reduce_and.get(bin), initVal);

    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), initVal);
    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), initVal);
  }
}

template <typename MultiReducePolicy,
          typename NumericType,
          std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerSingleInitConstructor(size_t num_bins, NumericType initVal)
{
  testMultiReducerSingleInitConstructorRegular< MultiReducePolicy, NumericType >(num_bins, initVal);
  testMultiReducerSingleInitConstructorBitwise< MultiReducePolicy, NumericType >(num_bins, initVal);
}
///
template <typename MultiReducePolicy,
          typename NumericType,
          std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerSingleInitConstructor(size_t num_bins, NumericType initVal)
{
  testMultiReducerSingleInitConstructorRegular< MultiReducePolicy, NumericType >(num_bins, initVal);
}

TYPED_TEST_P(MultiReducerSingleInitConstructorUnitTest, MultiReducerConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  testMultiReducerSingleInitConstructor< MultiReducePolicy, NumericType >(0, NumericType(2));
  testMultiReducerSingleInitConstructor< MultiReducePolicy, NumericType >(1, NumericType(4));
  testMultiReducerSingleInitConstructor< MultiReducePolicy, NumericType >(2, NumericType(0));
  testMultiReducerSingleInitConstructor< MultiReducePolicy, NumericType >(10, NumericType(9));
}


template <typename MultiReducePolicy,
          typename NumericType,
          typename Container>
void testMultiReducerContainerInitConstructorRegular(Container const& container)
{
  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(container);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(container);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(container);

  ASSERT_EQ(multi_reduce_sum.size(), container.size());
  ASSERT_EQ(multi_reduce_min.size(), container.size());
  ASSERT_EQ(multi_reduce_max.size(), container.size());

  size_t bin = 0;
  for (NumericType val : container) {
    ASSERT_EQ(multi_reduce_sum.get(bin), val);
    ASSERT_EQ(multi_reduce_min.get(bin), val);
    ASSERT_EQ(multi_reduce_max.get(bin), val);

    ASSERT_EQ((NumericType)multi_reduce_sum[bin].get(), val);
    ASSERT_EQ((NumericType)multi_reduce_min[bin].get(), val);
    ASSERT_EQ((NumericType)multi_reduce_max[bin].get(), val);
    ++bin;
  }
}

template <typename MultiReducePolicy,
          typename NumericType,
          typename Container>
void testMultiReducerContainerInitConstructorBitwise(Container const& container)
{
  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(container);
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(container);

  ASSERT_EQ(multi_reduce_and.size(), container.size());
  ASSERT_EQ(multi_reduce_or.size(), container.size());

  size_t bin = 0;
  for (NumericType val : container) {
    ASSERT_EQ(multi_reduce_and.get(bin), val);
    ASSERT_EQ(multi_reduce_or.get(bin), val);

    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), val);
    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), val);
    ++bin;
  }
}

template <typename MultiReducePolicy,
          typename NumericType,
          typename Container,
          std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr>
void testMultiReducerContainerInitConstructor(Container const& container)
{
  testMultiReducerContainerInitConstructorRegular< MultiReducePolicy, NumericType >(container);
  testMultiReducerContainerInitConstructorBitwise< MultiReducePolicy, NumericType >(container);
}
///
template <typename MultiReducePolicy,
          typename NumericType,
          typename Container,
          std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr>
void testMultiReducerContainerInitConstructor(Container const& container)
{
  testMultiReducerContainerInitConstructorRegular< MultiReducePolicy, NumericType >(container);
}

TYPED_TEST_P(MultiReducerContainerInitConstructorUnitTest, MultiReducerConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;

  std::vector<NumericType> c0(0);
  std::vector<NumericType> c1(1, 3);
  std::set<NumericType> c2;
  c2.emplace(5);
  c2.emplace(8);
  std::list<NumericType> c10;
  for (size_t bin = 0; bin < size_t(10); ++bin) {
    c10.emplace_front(NumericType(bin));
  }
  testMultiReducerContainerInitConstructor< MultiReducePolicy, NumericType >(c0);
  testMultiReducerContainerInitConstructor< MultiReducePolicy, NumericType >(c1);
  testMultiReducerContainerInitConstructor< MultiReducePolicy, NumericType >(c2);
  testMultiReducerContainerInitConstructor< MultiReducePolicy, NumericType >(c10);
}


REGISTER_TYPED_TEST_SUITE_P(MultiReducerBasicConstructorUnitTest,
                            MultiReducerConstructor);

REGISTER_TYPED_TEST_SUITE_P(MultiReducerSingleInitConstructorUnitTest,
                            MultiReducerConstructor);

REGISTER_TYPED_TEST_SUITE_P(MultiReducerContainerInitConstructorUnitTest,
                            MultiReducerConstructor);

#endif  //__TEST_MULTI_REDUCER_CONSTRUCTOR__
