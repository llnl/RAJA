//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA multi reducer reset.
///

#ifndef __TEST_MULTI_REDUCER_RESET__
#define __TEST_MULTI_REDUCER_RESET__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-multi-reducer.hpp"

#include <vector>
#include <list>
#include <set>

template <typename T>
class MultiReducerBasicResetUnitTest : public ::testing::Test
{
};

template <typename T>
class MultiReducerSingleResetUnitTest : public ::testing::Test
{
};

template <typename T>
class MultiReducerContainerResetUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(MultiReducerBasicResetUnitTest);
TYPED_TEST_SUITE_P(MultiReducerSingleResetUnitTest);
TYPED_TEST_SUITE_P(MultiReducerContainerResetUnitTest);



template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerBasicResetRegular(bool use_reducer, size_t num_bins)
{
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(num_bins, initVal);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(num_bins, initVal);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(num_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < num_bins; ++bin) {
        multi_reduce_sum[bin] += initVal;
        multi_reduce_min[bin].min(initVal-1);
        multi_reduce_max[bin].max(initVal+1);
      }
    });
  }

  multi_reduce_sum.reset();
  multi_reduce_min.reset();
  multi_reduce_max.reset();

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

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerBasicResetBitwise(bool use_reducer, size_t num_bins)
{
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(num_bins, initVal);
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(num_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < num_bins; ++bin) {
        multi_reduce_and[bin] &= initVal-1;
        multi_reduce_or[bin] |= initVal+1;
      }
    });
  }

  multi_reduce_and.reset();
  multi_reduce_or.reset();

  ASSERT_EQ(multi_reduce_and.size(), num_bins);
  ASSERT_EQ(multi_reduce_or.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_and.get(bin), get_op_identity(multi_reduce_and));
    ASSERT_EQ(multi_reduce_or.get(bin), get_op_identity(multi_reduce_or));

    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), get_op_identity(multi_reduce_and));
    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), get_op_identity(multi_reduce_or));
  }
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerBasicReset(size_t num_bins)
{
  testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, num_bins);
  testMultiReducerBasicResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(false, num_bins);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, num_bins);
  // testMultiReducerBasicResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(true, num_bins);
}
///
template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerBasicReset(size_t num_bins)
{
  testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, num_bins);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, num_bins);
}

TYPED_TEST_P(MultiReducerBasicResetUnitTest, MultiReducerReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(0);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(1);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(2);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(10);
}



template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerSingleResetRegular(bool use_reducer, size_t init_bins, size_t num_bins, NumericType initVal)
{
  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(init_bins, initVal);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(init_bins, initVal);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(init_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < init_bins; ++bin) {
        multi_reduce_sum[bin] += initVal;
        multi_reduce_min[bin].min(initVal-1);
        multi_reduce_max[bin].max(initVal+1);
      }
    });
  }

  multi_reduce_sum.reset(num_bins, initVal);
  multi_reduce_min.reset(num_bins, initVal);
  multi_reduce_max.reset(num_bins, initVal);

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

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerSingleResetBitwise(bool use_reducer, size_t init_bins, size_t num_bins, NumericType initVal)
{
  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(init_bins, initVal);
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(init_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < init_bins; ++bin) {
        multi_reduce_and[bin] &= initVal-1;
        multi_reduce_or[bin] |= initVal+1;
      }
    });
  }

  multi_reduce_and.reset(num_bins, initVal);
  multi_reduce_or.reset(num_bins, initVal);

  ASSERT_EQ(multi_reduce_and.size(), num_bins);
  ASSERT_EQ(multi_reduce_or.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_and.get(bin), initVal);
    ASSERT_EQ(multi_reduce_or.get(bin), initVal);

    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), initVal);
    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), initVal);
  }
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerSingleResetSize(size_t init_bins, size_t num_bins, NumericType initVal)
{
  testMultiReducerSingleResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, num_bins, initVal);
  testMultiReducerSingleResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, num_bins, initVal);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerSingleResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, num_bins, initVal);
  // testMultiReducerSingleResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, num_bins, initVal);
}
///
template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerSingleResetSize(size_t init_bins, size_t num_bins, NumericType initVal)
{
  testMultiReducerSingleResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, num_bins, initVal);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerSingleResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, num_bins, initVal);
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol >
void testMultiReducerSingleReset(size_t num_bins, NumericType initVal)
{
  testMultiReducerSingleResetSize< MultiReducePolicy, NumericType, ForOnePol >(0, num_bins, initVal);
  testMultiReducerSingleResetSize< MultiReducePolicy, NumericType, ForOnePol >(4, num_bins, initVal);
  testMultiReducerSingleResetSize< MultiReducePolicy, NumericType, ForOnePol >(num_bins, num_bins, initVal);
}

TYPED_TEST_P(MultiReducerSingleResetUnitTest, MultiReducerReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  testMultiReducerSingleReset< MultiReducePolicy, NumericType, ForOnePol >(0, NumericType(3));
  testMultiReducerSingleReset< MultiReducePolicy, NumericType, ForOnePol >(1, NumericType(5));
  testMultiReducerSingleReset< MultiReducePolicy, NumericType, ForOnePol >(2, NumericType(0));
  testMultiReducerSingleReset< MultiReducePolicy, NumericType, ForOnePol >(10, NumericType(8));
}



template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container  >
void testMultiReducerContainerResetRegular(bool use_reducer, size_t init_bins, Container const& container)
{
  const size_t num_bins = container.size();
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(init_bins, initVal);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(init_bins, initVal);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(init_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < init_bins; ++bin) {
        multi_reduce_sum[bin] += initVal;
        multi_reduce_min[bin].min(initVal-1);
        multi_reduce_max[bin].max(initVal+1);
      }
    });
  }

  multi_reduce_sum.reset(container);
  multi_reduce_min.reset(container);
  multi_reduce_max.reset(container);

  ASSERT_EQ(multi_reduce_sum.size(), num_bins);
  ASSERT_EQ(multi_reduce_min.size(), num_bins);
  ASSERT_EQ(multi_reduce_max.size(), num_bins);

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

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container >
void testMultiReducerContainerResetBitwise(bool use_reducer, size_t init_bins, Container const& container)
{
  const size_t num_bins = container.size();
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(init_bins, initVal);
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(init_bins, initVal);

  if (use_reducer) {
    forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
      for (size_t bin = 0; bin < init_bins; ++bin) {
        multi_reduce_and[bin] &= initVal-1;
        multi_reduce_or[bin] |= initVal+1;
      }
    });
  }

  multi_reduce_and.reset(container);
  multi_reduce_or.reset(container);

  ASSERT_EQ(multi_reduce_and.size(), num_bins);
  ASSERT_EQ(multi_reduce_or.size(), num_bins);

  size_t bin = 0;
  for (NumericType val : container) {
    ASSERT_EQ(multi_reduce_and.get(bin), val);
    ASSERT_EQ(multi_reduce_or.get(bin), val);

    ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), val);
    ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), val);
    ++bin;
  }
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container,
            std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerContainerResetSize(size_t init_bins, Container const& container)
{
  testMultiReducerContainerResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, container);
  testMultiReducerContainerResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, container);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerContainerResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, container);
  // testMultiReducerContainerResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, container);
}
///
template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container,
            std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerContainerResetSize(size_t init_bins, Container const& container)
{
  testMultiReducerContainerResetRegular< MultiReducePolicy, NumericType, ForOnePol >(false, init_bins, container);
  // avoid using the reducer as forone does not handle reducers correctly
  // forone does not make_lambda_body or privatize the body
  // testMultiReducerContainerResetRegular< MultiReducePolicy, NumericType, ForOnePol >(true, init_bins, container);
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container >
void testMultiReducerContainerReset(Container const& container)
{
  testMultiReducerContainerResetSize< MultiReducePolicy, NumericType, ForOnePol >(0, container);
  testMultiReducerContainerResetSize< MultiReducePolicy, NumericType, ForOnePol >(4, container);
  testMultiReducerContainerResetSize< MultiReducePolicy, NumericType, ForOnePol >(container.size(), container);
}

TYPED_TEST_P(MultiReducerContainerResetUnitTest, MultiReducerReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  std::vector<NumericType> c0(0);
  std::vector<NumericType> c1(1, 3);
  std::set<NumericType> c2;
  c2.emplace(5);
  c2.emplace(8);
  std::list<NumericType> c10;
  for (size_t bin = 0; bin < size_t(10); ++bin) {
    c10.emplace_front(NumericType(bin));
  }
  testMultiReducerContainerReset< MultiReducePolicy, NumericType, ForOnePol >(c0);
  testMultiReducerContainerReset< MultiReducePolicy, NumericType, ForOnePol >(c1);
  testMultiReducerContainerReset< MultiReducePolicy, NumericType, ForOnePol >(c2);
  testMultiReducerContainerReset< MultiReducePolicy, NumericType, ForOnePol >(c10);
}



REGISTER_TYPED_TEST_SUITE_P(MultiReducerBasicResetUnitTest,
                            MultiReducerReset);

REGISTER_TYPED_TEST_SUITE_P(MultiReducerSingleResetUnitTest,
                            MultiReducerReset);

REGISTER_TYPED_TEST_SUITE_P(MultiReducerContainerResetUnitTest,
                            MultiReducerReset);

#endif  //__TEST_MULTI_REDUCER_RESET__
