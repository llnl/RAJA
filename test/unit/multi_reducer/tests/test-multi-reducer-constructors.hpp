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
#include "RAJA_test-reducer-api.hpp"

#include "../test-multi-reducer.hpp"

#include <vector>
#include <list>
#include <set>

template <typename T>
class MultiReducerConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(MultiReducerConstructorUnitTest);


template <typename MultiReducePolicy,
          typename NumericType>
struct TestBasicMultiReducerConstructor
{
  const size_t num_bins;

  template < typename Api >
  void test_core(Api api)
  {
    auto multi_reduce_sum =
        api.template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
            num_bins);
    auto multi_reduce_min =
        api.template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
            num_bins);
    auto multi_reduce_max =
        api.template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
            num_bins);

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

  template < typename Api >
  void test_bitwise(Api api)
  {
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            num_bins);
    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            num_bins);

    ASSERT_EQ(multi_reduce_or.size(), num_bins);
    ASSERT_EQ(multi_reduce_and.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(multi_reduce_or.get(bin), get_op_identity(multi_reduce_or));
      ASSERT_EQ(multi_reduce_and.get(bin), get_op_identity(multi_reduce_and));

      ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), get_op_identity(multi_reduce_or));
      ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), get_op_identity(multi_reduce_and));
    }
  }
};

template <typename MultiReducePolicy,
          typename NumericType>
struct TestMultiReducerSingleInitConstructor
{
  const size_t num_bins;
  const NumericType initVal;

  template < typename Api >
  void test_core(Api api)
  {
    auto multi_reduce_sum =
        api.template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
            num_bins, initVal);
    auto multi_reduce_min =
        api.template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
            num_bins, initVal);
    auto multi_reduce_max =
        api.template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
            num_bins, initVal);

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

  template < typename Api >
  void test_bitwise(Api api)
  {
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            num_bins, initVal);
    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            num_bins, initVal);

    ASSERT_EQ(multi_reduce_or.size(), num_bins);
    ASSERT_EQ(multi_reduce_and.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(multi_reduce_or.get(bin), initVal);
      ASSERT_EQ(multi_reduce_and.get(bin), initVal);

      ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), initVal);
      ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), initVal);
    }
  }
};

template <typename MultiReducePolicy,
          typename NumericType,
          typename Container>
struct TestMultiReducerContainerInitConstructor
{
  Container const& container;

  template < typename Api >
  void test_core(Api api)
  {
    auto multi_reduce_sum =
        api.template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
            container);
    auto multi_reduce_min =
        api.template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
            container);
    auto multi_reduce_max =
        api.template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
            container);

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

  template < typename Api >
  void test_bitwise(Api api)
  {
    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            container);
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            container);

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
};


TYPED_TEST_P(MultiReducerConstructorUnitTest, MultiReducerBasicConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>> runtime_api{};

  auto tester = [](size_t num_bins)
  {
    TestBasicMultiReducerConstructor< MultiReducePolicy, NumericType > test{num_bins};

    test.test_core(legacy_api);
    test.test_core(runtime_api);
    if constexpr (std::is_integral_v<NumericType>) {
      test.test_bitwise(legacy_api);
      test.test_bitwise(runtime_api);
    }
  };

  tester(0);
  tester(1);
  tester(2);
  tester(10);
}

TYPED_TEST_P(MultiReducerConstructorUnitTest, MultiReducerSingleInitConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>> runtime_api{};

  auto tester = [](size_t num_bins, NumericType initVal)
  {
    TestMultiReducerSingleInitConstructor< MultiReducePolicy, NumericType > test{num_bins, initVal};

    test.test_core(legacy_api);
    test.test_core(runtime_api);
    if constexpr (std::is_integral_v<NumericType>) {
      test.test_bitwise(legacy_api);
      test.test_bitwise(runtime_api);
    }
  };

  tester(0, NumericType(2));
  tester(1, NumericType(4));
  tester(2, NumericType(0));
  tester(10, NumericType(9));
}

TYPED_TEST_P(MultiReducerConstructorUnitTest, MultiReducerContainerInitConstructor)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>> runtime_api{};

  std::vector<NumericType> c0(0);
  std::vector<NumericType> c1(1, 3);
  std::set<NumericType> c2;
  c2.emplace(5);
  c2.emplace(8);
  std::list<NumericType> c10;
  for (size_t bin = 0; bin < size_t(10); ++bin) {
    c10.emplace_front(NumericType(bin));
  }

  auto tester = [&](auto const& c)
  {
    TestMultiReducerContainerInitConstructor< MultiReducePolicy, NumericType, std::decay_t<decltype(c)> > test{c};

    test.test_core(legacy_api);
    test.test_core(runtime_api);
    if constexpr (std::is_integral_v<NumericType>) {
      test.test_bitwise(legacy_api);
      test.test_bitwise(runtime_api);
    }
  };

  tester(c0);
  tester(c1);
  tester(c2);
  tester(c10);
}


REGISTER_TYPED_TEST_SUITE_P(MultiReducerConstructorUnitTest,
                            MultiReducerBasicConstructor,
                            MultiReducerSingleInitConstructor,
                            MultiReducerContainerInitConstructor);

#endif  //__TEST_MULTI_REDUCER_CONSTRUCTOR__
