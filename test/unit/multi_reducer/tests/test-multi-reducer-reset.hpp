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
class MultiReducerResetUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(MultiReducerResetUnitTest);



template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
struct TestMultiReducerBasicReset
{
  const size_t num_bins;
  const NumericType initVal = NumericType(5);

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(runtime_policy_or_not..., num_bins, initVal);
    RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(runtime_policy_or_not..., num_bins, initVal);
    RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(runtime_policy_or_not..., num_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    }

    multi_reduce_sum.reset(runtime_policy_or_not...);
    multi_reduce_min.reset(runtime_policy_or_not...);
    multi_reduce_max.reset(runtime_policy_or_not...);

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

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(runtime_policy_or_not..., num_bins, initVal);
    RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(runtime_policy_or_not..., num_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_and[bin] &= initVal-1;
          multi_reduce_or[bin] |= initVal+1;
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_and[bin] &= initVal-1;
          multi_reduce_or[bin] |= initVal+1;
        }
      });
    }

    multi_reduce_and.reset(runtime_policy_or_not...);
    multi_reduce_or.reset(runtime_policy_or_not...);

    ASSERT_EQ(multi_reduce_and.size(), num_bins);
    ASSERT_EQ(multi_reduce_or.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(multi_reduce_and.get(bin), get_op_identity(multi_reduce_and));
      ASSERT_EQ(multi_reduce_or.get(bin), get_op_identity(multi_reduce_or));

      ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), get_op_identity(multi_reduce_and));
      ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), get_op_identity(multi_reduce_or));
    }
  }
};

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
struct TestMultiReducerSingleResetSize
{
  const size_t init_bins;
  const size_t num_bins;
  const NumericType initVal;

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(runtime_policy_or_not..., init_bins, initVal);
    RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(runtime_policy_or_not..., init_bins, initVal);
    RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(runtime_policy_or_not..., init_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    }

    multi_reduce_sum.reset(runtime_policy_or_not..., num_bins, initVal);
    multi_reduce_min.reset(runtime_policy_or_not..., num_bins, initVal);
    multi_reduce_max.reset(runtime_policy_or_not..., num_bins, initVal);

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

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(runtime_policy_or_not..., init_bins, initVal);
    RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(runtime_policy_or_not..., init_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_and[bin] &= initVal-1;
          multi_reduce_or[bin] |= initVal+1;
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_and[bin] &= initVal-1;
          multi_reduce_or[bin] |= initVal+1;
        }
      });
    }

    multi_reduce_and.reset(runtime_policy_or_not..., num_bins, initVal);
    multi_reduce_or.reset(runtime_policy_or_not..., num_bins, initVal);

    ASSERT_EQ(multi_reduce_and.size(), num_bins);
    ASSERT_EQ(multi_reduce_or.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(multi_reduce_and.get(bin), initVal);
      ASSERT_EQ(multi_reduce_or.get(bin), initVal);

      ASSERT_EQ((NumericType)multi_reduce_and[bin].get(), initVal);
      ASSERT_EQ((NumericType)multi_reduce_or[bin].get(), initVal);
    }
  }
};

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            typename Container  >
struct TestMultiReducerContainerResetSize
{
  const size_t init_bins;
  Container const& container;

  template < RAJA::Policy... runtime_policy_or_not >
  void test_core(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    const size_t num_bins = container.size();
    NumericType initVal = NumericType(5);

    RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(runtime_policy_or_not..., init_bins, initVal);
    RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(runtime_policy_or_not..., init_bins, initVal);
    RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(runtime_policy_or_not..., init_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_sum[bin] += initVal;
          multi_reduce_min[bin].min(initVal-1);
          multi_reduce_max[bin].max(initVal+1);
        }
      });
    }

    multi_reduce_sum.reset(runtime_policy_or_not..., container);
    multi_reduce_min.reset(runtime_policy_or_not..., container);
    multi_reduce_max.reset(runtime_policy_or_not..., container);

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

  template < RAJA::Policy... runtime_policy_or_not >
  void test_bitwise(RAJA::PolicyList<runtime_policy_or_not...>)
  {
    const size_t num_bins = container.size();
    NumericType initVal = NumericType(5);

    RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(init_bins, initVal);
    RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(init_bins, initVal);

    if constexpr (std::is_base_of_v<RunOnDevice, ForOnePol>) {
      forone<ForOnePol>( [=, *this] RAJA_HOST_DEVICE() {
        for (size_t bin = 0; bin < init_bins; ++bin) {
          multi_reduce_and[bin] &= initVal-1;
          multi_reduce_or[bin] |= initVal+1;
        }
      });
    } else {
      forone<ForOnePol>( [=, *this] () {
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
};


TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerBasicReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value> runtime_policy{};

  auto tester = [](size_t num_bins)
  {
    TestMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol > test{num_bins};

    test.test_core(not_policy);
    test.test_core(runtime_policy);
    if constexpr (std::is_integral_v<NumericType>) {
      test.test_bitwise(not_policy);
      test.test_bitwise(runtime_policy);
    }
  };

  tester(0);
  tester(1);
  tester(2);
  tester(10);
}

TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerSingleReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value> runtime_policy{};

  auto tester = [](size_t num_bins, NumericType initVal)
  {
    auto tester = [&](size_t init_bins)
    {
      TestMultiReducerSingleResetSize< MultiReducePolicy, NumericType, ForOnePol > test{init_bins, num_bins, initVal};

      test.test_core(not_policy);
      test.test_core(runtime_policy);
      if constexpr (std::is_integral_v<NumericType>) {
        test.test_bitwise(not_policy);
        test.test_bitwise(runtime_policy);
      }
    };

    tester(0);
    tester(4);
    tester(num_bins);
  };

  tester(0, NumericType(3));
  tester(1, NumericType(5));
  tester(2, NumericType(0));
  tester(10, NumericType(8));
}

TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerContainerReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr RAJA::PolicyList<> not_policy{};
  static constexpr RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value> runtime_policy{};

  std::vector<NumericType> c0(0);
  std::vector<NumericType> c1(1, 3);
  std::set<NumericType> c2;
  c2.emplace(5);
  c2.emplace(8);
  std::list<NumericType> c10;
  for (size_t bin = 0; bin < size_t(10); ++bin) {
    c10.emplace_front(NumericType(bin));
  }

  auto tester = [](auto const& c)
  {
    auto tester = [&](size_t init_bins)
    {
      TestMultiReducerContainerResetSize< MultiReducePolicy, NumericType, ForOnePol, std::decay_t<decltype(c)> > test{init_bins, c};

      test.test_core(not_policy);
      test.test_core(runtime_policy);
      if constexpr (std::is_integral_v<NumericType>) {
        test.test_bitwise(not_policy);
        test.test_bitwise(runtime_policy);
      }
    };

    tester(0);
    tester(4);
    tester(c.size());
  };

  tester(c0);
  tester(c1);
  tester(c2);
  tester(c10);
}



REGISTER_TYPED_TEST_SUITE_P(MultiReducerResetUnitTest,
                            MultiReducerBasicReset,
                            MultiReducerSingleReset,
                            MultiReducerContainerReset);

#endif  //__TEST_MULTI_REDUCER_RESET__
