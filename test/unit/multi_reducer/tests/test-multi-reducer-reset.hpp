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
#include "RAJA_test-reducer-api.hpp"

#include "../test-multi-reducer.hpp"

#include <type_traits>
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

    api.reset(multi_reduce_sum);
    api.reset(multi_reduce_min);
    api.reset(multi_reduce_max);

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
    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            num_bins, initVal);
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            num_bins, initVal);

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

    api.reset(multi_reduce_and);
    api.reset(multi_reduce_or);

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

  template < typename Api >
  void test_core(Api api)
  {
    auto multi_reduce_sum =
        api.template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_min =
        api.template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_max =
        api.template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
            init_bins, initVal);

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

    api.reset(multi_reduce_sum, num_bins, initVal);
    api.reset(multi_reduce_min, num_bins, initVal);
    api.reset(multi_reduce_max, num_bins, initVal);

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
    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            init_bins, initVal);

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

    api.reset(multi_reduce_and, num_bins, initVal);
    api.reset(multi_reduce_or, num_bins, initVal);

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

  template < typename Api >
  void test_core(Api api)
  {
    const size_t num_bins = container.size();
    NumericType initVal = NumericType(5);

    auto multi_reduce_sum =
        api.template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_min =
        api.template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_max =
        api.template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
            init_bins, initVal);

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

    api.reset(multi_reduce_sum, container);
    api.reset(multi_reduce_min, container);
    api.reset(multi_reduce_max, container);

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

  template < typename Api >
  void test_bitwise(Api api)
  {
    const size_t num_bins = container.size();
    NumericType initVal = NumericType(5);

    auto multi_reduce_and =
        api.template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
            init_bins, initVal);
    auto multi_reduce_or =
        api.template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
            init_bins, initVal);

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

    api.reset(multi_reduce_and, container);
    api.reset(multi_reduce_or, container);

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


template <typename MultiReducePolicy,
          typename NumericType>
struct CoreMultiReducerTransitionHelper
{
  static constexpr size_t num_bins = 4;

  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> transition_sum;
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> transition_min;
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> transition_max;

  template <typename ConstructPhase>
  CoreMultiReducerTransitionHelper(ConstructPhase, NumericType initVal)
      : transition_sum(
            ConstructPhase::Api::template make<RAJA::MultiReduceSum<MultiReducePolicy, NumericType>>(
                num_bins, initVal)),
        transition_min(
            ConstructPhase::Api::template make<RAJA::MultiReduceMin<MultiReducePolicy, NumericType>>(
                num_bins, initVal)),
        transition_max(
            ConstructPhase::Api::template make<RAJA::MultiReduceMax<MultiReducePolicy, NumericType>>(
                num_bins, initVal))
  {
  }

  template <typename Phase>
  void phase(Phase, NumericType initVal)
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    Api::reset(transition_sum, num_bins, initVal);
    Api::reset(transition_min, num_bins, initVal);
    Api::reset(transition_max, num_bins, initVal);

    ASSERT_EQ(transition_sum.size(), num_bins);
    ASSERT_EQ(transition_min.size(), num_bins);
    ASSERT_EQ(transition_max.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(transition_sum.get(bin), initVal);
      ASSERT_EQ(transition_min.get(bin), initVal);
      ASSERT_EQ(transition_max.get(bin), initVal);

      ASSERT_EQ((NumericType)transition_sum[bin].get(), initVal);
      ASSERT_EQ((NumericType)transition_min[bin].get(), initVal);
      ASSERT_EQ((NumericType)transition_max[bin].get(), initVal);
    }

    auto& multi_reduce_sum = transition_sum;
    auto& multi_reduce_min = transition_min;
    auto& multi_reduce_max = transition_max;

    const NumericType addend = NumericType(3);
    const NumericType delta = NumericType(2);

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_sum[bin] += addend;
          multi_reduce_min[bin].min(initVal - delta);
          multi_reduce_max[bin].max(initVal + delta);
        }
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_sum[bin] += addend;
          multi_reduce_min[bin].min(initVal - delta);
          multi_reduce_max[bin].max(initVal + delta);
        }
      });
    }

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(transition_sum.get(bin), NumericType(initVal + addend));
      ASSERT_EQ(transition_min.get(bin), NumericType(initVal - delta));
      ASSERT_EQ(transition_max.get(bin), NumericType(initVal + delta));

      ASSERT_EQ((NumericType)transition_sum[bin].get(),
                NumericType(initVal + addend));
      ASSERT_EQ((NumericType)transition_min[bin].get(),
                NumericType(initVal - delta));
      ASSERT_EQ((NumericType)transition_max[bin].get(),
                NumericType(initVal + delta));
    }
  }
};

template <typename MultiReducePolicy,
          typename NumericType>
struct BitwiseMultiReducerTransitionHelper
{
  static constexpr size_t num_bins = 4;

  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> transition_bitand;
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> transition_bitor;

  template <typename ConstructPhase>
  BitwiseMultiReducerTransitionHelper(ConstructPhase, NumericType initVal)
      : transition_bitand(
            ConstructPhase::Api::template make<RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType>>(
                num_bins, initVal)),
        transition_bitor(
            ConstructPhase::Api::template make<RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType>>(
                num_bins, initVal))
  {
  }

  template <typename Phase>
  void phase(Phase, NumericType initVal)
  {
    using Api = typename Phase::Api;
    using PhaseForOnePol = typename Phase::ForOne;

    Api::reset(transition_bitand, num_bins, initVal);
    Api::reset(transition_bitor, num_bins, initVal);

    ASSERT_EQ(transition_bitand.size(), num_bins);
    ASSERT_EQ(transition_bitor.size(), num_bins);

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(transition_bitand.get(bin), initVal);
      ASSERT_EQ(transition_bitor.get(bin), initVal);

      ASSERT_EQ((NumericType)transition_bitand[bin].get(), initVal);
      ASSERT_EQ((NumericType)transition_bitor[bin].get(), initVal);
    }

    auto& multi_reduce_bitand = transition_bitand;
    auto& multi_reduce_bitor = transition_bitor;

    const NumericType andMask = NumericType(7);
    const NumericType orMask = NumericType(2);

    if constexpr (std::is_base_of_v<RunOnDevice, PhaseForOnePol>) {
      forone<PhaseForOnePol>( [=] RAJA_HOST_DEVICE () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_bitand[bin] &= andMask;
          multi_reduce_bitor[bin] |= orMask;
        }
      });
    } else {
      forone<PhaseForOnePol>( [=] () {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          multi_reduce_bitand[bin] &= andMask;
          multi_reduce_bitor[bin] |= orMask;
        }
      });
    }

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(transition_bitand.get(bin), NumericType(initVal & andMask));
      ASSERT_EQ(transition_bitor.get(bin), NumericType(initVal | orMask));

      ASSERT_EQ((NumericType)transition_bitand[bin].get(),
                NumericType(initVal & andMask));
      ASSERT_EQ((NumericType)transition_bitor[bin].get(),
                NumericType(initVal | orMask));
    }
  }
};

template <typename MultiReducePolicy,
          typename NumericType,
          typename ForOnePol,
          template <typename, typename> class Helper>
void run_multi_reducer_reset_transition_chains()
{
  using SeqPhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<RAJA::Policy::sequential>>,
                      test_seq>;
  using RuntimePhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>>,
                      ForOnePol>;
  using LegacySeqPhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<>>,
                      test_seq>;
  using LegacyRuntimePhase =
      TransitionPhase<ReducerApi<RAJA::PolicyList<>>,
                      ForOnePol>;

  {
    Helper<MultiReducePolicy, NumericType> tester(
                 SeqPhase{},           NumericType(10));
    tester.phase(RuntimePhase{},       NumericType(20));
    tester.phase(SeqPhase{},           NumericType(30));
    tester.phase(RuntimePhase{},       NumericType(40));
    tester.phase(LegacyRuntimePhase{}, NumericType(50));
    tester.phase(RuntimePhase{},       NumericType(60));
  }

  {
    Helper<MultiReducePolicy, NumericType> tester(
                 RuntimePhase{},       NumericType(110));
    tester.phase(SeqPhase{},           NumericType(120));
    tester.phase(RuntimePhase{},       NumericType(130));
    tester.phase(LegacyRuntimePhase{}, NumericType(140));
  }

  {
    Helper<MultiReducePolicy, NumericType> tester(
                 LegacyRuntimePhase{}, NumericType(210));
    tester.phase(LegacySeqPhase{},     NumericType(220));
    tester.phase(LegacyRuntimePhase{}, NumericType(230));
    tester.phase(SeqPhase{},           NumericType(240));
    tester.phase(LegacySeqPhase{},     NumericType(250));
  }
}


TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerBasicReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>> runtime_api{};

  auto tester = [](size_t num_bins)
  {
    TestMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol > test{num_bins};

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

TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerSingleReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;
  static constexpr ReducerApi<RAJA::PolicyList<>> legacy_api{};
  static constexpr ReducerApi<RAJA::PolicyList<RAJA::policy_of<MultiReducePolicy>::value>> runtime_api{};

  auto tester = [](size_t num_bins, NumericType initVal)
  {
    auto tester = [&](size_t init_bins)
    {
      TestMultiReducerSingleResetSize< MultiReducePolicy, NumericType, ForOnePol > test{init_bins, num_bins, initVal};

      test.test_core(legacy_api);
      test.test_core(runtime_api);
      if constexpr (std::is_integral_v<NumericType>) {
        test.test_bitwise(legacy_api);
        test.test_bitwise(runtime_api);
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

  auto tester = [](auto const& c)
  {
    auto tester = [&](size_t init_bins)
    {
      TestMultiReducerContainerResetSize< MultiReducePolicy, NumericType, ForOnePol, std::decay_t<decltype(c)> > test{init_bins, c};

      test.test_core(legacy_api);
      test.test_core(runtime_api);
      if constexpr (std::is_integral_v<NumericType>) {
        test.test_bitwise(legacy_api);
        test.test_bitwise(runtime_api);
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

TYPED_TEST_P(MultiReducerResetUnitTest, MultiReducerResetTransition)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  run_multi_reducer_reset_transition_chains<
      MultiReducePolicy, NumericType, ForOnePol,
      CoreMultiReducerTransitionHelper>();
  if constexpr (std::is_integral_v<NumericType>) {
    run_multi_reducer_reset_transition_chains<
        MultiReducePolicy, NumericType, ForOnePol,
        BitwiseMultiReducerTransitionHelper>();
  }
}



REGISTER_TYPED_TEST_SUITE_P(MultiReducerResetUnitTest,
                            MultiReducerBasicReset,
                            MultiReducerSingleReset,
                            MultiReducerContainerReset,
                            MultiReducerResetTransition);

#endif  //__TEST_MULTI_REDUCER_RESET__
