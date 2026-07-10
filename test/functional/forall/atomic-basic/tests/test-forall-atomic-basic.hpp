//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall.
///

#ifndef __TEST_FORALL_ATOMIC_BASIC_HPP__
#define __TEST_FORALL_ATOMIC_BASIC_HPP__

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

// segment multiplexer
template< typename IdxType, typename SegType >
struct RSMultiplexer {};

template< typename IdxType >
struct RSMultiplexer<IdxType, RAJA::TypedRangeSegment<IdxType>>
{
  RAJA::TypedRangeSegment<IdxType>
  makeseg(IdxType N, camp::resources::Resource RAJA_UNUSED_ARG(work_res))
  {
    return RAJA::TypedRangeSegment<IdxType>(0, N);
  }
};

template< typename IdxType >
struct RSMultiplexer<IdxType, RAJA::TypedRangeStrideSegment<IdxType>>
{
  RAJA::TypedRangeStrideSegment<IdxType>
  makeseg(IdxType N, camp::resources::Resource RAJA_UNUSED_ARG(work_res))
  {
    return RAJA::TypedRangeStrideSegment<IdxType>(0, N, 1);
  }
};

template< typename IdxType >
struct RSMultiplexer<IdxType, RAJA::TypedListSegment<IdxType>>
{
  RAJA::TypedListSegment<IdxType>
  makeseg(IdxType N, camp::resources::Resource work_res)
  {
    std::vector<IdxType> temp(N);
    std::iota(std::begin(temp), std::end(temp), 0);
    return RAJA::TypedListSegment<IdxType>(&temp[0],
                                           static_cast<size_t>(temp.size()),
                                           work_res);
  }
};
// end segment multiplexer

namespace forall_atomic_basic_test
{

constexpr RAJA::Index_type atomic_test_length = 16384;

template <typename IdxType>
RAJA_HOST_DEVICE RAJA_INLINE size_t stripIndex(IdxType idx)
{
  return static_cast<size_t>(RAJA::stripIndexType(idx));
}

template <typename T>
std::vector<T> sortedValues(const T* values, size_t len)
{
  std::vector<T> sorted(values, values + len);
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

template <typename T>
void expectUnorderedValuesEq(const T* actual_values,
                             size_t len,
                             std::vector<T> expected_values)
{
  std::sort(expected_values.begin(), expected_values.end());
  EXPECT_EQ(expected_values, sortedValues(actual_values, len));
}

template <typename T>
void expectAllValuesEq(const T* actual_values, size_t len, T expected_value)
{
  for (size_t i = 0; i < len; ++i) {
    EXPECT_EQ(expected_value, actual_values[i]);
  }
}

template <typename T>
struct AtomicTestData
{
  explicit AtomicTestData(size_t num_returns, camp::resources::Resource resource)
    : len(num_returns), work_res(resource)
  {
    allocateForallTestData<T>(1,
                              work_res,
                              &work_value,
                              &check_value,
                              &test_value);
    allocateForallTestData<T>(len,
                              work_res,
                              &work_returns,
                              &check_returns,
                              &test_returns);
  }

  ~AtomicTestData()
  {
    deallocateForallTestData<T>(work_res,
                                work_value,
                                check_value,
                                test_value);
    deallocateForallTestData<T>(work_res,
                                work_returns,
                                check_returns,
                                test_returns);
  }

  void copyValueToWork()
  {
    work_res.memcpy(work_value, test_value, sizeof(T));
  }

  void copyBack(bool copy_returns = true)
  {
    work_res.memcpy(check_value, work_value, sizeof(T));
    if (copy_returns) {
      work_res.memcpy(check_returns, work_returns, sizeof(T) * len);
    }
    work_res.wait();
  }

  size_t len;
  camp::resources::Resource work_res;
  T* work_value {};
  T* check_value {};
  T* test_value {};
  T* work_returns {};
  T* check_returns {};
  T* test_returns {};
};

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicAddTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicAdd<AtomicPolicy>(work_value, static_cast<T>(1));
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = static_cast<T>(i);
  }

  EXPECT_EQ(static_cast<T>(seglimit), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicSubTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(seglimit);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicSub<AtomicPolicy>(work_value, static_cast<T>(1));
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = static_cast<T>(i + 1);
  }

  EXPECT_EQ(static_cast<T>(0), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicMinTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  // Use a uniform update value so the return multiset is deterministic:
  // exactly one thread observes the initial value and all others observe 0.
  data.test_value[0] = static_cast<T>(1);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    RAJA_UNUSED_VAR(i);
    work_returns[stripIndex(i)] =
      RAJA::atomicMin<AtomicPolicy>(work_value, static_cast<T>(0));
  });

  data.copyBack();

  std::vector<T> expected(data.len, static_cast<T>(0));
  expected[0] = static_cast<T>(1);

  EXPECT_EQ(static_cast<T>(0), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicMaxTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  // Use a uniform update value so the return multiset is deterministic:
  // exactly one thread observes the initial value and all others observe 1.
  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    RAJA_UNUSED_VAR(i);
    work_returns[stripIndex(i)] =
      RAJA::atomicMax<AtomicPolicy>(work_value, static_cast<T>(1));
  });

  data.copyBack();

  std::vector<T> expected(data.len, static_cast<T>(1));
  expected[0] = static_cast<T>(0);

  EXPECT_EQ(static_cast<T>(1), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicIncTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicInc<AtomicPolicy>(work_value);
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = static_cast<T>(i);
  }

  EXPECT_EQ(static_cast<T>(seglimit), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicDecTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(seglimit);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicDec<AtomicPolicy>(work_value);
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = static_cast<T>(i + 1);
  }

  EXPECT_EQ(static_cast<T>(0), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicIncBoundedTestImpl(IdxType seglimit)
{
  constexpr T compare = static_cast<T>(16);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicInc<AtomicPolicy>(work_value, compare);
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  T current = static_cast<T>(0);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = current;
    current = compare <= current ? static_cast<T>(0) : current + static_cast<T>(1);
  }

  EXPECT_EQ(current, data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicDecBoundedTestImpl(IdxType seglimit)
{
  constexpr T compare = static_cast<T>(16);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = compare;
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicDec<AtomicPolicy>(work_value, compare);
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  T current = compare;
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = current;
    current = current == static_cast<T>(0) || compare < current
                ? compare
                : current - static_cast<T>(1);
  }

  EXPECT_EQ(current, data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicExchangeTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(seglimit);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicExchange<AtomicPolicy>(work_value, static_cast<T>(i));
  });

  data.copyBack();

  EXPECT_LE(static_cast<T>(0), data.check_value[0]);
  EXPECT_GT(static_cast<T>(seglimit), data.check_value[0]);

  std::vector<T> expected;
  expected.reserve(data.len);
  expected.push_back(static_cast<T>(seglimit));
  for (size_t i = 0; i < data.len; ++i) {
    T value = static_cast<T>(i);
    if (value != data.check_value[0]) {
      expected.push_back(value);
    }
  }

  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicCASTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicCAS<AtomicPolicy>(work_value,
                                    static_cast<T>(0),
                                    static_cast<T>(1));
  });

  data.copyBack();

  std::vector<T> expected(data.len, static_cast<T>(1));
  expected[0] = static_cast<T>(0);

  EXPECT_EQ(static_cast<T>(1), data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicLoadTestImpl(IdxType seglimit)
{
  constexpr T loaded_value = static_cast<T>(7);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = loaded_value;
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicLoad<AtomicPolicy>(work_value);
  });

  data.copyBack();

  EXPECT_EQ(loaded_value, data.check_value[0]);
  expectAllValuesEq(data.check_returns, data.len, loaded_value);
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicStoreTestImpl(IdxType seglimit)
{
  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    RAJA_UNUSED_VAR(i);
    RAJA::atomicStore<AtomicPolicy>(work_value, static_cast<T>(1));
  });

  data.copyBack(false);

  EXPECT_EQ(static_cast<T>(1), data.check_value[0]);
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename SegmentType,
          typename T>
void ForallAtomicGenericTestImpl(IdxType seglimit)
{
  constexpr T cycle_limit = static_cast<T>(9);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  SegmentType seg = RSMultiplexer<IdxType, SegmentType>().makeseg(seglimit, work_res);
  AtomicTestData<T> data(stripIndex(seglimit), work_res);

  data.test_value[0] = static_cast<T>(0);
  data.copyValueToWork();

  T* work_value = data.work_value;
  T* work_returns = data.work_returns;

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    work_returns[stripIndex(i)] =
      RAJA::atomicGeneric<AtomicPolicy>(work_value, [=](T old) {
        return old >= cycle_limit ? static_cast<T>(0) : old + static_cast<T>(1);
      });
  });

  data.copyBack();

  std::vector<T> expected(data.len);
  T current = static_cast<T>(0);
  for (size_t i = 0; i < data.len; ++i) {
    expected[i] = current;
    current = current >= cycle_limit ? static_cast<T>(0) : current + static_cast<T>(1);
  }

  EXPECT_EQ(current, data.check_value[0]);
  expectUnorderedValuesEq(data.check_returns, data.len, std::move(expected));
}

}  // namespace forall_atomic_basic_test

#define FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(TEST_IMPL, SEGLIMIT)               \
  TEST_IMPL<AExec, APol, ResType, IdxType, RAJA::TypedRangeSegment<IdxType>,       \
            DType>(SEGLIMIT);                                                      \
  TEST_IMPL<AExec, APol, ResType, IdxType, RAJA::TypedRangeStrideSegment<IdxType>, \
            DType>(SEGLIMIT);                                                      \
  TEST_IMPL<AExec, APol, ResType, IdxType, RAJA::TypedListSegment<IdxType>,        \
            DType>(SEGLIMIT)

TYPED_TEST_SUITE_P(ForallAtomicBasicTest);
template <typename T>
class ForallAtomicBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicBasicTest, AtomicAddForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicAddTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicSubForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicSubTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicMinForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicMinTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicMaxForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicMaxTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicIncForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicIncTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicDecForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicDecTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicIncBoundedForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(
    forall_atomic_basic_test::ForallAtomicIncBoundedTestImpl,
    forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicDecBoundedForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(
    forall_atomic_basic_test::ForallAtomicDecBoundedTestImpl,
    forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicExchangeForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(
    forall_atomic_basic_test::ForallAtomicExchangeTestImpl,
    forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicCASForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicCASTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicLoadForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicLoadTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicStoreForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicStoreTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

TYPED_TEST_P(ForallAtomicBasicTest, AtomicGenericForall)
{
  using AExec = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType = typename camp::at<TypeParam, camp::num<4>>::type;

  FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS(forall_atomic_basic_test::ForallAtomicGenericTestImpl,
                                          forall_atomic_basic_test::atomic_test_length);
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicBasicTest,
                            AtomicAddForall,
                            AtomicSubForall,
                            AtomicMinForall,
                            AtomicMaxForall,
                            AtomicIncForall,
                            AtomicDecForall,
                            AtomicIncBoundedForall,
                            AtomicDecBoundedForall,
                            AtomicExchangeForall,
                            AtomicCASForall,
                            AtomicLoadForall,
                            AtomicStoreForall,
                            AtomicGenericForall);

#undef FORALL_ATOMIC_BASIC_RUN_ON_ALL_SEGMENTS

#endif  // __TEST_FORALL_ATOMIC_BASIC_HPP__
