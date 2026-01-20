//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RangeSegment
///

#include "RAJA_test-base.hpp"

#include "RAJA_unit-test-types.hpp"

template<typename T>
class RangeSegmentUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(RangeSegmentUnitTest, UnitIndexTypes);


template< typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void NegativeRangeSegConstructorsTest()
{
}

template< typename T, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr>
void NegativeRangeSegConstructorsTest()
{
  RAJA::TypedRangeSegment<T> r1(-10, 7);
  RAJA::TypedRangeSegment<T> r3(-13, -1);
  ASSERT_EQ(17, r1.size());
  ASSERT_EQ(12, r3.size());
  // Test clamping when begin > end
  RAJA::TypedRangeSegment<T> smaller(T(0), T(-50));
  ASSERT_EQ(smaller.begin(), smaller.end());
}

TYPED_TEST(RangeSegmentUnitTest, Constructors)
{
  RAJA::TypedRangeSegment<TypeParam> first(0, 10);
  RAJA::TypedRangeSegment<TypeParam> copied(first);

  ASSERT_EQ(first, copied);

  RAJA::TypedRangeSegment<TypeParam> moved(std::move(first));

  ASSERT_EQ(moved, copied);

  // Test clamping when begin > end
  RAJA::TypedRangeSegment<TypeParam> smaller(20, 19);
  ASSERT_EQ(smaller.begin(), smaller.end());

  NegativeRangeSegConstructorsTest<TypeParam>();
}

TYPED_TEST(RangeSegmentUnitTest, Assignments)
{
  auto r = RAJA::TypedRangeSegment<TypeParam>(RAJA::Index_type(), 5);
  RAJA::TypedRangeSegment<TypeParam> seg1 = r;
  ASSERT_EQ(r, seg1);
  RAJA::TypedRangeSegment<TypeParam> seg2 = std::move(r);
  ASSERT_EQ(seg2, seg1);
}

TYPED_TEST(RangeSegmentUnitTest, Swaps)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 5);
  RAJA::TypedRangeSegment<TypeParam> r2(1, 6);
  RAJA::TypedRangeSegment<TypeParam> r3(r1);
  RAJA::TypedRangeSegment<TypeParam> r4(r2);
  std::swap(r1, r2);
  ASSERT_EQ(r1, r4);
  ASSERT_EQ(r2, r3);
}

template< typename T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void NegativeRangeSegIteratorsTest()
{
}

template< typename T, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr>
void NegativeRangeSegIteratorsTest()
{
  RAJA::TypedRangeSegment<T> r3(-2, 100);
  ASSERT_EQ(T(-2), *r3.begin());
}

TYPED_TEST(RangeSegmentUnitTest, Iterators)
{
  RAJA::TypedRangeSegment<TypeParam> r1(0, 100);
  ASSERT_EQ(TypeParam(0), *r1.begin());
  ASSERT_EQ(TypeParam(99), *(--r1.end()));
  ASSERT_EQ(TypeParam(100), r1.end() - r1.begin());
  using difftype_t = decltype(std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(difftype_t(100), std::distance(r1.begin(), r1.end()));
  ASSERT_EQ(difftype_t(100), r1.size());

  NegativeRangeSegIteratorsTest<TypeParam>();
}

template <typename IDX_TYPE,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<IDX_TYPE>>::value>::type* = nullptr>
void runNegativeIndexSliceTests()
{
}

template <typename IDX_TYPE,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<IDX_TYPE>>::value>::type* = nullptr>
void runNegativeIndexSliceTests()
{
  auto r1 = RAJA::TypedRangeSegment<IDX_TYPE>(-4, 4);
  auto s1 = r1.slice(0, 5);

  ASSERT_EQ(IDX_TYPE(-4), *s1.begin());
  ASSERT_EQ(IDX_TYPE(1), *(s1.end()));
  ASSERT_EQ(IDX_TYPE(5), s1.size());


  auto r2 = RAJA::TypedRangeSegment<IDX_TYPE>(-8, -2);
  auto s2 = r2.slice(1, 7);

  ASSERT_EQ(IDX_TYPE(-7), *s2.begin());
  ASSERT_EQ(IDX_TYPE(-2), *(s2.end()));
  ASSERT_EQ(IDX_TYPE(5), s2.size());
}

TYPED_TEST(RangeSegmentUnitTest, Slices)
{
  auto r1 = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  auto s1 = r1.slice(10,100);

  ASSERT_EQ(TypeParam(10), *s1.begin());
  ASSERT_EQ(TypeParam(110), *(s1.end()));
  ASSERT_EQ(TypeParam(100), s1.size());

 
  auto r2 = RAJA::TypedRangeSegment<TypeParam>(0, 12);
  auto s2 = r2.slice(1,13);

  ASSERT_EQ(TypeParam(1), *s2.begin());
  ASSERT_EQ(TypeParam(12), *(s2.end()));
  ASSERT_EQ(TypeParam(11), s2.size());


  auto r3 = RAJA::TypedRangeSegment<TypeParam>(1, 125);
  auto s3 = r3.slice(10,100);

  ASSERT_EQ(TypeParam(11), *s3.begin());
  ASSERT_EQ(TypeParam(111), *(s3.end()));
  ASSERT_EQ(TypeParam(100), s3.size());

  runNegativeIndexSliceTests<TypeParam>();
}

TYPED_TEST(RangeSegmentUnitTest, Equality)
{
  auto r1 = RAJA::TypedRangeSegment<TypeParam>(0, 125);
  auto r2 = RAJA::TypedRangeSegment<TypeParam>(0, 125);

  ASSERT_EQ(r1, r2);

  auto r3 = RAJA::TypedRangeSegment<TypeParam>(10,15);

  ASSERT_NE(r1, r3);
}
