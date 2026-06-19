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

RAJA_INDEX_VALUE(RangeStrongIndex, "RangeStrongIndex");
RAJA_INDEX_VALUE(AnotherRangeStrongIndex, "AnotherRangeStrongIndex");

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

TEST(RangeSegmentUnitTest, RangeEnd)
{
  auto r = RAJA::range(RAJA::Index_type(17));

  ASSERT_EQ(RAJA::RangeSegment(RAJA::Index_type(0), RAJA::Index_type(17)), r);
  ASSERT_EQ(RAJA::Index_type(0), *r.begin());
  ASSERT_EQ(RAJA::Index_type(17), r.size());
}

TEST(RangeSegmentUnitTest, LongRangeEnd)
{
  auto r = RAJA::range(long {10});

  static_assert(std::is_same<decltype(r), RAJA::TypedRangeSegment<long>>::value,
                "range(long) should deduce the segment storage type.");
  ASSERT_EQ((RAJA::TypedRangeSegment<long>(0, 10)), r);
  ASSERT_EQ(0, *r.begin());
  ASSERT_EQ(10, r.size());
}

TEST(RangeSegmentUnitTest, LongRangeEndLValue)
{
  const long end = 10;
  auto r         = RAJA::range(end);

  static_assert(std::is_same<decltype(r), RAJA::TypedRangeSegment<long>>::value,
                "range(const long&) should decay the deduced storage type.");
  ASSERT_EQ((RAJA::TypedRangeSegment<long>(0, 10)), r);
  ASSERT_EQ(0, *r.begin());
  ASSERT_EQ(10, r.size());
}

TEST(RangeSegmentUnitTest, TypedRangeEnd)
{
  auto r = RAJA::range<RangeStrongIndex>(RangeStrongIndex(17));

  ASSERT_EQ((RAJA::TypedRangeSegment<RangeStrongIndex>(0, 17)), r);
  ASSERT_EQ(RangeStrongIndex(0), *r.begin());
  ASSERT_EQ(RAJA::Index_type(17), r.size());
}

TEST(RangeSegmentUnitTest, DeducedStrongRangeEnd)
{
  auto r = RAJA::range(RangeStrongIndex(17));

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeSegment<RangeStrongIndex>>::value,
      "range(StrongIndex) should deduce the segment storage type.");
  ASSERT_EQ((RAJA::TypedRangeSegment<RangeStrongIndex>(0, 17)), r);
  ASSERT_EQ(RangeStrongIndex(0), *r.begin());
  ASSERT_EQ(RAJA::Index_type(17), r.size());
}

TEST(RangeSegmentUnitTest, ExplicitTypedRangeEnd)
{
  auto r = RAJA::range<long>(10);

  static_assert(std::is_same<decltype(r), RAJA::TypedRangeSegment<long>>::value,
                "range<T>(end) should use the explicit storage type.");
  ASSERT_EQ((RAJA::TypedRangeSegment<long>(0, 10)), r);
  ASSERT_EQ(0, *r.begin());
  ASSERT_EQ(10, r.size());
}

TEST(RangeSegmentUnitTest, RangeBeginEnd)
{
  auto r = RAJA::range(3, 17);

  ASSERT_EQ((RAJA::TypedRangeSegment<int>(3, 17)), r);
  ASSERT_EQ(14, r.size());
}

TEST(RangeSegmentUnitTest, StrongRangeBeginEnd)
{
  RangeStrongIndex begin(3);
  RangeStrongIndex end(17);
  auto r = RAJA::range(begin, end);

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeSegment<RangeStrongIndex>>::value,
      "range(StrongIndex, StrongIndex) should deduce the strong storage type.");
  ASSERT_EQ((RAJA::TypedRangeSegment<RangeStrongIndex>(3, 17)), r);
  ASSERT_EQ(RangeStrongIndex(3), *r.begin());
  ASSERT_EQ(RAJA::Index_type(14), r.size());
}

TEST(RangeSegmentUnitTest, MixedStrongRangeBeginEnd)
{
  static_assert(!requires { RAJA::range(3, RangeStrongIndex(17)); });
}

static_assert(RAJA::concepts::RangeConstructible<RangeStrongIndex,
                                                 RangeStrongIndex>);
static_assert(!RAJA::concepts::RangeConstructible<RangeStrongIndex, int>);
static_assert(!RAJA::concepts::RangeConstructible<RangeStrongIndex, long>);
static_assert(!RAJA::concepts::RangeConstructible<long, RangeStrongIndex>);
static_assert(requires {
  RAJA::range<RangeStrongIndex>(RangeStrongIndex(3), RangeStrongIndex(17));
});
static_assert(!requires { RAJA::range<int>(RangeStrongIndex(3), 17); });
static_assert(!requires {
  RAJA::range<RangeStrongIndex>(RangeStrongIndex(3), 17);
});
static_assert(!requires { RAJA::range<RangeStrongIndex>(AnotherRangeStrongIndex(3), 17); });

TEST(RangeSegmentUnitTest, MixedStrongRangeStrongBegin)
{
  static_assert(!requires { RAJA::range(RangeStrongIndex(3), 17); });
}

static_assert(RAJA::concepts::RangeStrideConstructible<RangeStrongIndex,
                                                       RangeStrongIndex,
                                                       RangeStrongIndex>);
static_assert(!RAJA::concepts::RangeStrideConstructible<RangeStrongIndex,
                                                        int,
                                                        int>);
static_assert(!RAJA::concepts::RangeStrideConstructible<RangeStrongIndex,
                                                        long,
                                                        int>);
static_assert(!RAJA::concepts::RangeStrideConstructible<long,
                                                        RangeStrongIndex,
                                                        int>);
static_assert(!requires {
  RAJA::range<int>(RangeStrongIndex(2), 11, 3);
});
static_assert(!requires {
  RAJA::range<RangeStrongIndex>(RangeStrongIndex(2), 11, 3);
});
static_assert(!requires {
  RAJA::range<RangeStrongIndex>(AnotherRangeStrongIndex(2), 11, 3);
});
static_assert(!requires { RAJA::range<int>(2, 11, RangeStrongIndex(3)); });
static_assert(!requires {
  RAJA::range<RangeStrongIndex>(2, 11, AnotherRangeStrongIndex(3));
});
static_assert(requires {
  RAJA::range<RangeStrongIndex>(RangeStrongIndex(2),
                                RangeStrongIndex(11),
                                RangeStrongIndex(3));
});

TEST(RangeSegmentUnitTest, RangeBeginEndStridePositive)
{
  auto r = RAJA::range(2, 11, 3);

  ASSERT_EQ((RAJA::TypedRangeStrideSegment<int>(2, 11, 3)), r);
  ASSERT_EQ(3, r.size());
  ASSERT_EQ(2, *r.begin());
}

TEST(RangeSegmentUnitTest, MixedStrongRangeBeginEndStrideStrongEnd)
{
  static_assert(!requires { RAJA::range(2, RangeStrongIndex(11), 3); });
}

TEST(RangeSegmentUnitTest, MixedStrongRangeBeginEndStrideStrongBegin)
{
  static_assert(!requires { RAJA::range(RangeStrongIndex(2), 11, 3); });
}

TEST(RangeSegmentUnitTest, RangeBeginEndStrongStride)
{
  static_assert(!requires { RAJA::range(2, 11, RangeStrongIndex(3)); });
}

TEST(RangeSegmentUnitTest, RangeBeginEndStrideNegative)
{
  auto r = RAJA::range(10, -1, -2);

  ASSERT_EQ((RAJA::TypedRangeStrideSegment<int>(10, -1, -2)), r);
  ASSERT_EQ(6, r.size());
  ASSERT_EQ(10, *r.begin());
}

TEST(RangeSegmentUnitTest, RangeBeginEndStrideDeducesFromStride)
{
  auto r = RAJA::range(2, 11, long {3});

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeStrideSegment<long>>::value,
      "range(begin, end, stride) should include stride in the storage type.");
  ASSERT_EQ((RAJA::TypedRangeStrideSegment<long>(2, 11, 3)), r);
  ASSERT_EQ(3, r.size());
  ASSERT_EQ(2, *r.begin());
}

TEST(RangeSegmentUnitTest, RangeBeginEndUnsignedStrideWorks)
{
  auto r = RAJA::range(-2, 11, 3u);

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeStrideSegment<int>>::value,
      "range(begin, end, unsigned stride) should preserve signed storage when "
      "begin/end are signed.");
  ASSERT_EQ((RAJA::TypedRangeStrideSegment<int>(-2, 11, 3)), r);
  ASSERT_EQ(5, r.size());
  ASSERT_EQ(-2, *r.begin());
}

TEST(RangeSegmentUnitTest, MakeStridedRangeDeducesFromStride)
{
  auto r = RAJA::make_strided_range(2, 11, long {3});

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeStrideSegment<long>>::value,
      "make_strided_range(begin, end, stride) should include stride in the "
      "storage type.");
  ASSERT_EQ((RAJA::TypedRangeStrideSegment<long>(2, 11, 3)), r);
  ASSERT_EQ(3, r.size());
  ASSERT_EQ(2, *r.begin());
}

TEST(RangeSegmentUnitTest, MakeStridedRangeUnsignedStrideWorks)
{
  auto r = RAJA::make_strided_range(-2, 11, 3u);

  static_assert(
      std::is_same<decltype(r), RAJA::TypedRangeStrideSegment<int>>::value,
      "make_strided_range(begin, end, unsigned stride) should preserve signed "
      "storage when begin/end are signed.");
  ASSERT_EQ((RAJA::TypedRangeStrideSegment<int>(-2, 11, 3)), r);
  ASSERT_EQ(5, r.size());
  ASSERT_EQ(-2, *r.begin());
}
