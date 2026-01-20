//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for IndexValue
///

#include "RAJA_test-base.hpp"

#include "RAJA_unit-test-types.hpp"

template<typename T>
class IndexValueUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(IndexValueUnitTest, UnitIndexTypes);


RAJA_INDEX_VALUE(StrongTypeIndex, "Strong Type")

TYPED_TEST(IndexValueUnitTest, Construct)
{
  StrongTypeIndex a;
  ASSERT_EQ(0l, *a);
  const StrongTypeIndex b(5);
  ASSERT_EQ(5l, *b);
  ASSERT_EQ(std::string("Strong Type"), StrongTypeIndex::getName());

  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType c;
  ASSERT_EQ((TypeParam)0, *c);
  const TestType d(5);
  ASSERT_EQ((TypeParam)5, *d);
  ASSERT_EQ(std::string("Test Type"), TestType::getName());
}

TYPED_TEST(IndexValueUnitTest, PrePostIncrement)
{
  StrongTypeIndex a;
  ASSERT_EQ(0l, *a++);
  ASSERT_EQ(1l, *a);
  ASSERT_EQ(2l, *++a);
  ASSERT_EQ(2l, *a);

  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType b;
  ASSERT_EQ((TypeParam)0, *b++);
  ASSERT_EQ((TypeParam)1, *b);
  ASSERT_EQ((TypeParam)2, *++b);
  ASSERT_EQ((TypeParam)2, *b);
}

TYPED_TEST(IndexValueUnitTest, PrePostDecrement)
{
  StrongTypeIndex a(3);
  ASSERT_EQ(3l, *a--);
  ASSERT_EQ(2l, *a);
  ASSERT_EQ(1l, *--a);
  ASSERT_EQ(1l, *a);

  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType b(3);
  ASSERT_EQ((TypeParam)3, *b--);
  ASSERT_EQ((TypeParam)2, *b);
  ASSERT_EQ((TypeParam)1, *--b);
  ASSERT_EQ((TypeParam)1, *b);
}

TYPED_TEST(IndexValueUnitTest, StrongTypesArith)
{
  StrongTypeIndex a(8);
  StrongTypeIndex b(2);

  ASSERT_EQ(StrongTypeIndex(10), a + b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(6), a - b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(16), a * b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(4), a / b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a += b;
  ASSERT_EQ(StrongTypeIndex(10), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a -= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a *= b;
  ASSERT_EQ(StrongTypeIndex(16), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  a /= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);


  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType c(8);
  TestType d(2);

  ASSERT_EQ(TestType(10), c + d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  ASSERT_EQ(TestType(6), c - d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  ASSERT_EQ(TestType(16), c * d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  ASSERT_EQ(TestType(4), c / d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  c += d;
  ASSERT_EQ(TestType(10), c);
  ASSERT_EQ(TestType(2), d);

  c -= d;
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  c *= d;
  ASSERT_EQ(TestType(16), c);
  ASSERT_EQ(TestType(2), d);

  c /= d;
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);
}

TYPED_TEST(IndexValueUnitTest, IndexTypeArith)
{
  StrongTypeIndex a(8);
  RAJA::Index_type b(2);

  ASSERT_EQ(StrongTypeIndex(10), a + b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(StrongTypeIndex(2), b);

  ASSERT_EQ(StrongTypeIndex(6), a - b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  ASSERT_EQ(StrongTypeIndex(16), a * b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  ASSERT_EQ(StrongTypeIndex(4), a / b);
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a += b;
  ASSERT_EQ(StrongTypeIndex(10), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a -= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a *= b;
  ASSERT_EQ(StrongTypeIndex(16), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  a /= b;
  ASSERT_EQ(StrongTypeIndex(8), a);
  ASSERT_EQ(RAJA::Index_type(2), b);

  
  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType c(8);
  RAJA::Index_type d(2);

  ASSERT_EQ(TestType(10), c + d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(TestType(2), d);

  ASSERT_EQ(TestType(6), c - d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  ASSERT_EQ(TestType(16), c * d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  ASSERT_EQ(TestType(4), c / d);
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  c += d;
  ASSERT_EQ(TestType(10), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  c -= d;
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  c *= d;
  ASSERT_EQ(TestType(16), c);
  ASSERT_EQ(RAJA::Index_type(2), d);

  c /= d;
  ASSERT_EQ(TestType(8), c);
  ASSERT_EQ(RAJA::Index_type(2), d);
}

TYPED_TEST(IndexValueUnitTest, StrongTypeCompare)
{
  StrongTypeIndex v1(5);
  StrongTypeIndex v2(6);
  ASSERT_LT(v1, v2);
  ASSERT_LE(v1, v2);
  ASSERT_LE(v1, v1);
  ASSERT_EQ(v1, v1);
  ASSERT_EQ(v2, v2);
  ASSERT_GE(v1, v1);
  ASSERT_GE(v2, v1);
  ASSERT_GT(v2, v1);
  ASSERT_NE(v1, v2);

  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType v3(5);
  TestType v4(6);
  ASSERT_LT(v3, v4);
  ASSERT_LE(v3, v4);
  ASSERT_LE(v3, v3);
  ASSERT_EQ(v3, v3);
  ASSERT_EQ(v4, v4);
  ASSERT_GE(v3, v3);
  ASSERT_GE(v4, v3);
  ASSERT_GT(v4, v3);
  ASSERT_NE(v3, v4);
}

TYPED_TEST(IndexValueUnitTest, IndexTypeCompare)
{
  StrongTypeIndex v(5);
  RAJA::Index_type v_lower(4);
  RAJA::Index_type v_higher(6);
  RAJA::Index_type v_same(5);
  ASSERT_LT(v, v_higher);
  ASSERT_LE(v, v_higher);
  ASSERT_LE(v, v_same);
  ASSERT_EQ(v, v_same);
  ASSERT_GE(v, v_same);
  ASSERT_GE(v, v_lower);
  ASSERT_GT(v, v_lower);
  ASSERT_NE(v, v_lower);
  ASSERT_NE(v, v_higher);

  RAJA_INDEX_VALUE_T(TestType, TypeParam, "Test Type");
  TestType x(5);
  RAJA::Index_type x_lower(4);
  RAJA::Index_type x_higher(6);
  RAJA::Index_type x_same(5);
  ASSERT_LT(x, x_higher);
  ASSERT_LE(x, x_higher);
  ASSERT_LE(x, x_same);
  ASSERT_EQ(x, x_same);
  ASSERT_GE(x, x_same);
  ASSERT_GE(x, x_lower);
  ASSERT_GT(x, x_lower);
  ASSERT_NE(x, x_lower);
  ASSERT_NE(x, x_higher);
}
