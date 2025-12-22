//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for numeric_iterator
///

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

#include <limits>

template<typename T>
class NumericIteratorUnitTest : public ::testing::Test {};

template<typename T>
class StridedNumericIteratorUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(NumericIteratorUnitTest, UnitExpandedIntegralTypes);
TYPED_TEST_SUITE(StridedNumericIteratorUnitTest, UnitExpandedIntegralTypes);

TYPED_TEST(NumericIteratorUnitTest, simple)
{
  RAJA::Iterators::numeric_iterator<TypeParam> i;
  ASSERT_EQ(TypeParam(0), *i);
  ++i;
  ASSERT_EQ(TypeParam(1), *i);
  --i;
  ASSERT_EQ(TypeParam(0), *i);
  ASSERT_EQ(TypeParam(0), *i++);
  ASSERT_EQ(TypeParam(1), *i);
  ASSERT_EQ(TypeParam(1), *i--);
  ASSERT_EQ(TypeParam(0), *i);
  i += 2;
  ASSERT_EQ(TypeParam(2), *i);
  i -= 1;
  ASSERT_EQ(TypeParam(1), *i);
  RAJA::Iterators::numeric_iterator<TypeParam> five(5);
  i += five;
  ASSERT_EQ(TypeParam(6), *i);
  i -= five;
  ASSERT_EQ(TypeParam(1), *i);
  RAJA::Iterators::numeric_iterator<TypeParam> three(3);
  ASSERT_LE(three, three);
  ASSERT_LE(three, five);
  ASSERT_LT(three, five);
  ASSERT_GE(five, three);
  ASSERT_GT(five, three);
  ASSERT_NE(five, three);
  ASSERT_EQ(three + 2, five);
  ASSERT_EQ(2 + three, five);
  ASSERT_EQ(five - 2, three);
  ASSERT_EQ(8 - five, three);
}

TYPED_TEST(StridedNumericIteratorUnitTest, simple)
{
  RAJA::Iterators::strided_numeric_iterator<TypeParam> i(0, 2);
  ASSERT_EQ(TypeParam(0), *i);
  ++i;
  ASSERT_EQ(TypeParam(2), *i);
  --i;
  ASSERT_EQ(TypeParam(0), *i);
  i += 2;
  ASSERT_EQ(TypeParam(4), *i);
  i -= 1;
  ASSERT_EQ(TypeParam(2), *i);
  RAJA::Iterators::strided_numeric_iterator<TypeParam> three(3, 2);
  RAJA::Iterators::strided_numeric_iterator<TypeParam> five(5, 2);
  ASSERT_LE(three, three);
  ASSERT_LE(three, five);
  ASSERT_LT(three, five);
  ASSERT_GE(five, three);
  ASSERT_GT(five, three);
  ASSERT_NE(five, three);
  ASSERT_EQ(three + 1, five);
  ASSERT_EQ(five - 1, three);
}

#if defined(RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG)
TYPED_TEST(NumericIteratorUnitTest, overflow)
{
  if (std::is_unsigned<TypeParam>::value) {
    ASSERT_ANY_THROW({
      TypeParam val = 10;
      RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      of_it -= 11;
    });
    ASSERT_ANY_THROW({
      TypeParam val = std::numeric_limits<TypeParam>::max() - 10;
      RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      of_it += 11;
    });
  
    ASSERT_ANY_THROW({
      TypeParam val = 10;
      RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      auto sum = of_it - 11u;
      (void)sum;
    });
    ASSERT_ANY_THROW({
      TypeParam val = std::numeric_limits<TypeParam>::max() - 10;
      RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      auto sum = of_it + 11;
      (void)sum;
    });
  
    ASSERT_ANY_THROW({
      TypeParam val = 10;
      const RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      auto sum = 8 - of_it;
      (void)sum;
    });
    ASSERT_ANY_THROW({
      TypeParam val = std::numeric_limits<TypeParam>::max() - 10;
      const RAJA::Iterators::numeric_iterator<TypeParam> of_it(val);
      auto sum = 11 + of_it;
      (void)sum;
    });
  } 
}

TYPED_TEST(StridedNumericIteratorUnitTest, overflow)
{
  if (std::is_unsigned<TypeParam>::value){
    ASSERT_ANY_THROW({
      TypeParam val = 2;
      RAJA::Iterators::strided_numeric_iterator<TypeParam> of_it(val, 2);
      of_it -= 2;
    });
    ASSERT_ANY_THROW({
      TypeParam val = std::numeric_limits<TypeParam>::max() - 2;
      RAJA::Iterators::strided_numeric_iterator<TypeParam> of_it(val, 2);
      of_it += 2;
    });

    ASSERT_ANY_THROW({
      TypeParam val = 2;
      RAJA::Iterators::strided_numeric_iterator<TypeParam> of_it(val, 2);
      auto sum = of_it - 2;
      (void)sum;
    });
    ASSERT_ANY_THROW({
      TypeParam val = std::numeric_limits<TypeParam>::max() - 2;
      RAJA::Iterators::strided_numeric_iterator<TypeParam> of_it(val, 2);
      auto sum = of_it + 2;
      (void)sum;
    });
  }
}
#endif
