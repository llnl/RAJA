//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for Operators.
///

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

template<typename T>
class OperatorsUnitTestEquivalence : public ::testing::Test {};

TYPED_TEST_SUITE(OperatorsUnitTestEquivalence, UnitIntFloatTypes);

template<typename T>
void equal_test()
{
  using Eq = RAJA::operators::equal_to<T>;

  Eq eq;
  T i = static_cast<T>(5);
  T j = static_cast<T>(5);
  ASSERT_TRUE(eq(i,j));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-5);
    ASSERT_TRUE(eq(i,j));
  }
}

template<typename T>
void not_equal_test()
{
  using NEq = RAJA::operators::not_equal_to<T>;

  NEq neq;
  T i = static_cast<T>(5);
  T j = static_cast<T>(3);
  ASSERT_TRUE(neq(i,j));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-3);
    ASSERT_TRUE(neq(i,j));
  }
}

template<typename T>
void greater_test()
{
  using G = RAJA::operators::greater<T>;

  G g;
  T i = static_cast<T>(5);
  T j = static_cast<T>(4);
  ASSERT_TRUE(g(i,j));
  ASSERT_FALSE(g(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-4);
    j = static_cast<T>(-5);
    ASSERT_TRUE(g(i,j));
    ASSERT_FALSE(g(j,i));
  }
}

template<typename T>
void less_test()
{
  using L = RAJA::operators::less<T>;

  L l;
  T i = static_cast<T>(4);
  T j = static_cast<T>(5);
  ASSERT_TRUE(l(i,j));
  ASSERT_FALSE(l(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-4);
    ASSERT_TRUE(l(i,j));
    ASSERT_FALSE(l(j,i));
  }
}

template<typename T>
void greater_eq_test()
{
  using G = RAJA::operators::greater_equal<T>;

  G g;
  T i = static_cast<T>(5);
  T i2 = static_cast<T>(5);
  T j = static_cast<T>(4);
  ASSERT_TRUE(g(i,j));
  ASSERT_TRUE(g(i,i2));
  ASSERT_FALSE(g(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-4);
    i2 = static_cast<T>(-4);
    j = static_cast<T>(-5);
    ASSERT_TRUE(g(i,j));
    ASSERT_TRUE(g(i,i2));
    ASSERT_FALSE(g(j,i));
  }
}

template<typename T>
void less_eq_test()
{
  using L = RAJA::operators::less_equal<T>;

  L l;
  T i = static_cast<T>(4);
  T i2 = static_cast<T>(4);
  T j = static_cast<T>(5);
  ASSERT_TRUE(l(i,j));
  ASSERT_TRUE(l(i,i2));
  ASSERT_FALSE(l(j,i));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    i2 = static_cast<T>(-5);
    j = static_cast<T>(-4);
    ASSERT_TRUE(l(i,j));
    ASSERT_TRUE(l(i,i2));
    ASSERT_FALSE(l(j,i));
  }
}

template<typename T>
void maximum_test()
{
  using Max = RAJA::operators::maximum<T>;

  Max m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), i);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), j);
  }
}

template<typename T>
void minimum_test()
{
  using Min = RAJA::operators::minimum<T>;

  Min m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), j);

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), i);
  }
}

TYPED_TEST(OperatorsUnitTestEquivalence, equivalence) {
  minimum_test<TypeParam>();
  maximum_test<TypeParam>();
  equal_test<TypeParam>();
  not_equal_test<TypeParam>();
  greater_test<TypeParam>();
  less_test<TypeParam>();
  greater_eq_test<TypeParam>();
  less_eq_test<TypeParam>();
}
