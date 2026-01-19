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
class OperatorsUnitTestMath : public ::testing::Test {};
TYPED_TEST_SUITE(OperatorsUnitTestMath, UnitIntFloatTypes);

template<typename T>
void plus_test()
{
  using Plus = RAJA::operators::plus<T>;
  auto ident = Plus::identity();
  ASSERT_EQ(ident, T(0));

  Plus p;
  T i = static_cast<T>(1);
  T j = static_cast<T>(2);
  ASSERT_EQ(p(i,j), T(3));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(p(i,j), T(-7));
  }
}

template<typename T>
void minus_test()
{
  using Minus = RAJA::operators::minus<T>;

  Minus m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), T(3));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), T(-3));
  }
}

template<typename T>
void multiplies_test()
{
  using Mult = RAJA::operators::multiplies<T>;
  auto ident = Mult::identity();
  ASSERT_EQ(ident, T(1));

  Mult m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), T(10));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), T(10));
  }
}

template<typename T>
void divides_test()
{
  using Div = RAJA::operators::divides<T>;

  Div d;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  if(std::is_floating_point<T>::value) 
    ASSERT_EQ(d(i,j), T(2.5));
  else
    ASSERT_EQ(d(i,j), T(2));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    if(std::is_floating_point<T>::value) 
      ASSERT_EQ(d(i,j), T(2.5));
    else
      ASSERT_EQ(d(i,j), T(2));
  }
}

TYPED_TEST(OperatorsUnitTestMath, math) {
  plus_test<TypeParam>();
  minus_test<TypeParam>();
  multiplies_test<TypeParam>();
  divides_test<TypeParam>();
}
