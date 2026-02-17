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
class OperatorsUnitTestLogical : public ::testing::Test {};
TYPED_TEST_SUITE(OperatorsUnitTestLogical, UnitIntFloatTypes);

template<typename T>
void logical_and_test()
{
  using And = RAJA::operators::logical_and<T>;

  And a;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  T i2 = static_cast<T>(2);
  T j0 = static_cast<T>(0);
  T j1 = static_cast<T>(1);
  T j2 = static_cast<T>(2);
  ASSERT_FALSE(a(i0,j0));
  ASSERT_FALSE(a(i0,j1));
  ASSERT_FALSE(a(i1,j0));
  ASSERT_TRUE(a(i1,j1));
  ASSERT_TRUE(a(i2,j2));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    j1 = static_cast<T>(-1);
    ASSERT_FALSE(a(i0,j1));
    ASSERT_FALSE(a(i1,j0));
    ASSERT_TRUE(a(i1,j1));
  }
}

template<typename T>
void logical_or_test()
{
  using Or = RAJA::operators::logical_or<T>;

  Or o;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  T i2 = static_cast<T>(2);
  T j0 = static_cast<T>(0);
  T j1 = static_cast<T>(1);
  T j2 = static_cast<T>(2);
  ASSERT_FALSE(o(i0,j0));
  ASSERT_TRUE(o(i0,j1));
  ASSERT_TRUE(o(i1,j0));
  ASSERT_TRUE(o(i1,j1));
  ASSERT_TRUE(o(i2,j2));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    j1 = static_cast<T>(-1);
    ASSERT_TRUE(o(i0,j1));
    ASSERT_TRUE(o(i1,j0));
    ASSERT_TRUE(o(i1,j1));
  }
}

template<typename T>
void logical_not_test()
{
  using Not = RAJA::operators::logical_not<T>;

  Not n;
  T i0 = static_cast<T>(0);
  T i1 = static_cast<T>(1);
  ASSERT_FALSE(n(i1));
  ASSERT_TRUE(n(i0));
  if (std::is_signed<T>::value) {
    i1 = static_cast<T>(-1);
    ASSERT_FALSE(n(i1));
  }
}

TYPED_TEST(OperatorsUnitTestLogical, logical) {
  logical_and_test<TypeParam>();
  logical_or_test<TypeParam>();
  logical_not_test<TypeParam>();
}
