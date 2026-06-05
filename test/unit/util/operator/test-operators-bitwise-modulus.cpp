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
class OperatorsIntegralUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(OperatorsIntegralUnitTest, UnitExpandedIntegralTypes);

template<typename T>
void modulus_test()
{
  using Mod = RAJA::operators::modulus<T>;

  Mod m;
  T i = static_cast<T>(5);
  T j = static_cast<T>(2);
  ASSERT_EQ(m(i,j), T(1));

  if (std::is_signed<T>::value) {
    i = static_cast<T>(-5);
    j = static_cast<T>(-2);
    ASSERT_EQ(m(i,j), T(-1));
  }
}

template<typename T>
void bit_or_test()
{
  using Or = RAJA::operators::bit_or<T>;

  Or o;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(o(i,j), T(0011));
  ASSERT_EQ(o(i,k), T(0111));
  ASSERT_EQ(o(j,k), T(0111));
}

template<typename T>
void bit_and_test()
{
  using And = RAJA::operators::bit_and<T>;

  And a;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(a(i,j), T(0000));
  ASSERT_EQ(a(i,k), T(0010));
  ASSERT_EQ(a(j,k), T(0001));
}

template<typename T>
void bit_xor_test()
{
  using Xor = RAJA::operators::bit_xor<T>;

  Xor x;
  T i = static_cast<T>(0010);
  T j = static_cast<T>(0001);
  T k = static_cast<T>(0111);
  ASSERT_EQ(x(i,j), T(0011));
  ASSERT_EQ(x(i,k), T(0101));
  ASSERT_EQ(x(j,k), T(0110));
}

TYPED_TEST(OperatorsIntegralUnitTest, bitwise_modulus) {
  bit_or_test<TypeParam>();
  bit_and_test<TypeParam>();
  bit_xor_test<TypeParam>();
  modulus_test<TypeParam>();
}
