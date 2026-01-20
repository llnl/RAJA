//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Fraction
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include <type_traits>

template <typename IntegerType, IntegerType numerator, IntegerType denominator>
void testFractionMultiplyTypesValues()
{
  using Frac = RAJA::Fraction<IntegerType, numerator, denominator>;

  ASSERT_EQ(Frac::multiply(IntegerType(0)), IntegerType(0));

  ASSERT_EQ(Frac::multiply(IntegerType(1)),
            IntegerType(double(numerator) / double(denominator)));

  ASSERT_EQ(Frac::multiply(IntegerType(100)),
            IntegerType(double(numerator) / double(denominator) * double(100)));

  ASSERT_EQ(Frac::multiply(IntegerType(101)),
            IntegerType(double(numerator) / double(denominator) * double(101)));

  // Test where naive algorithm causes overflow, when within precision of double
  if /*constexpr*/ (sizeof(IntegerType) < sizeof(double)) {

    static constexpr IntegerType max = std::numeric_limits<IntegerType>::max();
    static constexpr IntegerType val = (numerator > denominator) ?
        (max / numerator * denominator) : max;

    ASSERT_EQ(Frac::multiply(IntegerType(val)),
              IntegerType(double(numerator) / double(denominator) * double(val)));
  }

}

template <typename IntegerType>
void testFractionMultiplyTypes()
{
  testFractionMultiplyTypesValues<IntegerType, 1, 1>();
  testFractionMultiplyTypesValues<IntegerType, 1, 2>();
  testFractionMultiplyTypesValues<IntegerType, 1, 3>();
  testFractionMultiplyTypesValues<IntegerType, 2, 3>();
  testFractionMultiplyTypesValues<IntegerType, 12, 7>();
  testFractionMultiplyTypesValues<IntegerType, 0, 100>();
}


#define RAJA_FRACTION_RUN_TEST(test) \
  test<int>(); \
  test<size_t>();

TEST(Fraction, basic_multiply_Fraction)
{
  RAJA_FRACTION_RUN_TEST(testFractionMultiplyTypes)
}
