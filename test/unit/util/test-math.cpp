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

template < typename T >
void test_log2()
{
  ASSERT_EQ(RAJA::log2(T(257)), T(8));
  ASSERT_EQ(RAJA::log2(T(256)), T(8));
  ASSERT_EQ(RAJA::log2(T(255)), T(7));
  ASSERT_EQ(RAJA::log2(T(4)), T(2));
  ASSERT_EQ(RAJA::log2(T(3)), T(1));
  ASSERT_EQ(RAJA::log2(T(2)), T(1));
  ASSERT_EQ(RAJA::log2(T(1)), T(0));
  ASSERT_EQ(RAJA::log2(T(0)), T(0));
  if (std::is_signed<T>::value) {
    ASSERT_EQ(RAJA::log2(T(-1)), T(0));
    ASSERT_EQ(RAJA::log2(T(-100)), T(0));
  }
}

TEST(math, log2)
{
  test_log2<int>();
  test_log2<size_t>();
}


template < typename T >
void test_next_pow2()
{
  ASSERT_EQ(RAJA::next_pow2(T(257)), T(512));
  ASSERT_EQ(RAJA::next_pow2(T(256)), T(256));
  ASSERT_EQ(RAJA::next_pow2(T(255)), T(256));
  ASSERT_EQ(RAJA::next_pow2(T(4)), T(4));
  ASSERT_EQ(RAJA::next_pow2(T(3)), T(4));
  ASSERT_EQ(RAJA::next_pow2(T(2)), T(2));
  ASSERT_EQ(RAJA::next_pow2(T(1)), T(1));
  ASSERT_EQ(RAJA::next_pow2(T(0)), T(0));
  if (std::is_signed<T>::value) {
    ASSERT_EQ(RAJA::next_pow2(T(-1)), T(0));
    ASSERT_EQ(RAJA::next_pow2(T(-100)), T(0));
  }
}

TEST(math, next_pow2)
{
  test_next_pow2<int>();
  test_next_pow2<size_t>();
}


template < typename T >
void test_prev_pow2()
{
  ASSERT_EQ(RAJA::prev_pow2(T(257)), T(256));
  ASSERT_EQ(RAJA::prev_pow2(T(256)), T(256));
  ASSERT_EQ(RAJA::prev_pow2(T(255)), T(128));
  ASSERT_EQ(RAJA::prev_pow2(T(4)), T(4));
  ASSERT_EQ(RAJA::prev_pow2(T(3)), T(2));
  ASSERT_EQ(RAJA::prev_pow2(T(2)), T(2));
  ASSERT_EQ(RAJA::prev_pow2(T(1)), T(1));
  ASSERT_EQ(RAJA::prev_pow2(T(0)), T(0));
  if (std::is_signed<T>::value) {
    ASSERT_EQ(RAJA::prev_pow2(T(-1)), T(0));
    ASSERT_EQ(RAJA::prev_pow2(T(-100)), T(0));
  }
}

TEST(math, prev_pow2)
{
  test_prev_pow2<int>();
  test_prev_pow2<size_t>();
}


template < typename T >
void test_power_of_2_mod()
{
  ASSERT_EQ(RAJA::power_of_2_mod(T(257), T(256)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(256), T(256)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(255), T(256)), T(255));
  ASSERT_EQ(RAJA::power_of_2_mod(T(128), T(256)), T(128));
  ASSERT_EQ(RAJA::power_of_2_mod(T(256), T(4)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(95), T(4)), T(3));
  ASSERT_EQ(RAJA::power_of_2_mod(T(94), T(4)), T(2));
  ASSERT_EQ(RAJA::power_of_2_mod(T(93), T(4)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(92), T(4)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(7), T(4)), T(3));
  ASSERT_EQ(RAJA::power_of_2_mod(T(6), T(4)), T(2));
  ASSERT_EQ(RAJA::power_of_2_mod(T(5), T(4)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(4), T(4)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(3), T(4)), T(3));
  ASSERT_EQ(RAJA::power_of_2_mod(T(2), T(4)), T(2));
  ASSERT_EQ(RAJA::power_of_2_mod(T(1), T(4)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(0), T(4)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(3), T(2)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(2), T(2)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(1), T(2)), T(1));
  ASSERT_EQ(RAJA::power_of_2_mod(T(0), T(2)), T(0));
  ASSERT_EQ(RAJA::power_of_2_mod(T(1), T(1)), T(0));
}

TEST(math, power_of_2_mod)
{
  test_power_of_2_mod<int>();
  test_power_of_2_mod<size_t>();
}
