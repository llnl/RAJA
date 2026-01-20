//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for for_each
///

#include "RAJA_test-base.hpp"

#include "RAJA_unit-test-types.hpp"

#include "camp/resource.hpp"

#include <type_traits>
#include <vector>
#include <set>

template<typename T>
class ForEachUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(ForEachUnitTest, UnitIndexTypes);


TYPED_TEST(ForEachUnitTest, EmptyRange)
{
  std::vector<TypeParam> numbers;

  std::vector<TypeParam> copies;
  RAJA::for_each(numbers, [&](TypeParam& number) {
    number += 1;
    copies.push_back(number);
  });

  ASSERT_EQ(copies.size(), 0);
  ASSERT_EQ(numbers.size(), 0);
}

TYPED_TEST(ForEachUnitTest, VectorRange)
{
  std::vector<TypeParam> numbers;
  for (TypeParam i = 0; i < 13; ++i) {
    numbers.push_back(i);
  }

  std::vector<TypeParam> copies;
  RAJA::for_each(numbers, [&](TypeParam& number) {
    copies.push_back(number);
    number += 1;
  });

  ASSERT_EQ(copies.size(), 13);
  for (TypeParam i = 0; i < 13; ++i) {
    ASSERT_EQ(numbers[i], copies[i]+1);
  }
}

TYPED_TEST(ForEachUnitTest, RajaSpanRange)
{
  std::vector<TypeParam> numbers;
  for (TypeParam i = 0; i < 11; ++i) {
    numbers.push_back(i);
  }

  std::vector<TypeParam> copies;
  RAJA::for_each(RAJA::make_span(numbers.data(), 11), [&](TypeParam& number) {
    copies.push_back(number);
    number += 1;
  });

  ASSERT_EQ(copies.size(), 11);
  for (TypeParam i = 0; i < 11; ++i) {
    ASSERT_EQ(numbers[i], copies[i]+1);
  }
}

TYPED_TEST(ForEachUnitTest, SetRange)
{
  std::set<TypeParam> numbers;
  for (TypeParam i = 0; i < 6; ++i) {
    numbers.insert(i);
  }

  std::vector<TypeParam> copies;
  RAJA::for_each(numbers, [&](TypeParam const& number) {
    copies.push_back(number);
  });

  ASSERT_EQ(copies.size(), 6);
  for (TypeParam i = 0; i < 6; ++i) {
    ASSERT_EQ(i, copies[i]);
    ASSERT_EQ(numbers.count(i), 1);
  }
}


TYPED_TEST(ForEachUnitTest, EmptyTypeList)
{
  using numbers = camp::list<>;

  std::vector<TypeParam> copies;
  RAJA::for_each_type(numbers{}, [&](auto number) {
    copies.push_back(number);
  });

  ASSERT_EQ(copies.size(), 0);
}


template < typename T, T val >
T get_num(std::integral_constant<T, val>)
{
  return val;
}

template < typename TypeParam,
           std::enable_if_t<std::is_integral<TypeParam>::value>* = nullptr >
void run_int_type_test()
{
  using numbers = camp::list<std::integral_constant<TypeParam, 0>,
                             std::integral_constant<TypeParam, 1>,
                             std::integral_constant<TypeParam, 2>,
                             std::integral_constant<TypeParam, 3>,
                             std::integral_constant<TypeParam, 4>>;

  std::vector<TypeParam> copies;
  RAJA::for_each_type(numbers{}, [&](auto number) {
    copies.push_back(get_num(number));
  });

  ASSERT_EQ(copies.size(), 5);
  for (TypeParam i = 0; i < 5; ++i) {
    ASSERT_EQ(i, copies[i]);
  }
}
///
template < typename TypeParam,
           std::enable_if_t<!std::is_integral<TypeParam>::value>* = nullptr >
void run_int_type_test()
{
  // ignore non-ints
}

TYPED_TEST(ForEachUnitTest, IntTypeList)
{
  run_int_type_test<TypeParam>();
}
