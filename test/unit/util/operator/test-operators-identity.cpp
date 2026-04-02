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
class OperatorsUnitTestIdentity: public ::testing::Test {};

TYPED_TEST_SUITE(OperatorsUnitTestIdentity, UnitIntFloatTypes);

template<typename T>
void identity_test()
{
  using Ident = RAJA::operators::identity<T>;

  Ident id;
  T i = static_cast<T>(0);
  T j = static_cast<T>(1);
  ASSERT_EQ(id(i), T(0));
  ASSERT_EQ(id(j), T(1));

  if (std::is_signed<T>::value) {
    j = static_cast<T>(-1);
    ASSERT_EQ(id(j), T(-1));
  }
}

template<typename T>
void project1st_test()
{
  using Proj1 = RAJA::operators::project1st<T, T>;

  Proj1 p;
  T i = static_cast<T>(0);
  T j = static_cast<T>(1);
  ASSERT_EQ(p(i,j), T(0));
  ASSERT_EQ(p(j,i), T(1));

  if (std::is_signed<T>::value) {
    j = static_cast<T>(-1);
    ASSERT_EQ(p(i,j), T(0));
    ASSERT_EQ(p(j,i), T(-1));
  }
}

template<typename T>
void project2nd_test()
{
  using Proj2 = RAJA::operators::project2nd<T, T>;

  Proj2 p;
  T i = static_cast<T>(0);
  T j = static_cast<T>(1);
  ASSERT_EQ(p(i,j), T(1));
  ASSERT_EQ(p(j,i), T(0));

#ifdef RAJA_COMPILER_MSVC
#pragma warning( disable : 4245 )  // Force msvc to not emit signed conversion warning
#endif
  if (std::is_signed<T>::value) {
    j = static_cast<T>(-1);
    ASSERT_EQ(p(i,j), T(-1));
    ASSERT_EQ(p(j,i), T(0));
  }
#ifdef RAJA_COMPILER_MSVC
#pragma warning( default : 4245 )
#endif
}

TYPED_TEST(OperatorsUnitTestIdentity, identity_project) {
  identity_test<TypeParam>();
  project1st_test<TypeParam>();
  project2nd_test<TypeParam>();
}
