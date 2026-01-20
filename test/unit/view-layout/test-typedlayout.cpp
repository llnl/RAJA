//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

template<typename T>
class TypedLayoutUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(TypedLayoutUnitTest, UnitIndexTypes);


TYPED_TEST(TypedLayoutUnitTest, TypedLayoutConstructors)
{

  const RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>> l(10,5);

  ASSERT_EQ(TypeParam{0}, l(TypeParam{0}, TypeParam{0}));

  ASSERT_EQ(TypeParam{2}, l(TypeParam{0}, TypeParam{2}));

  ASSERT_EQ(TypeParam{10}, l(TypeParam{2}, TypeParam{0}));

  TypeParam x{5};
  TypeParam y{0};
  l.toIndices(TypeParam{10}, y, x);
  ASSERT_EQ(x, TypeParam{0});
  ASSERT_EQ(y, TypeParam{2});
}

TYPED_TEST(TypedLayoutUnitTest, 2D_accessor)
{
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  /*
   * Construct a 2D layout:
   *
   * I is stride 5
   * J is stride 1
   *
   * Linear indices range from [0, 15)
   *
   */

  // Construct using variadic "sizes" ctor
  const my_layout layout_a(3, 5);

  // Construct using copy ctor
  const my_layout layout_b(layout_a);

  // Test default ctor and assignment operator
  my_layout layout;
  layout = layout_b;


  ASSERT_EQ(TypeParam(0), layout(0, 0));

  ASSERT_EQ(TypeParam(5), layout(1, 0));
  ASSERT_EQ(TypeParam(14), layout(2, 4));

  ASSERT_EQ(TypeParam(1), layout(0, 1));
  ASSERT_EQ(TypeParam(4), layout(0, 4));

  // Check that we get the identity
  for (int k = 0; k < 15; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(k, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k, k2);

    // check with a and b
    ASSERT_EQ(k2, layout_a(i, j));
    ASSERT_EQ(k2, layout_b(i, j));
  }

}

TYPED_TEST(TypedLayoutUnitTest, 2D_IJ_ProjJ)
{
  using my_layout = RAJA::TypedLayout<TypeParam, RAJA::tuple<TypeParam, TypeParam>>;

  /*
   * Construct a 2D projective layout:
   *
   * I is stride 1
   * J is stride 0  -  projected out
   *
   * Linear indices range from [0, 7)
   *
   * values of J should have no effect on linear index
   *
   * All linear indices should produce J=0
   *
   */

  // Construct using variadic "sizes" ctor
  // Zero for J size should correctly produce projective layout
  const my_layout layout(7, 0);

  ASSERT_EQ(TypeParam(0), layout(0, 0));

  ASSERT_EQ(TypeParam(1), layout(1, 0));
  ASSERT_EQ(TypeParam(3), layout(3, 0));

  // J should be projected out
  ASSERT_EQ(TypeParam(0), layout(0, 1));
  ASSERT_EQ(TypeParam(0), layout(0, 5));

  TypeParam pK = 0;
  // Check that we get the identity (mod 7)
  for (int k = 0; k < 20; ++k) {

    // inverse map
    TypeParam i, j;
    layout.toIndices(pK, i, j);

    // forward map
    TypeParam k2 = layout(i, j);

    // check ident
    ASSERT_EQ(pK % 7, k2);

    // check projection of j
    ASSERT_EQ(j, TypeParam(0));
    pK++;
  }
}

TYPED_TEST(TypedLayoutUnitTest, 2D_StaticLayout)
{
  RAJA::Layout<2, TypeParam> dynamic_layout(7, 5);
  using static_layout = RAJA::TypedStaticLayout<RAJA::PERM_IJ,TypeParam,RAJA::list<TypeParam,TypeParam>,7,5>;

  // Check that we get the same layout
  for (TypeParam i = 0; i < 7; ++i) {
    for (TypeParam j = 0; j < 5; ++j) {

      ASSERT_EQ(dynamic_layout(i, j), static_layout::s_oper(i,j));
    }
  }

}

TYPED_TEST(TypedLayoutUnitTest, 2D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 5}},
                               RAJA::as_array<RAJA::PERM_JI>::get());
  using static_layout = RAJA::TypedStaticLayout<RAJA::PERM_JI,
                                                TypeParam,
                                                RAJA::list<TypeParam,TypeParam>, 7,5>;

  // Check that we get the same layout
  for (TypeParam i = 0; i < 7; ++i) {
    for (TypeParam j = 0; j < 5; ++j) {
      ASSERT_EQ(TypeParam(dynamic_layout(i, j)), static_layout::s_oper(i,j));
    }
  }
}

TYPED_TEST(TypedLayoutUnitTest, 3D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 13, 5}},
                               RAJA::as_array<RAJA::PERM_JKI>::get());
  using static_layout = RAJA::TypedStaticLayout<RAJA::PERM_JKI,
                                                TypeParam,
                                                RAJA::list<TypeParam,TypeParam,TypeParam>,
                                                7,13,5>;

  // Check that we get the same layout
  for (TypeParam i = 0; i < 7; ++i) {
    for (TypeParam j = 0; j < 9; ++j) {
      for (TypeParam k = 0; k < 5; ++k) {
        ASSERT_EQ(TypeParam(dynamic_layout(i, j, k)), static_layout::s_oper(i,j,k));
      }
    }
  }
}


TYPED_TEST(TypedLayoutUnitTest, 4D_PermutedStaticLayout)
{
  auto dynamic_layout =
    RAJA::make_permuted_layout({{7, 13, 5, 17}},
                               RAJA::as_array<RAJA::PERM_LJKI>::get());
  using static_layout = RAJA::TypedStaticLayout<RAJA::PERM_LJKI,
                                                TypeParam,
                                                RAJA::list<TypeParam,TypeParam,TypeParam,TypeParam>,
                                                7,13,5,17>;

  // Check that we get the same layout
  for (TypeParam i = 0; i < 7; ++i) {
    for (TypeParam j = 0; j < 8; ++j) {
      for (TypeParam k = 0; k < 5; ++k) {
        for (TypeParam l = 0; l < 5; ++l) {
          ASSERT_EQ(TypeParam(dynamic_layout(i, j, k, l)), static_layout::s_oper(i,j,k,l));
        }
      }
    }
  }
}
