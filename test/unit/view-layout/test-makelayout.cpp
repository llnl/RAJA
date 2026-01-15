//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

TEST(LayoutUnitTest, OffsetVsRegular)
{
  const auto layout =
      RAJA::make_permuted_layout({{6, 6}},
                                 RAJA::as_array<RAJA::Perm<1, 0>>::get());
  const auto offset =
      RAJA::make_permuted_offset_layout({{0, 0}},
                                        {{6, 6}},
                                        RAJA::as_array<RAJA::PERM_JI>::get());

  /*
   * OffsetLayout with 0 offset should function like the regular Layout.
   */
  for (int j = 0; j < 6; ++j) {
    for (int i = 0; i < 6; ++i) {
      ASSERT_EQ(offset(i, j), layout(i, j))
          << layout.strides[0] << layout.strides[1];
    }
  }
}

TEST(OffsetLayoutUnitTest, 2D_IJ)
{
  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const auto layout = RAJA::make_offset_layout<2>({{-1, -2}}, {{2, 1}});

  /*
   * First element, (-1, -2), should have index 0.
   */
  ASSERT_EQ(0, layout(-1, -2));

  /*
   * (0, -2) should have index 3.
   */
  ASSERT_EQ(3, layout(0, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1, 0));
}


TEST(OffsetLayoutUnitTest, 2D_JI)
{
  using my_layout = RAJA::OffsetLayout<2>;

  /*
   * Construct a 2D layout:
   *
   * (-1, 0), (0, 0), (1, 0)
   * (-1, -1), (0, -1), (1, -1)
   * (-1, -2), (0, -2), (1, -2)
   */
  const my_layout layout =
      RAJA::make_permuted_offset_layout({{-1, -2}},
                                        {{2, 1}},
                                        RAJA::as_array<RAJA::PERM_JI>::get());

  /*
   * First element, (-1, -2), should have index 0.
   * things.
   */
  ASSERT_EQ(0, layout(-1, -2));

  ASSERT_EQ(1, layout(-0, -2));

  /*
   * Last element, (1, 0), should have index 8.
   */
  ASSERT_EQ(8, layout(1, 0));
}

TEST(LayoutUnitTest, 3D_KJI_ProjJ)
{
  using my_layout = RAJA::Layout<3>;

  /*
   * Construct a 3D projective layout:
   *
   * I is stride 1
   * J is stride 0  -  projected out
   * K is stride 3
   *
   * Linear indices range from [0, 21)
   *
   * values of J should have no effect on linear index
   *
   * All linear indices should produce J=0
   *
   */

  // Construct using variadic "sizes" ctor
  // Zero for J size should correctly produce projective layout
  const my_layout layout =
      RAJA::make_permuted_layout({{3, 0, 7}},
                                 RAJA::as_array<RAJA::PERM_KJI>::get());

  ASSERT_EQ(0, layout(0, 0, 0));

  ASSERT_EQ(1, layout(1, 0, 0));
  ASSERT_EQ(2, layout(2, 0, 0));

  // J should be projected out
  ASSERT_EQ(0, layout(0, 1, 0));
  ASSERT_EQ(0, layout(0, 5, 0));

  ASSERT_EQ(6, layout(0, 0, 2));
  ASSERT_EQ(12, layout(0, 0, 4));

  // Check that we get the identity (mod 21)
  for (int x = 0; x < 40; ++x) {

    // inverse map
    int i, j, k;
    layout.toIndices(x, i, j, k);

    // forward map
    int x2 = layout(i, j, k);

    // check ident
    ASSERT_EQ(x % 21, x2);

    // check projection of j
    ASSERT_EQ(j, 0);
  }
}

TEST(LayoutUnitTest, 2D_StrideOne)
{
  using my_layout = RAJA::Layout<2>;
  using my_layout_s1 = RAJA::Layout<2, ptrdiff_t, 0>; // first index is stride-1

  /*
   * Construct a 2D layout:
   *
   * I is stride 1
   * J is stride 3
   *
   * Linear indices range from [0, 15)
   *
   */
  const my_layout layout =
      RAJA::make_permuted_layout({{3, 5}},
                                 RAJA::as_array<RAJA::PERM_JI>::get());


  /*
   * Construct another 2D layout that forces J to be stride-1
   */
  const my_layout_s1 layout_s1 = layout;


  // Check that we get the same layout
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {

      ASSERT_EQ(layout(i, j), layout_s1(i, j));
    }
  }
}

TEST(StaticLayoutUnitTest, 2D_StaticLayout)
{
  RAJA::Layout<2> dynamic_layout(7, 5);
  using static_layout = RAJA::StaticLayout<RAJA::PERM_IJ,7,5>;
  
  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 5; ++j) {

      ASSERT_EQ(dynamic_layout(i, j), static_layout::s_oper(i,j));
    }
  }
}

TEST(StaticLayoutUnitTest, 2D_PermutedStaticLayout)
{
  auto dynamic_layout = 
    RAJA::make_permuted_layout({{7, 5}},
                               RAJA::as_array<RAJA::PERM_JI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_JI, 7,5>;
  
  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(dynamic_layout(i, j), static_layout::s_oper(i,j));
    }
  }
}

TEST(StaticLayoutUnitTest, 3D_PermutedStaticLayout)
{
  auto dynamic_layout = 
    RAJA::make_permuted_layout({{7, 13, 5}},
                               RAJA::as_array<RAJA::PERM_JKI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_JKI, 7,13,5>;

  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 13; ++j) {
      for (int k = 0; k < 5; ++k) {
        ASSERT_EQ(dynamic_layout(i, j, k), static_layout::s_oper(i,j,k));
      }
    }
  }
}


TEST(StaticLayoutUnitTest, 4D_PermutedStaticLayout)
{
  auto dynamic_layout = 
    RAJA::make_permuted_layout({{7, 13, 5, 17}},
                               RAJA::as_array<RAJA::PERM_LJKI>::get());
  using static_layout = RAJA::StaticLayout<RAJA::PERM_LJKI, 7,13,5,17>;

  // Check that we get the same layout
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 13; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 5; ++l) {
          ASSERT_EQ(dynamic_layout(i, j, k, l), static_layout::s_oper(i,j,k,l));
        } 
      }
    }
  }
}


