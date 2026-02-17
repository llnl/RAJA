//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

/*
 * Basic layout test
 */

TEST(OffsetLayoutUnitTest, Constructors)
{
  using layout = RAJA::OffsetLayout<1>;

  /*
   * Construct a 1D view with  with the following indices:
   *
   * 10, 11, 12, 13, 14
   */
  const layout l({{10}}, {{15}});

  /*
   * First element, 10, should have index 0.
   */
  ASSERT_EQ(0, l(10));

  ASSERT_EQ(2, l(12));

  /*
   * Last element, 14, should have index 5.
   */
  ASSERT_EQ(4, l(14));
}

TEST(LayoutUnitTest, 2D_IJ)
{
  using my_layout = RAJA::Layout<2>;

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


  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(5, layout(1, 0));
  ASSERT_EQ(14, layout(2, 4));

  ASSERT_EQ(1, layout(0, 1));
  ASSERT_EQ(4, layout(0, 4));

  // Check that we get the identity
  for (int k = 0; k < 15; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k % 15, k2);

    // check with a and b
    ASSERT_EQ(k2, layout_a(i, j));
    ASSERT_EQ(k2, layout_b(i, j));
  }
}

TEST(LayoutUnitTest, 2D_JI)
{
  using my_layout = RAJA::Layout<2>;

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

  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(1, layout(1, 0));
  ASSERT_EQ(2, layout(2, 0));

  ASSERT_EQ(3, layout(0, 1));
  ASSERT_EQ(14, layout(2, 4));

  // Check that we get the identity (mod 15)
  for (int k = 0; k < 15; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    ASSERT_EQ(k % 15, k2);
  }
}

TEST(LayoutUnitTest, 2D_IJ_ProjJ)
{
  using my_layout = RAJA::Layout<2>;

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

  ASSERT_EQ(0, layout(0, 0));

  ASSERT_EQ(1, layout(1, 0));
  ASSERT_EQ(3, layout(3, 0));

  // J should be projected out
  ASSERT_EQ(0, layout(0, 1));
  ASSERT_EQ(0, layout(0, 5));

  // Check that we get the identity (mod 7)
  for (int k = 0; k < 20; ++k) {

    // inverse map
    int i, j;
    layout.toIndices(k, i, j);

    // forward map
    int k2 = layout(i, j);

    // check ident
    ASSERT_EQ(k % 7, k2);

    // check projection of j
    ASSERT_EQ(j, 0);
  }
}

