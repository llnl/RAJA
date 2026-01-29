//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include <cstdio>
#include <array>

/*
 * MultiView Usage Example
 *
 * A RAJA::MultiView object wraps an array-of-pointers,
 * or a pointer-to-pointers, whereas a RAJA::View wraps a single
 * pointer or array. This allows a single RAJA::Layout to be applied to
 * multiple arrays internal to the MultiView, allowing multiple arrays to share indexing
 * arithmetic when their access patterns are the same.
 * 
 * The instantiation of a MultiView works exactly like a standard View,
 * except that it takes an array-of-pointers. In the following example, a MultiView
 * applies a 1-D layout of length 4 to 2 internal arrays in myarr:
 *
 *   // Arrays of the same size, which will become internal to the MultiView.
 *   int a1[4] = {5,6,7,8};
 *   int a2[4] = {9,10,11,12};
 *
 *   // Array-of-pointers which will be passed into MultiView.
 *   int * myarr[2];
 *   myarr[0] = a1;
 *   myarr[1] = a2;
 *
 *   // This MultiView applies a 1-D layout of length 4 to each internal array in myarr.
 *   RAJA::MultiView< int, RAJA::Layout<1> > MView(myarr, 4);
 * 
 * The default MultiView accesses internal arrays via the 0th index of the MultiView:
 * 
 *   MView( 0, 4 ); // accesses the 4th index of the 0th internal array a1, returns value of 8
 *   MView( 1, 2 ); // accesses 2nd index of the 1st internal array a2, returns value of 10
 * 
 * The index into the array-of-pointers can be moved to different
 * indices of the MultiView () access operator, rather than the default 0th index. By 
 * passing a third template parameter to the MultiView constructor, the internal array index
 * and the integer indicating which array to access can be reversed:
 *
 *   // MultiView with array-of-pointers index in 1st position
 *   RAJA::MultiView< int, RAJA::Layout<1>, 1 > MView1(myarr, 4);
 *
 *   MView1( 4, 0 ); // accesses the 4th index of the 0th internal array a1, returns value of 8
 *   MView1( 2, 1 ); // accesses 2nd index of the 1st internal array a2, returns value of 10
 * 
 * As the number of Layout dimensions increases, the index into the array-of-pointers can be
 * moved to more distinct locations in the MultiView () access operator. Here is an example
 * which compares the accesses of a 2-D layout on a normal RAJA::View with a RAJA::MultiView
 * with the array-of-pointers index set to the 2nd position:
 *  
 *   RAJA::View< int, RAJA::Layout<2> > normalView(a1, 2, 2);
 *
 *   normalView( 2, 1 ); // accesses 3rd index of the a1 array, value = 7
 *
 *   // MultiView with array-of-pointers index in 2nd position
 *   RAJA::MultiView< int, RAJA::Layout<2>, 2 > MView2(myarr, 2, 2);
 *
 *   MView2( 2, 1, 0 ); // accesses the 3rd index of the 0th internal array a1, returns value of 7 (same as normaView(2,1))
 *   MView2( 2, 1, 1 ); // accesses the 3rd index of the 1st internal array a2, returns value of 11
 *
 * The following code demonstrates 2 aspects of RAJA::MultiView usage:
 * - Basic usage
 * - Moving of the array-of-pointers index
 */

void docs_example()
{
  // temporaries
  int t1, t2, t3, t4;

  printf( "MultiView Example from RAJA Documentation:\n" );

  // _multiview_example_1Dinit_start
  // Arrays of the same size, which will become internal to the MultiView.
  int a1[4] = {5,6,7,8};
  int a2[4] = {9,10,11,12};

  // Array-of-pointers which will be passed into MultiView.
  int * myarr[2];
  myarr[0] = a1;
  myarr[1] = a2;

  // This MultiView applies a 1-D layout of length 4 to each internal array in myarr.
  RAJA::MultiView< int, RAJA::Layout<1> > MView(myarr, 4);
  // _multiview_example_1Dinit_end

  // _multiview_example_1Daccess_start
  t1 = MView( 0, 3 ); // accesses the 4th index of the 0th internal array a1, returns value of 8
  t2 = MView( 1, 2 ); // accesses 3rd index of the 1st internal array a2, returns value of 11
  // _multiview_example_1Daccess_end

  // _multiview_example_1Daopindex_start
  // MultiView with array-of-pointers index in 1st position.
  RAJA::MultiView< int, RAJA::Layout<1>, 1 > MView1(myarr, 4);

  t3 = MView1( 3, 0 ); // accesses the 4th index of the 0th internal array a1, returns value of 8
  t4 = MView1( 2, 1 ); // accesses 3rd index of the 1st internal array a2, returns value of 11
  // _multiview_example_1Daopindex_end

  printf( "Comparison of default MultiView with another MultiView that has the array-of-pointers index in the 1st position of the () accessor:\n" );
  printf( "MView( 0, 3 ) = %i, MView1( 3, 0 ) = %i\n", t1, t3 );
  printf( "MView( 1, 2 ) = %i, MView1( 2, 1 ) = %i\n", t2, t4 );

  // _multiview_example_2Daopindex_start
  RAJA::View< int, RAJA::Layout<2> > normalView(a1, 2, 2);

  t1 = normalView( 1, 1 ); // accesses 4th index of the a1 array, value = 8

  // MultiView with array-of-pointers index in 2nd position
  RAJA::MultiView< int, RAJA::Layout<2>, 2 > MView2(myarr, 2, 2);

  t2 = MView2( 1, 1, 0 ); // accesses the 4th index of the 0th internal array a1, returns value of 8 (same as normalView(1,1))
  t3 = MView2( 0, 0, 1 ); // accesses the 1st index of the 1st internal array a2, returns value of 9
  // _multiview_example_2Daopindex_end

  printf( "Comparison of 2D normal View with 2D MultiView that has the array-of-pointers index in the 2nd position of the () accessor:\n" );
  printf( "normalView( 1, 1 ) = %i, MView2( 1, 1, 0 ) = %i\n", t1, t2 );
}

int main()
{
  docs_example();

  constexpr int N = 12;
  int * myarr[2]; // two 3x4 arrays
  int arr1[N];
  int arr2[N];

  for ( int ii = 0; ii < N; ++ii )
  {
    arr1[ii] = 100 + ii;
    arr2[ii] = 200 + ii;
  }

  myarr[0] = arr1;
  myarr[1] = arr2;

  // 4x3 layout
  std::array<RAJA::idx_t, 2> perm { {0, 1} };
  RAJA::Layout<2> layout = RAJA::make_permuted_layout(
                              { {4, 3} }, perm
                           );

  // Basic MultiView usage
  // Default usage: no specified array-of-pointers index moving
  // 0th position is used as the array-of-pointers index
  RAJA::MultiView<int, RAJA::Layout<2, RAJA::Index_type>> arrView(myarr, layout);

  // Moved array-of-pointers index MultiView usage
  // Add an array-of-pointers index specifier
  constexpr int aopidx = 1;
  RAJA::MultiView<int, RAJA::Layout<2, RAJA::Index_type>, aopidx> arrViewMov(myarr, layout);

  // Comparing values of both views
  printf ( "Comparing values of both default and 1-index-ed MultiViews:\n" );
  for ( int pp = 0; pp < 2; ++pp )
  {
    for ( int kk = 0; kk < 4; ++kk )
    {
      for ( int jj = 0; jj < 3; ++jj )
      {
        printf ( "arr(%i, %i, %i) %d == arrmov(%i, %i, %i) %d\n", pp, kk, jj, arrView(pp, kk, jj), kk, pp, jj, arrViewMov(kk, pp, jj) );
      }
    }
  }

  // switch values
  printf ( "Switching values\n" );
  for ( int kk = 0; kk < 4; ++kk )
  {
    for ( int jj = 0; jj < 3; ++jj )
    {
      int temp = arrView(0, kk, jj);
      arrView(0, kk, jj) = arrView(1, kk, jj);
      arrView(1, kk, jj) = temp;
    }
  }

  // Comparing switched values of both views
  printf ( "Comparing switched values of both default and 1-index-ed MultiViews:\n" );
  for ( int pp = 0; pp < 2; ++pp )
  {
    for ( int kk = 0; kk < 4; ++kk )
    {
      for ( int jj = 0; jj < 3; ++jj )
      {
        printf ( "arr(%i, %i, %i) %d == arrmov(%i, %i, %i) %d\n", pp, kk, jj, arrView(pp, kk, jj), kk, pp, jj, arrViewMov(kk, pp, jj) );
      }
    }
  }

  return 0;
}
