//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <gtest/gtest.h>
#include "RAJA/util/types.hpp"
#include "RAJA_test-base.hpp"

using namespace RAJA;

TEST(IndexLayout, IndexList1D) {
  /*
   * Construct a 1D index layout with the index list {1,2,3}
   */

  Index_type arr[3] = {1,2,3};

  auto index_tuple = make_index_tuple(IndexList<>{&arr[0]});
  auto index_layout = make_index_layout(index_tuple, 3);

  EXPECT_EQ(index_layout(0), 1);
  EXPECT_EQ(index_layout(1), 2);
  EXPECT_EQ(index_layout(2), 3);

  EXPECT_EQ(index_layout.size(), 3);
  EXPECT_EQ(index_layout.size_noproj(), 3);
  EXPECT_EQ(index_layout.get_dim_stride<0>(), 1);
  EXPECT_EQ(index_layout.get_dim_size<0>(), 3);

}

TEST(IndexLayout, IndexList1DSubsetOfLayout) {
  /*
   * Construct a 1D index layout of arbitrary size greater than 3 
   * with the index list {2,3,4}.
   * The purpose of this test is to demonstrate the use case where
   * the index list contains a subset of its index layout
   */

  Index_type arr[3] = {2,3,4};

  auto index_tuple = make_index_tuple(IndexList<>{&arr[0]});
  auto index_layout = make_index_layout(index_tuple, 5);

  EXPECT_EQ(index_layout(0), 2);
  EXPECT_EQ(index_layout(1), 3);
  EXPECT_EQ(index_layout(2), 4);

}


TEST(IndexLayout, ExtractTwoIndices2DLayoutAxis0) {
  /*
   * Construct a 2D index layout of size 3x10 with 
   * the index list {1,2} used along the 0-axis and
   * the direct index used along the 1-axis
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(0,1)   -> layout(1,1)          -> 11
   *   index_layout(0,5)   -> layout(1,5)          -> 15
   *   index_layout(1,7)   -> layout(2,7)          -> 27
   */

  Index_type arr[2] = {1,2};

  auto index_tuple = make_index_tuple(IndexList<>{&arr[0]}, DirectIndex<>());
  auto index_layout = make_index_layout(index_tuple, 3, 10);

  for (int i = 0; i < 10; i++ ) {
    EXPECT_EQ(index_layout(0,i), i+10);
    EXPECT_EQ(index_layout(1,i), i+20);
  }

  EXPECT_EQ(index_layout.size(), 30);
  EXPECT_EQ(index_layout.size_noproj(), 30);
  EXPECT_EQ(index_layout.get_dim_stride<0>(), 10);
  EXPECT_EQ(index_layout.get_dim_stride<1>(), 1);
  EXPECT_EQ(index_layout.get_dim_size<0>(), 3);
  EXPECT_EQ(index_layout.get_dim_size<1>(), 10);

}

TEST(IndexLayout, ExtractTwoIndices2DLayoutAxis1) {
  /*
   * Construct a 2D index layout of size 3x10 with 
   * the direct index used along the 0-axis and
   * the index list {9,5} used along the 1-axis
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(0,1)   -> layout(0,5)          -> 5
   *   index_layout(2,0)   -> layout(2,9)          -> 29
   */

  Index_type arr[2] = {9,5};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>{&arr[0]});
  auto index_layout = make_index_layout(index_tuple, 3, 10);

  EXPECT_EQ(index_layout(0,0), 9);
  EXPECT_EQ(index_layout(0,1), 5);
  EXPECT_EQ(index_layout(1,0), 19);
  EXPECT_EQ(index_layout(1,1), 15);
  EXPECT_EQ(index_layout(2,0), 29);
  EXPECT_EQ(index_layout(2,1), 25);

}

TEST(IndexLayout, ExtractOneIndex2DLayoutAxis0) {
  /*
   * Construct a 2D index layout of size 3x3 with 
   * the index list {2} used along the 0-axis and
   * the direct index used along the 1-axis
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(0,1)   -> layout(2,1)          -> 7
   *   index_layout(0,2)   -> layout(2,2)          -> 8
   */

  Index_type arr[1] = {2};

  auto index_tuple = make_index_tuple(IndexList<>{&arr[0]}, DirectIndex<>());
  auto index_layout = make_index_layout(index_tuple, 3, 3);

  EXPECT_EQ(index_layout(0,0), 6);
  EXPECT_EQ(index_layout(0,1), 7);
  EXPECT_EQ(index_layout(0,2), 8);  

}

TEST(IndexLayout, IndexList2DLayoutExtractOneIndex) {
  /*
   * Construct a 2D index layout of size 3x3 with 
   * the direct index used along the 0-axis and
   * the index list {2} used along the 1-axis
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(1,0)   -> layout(1,2)          -> 5
   *   index_layout(2,0)   -> layout(2,2)          -> 8
   */

  Index_type arr[1] = {2};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>{&arr[0]});
  auto index_layout = make_index_layout(index_tuple, 3, 3);

  EXPECT_EQ(index_layout(0,0), 2);
  EXPECT_EQ(index_layout(1,0), 5);
  EXPECT_EQ(index_layout(2,0), 8);

}

TEST(IndexLayout, ConditionalIndexListNullPtr) {
  /*
   * Construct a 1D index layout of size 3 with 
   * the conditional index list that is a nullptr
   * (conditional index lists always evaluate nullptr to regular indexing)
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(0)     -> layout(0)            -> 0
   *   index_layout(2)     -> layout(2)            -> 2
   */

  Index_type* arr_ptr = nullptr;

  auto index_tuple = make_index_tuple(ConditionalIndexList<>{arr_ptr});
  auto index_layout = make_index_layout(index_tuple, 3);

  EXPECT_EQ(index_layout(0), 0);
  EXPECT_EQ(index_layout(1), 1);
  EXPECT_EQ(index_layout(2), 2);
}

TEST(IndexLayout, ConditionalIndexListWithIndexList) {
  /*
   * Construct a 1D index layout of size 3 with 
   * the conditional index list that is not a nullptr
   * (conditional index lists with index list act the same as IndexList)
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index)
   *   index_layout(0)     -> layout(1)            -> 1
   *   index_layout(1)     -> layout(2)            -> 2
   */

  Index_type arr[2] = {1,2};

  auto index_tuple = make_index_tuple(ConditionalIndexList<>{&arr[0]});
  auto index_layout = make_index_layout(index_tuple, 3);

  EXPECT_EQ(index_layout(0), 1);
  EXPECT_EQ(index_layout(1), 2);
}

TEST(IndexLayout, View1DLayout)
{
  /*
   * Construct a 1D index layout of size 5 with 
   * the index list {4,2,3} and pass to a 1D view with the data {5,10,15,20,25}
   * Examples: 
   *   (index layout index -> regular layout index -> unit stride index -> view at index)
   *   index_layout(0)     -> layout(4)            -> 4                 -> 25
   *   index_layout(2)     -> layout(3)            -> 3                 -> 20
   */
  
  Index_type data[5] = {5,10,15,20,25};
  Index_type index_list[3] = {4,2,3};

  auto index_tuple = make_index_tuple(IndexList<>{&index_list[0]});
  auto index_layout = make_index_layout(index_tuple, 5);

  auto view = make_index_view(&data[0], index_layout);

  EXPECT_EQ(view(0), 25);
  EXPECT_EQ(view(1), 15);
  EXPECT_EQ(view(2), 20);

}

TEST(IndexLayout, View2DLayout)
{
  /*
   * Construct a 2D index layout of size 2x3 with 
   * the direct index used along the 0-axis and
   * the index list {1,2} used along the 1-axis and
   * pass to a 2D view of size 2x3 with the each entry being i*j
   * for i,j in [0,2)x[0,3) (e.g. view(1,2) = 1*2, view(0,2) = 0*2, etc..)
   * Examples: 
   *   (index layout index -> view index -> view at index)
   *   index_layout(0,1)   -> view(0,2)  -> 0
   *   index_layout(1,0)   -> view(1,1)  -> 1
   */

  Index_type data[2][3];

  for (int i = 0; i < 2; i ++ ) {
    for (int j = 0; j < 3; j ++ ) {
      data[i][j] = i*j;
    }
  }

  Index_type index_list[2] = {1,2};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>{&index_list[0]});
  auto index_layout = make_index_layout(index_tuple, 2, 3);

  auto view = make_index_view(&data[0][0], index_layout);

  for (int i = 0; i < 2; i ++ ) {
    for (int j = 0; j < 2; j ++ ) {
      EXPECT_EQ(view(i,j), i*(j+1));
    }
  }

}

TEST(IndexLayout, View3DLayout)
{
  /*
   * Construct a 3D index layout of size 2x3x4 with 
   * the direct index used along the 0-axis and
   * the index list {1,2} used along the 1-axis and
   * the index list {2,3} used along the 2-axis and
   * pass to a 3D view of size 2x3x4 with the each entry being i*j*k
   * for i,j,k in [0,2)x[0,3)x[0,4) (e.g. view(1,2,3) = 1*2*3, view(0,2,2) = 0*2*2, etc..)
   * Examples: 
   *   (index layout index -> view index -> view at index)
   *   index_layout(0,1,0) -> view(0,2,2)-> 0
   *   index_layout(2,1,1) -> view(2,2,3)-> 12
   */
  
  Index_type data[2][3][4];

  for (int i = 0; i < 2; i ++ ) {
    for (int j = 0; j < 3; j ++ ) {
      for (int k = 0; k < 4; k ++ ) {
	data[i][j][k] = i*j*k;
      }
    }
  }

  Index_type index_list_j[2] = {1,2};
  Index_type index_list_k[2] = {2,3};

  auto index_tuple = make_index_tuple(DirectIndex<>(), 
                                      IndexList<>{&index_list_j[0]},
                                      IndexList<>{&index_list_k[0]});

  auto index_layout = make_index_layout(index_tuple, 2, 3, 4);

  auto view = make_index_view(&data[0][0][0], index_layout);

  for (int i = 0; i < 2; i ++ ) {
    for (int j = 0; j < 2; j ++ ) {
      for (int k = 0; k < 2; k ++ ) {
        EXPECT_EQ(view(i,j,k), i*(j+1)*(k+2));
      }
    }
  }

}

TEST(IndexLayout, MultiView1DLayout)
{
  /*
   * Construct a 1D index layout of size 4 with 
   * the index list {1,2} and pass to a 1D multiview containing two 1D views of size 4 with
   * the first view having each entry be the square of its index (e.g. view(2) = 2*2 = 4)
   * and the second view having each entry be the cube of its index (e.g. view(3) = 3*3*3 = 27)
   * Examples: 
   *   (index layout index -> mutiview index -> view at index)
   *   index_layout(0,1)   -> view(0,2)      -> 4
   *   index_layout(1,0)   -> view(1,1)      -> 1
   */

  Index_type data_squared[4];
  Index_type data_cubed[4];

  for (int i = 0; i < 4; i ++ ) {
    data_squared[i] = i*i;
  }
  
  for (int i = 0; i < 4; i ++ ) {
    data_cubed[i] = i*i*i;
  }

  Index_type* data_array[2];
  data_array[0] = data_squared;
  data_array[1] = data_cubed;

  Index_type index_list[2] = {1,2};

  auto index_tuple = make_index_tuple(IndexList<>{&index_list[0]});
  auto index_layout = make_index_layout(index_tuple, 4);

  auto view = MultiView<Index_type, IndexLayout<1, Index_type, IndexList<> > >(data_array, index_layout);

  for (int i = 0; i < 2; i ++ ) {
    EXPECT_EQ(view(0,i), data_squared[i+1]);
    EXPECT_EQ(view(1,i), data_cubed[i+1]);
  }

}

