//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__
#define __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelTileLocalArray2DTestImpl(const int rows, const int cols)
{
  // This test emulates matrix transposition with tiling.

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  // holds transposed matrices
  DATA_TYPE * work_array_t;
  DATA_TYPE * check_array_t;
  DATA_TYPE * test_array_t;

  INDEX_TYPE array_length = rows * cols;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array_t,
                                      &check_array_t,
                                      &test_array_t
                                    );

  RAJA::View<DATA_TYPE, RAJA::Layout<2>> HostView( test_array, rows, cols );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> HostTView( test_array_t, cols, rows );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> WorkView( work_array, rows, cols );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> WorkTView( work_array_t, cols, rows );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> CheckTView( check_array_t, cols, rows );

  // initialize local array (shared mem)
  using TILE_MEM = RAJA::LocalArray<DATA_TYPE, RAJA::Perm<0,1>, RAJA::SizeList<tile_dim_x, tile_dim_y>>;
  TILE_MEM Tile_Array;

  // initialize arrays
  std::iota( test_array, test_array + array_length, 1 );
  std::iota( test_array_t, test_array_t + array_length, 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );
  work_res.memcpy( work_array_t, test_array_t, sizeof(DATA_TYPE) * array_length );

  // transpose test_array on CPU
  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      HostTView( cc, rr ) = HostView( rr, cc ); 
    }
  }

  // transpose work_array
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows );
  RAJA::TypedRangeSegment<INDEX_TYPE> colrange( 0, cols );

  RAJA::kernel_param<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ), RAJA::make_tuple( (INDEX_TYPE)0, (INDEX_TYPE)0, Tile_Array ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr, INDEX_TYPE tx, INDEX_TYPE ty, TILE_MEM &_Tile_Array ) {
      _Tile_Array( ty, tx ) = WorkView( rr, cc );
    },

    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr, INDEX_TYPE tx, INDEX_TYPE ty, TILE_MEM &_Tile_Array ) {
      WorkTView( cc, rr ) = _Tile_Array( ty, tx );
    }
  );

  work_res.memcpy( check_array_t, work_array_t, sizeof(DATA_TYPE) * array_length );

  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      ASSERT_EQ(CheckTView(cc, rr), HostTView(cc, rr));
    }
  }

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array_t,
                                        check_array_t,
                                        test_array_t
                                      );
}


TYPED_TEST_SUITE_P(KernelTileLocalArray2DTest);
template <typename T>
class KernelTileLocalArray2DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelTileLocalArray2DTest, TileLocalArray2DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(10, 10);
  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(151, 111);
  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileLocalArray2DTest,
                            TileLocalArray2DKernel);

#endif  // __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__
