//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_FIXED2DMINMAX_HPP__
#define __TEST_KERNEL_TILE_FIXED2DMINMAX_HPP__

#include <numeric>
#include <vector>

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelTileFixed2DMinMaxTestImpl(const int rows, const int cols)
{
  // This test reduces min and max with tiling.

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  INDEX_TYPE array_length = rows * cols;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  // initialize arrays
  std::iota( test_array, test_array + array_length, 1 );

  // set min and max of the array
  test_array[4] = -1;
  test_array[8] = array_length+2;

  RAJA::View<DATA_TYPE, RAJA::Layout<2>> WorkView( work_array, rows, cols );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );

  RAJA::ReduceMin<REDUCE_POLICY, DATA_TYPE> workmin( DATA_TYPE(99999) ); 
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> workmax( DATA_TYPE(-1) ); 

  // mixed range types
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows );

  std::vector<INDEX_TYPE> colidx;
  for (INDEX_TYPE ii = INDEX_TYPE(0); ii < static_cast<INDEX_TYPE>(cols); ++ii)
  {
    colidx.push_back(ii);
  }

  RAJA::TypedListSegment<INDEX_TYPE> colrange( &colidx[0], colidx.size(), work_res );

  // find min and max on target platform
  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      workmin.min(WorkView(rr, cc));
      workmax.max(WorkView(rr, cc));
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(-1), static_cast<DATA_TYPE>(workmin.get()));
  ASSERT_EQ(static_cast<DATA_TYPE>(array_length+2), static_cast<DATA_TYPE>(workmax.get()));

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );
}


TYPED_TEST_SUITE_P(KernelTileFixed2DMinMaxTest);
template <typename T>
class KernelTileFixed2DMinMaxTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelTileFixed2DMinMaxTest, TileFixed2DMinMaxKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  KernelTileFixed2DMinMaxTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelTileFixed2DMinMaxTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(151, 111);
  KernelTileFixed2DMinMaxTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileFixed2DMinMaxTest,
                            TileFixed2DMinMaxKernel);

#endif  // __TEST_KERNEL_TILE_FIXED2DMINMAX_HPP__
