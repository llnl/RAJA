//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_FIXED2DSUM_HPP__
#define __TEST_KERNEL_TILE_FIXED2DSUM_HPP__

#include <numeric>
#include <vector>
#include <type_traits>

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelTileFixed2DSumTestImpl(const int rowsin, const int colsin)
{
  // This test reduces sums with tiling.

  int rows, cols;
  if ( std::is_same<DATA_TYPE, float>::value )
  {
    // Restrict to a small data size for better float precision.
    rows = 3;
    cols = 3;
  }
  else
  {
    rows = rowsin;
    cols = colsin;
  }

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE hostsum = 0;

  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> worksum( DATA_TYPE(0) ); 

  // sum on CPU in a tiled manner
  for ( int rr = 0; rr < rows; rr += tile_dim_x )
  {
    for ( int cc = 0; cc < cols; cc += tile_dim_y )
    {
      for ( int r = rr; r < std::min(rr+tile_dim_x, rows); ++r )
      {
        for ( int c = cc; c < std::min(cc+tile_dim_y, cols); ++c )
        {
          hostsum += (DATA_TYPE)(r * 1.1 + c);
        }
      }
    }
  }

  // mixed range types
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows );

  std::vector<INDEX_TYPE> colidx;
  for (INDEX_TYPE ii = INDEX_TYPE(0); ii < static_cast<INDEX_TYPE>(cols); ++ii)
  {
    colidx.push_back(ii);
  }

  RAJA::TypedListSegment<INDEX_TYPE> colrange( &colidx[0], colidx.size(), work_res );

  // sum on target platform
  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      worksum += (DATA_TYPE)(rr * 1.1 + cc);
  });

  ASSERT_FLOAT_EQ(hostsum, (DATA_TYPE)worksum.get());
}


TYPED_TEST_SUITE_P(KernelTileFixed2DSumTest);
template <typename T>
class KernelTileFixed2DSumTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelTileFixed2DSumTest, TileFixed2DSumKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(151, 111);
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileFixed2DSumTest,
                            TileFixed2DSumKernel);

#endif  // __TEST_KERNEL_TILE_FIXED2DSUM_HPP__
