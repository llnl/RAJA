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

namespace {
template <typename T>
struct val_t_impl {
  using type = T;
};

template <RAJA::concepts::IndexValued T>
struct val_t_impl<T> {
  using type = typename T::value_type;
};

template <typename T>
using VAL_T = typename val_t_impl<T>::type;

template<RAJA::concepts::IndexValued T>
RAJA_HOST_DEVICE typename T::value_type get_val(T index_val) { return *index_val; }

template<typename T>
requires (!RAJA::concepts::IndexValued<T>)
RAJA_HOST_DEVICE auto get_val(T index_val) { return index_val; }
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelTileFixed2DSumTestImpl(const INDEX_TYPE rowsin, const INDEX_TYPE colsin)
{
  // This test reduces sums with tiling.
  using raw_index_type = VAL_T<INDEX_TYPE>;

  raw_index_type rows, cols;
  if ( std::is_same<DATA_TYPE, float>::value )
  {
    // Restrict to a small data size for better float precision.
    rows = 3;
    cols = 3;
  }
  else
  {
    rows = get_val(rowsin);
    cols = get_val(colsin);
  }

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE hostsum = 0;

  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> worksum( DATA_TYPE(0) ); 

  // sum on CPU in a tiled manner
  for ( raw_index_type rr = 0; rr < rows; rr += tile_dim_x )
  {
    for ( raw_index_type cc = 0; cc < cols; cc += tile_dim_y )
    {
      for ( raw_index_type r = rr; r < std::min<raw_index_type>(rr + tile_dim_x, rows); ++r )
      {
        for ( raw_index_type c = cc; c < std::min<raw_index_type>(cc + tile_dim_y, cols); ++c )
        {
          hostsum += (DATA_TYPE)(r * 1.1 + c);
        }
      }
    }
  }

  // mixed range types
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, INDEX_TYPE(rows) );

  std::vector<INDEX_TYPE> colidx;
  for (raw_index_type ii = 0; ii < cols; ++ii)
  {
    colidx.push_back(INDEX_TYPE(ii));
  }

  RAJA::TypedListSegment<INDEX_TYPE> colrange( &colidx[0], colidx.size(), work_res );

  // sum on target platform
  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      worksum += (DATA_TYPE)(get_val(rr) * 1.1 + get_val(cc));
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

  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE(10), INDEX_TYPE(10));
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE(151), INDEX_TYPE(111));
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE(362), INDEX_TYPE(362));
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileFixed2DSumTest,
                            TileFixed2DSumKernel);

#endif  // __TEST_KERNEL_TILE_FIXED2DSUM_HPP__
