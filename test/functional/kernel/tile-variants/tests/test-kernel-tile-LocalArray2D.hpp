//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__
#define __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__

#include <numeric>

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

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelTileLocalArray2DTestImpl(const INDEX_TYPE rows_t, const INDEX_TYPE cols_t)
{
  // This test emulates matrix transposition with tiling.
  using raw_index_type = VAL_T<INDEX_TYPE>;

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  // holds transposed matrices
  DATA_TYPE * work_array_t;
  DATA_TYPE * check_array_t;
  DATA_TYPE * test_array_t;

  INDEX_TYPE array_length = rows_t * cols_t;

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

  using LayoutType =
      RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE>>;
  using ViewType = RAJA::View<DATA_TYPE, LayoutType>;

  ViewType HostView( test_array, rows_t, cols_t );
  ViewType HostTView( test_array_t, cols_t, rows_t );
  ViewType WorkView( work_array, rows_t, cols_t );
  ViewType WorkTView( work_array_t, cols_t, rows_t );
  ViewType CheckTView( check_array_t, cols_t, rows_t );

  // initialize local array (shared mem)
  using TILE_MEM = RAJA::LocalArray<DATA_TYPE, RAJA::Perm<0,1>, RAJA::SizeList<tile_dim_x, tile_dim_y>>;
  TILE_MEM Tile_Array;

  // initialize arrays
  std::iota( test_array, test_array + get_val(array_length), 1 );
  std::iota( test_array_t, test_array_t + get_val(array_length), 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * get_val(array_length) );
  work_res.memcpy( work_array_t, test_array_t, sizeof(DATA_TYPE) * get_val(array_length) );

  // transpose test_array on CPU
  for ( raw_index_type rr = 0; rr < get_val(rows_t); ++rr )
  {
    for ( raw_index_type cc = 0; cc < get_val(cols_t); ++cc )
    {
      HostTView( INDEX_TYPE(cc), INDEX_TYPE(rr) ) = HostView( INDEX_TYPE(rr), INDEX_TYPE(cc) );
    }
  }

  // transpose work_array
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows_t );
  RAJA::TypedRangeSegment<INDEX_TYPE> colrange( 0, cols_t );

  RAJA::kernel_param<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ), RAJA::make_tuple( INDEX_TYPE(0), INDEX_TYPE(0), Tile_Array ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr, INDEX_TYPE tx, INDEX_TYPE ty, TILE_MEM &_Tile_Array ) {
      _Tile_Array( get_val(ty), get_val(tx) ) = WorkView( rr, cc );
    },

    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr, INDEX_TYPE tx, INDEX_TYPE ty, TILE_MEM &_Tile_Array ) {
      WorkTView( cc, rr ) = _Tile_Array( get_val(ty), get_val(tx) );
    }
  );

  work_res.memcpy( check_array_t, work_array_t, sizeof(DATA_TYPE) * get_val(array_length) );

  for ( raw_index_type rr = 0; rr < get_val(rows_t); ++rr )
  {
    for ( raw_index_type cc = 0; cc < get_val(cols_t); ++cc )
    {
      ASSERT_EQ(CheckTView(INDEX_TYPE(cc), INDEX_TYPE(rr)),
                HostTView(INDEX_TYPE(cc), INDEX_TYPE(rr)));
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

  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(10), INDEX_TYPE(10));
  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(151), INDEX_TYPE(111));
  KernelTileLocalArray2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(362), INDEX_TYPE(362));
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileLocalArray2DTest,
                            TileLocalArray2DKernel);

#endif  // __TEST_KERNEL_TILE_LOCALARRAY2D_HPP__
