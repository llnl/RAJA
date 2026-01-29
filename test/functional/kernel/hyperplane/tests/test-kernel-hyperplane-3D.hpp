//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_HYPERPLANE_3D_HPP__
#define __TEST_KERNEL_HYPERPLANE_3D_HPP__

#include <numeric>
#include <type_traits>

template <typename INDEX_TYPE, typename DATA_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCER>
std::enable_if_t<USE_PARAM_REDUCER::value>
CallKernel(DATA_TYPE& trip_count,
           DATA_TYPE& oob_count,
           RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>>& WorkView,
           const int idim,
           const int jdim,
           const int kdim,
           const int groups)
{

  // perform array arithmetic with a 2D J-K hyperplane
  RAJA::TypedRangeSegment<INDEX_TYPE>   Grange( 0, groups );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Irange( 0, idim, 1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Jrange( jdim-1, -1, -1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Krange( 0, kdim, 1 );

  RAJA::kernel_param<EXEC_POLICY> (
    RAJA::make_tuple( Grange, Irange, Jrange, Krange ),
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus>(&trip_count),
      RAJA::expt::Reduce<RAJA::operators::plus>(&oob_count)
    ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj, INDEX_TYPE kk,
                           RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _trip_count,
                           RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _oob_count ) {
      if (g < 0 || g >= groups || ii < 0 || ii >= idim || jj < 0 || jj >= jdim || kk < 0 || kk >= kdim) {
        _oob_count += 1;
      }

      DATA_TYPE left = 1;
      if (ii > 0) {
        left = WorkView(g, ii - 1, jj, kk);
      }

      DATA_TYPE up = 1;
      if (jj > 0) {
        up = WorkView(g, ii, jj - 1, kk);
      }

      DATA_TYPE back = 1;
      if (kk > 0) {
        back = WorkView(g, ii, jj, kk - 1);
      }

      WorkView(g, ii, jj, kk) = left + up + back;

      _trip_count += 1;
  });
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCER>
std::enable_if_t<!USE_PARAM_REDUCER::value>
CallKernel(DATA_TYPE& _trip_count,
           DATA_TYPE& _oob_count,
           RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>>& WorkView,
           const int idim,
           const int jdim,
           const int kdim,
           const int groups)
{
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> trip_count (_trip_count);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> oob_count (_oob_count);
  // perform array arithmetic with a 2D J-K hyperplane
  RAJA::TypedRangeSegment<INDEX_TYPE>   Grange( 0, groups );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Irange( 0, idim, 1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Jrange( jdim-1, -1, -1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Krange( 0, kdim, 1 );

  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( Grange, Irange, Jrange, Krange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj, INDEX_TYPE kk ) {
      if (g < 0 || g >= groups || ii < 0 || ii >= idim || jj < 0 || jj >= jdim || kk < 0 || kk >= kdim) {
        oob_count += 1;
      }

      DATA_TYPE left = 1;
      if (ii > 0) {
        left = WorkView(g, ii - 1, jj, kk);
      }

      DATA_TYPE up = 1;
      if (jj > 0) {
        up = WorkView(g, ii, jj - 1, kk);
      }

      DATA_TYPE back = 1;
      if (kk > 0) {
        back = WorkView(g, ii, jj, kk - 1);
      }

      WorkView(g, ii, jj, kk) = left + up + back;

      trip_count += 1;
  });
  _trip_count = trip_count.get();
  _oob_count = oob_count.get();
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCERS>
std::enable_if_t<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>
KernelHyperplane3DTestImpl(const int RAJA_UNUSED_ARG(groups), const int RAJA_UNUSED_ARG(idim), const int RAJA_UNUSED_ARG(jdim), const int RAJA_UNUSED_ARG(kdim))
{
  // do nothing for unsigned index types
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCERS>
std::enable_if_t<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>
KernelHyperplane3DTestImpl(const int groups, const int idimin, const int jdimin, const int kdimin)
{
  // This test traverses "groups" number of 3D arrays, and modifies values in a 2D hyperplane manner.

  int idim, jdim, kdim;
  if ( std::is_same<DATA_TYPE, float>::value )
  {
    // Restrict to a small data size for better float precision.
    idim = 5;
    jdim = 5;
    kdim = 5;
  }
  else
  {
    idim = idimin;
    jdim = jdimin;
    kdim = kdimin;
  }

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  INDEX_TYPE array_length = groups * idim * jdim * kdim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> HostView( test_array, groups, idim, jdim, kdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> WorkView( work_array, groups, idim, jdim, kdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> CheckView( check_array, groups, idim, jdim, kdim );

  // initialize array
  std::iota( test_array, test_array + array_length, 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );

  DATA_TYPE trip_count(0);
  DATA_TYPE oob_count(0);

  CallKernel<INDEX_TYPE, DATA_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(trip_count, oob_count, WorkView, idim, jdim, kdim, groups);

  ASSERT_EQ((INDEX_TYPE)trip_count, (INDEX_TYPE)groups * idim * jdim * kdim);
  ASSERT_EQ((INDEX_TYPE)oob_count, (INDEX_TYPE)0);

  work_res.memcpy( check_array, work_array, sizeof(DATA_TYPE) * array_length );

  // perform array arithmetic on the CPU
  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = jdim - 1; j >= 0; --j) {
        for (int k = 0; k < kdim; ++k) {
          DATA_TYPE left = 1;
          if (i > 0) {
            left = HostView(g, i - 1, j, k);
          }

          DATA_TYPE up = 1;
          if (j > 0) {
            up = HostView(g, i, j - 1, k);
          }

          DATA_TYPE back = 1;
          if (k > 0) {
            back = HostView(g, i, j, k - 1);
          }

          HostView(g, i, j, k) = left + up + back;
        }
      }
    }
  }

  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = 0; j < jdim; ++j) {
        for (int k = 0; k < kdim; ++k) {
          ASSERT_FLOAT_EQ(CheckView(g, i, j, k), HostView(g, i, j, k));
        }
      }
    }
  }

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );
}


TYPED_TEST_SUITE_P(KernelHyperplane3DTest);
template <typename T>
class KernelHyperplane3DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelHyperplane3DTest, Hyperplane3DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using USE_PARAM_REDUCERS = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(1, 10, 10, 10);
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(2, 151, 111, 205);
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(3, 101, 213, 123);
}

REGISTER_TYPED_TEST_SUITE_P(KernelHyperplane3DTest,
                            Hyperplane3DKernel);

#endif  // __TEST_KERNEL_HYPERPLANE_3D_HPP__
