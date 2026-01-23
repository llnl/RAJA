//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_HYPERPLANE_2D_HPP__
#define __TEST_KERNEL_HYPERPLANE_2D_HPP__

#include <numeric>
#include <type_traits>

template <typename INDEX_TYPE, typename DATA_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCER>
std::enable_if_t<USE_PARAM_REDUCER::value>
CallKernel(DATA_TYPE& trip_count,
           DATA_TYPE& oob_count,
           RAJA::View<DATA_TYPE, RAJA::Layout<3, INDEX_TYPE>>& WorkView,
           const int idim,
           const int jdim,
           const int groups)
{
  RAJA::TypedRangeSegment<INDEX_TYPE>  Grange( 0, groups );
  RAJA::TypedRangeSegment<INDEX_TYPE>  Irange( 0, idim );
  RAJA::TypedRangeSegment<INDEX_TYPE>  Jrange( 0, jdim );

  RAJA::kernel_param<EXEC_POLICY> ( RAJA::make_tuple( Grange, Irange, Jrange ),
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus>(&trip_count),
      RAJA::expt::Reduce<RAJA::operators::plus>(&oob_count)
    ),
    [=] RAJA_HOST_DEVICE (INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj,
                          RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _trip_count,
                          RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _oob_count ) {
      if ((int)g < 0 || (int)g >= groups || (int)ii < 0 || (int)ii >= idim || (int)jj < 0 || (int)jj >= jdim) {
        _oob_count += 1;
      }

      DATA_TYPE left = 1;
      if (ii > 0) {
        left = WorkView(g, ii - 1, jj);
      }

      DATA_TYPE up = 1;
      if (jj > 0) {
        up = WorkView(g, ii, jj - 1);
      }

      WorkView(g, ii, jj) = left + up;

      _trip_count += 1;
  });

}

template <typename INDEX_TYPE, typename DATA_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCER>
std::enable_if_t<!USE_PARAM_REDUCER::value>
CallKernel(DATA_TYPE& _trip_count,
           DATA_TYPE& _oob_count,
           RAJA::View<DATA_TYPE, RAJA::Layout<3, INDEX_TYPE>>& WorkView,
           const int idim,
           const int jdim,
           const int groups)
{
  RAJA::TypedRangeSegment<INDEX_TYPE>  Grange( 0, groups );
  RAJA::TypedRangeSegment<INDEX_TYPE>  Irange( 0, idim );
  RAJA::TypedRangeSegment<INDEX_TYPE>  Jrange( 0, jdim );

  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> trip_count(_trip_count);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> oob_count(_oob_count);

  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( Grange, Irange, Jrange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj ) {
      if ((int)g < 0 || (int)g >= groups || (int)ii < 0 || (int)ii >= idim || (int)jj < 0 || (int)jj >= jdim) {
        oob_count += 1;
      }

      DATA_TYPE left = 1;
      if (ii > 0) {
        left = WorkView(g, ii - 1, jj);
      }

      DATA_TYPE up = 1;
      if (jj > 0) {
        up = WorkView(g, ii, jj - 1);
      }

      WorkView(g, ii, jj) = left + up;

      trip_count += 1;
  });
  _trip_count = trip_count.get();
  _oob_count = oob_count.get();
}


template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCERS>
void KernelHyperplane2DTestImpl(const int groups, const int idim, const int jdim)
{
  // This test traverses "groups" 2D arrays, and modifies values in a 1D hyperplane manner.

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  INDEX_TYPE array_length = groups * idim * jdim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  RAJA::View<DATA_TYPE, RAJA::Layout<3, INDEX_TYPE>> HostView( test_array, groups, idim, jdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<3, INDEX_TYPE>> WorkView( work_array, groups, idim, jdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<3, INDEX_TYPE>> CheckView( check_array, groups, idim, jdim );

  // initialize array
  std::iota( test_array, test_array + array_length, 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );

  DATA_TYPE trip_count(0);
  DATA_TYPE oob_count(0);

  // perform array arithmetic with a 1D hyperplane, in either the I or J direction
  CallKernel<INDEX_TYPE, DATA_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(trip_count, oob_count, WorkView, idim, jdim, groups);

  ASSERT_EQ((INDEX_TYPE)trip_count, (INDEX_TYPE)groups * idim * jdim);
  ASSERT_EQ((INDEX_TYPE)oob_count, (INDEX_TYPE)0);

  work_res.memcpy( check_array, work_array, sizeof(DATA_TYPE) * array_length );

  // perform array arithmetic on the CPU
  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = 0; j < jdim; ++j) {
        DATA_TYPE left = 1;
        if (i > 0) {
          left = HostView(g, i - 1, j);
        }

        DATA_TYPE up = 1;
        if (j > 0) {
          up = HostView(g, i, j - 1);
        }

        HostView(g, i, j) = left + up;
      }
    }
  }

  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = 0; j < jdim; ++j) {
        ASSERT_FLOAT_EQ(CheckView(g, i, j), HostView(g, i, j));
      }
    }
  }

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );
}


TYPED_TEST_SUITE_P(KernelHyperplane2DTest);
template <typename T>
class KernelHyperplane2DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelHyperplane2DTest, Hyperplane2DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using USE_PARAM_REDUCERS = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelHyperplane2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(1, 10, 10);
  KernelHyperplane2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(2, 111, 205);
  KernelHyperplane2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(3, 213, 123);
}

REGISTER_TYPED_TEST_SUITE_P(KernelHyperplane2DTest,
                            Hyperplane2DKernel);

#endif  // __TEST_KERNEL_HYPERPLANE_2D_HPP__
