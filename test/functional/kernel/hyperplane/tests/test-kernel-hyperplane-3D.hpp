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
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCER>
std::enable_if_t<USE_PARAM_REDUCER::value>
CallKernel(DATA_TYPE& trip_count,
           DATA_TYPE& oob_count,
           RAJA::View<DATA_TYPE, RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>>>& WorkView,
           const INDEX_TYPE idim,
           const INDEX_TYPE jdim,
           const INDEX_TYPE kdim,
           const INDEX_TYPE groups)
{

  // perform array arithmetic with a 2D J-K hyperplane
  RAJA::TypedRangeSegment<INDEX_TYPE>   Grange( 0, RAJA::stripIndexType(groups) );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Irange( 0, RAJA::stripIndexType(idim), 1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Jrange( RAJA::stripIndexType(jdim) - 1, -1, -1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Krange( 0, RAJA::stripIndexType(kdim), 1 );

  RAJA::kernel_param<EXEC_POLICY> (
    RAJA::make_tuple( Grange, Irange, Jrange, Krange ),
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus>(&trip_count),
      RAJA::expt::Reduce<RAJA::operators::plus>(&oob_count)
    ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj, INDEX_TYPE kk,
                           RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _trip_count,
                           RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>& _oob_count ) {
      if (RAJA::stripIndexType(g) >= RAJA::stripIndexType(groups) || RAJA::stripIndexType(ii) >= RAJA::stripIndexType(idim) ||
          RAJA::stripIndexType(jj) >= RAJA::stripIndexType(jdim) || RAJA::stripIndexType(kk) >= RAJA::stripIndexType(kdim)) {
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
           RAJA::View<DATA_TYPE, RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>>>& WorkView,
           const INDEX_TYPE idim,
           const INDEX_TYPE jdim,
           const INDEX_TYPE kdim,
           const INDEX_TYPE groups)
{
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> trip_count (_trip_count);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> oob_count (_oob_count);
  // perform array arithmetic with a 2D J-K hyperplane
  RAJA::TypedRangeSegment<INDEX_TYPE>   Grange( 0, RAJA::stripIndexType(groups) );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Irange( 0, RAJA::stripIndexType(idim), 1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Jrange( RAJA::stripIndexType(jdim - 1), -1, -1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Krange( 0, RAJA::stripIndexType(kdim), 1 );

  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( Grange, Irange, Jrange, Krange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj, INDEX_TYPE kk ) {
      if (RAJA::stripIndexType(g) >= RAJA::stripIndexType(groups) || RAJA::stripIndexType(ii) >= RAJA::stripIndexType(idim) ||
          RAJA::stripIndexType(jj) >= RAJA::stripIndexType(jdim) || RAJA::stripIndexType(kk) >= RAJA::stripIndexType(kdim)) {
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
KernelHyperplane3DTestImpl(const INDEX_TYPE RAJA_UNUSED_ARG(groups), const INDEX_TYPE RAJA_UNUSED_ARG(idim), const INDEX_TYPE RAJA_UNUSED_ARG(jdim), const INDEX_TYPE RAJA_UNUSED_ARG(kdim))
{
  // do nothing for unsigned index types
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY, typename USE_PARAM_REDUCERS>
std::enable_if_t<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>
KernelHyperplane3DTestImpl(const INDEX_TYPE groups, const INDEX_TYPE idimin, const INDEX_TYPE jdimin, const INDEX_TYPE kdimin)
{
  // This test traverses "groups" number of 3D arrays, and modifies values in a 2D hyperplane manner.

  using raw_index_type = VAL_T<INDEX_TYPE>;
  raw_index_type idim, jdim, kdim;
  if ( std::is_same<DATA_TYPE, float>::value )
  {
    // Restrict to a small data size for better float precision.
    idim = 5;
    jdim = 5;
    kdim = 5;
  }
  else
  {
    idim = RAJA::stripIndexType(idimin);
    jdim = RAJA::stripIndexType(jdimin);
    kdim = RAJA::stripIndexType(kdimin);
  }

  INDEX_TYPE idim_t(idim);
  INDEX_TYPE jdim_t(jdim);
  INDEX_TYPE kdim_t(kdim);

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  INDEX_TYPE array_length = groups * idim_t * jdim_t * kdim_t;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  using LayoutType = RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>>;
  using ViewType = RAJA::View<DATA_TYPE, LayoutType>;
  ViewType HostView( test_array, groups, idim_t, jdim_t, kdim_t );
  ViewType WorkView( work_array, groups, idim_t, jdim_t, kdim_t );
  ViewType CheckView( check_array, groups, idim_t, jdim_t, kdim_t );

  // initialize array
  std::iota( test_array, test_array + RAJA::stripIndexType(array_length), 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * RAJA::stripIndexType(array_length) );

  DATA_TYPE trip_count(0);
  DATA_TYPE oob_count(0);

  CallKernel<INDEX_TYPE, DATA_TYPE, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(trip_count, oob_count, WorkView, idim_t, jdim_t, kdim_t, groups);

  ASSERT_EQ((INDEX_TYPE)trip_count, groups * idim_t * jdim_t * kdim_t);
  ASSERT_EQ((INDEX_TYPE)oob_count, (INDEX_TYPE)0);

  work_res.memcpy( check_array, work_array, sizeof(DATA_TYPE) * RAJA::stripIndexType(array_length) );

  // perform array arithmetic on the CPU
  for (INDEX_TYPE g(0); g < groups; ++g) {
    for (INDEX_TYPE i(0); i < idim_t; ++i) {
      for (INDEX_TYPE j(jdim_t - 1); j >= INDEX_TYPE(0); --j) {
        for (INDEX_TYPE k(0); k < kdim_t; ++k) {
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

  for (INDEX_TYPE g(0); g < groups; ++g) {
    for (INDEX_TYPE i(0); i < idim_t; ++i) {
      for (INDEX_TYPE j(0); j < jdim_t; ++j) {
        for (INDEX_TYPE k(0); k < kdim_t; ++k) {
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

  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(INDEX_TYPE{1}, INDEX_TYPE{10}, INDEX_TYPE{10}, INDEX_TYPE{10});
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(INDEX_TYPE{2}, INDEX_TYPE{151}, INDEX_TYPE{111}, INDEX_TYPE{205});
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY, USE_PARAM_REDUCERS>(INDEX_TYPE{3}, INDEX_TYPE{101}, INDEX_TYPE{213}, INDEX_TYPE{123});
}

REGISTER_TYPED_TEST_SUITE_P(KernelHyperplane3DTest,
                            Hyperplane3DKernel);

#endif  // __TEST_KERNEL_HYPERPLANE_3D_HPP__
