//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__
#define __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__

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

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelLocMax2DViewTupleTestImpl(const INDEX_TYPE xdim, const INDEX_TYPE ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE* work_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  // set rows to point to check and work _arrays
  RAJA::TypedRangeSegment<INDEX_TYPE> seg(0,ydim);
  using LayoutType = RAJA::TypedLayout<INDEX_TYPE, camp::tuple<INDEX_TYPE, INDEX_TYPE>>;
  using ViewType = RAJA::View<DATA_TYPE, LayoutType>;
  ViewType ArrView(work_array, xdim, ydim);
  ViewType CheckView(check_array, xdim, ydim);

  // initializing  values
  RAJA::forall<RAJA::seq_exec>(seg, [=] (INDEX_TYPE zz)
  {
    for ( INDEX_TYPE xx(0); xx < xdim; ++xx )
    {
      CheckView(zz, xx) = RAJA::stripIndexType(zz * xdim + xx);
    }
    CheckView(ydim - 1, xdim - 1) = 0;
  });

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * RAJA::stripIndexType(array_length));

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  //#pragma omp target data map(to:work_array[0:array_length])
#endif

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange(0, xdim);
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange(0, ydim);

  RAJA::tuple<DATA_TYPE, DATA_TYPE> LocTup(0, 0);

  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, RAJA::tuple<DATA_TYPE, DATA_TYPE>> maxloc_reducer((DATA_TYPE)0, LocTup);

  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(colrange, rowrange),
                           [=] RAJA_HOST_DEVICE (INDEX_TYPE c, INDEX_TYPE r) {
                             maxloc_reducer.maxloc(ArrView(r, c), RAJA::make_tuple((DATA_TYPE)RAJA::stripIndexType(c), (DATA_TYPE)RAJA::stripIndexType(r)));
                           });

  // CPU answer
  RAJA::ReduceMaxLoc<RAJA::seq_reduce, DATA_TYPE, Index2D<INDEX_TYPE>> checkmaxloc_reducer((DATA_TYPE)0, Index2D<INDEX_TYPE>(INDEX_TYPE{0}, INDEX_TYPE{0}));

  RAJA::forall<RAJA::seq_exec>(colrange, [=] (INDEX_TYPE c) {
    for (INDEX_TYPE r(0); r < ydim; ++r)
    {
      checkmaxloc_reducer.maxloc(CheckView(r, c), Index2D<INDEX_TYPE>(c, r));
    }
  });

  RAJA::tuple<DATA_TYPE, DATA_TYPE> raja_loc = maxloc_reducer.getLoc();
  DATA_TYPE raja_max = (DATA_TYPE)maxloc_reducer.get();
  Index2D<INDEX_TYPE> checkraja_loc = checkmaxloc_reducer.getLoc();
  DATA_TYPE checkraja_max = (DATA_TYPE)checkmaxloc_reducer.get();

  ASSERT_DOUBLE_EQ((DATA_TYPE)checkraja_max, (DATA_TYPE)raja_max);
  ASSERT_EQ(checkraja_loc.idx, RAJA::get<0>(raja_loc));
  ASSERT_EQ(checkraja_loc.idy, RAJA::get<1>(raja_loc));

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );
}


TYPED_TEST_SUITE_P(KernelLocMax2DViewTupleTest);
template <typename T>
class KernelLocMax2DViewTupleTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelLocMax2DViewTupleTest, LocMax2DViewTupleKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE{10}, INDEX_TYPE{10});
  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE{151}, INDEX_TYPE{151});
  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(INDEX_TYPE{362}, INDEX_TYPE{362});
}

REGISTER_TYPED_TEST_SUITE_P(KernelLocMax2DViewTupleTest,
                            LocMax2DViewTupleKernel);

#endif  // __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__
