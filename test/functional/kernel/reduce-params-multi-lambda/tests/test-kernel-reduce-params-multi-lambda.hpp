//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REDUCELOC_MAX2D_HPP__
#define __TEST_KERNEL_REDUCELOC_MAX2D_HPP__

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY>
void KernelParamReduceMultiLambda(const int xdim, const int ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE ** workarr2D;
  DATA_TYPE ** checkarr2D;
  DATA_TYPE ** testarr2D;
  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  allocateForallTestData<DATA_TYPE *> ( ydim,
                                        work_res,
                                        &workarr2D,
                                        &checkarr2D,
                                        &testarr2D
                                      );

  // set rows to point to check and work _arrays
  RAJA::TypedRangeSegment<INDEX_TYPE> seg(0,ydim);
  RAJA::forall<FORALL_POLICY>(seg, [=] RAJA_HOST_DEVICE (INDEX_TYPE zz)
  {
    workarr2D[zz] = work_array + zz * ydim;
  });

  RAJA::forall<RAJA::seq_exec>(seg, [=] RAJA_HOST_DEVICE (INDEX_TYPE zz)
  {
    checkarr2D[zz] = check_array + zz * ydim;
  });

  // initializing  values
  RAJA::forall<RAJA::seq_exec>(seg, [=] RAJA_HOST_DEVICE (INDEX_TYPE zz)
  {
    for ( int xx = 0; xx < xdim; ++xx )
    {
      checkarr2D[zz][xx] = (zz*xdim + xx ) % 100 + 1;
    }
    // Make a unique min
    checkarr2D[ydim-1][xdim-1] = 0;
    // Make a unique max
    checkarr2D[ydim/2][xdim/2] = 101;
  });

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * array_length);

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange(0, xdim);
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange(0, ydim);

  using VALOP_DATA_TYPE_SUM = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>;
  using VALOP_DATA_TYPE_MIN = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::minimum>;
  using VALOP_DATA_TYPE_MAX = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::maximum>;

  DATA_TYPE sum_1 = 0;
  DATA_TYPE sum_2 = 0;
  DATA_TYPE mutual_sum = 0;
  DATA_TYPE sum_seq = 0;

  DATA_TYPE min_1 = 0;
  DATA_TYPE min_2 = 0;
  DATA_TYPE min_seq = 0;

  DATA_TYPE max_1 = 0;
  DATA_TYPE max_2 = 0;
  DATA_TYPE max_seq = 0;

  // Test that each lambda only performs a single reduction, and that
  // those reductions are identical given the same data.
  RAJA::kernel_param<EXEC_POLICY>(
    // segs
    RAJA::make_tuple(colrange, rowrange),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus>(&sum_1),
      RAJA::expt::Reduce<RAJA::operators::plus>(&sum_2),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&min_1),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&min_2),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&max_1),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&max_2),
      double {0.0},
      // This reducer capture tests that our calls to init/resolve are
      // not init or resolving reducers in between lambda calls.
      RAJA::expt::Reduce<RAJA::operators::plus>(&mutual_sum)
    ),
      [=] RAJA_HOST_DEVICE (int c,
                            int r,
                            VALOP_DATA_TYPE_SUM &_sum,
                            VALOP_DATA_TYPE_MIN &_min,
                            VALOP_DATA_TYPE_MAX &_max,
                            VALOP_DATA_TYPE_SUM &_mutual_sum) {
        _sum += workarr2D[r][c];
        _mutual_sum += workarr2D[r][c];
        _min.min(workarr2D[r][c]);
        _max.max(workarr2D[r][c]);
      },
      [=] RAJA_HOST_DEVICE (int c,
                            int r,
                            VALOP_DATA_TYPE_SUM &_sum,
                            VALOP_DATA_TYPE_MIN &_min,
                            VALOP_DATA_TYPE_MAX &_max,
                            double,
                            VALOP_DATA_TYPE_SUM &_mutual_sum) {
        _sum += workarr2D[r][c];
        _mutual_sum += workarr2D[r][c];
        _min.min(workarr2D[r][c]);
        _max.max(workarr2D[r][c]);
      }
    );



  // CPU answer
  RAJA::forall<RAJA::seq_exec>(rowrange,
      RAJA::expt::Reduce<RAJA::operators::plus>(&sum_seq),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&min_seq),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&max_seq),
    [=] (INDEX_TYPE r,
         VALOP_DATA_TYPE_SUM &_sum,
         VALOP_DATA_TYPE_MIN &_min,
         VALOP_DATA_TYPE_MAX &_max
    ) {
    for (int c = 0; c < xdim; ++c)
    {
      _sum += checkarr2D[r][c];
      _min.min(checkarr2D[r][c]);
      _max.max(checkarr2D[r][c]);
    }
  });

  ASSERT_DOUBLE_EQ(sum_seq, sum_1);
  ASSERT_DOUBLE_EQ(sum_seq, sum_2);
  ASSERT_DOUBLE_EQ(sum_1, sum_2);

  ASSERT_DOUBLE_EQ(min_seq, min_1);
  ASSERT_DOUBLE_EQ(min_seq, min_2);
  ASSERT_DOUBLE_EQ(min_1, min_2);

  ASSERT_DOUBLE_EQ(max_seq, max_1);
  ASSERT_DOUBLE_EQ(max_seq, max_2);
  ASSERT_DOUBLE_EQ(max_1, max_2);

  ASSERT_DOUBLE_EQ(mutual_sum, sum_1 + sum_2);


  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );

  deallocateForallTestData<DATA_TYPE *> ( work_res,
                                          workarr2D,
                                          checkarr2D,
                                          testarr2D
                                        );
}


TYPED_TEST_SUITE_P(KernelParamReduceMultiLambdaTest);
template <typename T>
class KernelParamReduceMultiLambdaTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelParamReduceMultiLambdaTest, ParamReduceKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  KernelParamReduceMultiLambda<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY>(10, 10);
  KernelParamReduceMultiLambda<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY>(151, 151);
  KernelParamReduceMultiLambda<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelParamReduceMultiLambdaTest,
                            ParamReduceKernel);

#endif  // __TEST_KERNEL_REDUCELOC_MAX2D_HPP__
