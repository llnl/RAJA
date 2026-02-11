//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_MULTIPLE_REDUCESUM_HPP__
#define __TEST_FORALL_MULTIPLE_REDUCESUM_HPP__

#include <cstdlib>
#include <numeric>

template <typename IDX_TYPE,
          typename DATA_TYPE, typename WORKING_RES,
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceSumMultipleStaggeredTestImpl(IDX_TYPE first, 
                                              IDX_TYPE last)
{
  RAJA::TypedRangeSegment<IDX_TYPE> r1(first, last);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const DATA_TYPE initval = 2;

  for (IDX_TYPE i = first; i < last; ++i) {
    test_array[i] = initval;
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);


  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum0(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum1(initval * 1);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum2(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum3(initval * 3);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum4(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum5(initval * 5);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum6(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum7(initval * 7);

  const DATA_TYPE index_len = static_cast<DATA_TYPE>(last - first);

  const int nloops = 2;
  for (int j = 0; j < nloops; ++j) {

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      sum0 += working_array[idx];
      sum1 += working_array[idx] * 2;
      sum2 += working_array[idx] * 3;
      sum3 += working_array[idx] * 4;
      sum4 += working_array[idx] * 5;
      sum5 += working_array[idx] * 6;
      sum6 += working_array[idx] * 7;
      sum7 += working_array[idx] * 8;
    });

    DATA_TYPE check_val = initval * index_len * (j + 1);

    ASSERT_EQ(1 * check_val, static_cast<DATA_TYPE>(sum0.get()));
    ASSERT_EQ(2 * check_val + (initval*1), static_cast<DATA_TYPE>(sum1.get()));
    ASSERT_EQ(3 * check_val, static_cast<DATA_TYPE>(sum2.get()));
    ASSERT_EQ(4 * check_val + (initval*3), static_cast<DATA_TYPE>(sum3.get()));
    ASSERT_EQ(5 * check_val, static_cast<DATA_TYPE>(sum4.get()));
    ASSERT_EQ(6 * check_val + (initval*5), static_cast<DATA_TYPE>(sum5.get()));
    ASSERT_EQ(7 * check_val, static_cast<DATA_TYPE>(sum6.get()));
    ASSERT_EQ(8 * check_val + (initval*7), static_cast<DATA_TYPE>(sum7.get()));

  }

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

template <typename IDX_TYPE,
          typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceSumMultipleStaggered2TestImpl(IDX_TYPE first, 
			                       IDX_TYPE last)
{
  RAJA::TypedRangeSegment<IDX_TYPE> r1(first, last);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData<DATA_TYPE>(last,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const DATA_TYPE initval = 2;

  for (IDX_TYPE i = first; i < last; ++i) {
    test_array[i] = initval;
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

  const DATA_TYPE index_len = static_cast<DATA_TYPE>(last - first);


  // Workaround for broken omp-target reduction interface.
  // This should be `sumX;` not `sumX(0);`
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum0(initval);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum1(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum2(initval);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum3(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum4(initval);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum5(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum6(initval);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> sum7(0);

  sum0.reset(0);
  sum1.reset(initval * 1);
  sum2.reset(0);
  sum3.reset(initval * 3);
  sum4.reset(0.0);
  sum5.reset(initval * 5);
  sum6.reset(0.0);
  sum7.reset(initval * 7);

  const int nloops = 3;
  for (int j = 0; j < nloops; ++j) {

    RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      sum0 += working_array[idx];
      sum1 += working_array[idx] * 2;
      sum2 += working_array[idx] * 3;
      sum3 += working_array[idx] * 4;
      sum4 += working_array[idx] * 5;
      sum5 += working_array[idx] * 6;
      sum6 += working_array[idx] * 7;
      sum7 += working_array[idx] * 8;
    });

    DATA_TYPE check_val = initval * index_len * (j + 1);

    ASSERT_EQ(1 * check_val, static_cast<DATA_TYPE>(sum0.get()));
    ASSERT_EQ(2 * check_val + (initval*1), static_cast<DATA_TYPE>(sum1.get()));
    ASSERT_EQ(3 * check_val, static_cast<DATA_TYPE>(sum2.get()));
    ASSERT_EQ(4 * check_val + (initval*3), static_cast<DATA_TYPE>(sum3.get()));
    ASSERT_EQ(5 * check_val, static_cast<DATA_TYPE>(sum4.get()));
    ASSERT_EQ(6 * check_val + (initval*5), static_cast<DATA_TYPE>(sum5.get()));
    ASSERT_EQ(7 * check_val, static_cast<DATA_TYPE>(sum6.get()));
    ASSERT_EQ(8 * check_val + (initval*7), static_cast<DATA_TYPE>(sum7.get()));

  }

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceSumMultipleTest);
template <typename T>
class ForallReduceSumMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceSumMultipleTest, ReduceSumMultipleForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceSumMultipleStaggeredTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                           EXEC_POLICY, REDUCE_POLICY>(0, 2115);

  ForallReduceSumMultipleStaggered2TestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES, 
                                            EXEC_POLICY, REDUCE_POLICY>(0, 2115);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceSumMultipleTest,
                            ReduceSumMultipleForall);

#endif  // __TEST_FORALL_MULTIPLE_REDUCESUM_HPP__
