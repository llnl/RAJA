//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_MULTIPLE_REDUCEMAX_HPP__
#define __TEST_FORALL_MULTIPLE_REDUCEMAX_HPP__

#include <cfloat>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <random>

template <typename IDX_TYPE, 
          typename DATA_TYPE, typename WORKING_RES, 
          typename EXEC_POLICY, typename REDUCE_POLICY>
void ForallReduceMaxMultipleTestImpl(IDX_TYPE first, 
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

  const DATA_TYPE default_val = static_cast<DATA_TYPE>(-SHRT_MAX);
  const DATA_TYPE big_val = 500;
  
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<double> dist(-100, 100);
  static std::uniform_int_distribution<int> dist2(static_cast<int>(first), static_cast<int>(last) - 1);

  // Workaround for broken omp-target reduction interface.
  // This should be `max0;` not `max0(0);`
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> max0(0);
  max0.reset(default_val);
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> max1(default_val);
  RAJA::ReduceMax<REDUCE_POLICY, DATA_TYPE> max2(big_val);

  const int nOuterLoops = 2;
  for (int l = 0; l < nOuterLoops; ++l) {

    ASSERT_EQ(default_val, static_cast<DATA_TYPE>(max0.get()));
    ASSERT_EQ(default_val, static_cast<DATA_TYPE>(max1.get()));
    ASSERT_EQ(big_val, static_cast<DATA_TYPE>(max2.get()));

    DATA_TYPE current_max = default_val;

    const int nMiddleLoops = 2;
    for (int k = 0; k < nMiddleLoops; ++k) {

      for (IDX_TYPE i = 0; i < last; ++i) {
        test_array[i] = default_val;
      }
      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * last);

      const int nloops = 6;
      for (int j = 0; j < nloops; ++j) {

        DATA_TYPE roll = static_cast<DATA_TYPE>( dist(mt) );
        IDX_TYPE max_index = static_cast<IDX_TYPE>(dist2(mt));

        test_array[max_index] = roll;
        working_res.memcpy(&working_array[max_index], &test_array[max_index], sizeof(DATA_TYPE));

        if ( current_max < roll ) {
          current_max = roll ;
        }

        RAJA::forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
          max0.max(working_array[idx]);
          max1.max(2 * working_array[idx]);
          max2.max(working_array[idx]);
        });

        ASSERT_EQ(current_max, static_cast<DATA_TYPE>(max0.get()));
        ASSERT_EQ(current_max * 2, static_cast<DATA_TYPE>(max1.get()));
        ASSERT_EQ(big_val, static_cast<DATA_TYPE>(max2.get()));

      }

    }

    max0.reset(default_val);
    max1.reset(default_val);
    max2.reset(big_val);

  }

  ASSERT_EQ(default_val, static_cast<DATA_TYPE>(max0.get()));
  ASSERT_EQ(default_val, static_cast<DATA_TYPE>(max1.get()));
  ASSERT_EQ(big_val, static_cast<DATA_TYPE>(max2.get()));

  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}

TYPED_TEST_SUITE_P(ForallReduceMaxMultipleTest);
template <typename T>
class ForallReduceMaxMultipleTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallReduceMaxMultipleTest, ReduceMaxMultipleForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallReduceMaxMultipleTestImpl<IDX_TYPE, DATA_TYPE, WORKING_RES,
                                  EXEC_POLICY, REDUCE_POLICY>(0, 2115);
}

REGISTER_TYPED_TEST_SUITE_P(ForallReduceMaxMultipleTest,
                            ReduceMaxMultipleForall);

#endif  // __TEST_FORALL_MULTIPLE_REDUCEMAX_HPP__
