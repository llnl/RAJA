//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_PARAM_EXPT_BASIC_REDUCESUM_HPP__
#define __TEST_LAUNCH_PARAM_EXPT_BASIC_REDUCESUM_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename LAUNCH_POLICY,
          typename GLOBAL_THREAD_POLICY>
void LaunchParamExptReduceSumBasicTestImpl(const SEG_TYPE& seg,
                                           const std::vector<IDX_TYPE>& seg_idx,
                                           camp::resources::Resource working_res)
{
  using REF_SUM = RAJA::expt::ValOp<DATA_TYPE, RAJA::operators::plus>;

  IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  constexpr int threads = 256;
  int blocks = (seg.size() - 1)/threads + 1;

  allocateForallTestData<DATA_TYPE>(data_len,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  const int modval = 100;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_sum = 0;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    ref_sum += test_array[ seg_idx[i] ];
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);


  DATA_TYPE sum(0), sum2(2);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     RAJA::expt::Reduce<RAJA::operators::plus>(&sum),
     RAJA::expt::Reduce<RAJA::operators::plus>(&sum2),
     RAJA::Name("LaunchSumBasicTest"),     
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx, REF_SUM &_sum, REF_SUM &_sum2) {

      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {
          _sum  += working_array[idx];
          _sum2 += working_array[idx];
     });

  });

  ASSERT_EQ(static_cast<DATA_TYPE>(sum), ref_sum);
  ASSERT_EQ(static_cast<DATA_TYPE>(sum2), ref_sum + 2);

  sum = 0;

  const int nloops = 2;

  for (int j = 0; j < nloops; ++j) {
    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
       RAJA::expt::Reduce<RAJA::operators::plus>(&sum),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx, REF_SUM &_sum) {

        RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {
            _sum += working_array[idx];
          });
      });
  }

  ASSERT_EQ(static_cast<DATA_TYPE>(sum), nloops * ref_sum);


  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(LaunchParamExptReduceSumBasicTest);
template <typename T>
class LaunchParamExptReduceSumBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchParamExptReduceSumBasicTest, ReduceSumBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<1>>::type;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                        RAJA::TypedRangeSegment<IDX_TYPE>,
                                        LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                        r1, seg_idx, working_res);
     
  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                       RAJA::TypedRangeSegment<IDX_TYPE>,
                                       LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                       r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                       RAJA::TypedRangeSegment<IDX_TYPE>,
                                       LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                       r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                       RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                       LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                       r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                       RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                                       LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                       r5, seg_idx, working_res);

// List segment tests
  seg_idx.clear();
  IDX_TYPE last = 10567;
  srand( time(NULL) );
  for (IDX_TYPE i = 0; i < last; ++i) {
    IDX_TYPE randval = IDX_TYPE( rand() % RAJA::stripIndexType(last) );
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(),
                                       working_res );
  LaunchParamExptReduceSumBasicTestImpl<IDX_TYPE, DATA_TYPE,
                                       RAJA::TypedListSegment<IDX_TYPE>,
                                      LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(
                                      l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(LaunchParamExptReduceSumBasicTest,
                            ReduceSumBasicForall);

#endif  // __TEST_LAUNCH_BASIC_REDUCESUM_HPP__
