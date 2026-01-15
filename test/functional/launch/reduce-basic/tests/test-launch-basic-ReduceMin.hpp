//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_BASIC_REDUCEMIN_HPP__
#define __TEST_LAUNCH_BASIC_REDUCEMIN_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename DATA_TYPE,
          typename SEG_TYPE,
          typename LAUNCH_POLICY,
          typename GLOBAL_THREAD_POLICY,
          typename REDUCE_POLICY>
void LaunchReduceMinBasicTestImpl(const SEG_TYPE& seg,
                                  const std::vector<IDX_TYPE>& seg_idx,
                                  camp::resources::Resource working_res)
{
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
  const DATA_TYPE min_init = modval+1;
  const DATA_TYPE small_min = -modval;

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = static_cast<DATA_TYPE>( rand() % modval );
  }

  DATA_TYPE ref_min = min_init;
  for (IDX_TYPE i = 0; i < idx_len; ++i) {
    ref_min = RAJA_MIN(test_array[ seg_idx[i] ], ref_min);
  }

  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);


  RAJA::ReduceMin<REDUCE_POLICY, DATA_TYPE> mininit(small_min);
  RAJA::ReduceMin<REDUCE_POLICY, DATA_TYPE> min(min_init);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     RAJA::Name("launch ReduceMin"),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {

          mininit.min( working_array[idx] );
          min.min( working_array[idx] );
    });
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(mininit.get()), small_min);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min);

  min.reset(min_init);
  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), min_init);

  DATA_TYPE factor = 3;
  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {

          min.min( working_array[idx] * factor);
    });
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);

  factor = 2;
  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
      RAJA::loop<GLOBAL_THREAD_POLICY>(ctx, seg, [&](IDX_TYPE idx) {

          min.min( working_array[idx] * factor);
      });
  });

  ASSERT_EQ(static_cast<DATA_TYPE>(min.get()), ref_min * factor);


  deallocateForallTestData<DATA_TYPE>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


TYPED_TEST_SUITE_P(LaunchReduceMinBasicTest);
template <typename T>
class LaunchReduceMinBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchReduceMinBasicTest, ReduceMinBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<3>>::type, camp::num<1>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
                                 r1, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
                                 r2, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
                                 r3, seg_idx, working_res);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
                                 r4, seg_idx, working_res);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
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
  LaunchReduceMinBasicTestImpl<IDX_TYPE, DATA_TYPE,
                               RAJA::TypedListSegment<IDX_TYPE>,
                               LAUNCH_POLICY, GLOBAL_THREAD_POLICY, REDUCE_POLICY>(
                                 l1, seg_idx, working_res);
}

REGISTER_TYPED_TEST_SUITE_P(LaunchReduceMinBasicTest,
                            ReduceMinBasicForall);

#endif  // __TEST_LAUNCH_BASIC_REDUCEMIN_HPP__
