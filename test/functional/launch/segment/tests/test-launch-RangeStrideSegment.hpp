//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_RANGESTRIDESEGMENT_HPP__
#define __TEST_LAUNCH_RANGESTRIDESEGMENT_HPP__

#include <cstring>

template <typename INDEX_TYPE, typename DIFF_TYPE,
          typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POICY>
void LaunchRangeStrideSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last,
                                      DIFF_TYPE stride)
{
  RAJA::TypedRangeStrideSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last), stride);
  INDEX_TYPE N = INDEX_TYPE(r1.size());

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N);
  if ( data_len == 0 ) {
    data_len = 1;
  }

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

  working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

  constexpr int threads = 256;
  int blocks = (data_len - 1)/threads + 1;

  if ( RAJA::stripIndexType(N) > 0 ) {

    INDEX_TYPE index = first;
    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i) {
      test_array[ RAJA::stripIndexType((index-first)/stride) ] = index;
      index += stride;
    }

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)), [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<GLOBAL_THREAD_POICY>(
          ctx, r1, [&](INDEX_TYPE idx) {
            working_array[ RAJA::stripIndexType((idx-first)/stride) ] = idx;
          }
        );

      }
    );

  } else { // zero-length segment

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)), [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<GLOBAL_THREAD_POICY>(
          ctx, r1, [&](INDEX_TYPE RAJA_UNUSED_ARG(idx)) {
            working_array[0]++;
          }
        );

      }
    );
  }

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  if (RAJA::stripIndexType(N) > 0) {

    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
    }

  } else {

    ASSERT_EQ(test_array[0], check_array[0]);

  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(LaunchRangeStrideSegmentTest);
template <typename T>
class LaunchRangeStrideSegmentTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
}

template <typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-10), INDEX_TYPE(-1), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5), DIFF_TYPE(3));

// Test negative strides
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(10), INDEX_TYPE(-1), DIFF_TYPE(-1));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(10), INDEX_TYPE(0), DIFF_TYPE(-2));
}


TYPED_TEST_P(LaunchRangeStrideSegmentTest, RangeStrideSegmentTeams)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using DIFF_TYPE   = typename std::make_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::type;

  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(1));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(1));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0), INDEX_TYPE(21), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(21), DIFF_TYPE(2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(255), DIFF_TYPE(2));

// Test size zero segments
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(-2));
  LaunchRangeStrideSegmentTestImpl<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(-2));

  runNegativeStrideTests<INDEX_TYPE, DIFF_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(LaunchRangeStrideSegmentTest,
                            RangeStrideSegmentTeams);

#endif  // __TEST_TEAMS_RANGESTRIDESEGMENT_HPP__
