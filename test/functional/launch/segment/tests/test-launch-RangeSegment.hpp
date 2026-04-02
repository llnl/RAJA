//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_RANGE_SEGMENT_HPP__
#define __TEST_LAUNCH_RANGE_SEGMENT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POICY>
void LaunchRangeSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last));
  INDEX_TYPE N = static_cast<INDEX_TYPE>(r1.end() - r1.begin());

  camp::resources::Resource working_res{WORKING_RES::get_default()};
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

  constexpr int threads = 256;
  int blocks = (data_len - 1)/threads + 1;

  if ( RAJA::stripIndexType(N) > 0 ) {

    const INDEX_TYPE rbegin = *r1.begin();

    std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<GLOBAL_THREAD_POICY>(
          ctx, r1, [&](INDEX_TYPE idx) {
            working_array[RAJA::stripIndexType(idx - rbegin)] = idx;
          }
        );
      }
    );

  } else { // zero-length segment

    memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),  [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

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


TYPED_TEST_SUITE_P(LaunchRangeSegmentTest);
template <typename T>
class LaunchRangeSegmentTest : public ::testing::Test
{
};

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
}

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeTests()
{
  // test zero-length range segment
  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(-5));

  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0));
  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5));
}

TYPED_TEST_P(LaunchRangeSegmentTest, RangeSegmentTeams)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;

  // test zero-length range segment
  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(3), INDEX_TYPE(3));

  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0), INDEX_TYPE(27));
  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(2047));
  LaunchRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(1), INDEX_TYPE(32000));

  runNegativeTests<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(LaunchRangeSegmentTest,
                            RangeSegmentTeams);

#endif  // __TEST_RANGE_SEGMENT_HPP__
