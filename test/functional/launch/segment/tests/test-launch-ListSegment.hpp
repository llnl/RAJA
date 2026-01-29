//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_LISTSEGMENT_HPP__
#define __TEST_LAUNCH_LISTSEGMENT_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename GLOBAL_THREAD_POICY>
void LaunchListSegmentTestImpl(INDEX_TYPE N)
{

  // Create and initialize indices in idx_array used to create list segment
  std::vector<INDEX_TYPE> idx_array;

  srand ( time(NULL) );

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i) {
    INDEX_TYPE randval = INDEX_TYPE(rand() % RAJA::stripIndexType(N));
    if ( i < randval ) {
      idx_array.push_back(i);
    }
  }

  size_t idxlen = idx_array.size();

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  // Create list segment for tests
  INDEX_TYPE* idx_vals = nullptr;
  if (N > 0) {
    idx_vals = &idx_array[0];
  }
  RAJA::TypedListSegment<INDEX_TYPE> lseg(idx_vals, idxlen,
                                          working_res);

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

    for (size_t i = 0; i < idxlen; ++i) {
      test_array[ RAJA::stripIndexType(idx_vals[i]) ] = idx_vals[i];
    }

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<GLOBAL_THREAD_POICY>(ctx, lseg, [&](INDEX_TYPE idx) {
            working_array[RAJA::stripIndexType(idx)] = idx;
          });
      });

  } else { // zero-length segment

    memset(static_cast<void*>(test_array), 0, sizeof(INDEX_TYPE) * data_len);

    working_res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * data_len);

    RAJA::launch<LAUNCH_POLICY>
      (RAJA::LaunchParams(RAJA::Teams(blocks), RAJA::Threads(threads)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<GLOBAL_THREAD_POICY>(ctx, lseg, [&](INDEX_TYPE idx) {
            (void) idx;
            working_array[0]++;
          });
      });

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


TYPED_TEST_SUITE_P(LaunchListSegmentTest);
template <typename T>
class LaunchListSegmentTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchListSegmentTest, ListSegmentTeams)
{
  using INDEX_TYPE       = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using GLOBAL_THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;

  // test zero-length list segment
  LaunchListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(0));

  LaunchListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(13));

  LaunchListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(2047));

  LaunchListSegmentTestImpl<INDEX_TYPE, WORKING_RESOURCE, LAUNCH_POLICY, GLOBAL_THREAD_POLICY>(INDEX_TYPE(32000));
}

REGISTER_TYPED_TEST_SUITE_P(LaunchListSegmentTest,
                            ListSegmentTeams);

#endif  // __TEST_TEAMS_LISTSEGMENT_HPP__
