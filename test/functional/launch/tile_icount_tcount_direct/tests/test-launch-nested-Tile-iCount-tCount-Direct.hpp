//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_NESTED_TILE_ICOUNT_TCOUNT_DIRECT_HPP__
#define __TEST_LAUNCH_NESTED_TILE_ICOUNT_TCOUNT_DIRECT_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY,
          typename THREAD_X_POLICY, typename TEAM_X_POLICY>
void LaunchNestedTileDirectTestImpl(INDEX_TYPE M)
{

  constexpr INDEX_TYPE threads_x   = 4;
  constexpr INDEX_TYPE blocks_x    = 4;

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, M*threads_x+1);

  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());

  INDEX_TYPE no_tiles = (N1-1)/threads_x + 1;

  INDEX_TYPE N = static_cast<INDEX_TYPE>(RAJA::stripIndexType(N1));

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_ttile_array;
  INDEX_TYPE* check_ttile_array;
  INDEX_TYPE* test_ttile_array;

  INDEX_TYPE* working_iloop_array;
  INDEX_TYPE* check_iloop_array;
  INDEX_TYPE* test_iloop_array;

  size_t data_len = RAJA::stripIndexType(N);
  if ( data_len == 0 ) {
    data_len = 1;
  }

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_ttile_array,
                                     &check_ttile_array,
                                     &test_ttile_array);

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_iloop_array,
                                     &check_iloop_array,
                                     &test_iloop_array);

  std::iota(test_ttile_array, test_ttile_array + data_len, 0);
  std::iota(test_iloop_array, test_iloop_array + data_len, 0);
  working_res.memset(working_ttile_array, 0, sizeof(INDEX_TYPE) * data_len);
  working_res.memset(working_iloop_array, 0, sizeof(INDEX_TYPE) * data_len);

  if ( RAJA::stripIndexType(N) > 0 ) {

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks_x), RAJA::Threads(threads_x)), [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::tile_tcount<TEAM_X_POLICY>(
          ctx, threads_x, r1, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &x_tile, INDEX_TYPE bx) {
            RAJA::loop_icount<THREAD_X_POLICY>(
              ctx, x_tile, [&](INDEX_TYPE tx, INDEX_TYPE ix) {

                working_ttile_array[tx] += bx;
                working_iloop_array[tx] += ix;

              }
            );
          }
        );
      }
    );

  } else { // zero-length segment

    RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks_x), RAJA::Threads(blocks_x)), [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::tile_tcount<TEAM_X_POLICY>(
          ctx, threads_x, r1, [&](RAJA::TypedRangeSegment<INDEX_TYPE> const &x_tile, INDEX_TYPE RAJA_UNUSED_ARG(bx)) {
            RAJA::loop_icount<THREAD_X_POLICY>(
              ctx, x_tile, [&](INDEX_TYPE RAJA_UNUSED_ARG(tx), INDEX_TYPE RAJA_UNUSED_ARG (ix)) {

                working_ttile_array[0]++;
                working_iloop_array[0]++;

              }
            );
          }
        );
      }
    );
  }

  working_res.memcpy(check_ttile_array, working_ttile_array, sizeof(INDEX_TYPE) * data_len);
  working_res.memcpy(check_iloop_array, working_iloop_array, sizeof(INDEX_TYPE) * data_len);
  working_res.wait();

  if (RAJA::stripIndexType(N) > 0) {

    INDEX_TYPE idx = 0;
    for (INDEX_TYPE bx = INDEX_TYPE(0); bx < no_tiles; ++bx) {
      for (INDEX_TYPE tx = INDEX_TYPE(0); tx < threads_x; ++tx) {

        if(idx >= N1) break;

        ASSERT_EQ(check_ttile_array[RAJA::stripIndexType(idx)], bx);
        ASSERT_EQ(check_iloop_array[RAJA::stripIndexType(idx)], tx);

        idx++;
      }
    }
    
  } else {
    
    ASSERT_EQ(check_ttile_array[0], check_ttile_array[0]);
    ASSERT_EQ(check_iloop_array[0], check_iloop_array[0]);
    
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_ttile_array,
                                       check_ttile_array,
                                       test_ttile_array);

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_iloop_array,
                                       check_iloop_array,
                                       test_iloop_array);
}


TYPED_TEST_SUITE_P(LaunchNestedTileDirectTest);
template <typename T>
class LaunchNestedTileDirectTest : public ::testing::Test
{
};


TYPED_TEST_P(LaunchNestedTileDirectTest, RangeSegmentTeams)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;

  using TEAM_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using THREAD_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;


  // test zero-length range segment
  LaunchNestedTileDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, TEAM_X_POLICY>
    (INDEX_TYPE(0));

  //Keep at one since we are doing a direct thread test
  LaunchNestedTileDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                                 THREAD_X_POLICY, TEAM_X_POLICY>
    (INDEX_TYPE(1));

    LaunchNestedTileDirectTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                                 THREAD_X_POLICY, TEAM_X_POLICY>
    (INDEX_TYPE(2));


}

REGISTER_TYPED_TEST_SUITE_P(LaunchNestedTileDirectTest,
                            RangeSegmentTeams);

#endif  // __TEST_LAUNCH_NESTED_TILE_ICOUNT_TCOUNT_DIRECT_HPP__
