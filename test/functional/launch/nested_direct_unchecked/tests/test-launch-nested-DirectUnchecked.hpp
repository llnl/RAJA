//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_NESTED_DIRECT_UNCHECKED_HPP__
#define __TEST_LAUNCH_NESTED_DIRECT_UNCHECKED_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY,
          typename THREAD_X_POLICY, typename THREAD_Y_POLICY, typename THREAD_Z_POLICY,
          typename TEAM_X_POLICY, typename TEAM_Y_POLICY, typename TEAM_Z_POLICY>
void LaunchNestedDirectUncheckedTestImpl(INDEX_TYPE M)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, 2*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(0, 3*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r3(0, 4*M);

  RAJA::TypedRangeSegment<INDEX_TYPE> r4(0, 4*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r5(0, 5*M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r6(0, 6*M);

  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N3 = static_cast<INDEX_TYPE>(r3.end() - r3.begin());

  INDEX_TYPE N4 = static_cast<INDEX_TYPE>(r4.end() - r4.begin());
  INDEX_TYPE N5 = static_cast<INDEX_TYPE>(r5.end() - r5.begin());
  INDEX_TYPE N6 = static_cast<INDEX_TYPE>(r6.end() - r6.begin());

  INDEX_TYPE N = static_cast<INDEX_TYPE>(N1 * N2 *                                          
                                         N3 * N4 *
                                         N5 * N6);

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N);

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  std::iota(test_array, test_array + data_len, 0);
  if ( data_len > 0 ) {
    working_res.memset(working_array, 0, sizeof(INDEX_TYPE) * data_len);
  }

  //6 threads total
  const int threads_x = 2*M;
  const int threads_y = 3*M;
  const int threads_z = 4*M;

  const int blocks_x = 4*M;
  const int blocks_y = 5*M;
  const int blocks_z = 6*M;

  const int DIM = 6;
  using layout_t = RAJA::Layout<DIM, INDEX_TYPE,DIM-1>;
  RAJA::View<INDEX_TYPE, layout_t> Aview(working_array, N6, N5, N4, N3, N2, N1);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(blocks_x, blocks_y, blocks_z), RAJA::Threads(threads_x, threads_y,threads_z)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::loop<TEAM_Z_POLICY>(ctx, r6, [&](INDEX_TYPE bz) {
          RAJA::loop<TEAM_Y_POLICY>(ctx, r5, [&](INDEX_TYPE by) {
              RAJA::loop<TEAM_X_POLICY>(ctx, r4, [&](INDEX_TYPE bx) {

                  RAJA::loop<THREAD_Z_POLICY>(ctx, r3, [&](INDEX_TYPE tz) {
                      RAJA::loop<THREAD_Y_POLICY>(ctx, r2, [&](INDEX_TYPE ty) {
                          RAJA::loop<THREAD_X_POLICY>(ctx, r1, [&](INDEX_TYPE tx) {

                              auto idx = tx + N1 * (ty + N2 * (tz + N3 * (bx + N4 * (by + N5 * bz))));


                              Aview(bz, by, bx, tz, ty, tx) += static_cast<INDEX_TYPE>(idx);
                            });
                        });
                    });

                });
            });
        });
  });

  if ( data_len > 0 ) {
    working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);
  }
  working_res.wait();
    
  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(LaunchNestedDirectUncheckedTest);
template <typename T>
class LaunchNestedDirectUncheckedTest : public ::testing::Test
{
};


TYPED_TEST_P(LaunchNestedDirectUncheckedTest, RangeSegmentTeams)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;

  using TEAM_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using TEAM_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;
  using TEAM_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<3>>::type;

  using THREAD_Z_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<4>>::type;
  using THREAD_Y_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<5>>::type;
  using THREAD_X_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<6>>::type;



  // test zero-length range segment
  LaunchNestedDirectUncheckedTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(0));

  //Keep at one since we are doing a direct unchecked thread test
  LaunchNestedDirectUncheckedTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY,
                           THREAD_X_POLICY, THREAD_Y_POLICY, THREAD_Z_POLICY,
                           TEAM_X_POLICY, TEAM_Y_POLICY, TEAM_Z_POLICY>
    (INDEX_TYPE(1));


}

REGISTER_TYPED_TEST_SUITE_P(LaunchNestedDirectUncheckedTest,
                            RangeSegmentTeams);

#endif  // __TEST_LAUNCH_NESTED_DIRECT_UNCHECKED_HPP__
