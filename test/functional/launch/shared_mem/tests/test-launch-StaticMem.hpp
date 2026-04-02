//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_STATIC_MEM_HPP__
#define __TEST_LAUNCH_STATIC_MEM_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename TEAM_POLICY, typename THREAD_POLICY,
int THREAD_RANGE>
void LaunchStaticMemTestImpl(INDEX_TYPE block_range)
{

  INDEX_TYPE thread_range(THREAD_RANGE);

  RAJA::TypedRangeSegment<INDEX_TYPE> outer_range(RAJA::stripIndexType(INDEX_TYPE(0)), RAJA::stripIndexType(block_range));
  RAJA::TypedRangeSegment<INDEX_TYPE> inner_range(RAJA::stripIndexType(INDEX_TYPE(0)), RAJA::stripIndexType(thread_range));

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(block_range)*RAJA::stripIndexType(thread_range);

  allocateForallTestData<INDEX_TYPE>(data_len,
                                     working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  //determine the underlying type of block_range
  using s_type = decltype(RAJA::stripIndexType(block_range));
  
  for(s_type b=0; b<RAJA::stripIndexType(block_range); ++b) {
    for(s_type c=0; c<RAJA::stripIndexType(thread_range); ++c) {
      s_type idx = c + RAJA::stripIndexType(thread_range)*b;
      test_array[idx] = INDEX_TYPE(idx);
    }
  }

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(RAJA::stripIndexType(block_range)),
                        RAJA::Threads(RAJA::stripIndexType(thread_range))),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::loop<TEAM_POLICY>(ctx, outer_range, [&](INDEX_TYPE bid) {

          //Since we are using custom index type we have to first use a
          //type that the device compiler can intialize, we can then use a
          //pointer to recast the shared memory to our desired type.
          //This enables us to work around the following warning:
          // warning #3019-D: dynamic initialization is not supported for
          //a function-scope static __shared__ variable within a __device__/__global__ function
          RAJA_TEAM_SHARED char char_Tile[THREAD_RANGE*sizeof(INDEX_TYPE)];
          INDEX_TYPE *Tile = (INDEX_TYPE *)char_Tile;

          RAJA::loop<THREAD_POLICY>(ctx, inner_range, [&](INDEX_TYPE tid) {
              Tile[RAJA::stripIndexType(thread_range)-RAJA::stripIndexType(tid)-1] = thread_range-tid-1 + thread_range*bid;
            });

          ctx.teamSync();

          RAJA::loop<THREAD_POLICY>(ctx, inner_range, [&](INDEX_TYPE tid) {
              INDEX_TYPE idx = tid + thread_range * bid;
              working_array[RAJA::stripIndexType(idx)] = Tile[RAJA::stripIndexType(tid)];
          });

          ctx.releaseSharedMemory();
        });

    });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);

  for (size_t i = 0; i < data_len; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(working_res,
                                       working_array,
                                       check_array,
                                       test_array);
}


TYPED_TEST_SUITE_P(LaunchStaticMemTest);
template <typename T>
class LaunchStaticMemTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchStaticMemTest, StaticMemLaunch)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using TEAM_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;


  LaunchStaticMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY, 2>
    (INDEX_TYPE(4));

  LaunchStaticMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY, 32>
    (INDEX_TYPE(5));

}

REGISTER_TYPED_TEST_SUITE_P(LaunchStaticMemTest,
                            StaticMemLaunch);

#endif  // __TEST_DYNAMIC_MEM_HPP__
