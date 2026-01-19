//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_DYNAMIC_MEM_HPP__
#define __TEST_LAUNCH_DYNAMIC_MEM_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename LAUNCH_POLICY, typename TEAM_POLICY, typename THREAD_POLICY>
void LaunchDynamicMemTestImpl(INDEX_TYPE block_range, INDEX_TYPE thread_range)
{

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
      test_array[idx] = INDEX_TYPE(idx) + INDEX_TYPE(c);
    }
  }

  size_t shared_mem_size = RAJA::stripIndexType(thread_range)*sizeof(INDEX_TYPE);

  //Use an int type to test the bump style allocator.
  //Key idea is that we are requesting different amounts.
  shared_mem_size += RAJA::stripIndexType(thread_range)*sizeof(int);

  RAJA::launch<LAUNCH_POLICY>
    (RAJA::LaunchParams(RAJA::Teams(RAJA::stripIndexType(block_range)),
                        RAJA::Threads(RAJA::stripIndexType(thread_range)), shared_mem_size),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

      RAJA::loop<TEAM_POLICY>(ctx, outer_range, [&](INDEX_TYPE bid) {

          INDEX_TYPE * tile_ptr = ctx.getSharedMemory<INDEX_TYPE>(RAJA::stripIndexType(thread_range));
          RAJA::View<INDEX_TYPE, RAJA::Layout<1>> Tile(tile_ptr, RAJA::stripIndexType(thread_range));

          int * int_tile_ptr = ctx.getSharedMemory<int>(RAJA::stripIndexType(thread_range));
          RAJA::View<int, RAJA::Layout<1>> Int_Tile(int_tile_ptr, RAJA::stripIndexType(thread_range));

          RAJA::loop<THREAD_POLICY>(ctx, inner_range, [&](INDEX_TYPE tid) {
              Int_Tile(RAJA::stripIndexType(tid)) = RAJA::stripIndexType(tid);
              Tile(RAJA::stripIndexType(thread_range)-RAJA::stripIndexType(tid)-1) = thread_range-tid-1 + thread_range*bid;
            });

          ctx.teamSync();

          RAJA::loop<THREAD_POLICY>(ctx, inner_range, [&](INDEX_TYPE tid) {
              INDEX_TYPE idx = tid + thread_range * bid;
              working_array[RAJA::stripIndexType(idx)] = Tile(RAJA::stripIndexType(tid)) + Int_Tile(RAJA::stripIndexType(tid));
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


TYPED_TEST_SUITE_P(LaunchDynamicMemTest);
template <typename T>
class LaunchDynamicMemTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchDynamicMemTest, DynamicMemLaunch)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using LAUNCH_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<0>>::type;
  using TEAM_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<1>>::type;
  using THREAD_POLICY = typename camp::at<typename camp::at<TypeParam,camp::num<2>>::type, camp::num<2>>::type;


  LaunchDynamicMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>
    (INDEX_TYPE(4), INDEX_TYPE(2));

  LaunchDynamicMemTestImpl<INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>
    (INDEX_TYPE(5), INDEX_TYPE(32));

}

REGISTER_TYPED_TEST_SUITE_P(LaunchDynamicMemTest,
                            DynamicMemLaunch);

#endif  // __TEST_DYNAMIC_MEM_HPP__
