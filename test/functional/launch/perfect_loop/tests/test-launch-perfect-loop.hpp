//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_PERFECT_LOOP_HPP__
#define __TEST_LAUNCH_PERFECT_LOOP_HPP__

#include <numeric>

template<typename PERFECT_LOOP_POLICY>
struct Prefix3PerfectLoopPolicy
{
  using host_policies = typename PERFECT_LOOP_POLICY::host_policy_t;
  using host_policy_t = camp::list<typename camp::at<host_policies, camp::num<0>>::type,
                                   typename camp::at<host_policies, camp::num<1>>::type,
                                   typename camp::at<host_policies, camp::num<2>>::type>;
#if defined(RAJA_GPU_ACTIVE)
  using device_policies = typename PERFECT_LOOP_POLICY::device_policy_t;
  using device_policy_t =
      camp::list<typename camp::at<device_policies, camp::num<0>>::type,
                 typename camp::at<device_policies, camp::num<1>>::type,
                 typename camp::at<device_policies, camp::num<2>>::type>;
#endif
};

template<typename RESOURCE>
RAJA_INLINE bool launch_test_resource_available()
{
  return true;
}

#if defined(RAJA_ENABLE_HIP)
template<>
RAJA_INLINE bool launch_test_resource_available<camp::resources::Hip>()
{
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}
#endif

#if defined(RAJA_ENABLE_CUDA)
template<>
RAJA_INLINE bool launch_test_resource_available<camp::resources::Cuda>()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#endif

template<typename INDEX_TYPE,
         typename WORKING_RES,
         typename LAUNCH_POLICY,
         typename PERFECT_LOOP_POLICY>
void LaunchPerfectLoopTestImpl(INDEX_TYPE M)
{
  if (!launch_test_resource_available<WORKING_RES>())
  {
    GTEST_SKIP() << "No device available for this launch backend.";
  }

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(0, 2 * M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(0, 3 * M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r3(0, 4 * M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r4(0, 5 * M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r5(0, 2 * M);
  RAJA::TypedRangeSegment<INDEX_TYPE> r6(0, 3 * M);

  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N3 = static_cast<INDEX_TYPE>(r3.end() - r3.begin());
  INDEX_TYPE N4 = static_cast<INDEX_TYPE>(r4.end() - r4.begin());
  INDEX_TYPE N5 = static_cast<INDEX_TYPE>(r5.end() - r5.begin());
  INDEX_TYPE N6 = static_cast<INDEX_TYPE>(r6.end() - r6.begin());

  INDEX_TYPE N = static_cast<INDEX_TYPE>(N1 * N2 * N3 * N4 * N5 * N6);

  camp::resources::Resource working_res {WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  size_t data_len = RAJA::stripIndexType(N);
  if (data_len == 0)
  {
    data_len = 1;
  }

  allocateForallTestData<INDEX_TYPE>(
      data_len, working_res, &working_array, &check_array, &test_array);

  std::iota(test_array, test_array + data_len, INDEX_TYPE(0));
  working_res.memset(working_array, 0, sizeof(INDEX_TYPE) * data_len);

  constexpr int threads_x = 2;
  constexpr int threads_y = 2;
  constexpr int threads_z = 2;

  constexpr int blocks_x = 2;
  constexpr int blocks_y = 2;
  constexpr int blocks_z = 2;

  RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(blocks_x, blocks_y, blocks_z),
                         RAJA::Threads(threads_x, threads_y, threads_z)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::perfect_loop<PERFECT_LOOP_POLICY>(
            ctx,
            r6,
            r5,
            r4,
            r3,
            r2,
            r1,
            [=](INDEX_TYPE bz,
                INDEX_TYPE by,
                INDEX_TYPE bx,
                INDEX_TYPE tz,
                INDEX_TYPE ty,
                INDEX_TYPE tx) {
              auto idx =
                  tx + N1 * (ty + N2 * (tz + N3 * (bx + N4 * (by + N5 * bz))));
              working_array[RAJA::stripIndexType(idx)] =
                  static_cast<INDEX_TYPE>(idx);
            });
      });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * data_len);
  working_res.wait();

  if (RAJA::stripIndexType(N) > 0)
  {
    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i)
    {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)],
                check_array[RAJA::stripIndexType(i)]);
    }
  }
  else
  {
    ASSERT_EQ(INDEX_TYPE(0), check_array[0]);
  }

  deallocateForallTestData<INDEX_TYPE>(
      working_res, working_array, check_array, test_array);
}

template<typename INDEX_TYPE,
         typename WORKING_RES,
         typename LAUNCH_POLICY,
         typename PERFECT_LOOP_POLICY>
void LaunchPerfectLoopInterchangeTestImpl()
{
  if (!launch_test_resource_available<WORKING_RES>())
  {
    GTEST_SKIP() << "No device available for this launch backend.";
  }

  INDEX_TYPE const begin0 = INDEX_TYPE(1);
  INDEX_TYPE const begin1 = INDEX_TYPE(3);
  INDEX_TYPE const begin2 = INDEX_TYPE(7);
  RAJA::TypedRangeSegment<INDEX_TYPE> r0(begin0, INDEX_TYPE(4));
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(begin1, INDEX_TYPE(5));
  RAJA::TypedRangeSegment<INDEX_TYPE> r2(begin2, INDEX_TYPE(10));

  INDEX_TYPE N0 = static_cast<INDEX_TYPE>(r0.end() - r0.begin());
  INDEX_TYPE N1 = static_cast<INDEX_TYPE>(r1.end() - r1.begin());
  INDEX_TYPE N2 = static_cast<INDEX_TYPE>(r2.end() - r2.begin());
  INDEX_TYPE N = static_cast<INDEX_TYPE>(N0 * N1 * N2);

  using interchange_policy = Prefix3PerfectLoopPolicy<PERFECT_LOOP_POLICY>;

  camp::resources::Resource working_res {WORKING_RES::get_default()};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(
      RAJA::stripIndexType(N), working_res, &working_array, &check_array, &test_array);

  for (INDEX_TYPE k = INDEX_TYPE(0); k < N2; ++k)
  {
    for (INDEX_TYPE j = INDEX_TYPE(0); j < N1; ++j)
    {
      for (INDEX_TYPE i = INDEX_TYPE(0); i < N0; ++i)
      {
        auto idx = i + N0 * (j + N1 * k);
        test_array[RAJA::stripIndexType(idx)] =
            (begin0 + i) + 10 * (begin1 + j) + 100 * (begin2 + k) +
            1000 * i + 10000 * j + 100000 * k;
      }
    }
  }
  working_res.memset(working_array, 0, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  RAJA::launch<LAUNCH_POLICY>(
      RAJA::LaunchParams(RAJA::Teams(3, 1, 1), RAJA::Threads(4, 1, 1)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::perfect_loop_icount<interchange_policy,
                                  RAJA::PerfectLoopInterchange<2, 0, 1>>(
            ctx,
            r0,
            r1,
            r2,
            [=](INDEX_TYPE i,
                INDEX_TYPE j,
                INDEX_TYPE k,
                INDEX_TYPE ci,
                INDEX_TYPE cj,
                INDEX_TYPE ck) {
              auto idx = ci + N0 * (cj + N1 * ck);
              working_array[RAJA::stripIndexType(idx)] =
                  i + 10 * j + 100 * k + 1000 * ci + 10000 * cj + 100000 * ck;
            });
      });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));
  working_res.wait();

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i)
  {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)],
              check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(
      working_res, working_array, check_array, test_array);
}

TYPED_TEST_SUITE_P(LaunchPerfectLoopTest);

template<typename T>
class LaunchPerfectLoopTest : public ::testing::Test
{};

TYPED_TEST_P(LaunchPerfectLoopTest, PerfectLoop)
{
  using INDEX_TYPE = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using POLICY_PAIR = typename camp::at<TypeParam, camp::num<2>>::type;
  using LAUNCH_POLICY = typename camp::at<POLICY_PAIR, camp::num<0>>::type;
  using PERFECT_LOOP_POLICY = typename camp::at<POLICY_PAIR, camp::num<1>>::type;

  LaunchPerfectLoopTestImpl<
      INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, PERFECT_LOOP_POLICY>(INDEX_TYPE(0));
  LaunchPerfectLoopTestImpl<
      INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, PERFECT_LOOP_POLICY>(INDEX_TYPE(1));
}

TYPED_TEST_P(LaunchPerfectLoopTest, PerfectLoopInterchange)
{
  using INDEX_TYPE = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using POLICY_PAIR = typename camp::at<TypeParam, camp::num<2>>::type;
  using LAUNCH_POLICY = typename camp::at<POLICY_PAIR, camp::num<0>>::type;
  using PERFECT_LOOP_POLICY = typename camp::at<POLICY_PAIR, camp::num<1>>::type;

  LaunchPerfectLoopInterchangeTestImpl<
      INDEX_TYPE, WORKING_RES, LAUNCH_POLICY, PERFECT_LOOP_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(LaunchPerfectLoopTest,
                            PerfectLoop,
                            PerfectLoopInterchange);

#endif  // __TEST_LAUNCH_PERFECT_LOOP_HPP__
