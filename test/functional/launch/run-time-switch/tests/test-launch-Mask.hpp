//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_LAUNCH_MASK_HPP__
#define __TEST_LAUNCH_MASK_HPP__

#include <cstring>

template <typename WORKING_RES, typename LAUNCH_POLICY, typename TEAM_POLICY, typename THREAD_POLICY>
void LaunchMaskTestImpl()
{
  constexpr int num_teams = 37;
  constexpr int threads_per_team = 32;

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  int* working_array;
  int* check_array;
  int* test_array;

  allocateForallTestData<int>(num_teams,
                              working_res,
                              &working_array,
                              &check_array,
                              &test_array);

  std::memset(test_array, 0, sizeof(int) * num_teams);
  working_res.memcpy(working_array, test_array, sizeof(int) * num_teams);

  RAJA::ExecPlace select_cpu_or_gpu =
      working_res.get_platform() == camp::resources::Platform::host
          ? RAJA::ExecPlace::HOST
          : RAJA::ExecPlace::DEVICE;

  RAJA::launch<LAUNCH_POLICY>(
      select_cpu_or_gpu,
      RAJA::LaunchParams(RAJA::Teams(num_teams), RAJA::Threads(threads_per_team)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<TEAM_POLICY>(ctx, RAJA::RangeSegment(0, num_teams), [&](int team) {
          RAJA::mask<THREAD_POLICY>(ctx, [&] {
            working_array[team] = team + 1;
          });
        });
      });

  working_res.memcpy(check_array, working_array, sizeof(int) * num_teams);
  working_res.wait();

  for (int team = 0; team < num_teams; ++team) {
    ASSERT_EQ(team + 1, check_array[team]);
  }

  deallocateForallTestData<int>(working_res,
                                working_array,
                                check_array,
                                test_array);
}

TYPED_TEST_SUITE_P(LaunchMaskTest);
template <typename T>
class LaunchMaskTest : public ::testing::Test
{
};

TYPED_TEST_P(LaunchMaskTest, MaskTeams)
{
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using LAUNCH_POLICY =
      typename camp::at<typename camp::at<TypeParam, camp::num<1>>::type,
                        camp::num<0>>::type;
  using TEAM_POLICY =
      typename camp::at<typename camp::at<TypeParam, camp::num<1>>::type,
                        camp::num<1>>::type;
  using THREAD_POLICY =
      typename camp::at<typename camp::at<TypeParam, camp::num<1>>::type,
                        camp::num<2>>::type;

  LaunchMaskTestImpl<WORKING_RES, LAUNCH_POLICY, TEAM_POLICY, THREAD_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(LaunchMaskTest, MaskTeams);

#endif  // __TEST_LAUNCH_MASK_HPP__
