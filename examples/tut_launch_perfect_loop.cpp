//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"

using host_launch = RAJA::seq_launch_t;

#if defined(RAJA_ENABLE_CUDA)
using device_launch = RAJA::cuda_launch_t<false>;
using teams_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>;
using teams_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>;
using threads_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_y_loop>;
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>;
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::hip_launch_t<false>;
using teams_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_y_loop>;
using teams_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_x_loop>;
using threads_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_y_loop>;
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_x_loop>;
#elif defined(RAJA_ENABLE_SYCL)
using device_launch = RAJA::sycl_launch_t<true>;
using teams_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_1_loop>;
using teams_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_2_loop>;
using threads_y = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_1_loop>;
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_2_loop>;
#else
using teams_y = RAJA::LoopPolicy<RAJA::seq_exec>;
using teams_x = RAJA::LoopPolicy<RAJA::seq_exec>;
using threads_y = RAJA::LoopPolicy<RAJA::seq_exec>;
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec>;
#endif

using launch_policy = RAJA::LaunchPolicy<
    host_launch
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
    ,
    device_launch
#endif
    >;

using perfect_loop_policy =
    RAJA::PerfectLoopPolicy<teams_y, teams_x, threads_y, threads_x>;

RAJA_INLINE bool device_available()
{
#if defined(RAJA_ENABLE_HIP)
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#elif defined(RAJA_ENABLE_CUDA)
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
  return false;
#endif
}

template<typename RESOURCE>
void run_example(RAJA::ExecPlace place, RESOURCE res)
{
  constexpr int num_team_y = 2;
  constexpr int num_team_x = 3;
  constexpr int num_thread_y = 4;
  constexpr int num_thread_x = 5;

  int const size = num_team_y * num_team_x * num_thread_y * num_thread_x;

  camp::resources::Resource work_res {res};
  camp::resources::Host host_res;

  int* data = work_res.allocate<int>(size);
  int* host_data = host_res.allocate<int>(size);

  work_res.memset(data, 0, sizeof(int) * size);

  auto team_y_range = RAJA::TypedRangeSegment<int>(0, num_team_y);
  auto team_x_range = RAJA::TypedRangeSegment<int>(0, num_team_x);
  auto thread_y_range = RAJA::TypedRangeSegment<int>(0, num_thread_y);
  auto thread_x_range = RAJA::TypedRangeSegment<int>(0, num_thread_x);

  // _launch_perfect_loop_start
  RAJA::launch<launch_policy>(
      place,
      RAJA::LaunchParams(RAJA::Teams(num_team_x, num_team_y),
                         RAJA::Threads(num_thread_x, num_thread_y)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::perfect_loop<perfect_loop_policy>(
            ctx,
            team_y_range,
            team_x_range,
            thread_y_range,
            thread_x_range,
            [=](int team_y, int team_x, int thread_y, int thread_x) {
              int idx = thread_x +
                        num_thread_x *
                            (thread_y + num_thread_y * (team_x + num_team_x * team_y));
              data[idx] = idx;
            });
      });
  // _launch_perfect_loop_end

  work_res.memcpy(host_data, data, sizeof(int) * size);
  work_res.wait();

  bool pass = true;
  for (int i = 0; i < size; ++i)
  {
    if (host_data[i] != i)
    {
      pass = false;
      break;
    }
  }

  std::cout << (place == RAJA::ExecPlace::HOST ? "host" : "device")
            << " perfect loop check: " << (pass ? "PASS" : "FAIL") << '\n';

  work_res.deallocate(data);
  host_res.deallocate(host_data);
}

int main()
{
  run_example(RAJA::ExecPlace::HOST, camp::resources::Host::get_default());

#if defined(RAJA_ENABLE_CUDA)
  if (device_available())
  {
    run_example(RAJA::ExecPlace::DEVICE, camp::resources::Cuda::get_default());
  }
#elif defined(RAJA_ENABLE_HIP)
  if (device_available())
  {
    run_example(RAJA::ExecPlace::DEVICE, camp::resources::Hip::get_default());
  }
#elif defined(RAJA_ENABLE_SYCL)
  run_example(RAJA::ExecPlace::DEVICE, camp::resources::Sycl::get_default());
#endif

  return EXIT_SUCCESS;
}
