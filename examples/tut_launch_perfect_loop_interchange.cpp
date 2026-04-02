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
using outer_k = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_y_loop>;
using outer_j = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_loop>;
using inner_i = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_thread_x_loop>;
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::hip_launch_t<false>;
using outer_k = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_y_loop>;
using outer_j = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_x_loop>;
using inner_i = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_thread_x_loop>;
#elif defined(RAJA_ENABLE_SYCL)
using device_launch = RAJA::sycl_launch_t<true>;
using outer_k = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_1_loop>;
using outer_j = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_2_loop>;
using inner_i = RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_local_2_loop>;
#else
using outer_k = RAJA::LoopPolicy<RAJA::seq_exec>;
using outer_j = RAJA::LoopPolicy<RAJA::seq_exec>;
using inner_i = RAJA::LoopPolicy<RAJA::seq_exec>;
#endif

using launch_policy = RAJA::LaunchPolicy<
    host_launch
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
    ,
    device_launch
#endif
    >;

using perfect_loop_policy = RAJA::PerfectLoopPolicy<outer_k, outer_j, inner_i>;
using interchange = RAJA::PerfectLoopInterchange<1, 2, 0>;

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
  camp::resources::Resource work_res {res};
  camp::resources::Host host_res;

  auto i_range = RAJA::TypedRangeSegment<int>(0, 4);
  auto j_range = RAJA::TypedRangeSegment<int>(0, 3);
  auto k_range = RAJA::TypedRangeSegment<int>(0, 2);

  int const ni = static_cast<int>(i_range.end() - i_range.begin());
  int const nj = static_cast<int>(j_range.end() - j_range.begin());
  int const nk = static_cast<int>(k_range.end() - k_range.begin());
  int const size = ni * nj * nk;

  int* data = work_res.allocate<int>(size);
  int* host_data = host_res.allocate<int>(size);
  work_res.memset(data, 0, sizeof(int) * size);

  // _launch_perfect_loop_interchange_start
  RAJA::launch<launch_policy>(
      place,
      RAJA::LaunchParams(RAJA::Teams(nj, nk), RAJA::Threads(ni)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::perfect_loop_icount<perfect_loop_policy, interchange>(
            ctx,
            i_range,
            j_range,
            k_range,
            [=](int i, int j, int k, int ii, int jj, int kk) {
              int idx = ii + ni * (jj + nj * kk);
              data[idx] = i + 10 * j + 100 * k;
            });
      });
  // _launch_perfect_loop_interchange_end

  work_res.memcpy(host_data, data, sizeof(int) * size);
  work_res.wait();

  bool pass = true;
  for (int k = 0; k < nk; ++k)
  {
    for (int j = 0; j < nj; ++j)
    {
      for (int i = 0; i < ni; ++i)
      {
        int idx = i + ni * (j + nj * k);
        if (host_data[idx] != i + 10 * j + 100 * k)
        {
          pass = false;
        }
      }
    }
  }

  std::cout << (place == RAJA::ExecPlace::HOST ? "host" : "device")
            << " perfect loop interchange check: " << (pass ? "PASS" : "FAIL")
            << '\n';

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
