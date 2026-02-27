//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
#if defined(RAJA_ENABLE_HIP)
  std::cout
      << "\n Running RAJA HIP launch-context IndicesAndDims caching example...\n";

  constexpr int N         = 1024;
  constexpr int BLOCK_DIM = 256;
  constexpr int GRID_DIM  = RAJA_DIVIDE_CEILING_INT(N, BLOCK_DIM);

  RAJA::resources::Hip device_res;
  RAJA::resources::Host host_res;

  int* d_array = device_res.allocate<int>(N);
  int* h_array = host_res.allocate<int>(N);

  for (int i = 0; i < N; ++i)
  {
    h_array[i] = -1;
  }
  device_res.memcpy(d_array, h_array, sizeof(int) * N);

  using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;

  using Ctx = RAJA::LaunchContextT<RAJA::hip::LaunchContextAllCachedIndicesAndDimsPolicy>;

  using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
  using threads_x = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;

  RAJA::launch<launch_policy>(
      device_res,
      RAJA::LaunchParams(RAJA::Teams(GRID_DIM), RAJA::Threads(BLOCK_DIM)),
      [=] RAJA_HOST_DEVICE(Ctx ctx) {
        auto const& idx = ctx.get_indices_and_dims();

        RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, GRID_DIM), [&](int bx) {
          RAJA_UNUSED_VAR(bx);

          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, BLOCK_DIM), [&](int tx) {
            RAJA_UNUSED_VAR(tx);

            int i = tx + BLOCK_DIM * bx;

            if (i < N)
            {
              d_array[i] = i;
            }
          });
        });
      });

  device_res.memcpy(h_array, d_array, sizeof(int) * N);

  int err_count = 0;
  for (int i = 0; i < N; ++i)
  {
    if (h_array[i] != i)
    {
      ++err_count;
    }
  }

  std::cout << "    Result -- " << (err_count ? "FAIL" : "PASS") << "\n";
  if (err_count)
  {
    std::cout << "      error count = " << err_count << "\n";
  }

  device_res.deallocate(d_array);
  host_res.deallocate(h_array);

  std::cout << "\n DONE!...\n";
  return (err_count ? 1 : 0);
#else
  std::cout << "Please build with HIP to run this example ...\n";
  return 0;
#endif
}
