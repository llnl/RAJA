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

template<typename Backend>
struct BackendTraits;

#if defined(RAJA_ENABLE_HIP)
struct HipBackend;

template<>
struct BackendTraits<HipBackend>
{
  static constexpr const char* name = "HIP";
  using device_res_t                = RAJA::resources::Hip;
  using launch_t                    = RAJA::hip_launch_t<false>;
  using ctx_policy_t = RAJA::hip::LaunchContextAllCachedIndicesAndDimsPolicy;
  using block_x_direct_t            = RAJA::hip_block_x_direct;
  using thread_x_direct_t           = RAJA::hip_thread_x_direct;
};
#endif

#if defined(RAJA_ENABLE_CUDA)
struct CudaBackend;

template<>
struct BackendTraits<CudaBackend>
{
  static constexpr const char* name = "CUDA";
  using device_res_t                = RAJA::resources::Cuda;
  using launch_t                    = RAJA::cuda_launch_t<false>;
  using ctx_policy_t = RAJA::cuda::LaunchContextAllCachedIndicesAndDimsPolicy;
  using block_x_direct_t            = RAJA::cuda_block_x_direct;
  using thread_x_direct_t           = RAJA::cuda_thread_x_direct;
};
#endif

template<typename Backend>
int run_example()
{
  using T = BackendTraits<Backend>;

  std::cout << "\n Running RAJA " << T::name
            << " launch-context indices/dims caching example...\n";

  constexpr int N         = 1024;
  constexpr int BLOCK_DIM = 256;
  constexpr int GRID_DIM  = RAJA_DIVIDE_CEILING_INT(N, BLOCK_DIM);

  typename T::device_res_t device_res;
  RAJA::resources::Host host_res;

  int* d_array = device_res.template allocate<int>(N);
  int* h_array = host_res.allocate<int>(N);

  for (int i = 0; i < N; ++i)
  {
    h_array[i] = -1;
  }
  device_res.memcpy(d_array, h_array, sizeof(int) * N);

  using launch_policy = RAJA::LaunchPolicy<typename T::launch_t>;
  using Ctx           = RAJA::LaunchContextT<typename T::ctx_policy_t>;
  using teams_x       = RAJA::LoopPolicy<typename T::block_x_direct_t>;
  using threads_x     = RAJA::LoopPolicy<typename T::thread_x_direct_t>;

  RAJA::launch<launch_policy>(
      device_res,
      RAJA::LaunchParams(RAJA::Teams(GRID_DIM), RAJA::Threads(BLOCK_DIM)),
      [=] RAJA_HOST_DEVICE(Ctx ctx) {     

        RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, GRID_DIM), [&](int bx) {

          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, BLOCK_DIM), [&](int tx) {                  

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

  return (err_count ? 1 : 0);
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
#if defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_CUDA)
  int err_count = 0;

  #if defined(RAJA_ENABLE_HIP)
  err_count += run_example<HipBackend>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  err_count += run_example<CudaBackend>();
  #endif

  std::cout << "\n DONE!...\n";
  return (err_count ? 1 : 0);
#else
  std::cout << "Please build with HIP or CUDA to run this example ...\n";
  return 0;
#endif
}
