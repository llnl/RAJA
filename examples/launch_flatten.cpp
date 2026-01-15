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
#include <limits>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Thread Flatten Example using RAJA Launch
 *
 *  This example illustrates use of the "flatten"
 *  policy inside RAJA Launch
 *
 *  The flatten policy enables reshaping
 *  multi-dimensional thread teams to 1D
 *  simplifying indexing patterns
 *
 *  RAJA features shown:
 *    - `launch' method to create kernel
 *    -  flatten_block_threads
 *
 */

/*
 * Define device launch policies
 */

#if defined(RAJA_ENABLE_CUDA)
using device_launch = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
using device_inner_pol0 =  RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
using device_inner_pol1 =  RAJA::LoopPolicy<RAJA::cuda_thread_y_direct>;
using device_flatten_pol =  RAJA::LoopPolicy<RAJA::cuda_flatten_block_threads_xy_direct>;
using reduce_policy = RAJA::cuda_reduce;
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;
using device_inner_pol0 =  RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;
using device_inner_pol1 =  RAJA::LoopPolicy<RAJA::hip_thread_y_direct>;
using device_flatten_pol =  RAJA::LoopPolicy<RAJA::hip_flatten_block_threads_xy_direct>;
using reduce_policy = RAJA::hip_reduce;
#endif

/*
 * Define device launch policies
 */

using host_launch = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
using host_loop = RAJA::LoopPolicy<RAJA::seq_exec>;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

  //
  // Problem size dimensions
  //
  constexpr int N = 4;
  constexpr int NN = N*N;

  //
  // Configure grid size
  //
  RAJA::LaunchParams launch_params(RAJA::Teams(1),
                                   RAJA::Threads(N, N));


  //
  // Resource object for host, used to allocate memory
  //
  camp::resources::Host host_res;
  int *h_A_ptr = host_res.allocate<int>(NN);

  //
  // Resource object for device, used to allocate memory
  //
#if defined(RAJA_ENABLE_CUDA)
  camp::resources::Cuda device_res;
#elif defined(RAJA_ENABLE_HIP)
  camp::resources::Hip device_res;
#endif

  int *d_A_ptr = device_res.allocate<int>(NN);

//----------------------------------------------------------------------------//

  std::cout << "\n Running device version of teams_flatten example ...\n";

  RAJA::ReduceSum<reduce_policy, int> device_kernel_sum(0);
  RAJA::View<int, RAJA::Layout<2>> d_A_2DView(d_A_ptr, N, N);
  RAJA::View<int, RAJA::Layout<1>> d_A_1DView(d_A_ptr, NN);

  RAJA::launch<device_launch>
    (launch_params, [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
     {

       RAJA::loop<device_inner_pol1>(ctx, RAJA::RangeSegment(0, N), [&] (int j) {
         RAJA::loop<device_inner_pol0>(ctx, RAJA::RangeSegment(0, N), [&] (int i) {
             d_A_2DView(j, i) = i + j;
           });
         });

       ctx.teamSync();

       // RAJA flatten policy will reshape a 2/3D thread team to 1D simplifying
       // accumulating memory contents
       RAJA::loop<device_flatten_pol>(ctx, RAJA::RangeSegment(0, NN), [&] (int i) {
           device_kernel_sum += d_A_1DView(i);
       });

     });

//----------------------------------------------------------------------------//

  std::cout << "\n Running host version of teams_flatten example ...\n";

  RAJA::ReduceSum<reduce_policy, int> host_kernel_sum(0);
  RAJA::View<int, RAJA::Layout<2>> h_A_2DView(h_A_ptr, N, N);
  RAJA::View<int, RAJA::Layout<1>> h_A_1DView(h_A_ptr, NN);

  RAJA::launch<host_launch>
    (launch_params, [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
    {

       RAJA::loop<host_loop>(ctx, RAJA::RangeSegment(0, N), [&] (int j) {
         RAJA::loop<host_loop>(ctx, RAJA::RangeSegment(0, N), [&] (int i) {
             h_A_2DView(j, i) = i + j;
           });
         });

       ctx.teamSync();

       //As loops are dispatched as standard C loops we can revert to using
       //a regular seq_exec policy
       RAJA::loop<host_loop>(ctx, RAJA::RangeSegment(0, NN), [&] (int i) {
           host_kernel_sum += h_A_1DView(i);
       });

     });

  if ( device_kernel_sum.get() == host_kernel_sum.get() ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }

#else
  std::cout << "Please build with CUDA or Hip to run this example ...\n";
#endif

  std::cout << "\n DONE!...\n";

  return 0;
}
