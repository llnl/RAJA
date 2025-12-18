//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"


/*
 * RAJA Launch Example: Upper Triangular Pattern + Shared Memory
 *
 * Launch introduces hierarchical parallelism through the concept of
 * teams and threads.  Computation is executed in a pre-defined grid
 * composed of threads and grouped into teams. The teams model enables
 * developers to express parallelism through loops over teams, and inner loops
 * over threads. Team loops are executed in parallel and
 * threads within a team should be treated as sub-parallel regions.
 *
 * Team shared memory is allocated between team and thread loops.
 * Memory allocated within thread loops are thread private.
 * The example below demonstrates composing an upper triangular
 * loop pattern, and using shared memory.
 *
 */

/*
 * Define host/device launch policies
 */
using launch_policy = RAJA::LaunchPolicy<
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::omp_launch_t
#else
    RAJA::seq_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::hip_launch_t<false>
#endif
    >;

/*
 * Define team policies.
 * Up to 3 dimension are supported: x,y,z
 */
using teams_x = RAJA::LoopPolicy<
#if defined(RAJA_ENABLE_OPENMP)
                                       RAJA::omp_parallel_for_exec
#else
                                       RAJA::seq_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_x_direct
#endif
                                       >;
/*
 * Define thread policies.
 * Up to 3 dimension are supported: x,y,z
 */
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_x_loop
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_x_loop
#endif
                                         >;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  // Resource object for host
  camp::resources::Host host_res;

  // Resource objects for CUDA or HIP
#if defined(RAJA_ENABLE_CUDA)
  camp::resources::Cuda device_res;
#endif

#if defined(RAJA_ENABLE_HIP)
  camp::resources::Hip device_res;
#endif

  std::cout << "\n Running RAJA-Launch examples...\n";
  int num_of_backends = 1;
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  num_of_backends++;
#endif

  // RAJA teams may switch between host and device policies at run time.
  // The loop below will execute through the available backends.

  for (int exec_place = 0; exec_place < num_of_backends; ++exec_place) {

    auto select_cpu_or_gpu = (RAJA::ExecPlace)exec_place;

    // Allocate memory for either host or device
    int N_tri = 5;

    int* Ddat = nullptr;
    if (select_cpu_or_gpu == RAJA::ExecPlace::HOST) {
      Ddat = host_res.allocate<int>(N_tri * N_tri);
    }

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (select_cpu_or_gpu == RAJA::ExecPlace::DEVICE) {
      Ddat = device_res.allocate<int>(N_tri * N_tri);
    }
#endif

    /*
     * RAJA::launch just starts a "kernel" and doesn't provide any looping.
     *
     * The first argument determines which policy should be executed,
     *
     * The second argument is the number of teams+threads needed for each of the
     * policies.
     *
     * Third argument is the lambda.
     *
     * The lambda takes a "resource" object, which has the teams+threads
     * and is used to perform thread synchronizations within a team.
     */

    if (select_cpu_or_gpu == RAJA::ExecPlace::HOST){
      std::cout << "\n Running upper triangular pattern example on the host...\n";
    } else {
      std::cout << "\n Running upper triangular pattern example on the device...\n";
    }


    RAJA::View<int, RAJA::Layout<2>> D(Ddat, N_tri, N_tri);

    RAJA::launch<launch_policy>
      (select_cpu_or_gpu,
       RAJA::LaunchParams(RAJA::Teams(N_tri), RAJA::Threads(N_tri)),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

         RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_tri), [&](int r) {

           // Array shared within threads of the same team
           RAJA_TEAM_SHARED int s_A[1];

           RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, 1), [&](int c) {
              s_A[c] = r;
           });  // loop c

           ctx.teamSync();

           RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(r, N_tri), [&](int c) {
               D(r, c) = r * N_tri + c;
               printf("r=%d, c=%d : D=%d : s_A = %d \n", r, c, D(r, c), s_A[0]);
           });  // loop c

         });  // loop r

       });  // outer lambda

    if (select_cpu_or_gpu == RAJA::ExecPlace::HOST) {
      host_res.deallocate(Ddat);
    }

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (select_cpu_or_gpu == RAJA::ExecPlace::DEVICE) {
      device_res.deallocate(Ddat);
    }
#endif

  }  // Execution places loop


}  // Main
