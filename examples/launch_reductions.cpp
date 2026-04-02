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
 *  Reduction Example using RAJA Launch
 *
 *  This example illustrates use of the RAJA reduction types: min, max,
 *  sum, min-loc, and max-loc.
 *
 *  RAJA features shown:
 *    - `launch' loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
 *
 */

#if defined(RAJA_ENABLE_OPENMP)
using host_launch = RAJA::omp_launch_t;
using host_loop = RAJA::omp_for_exec;
#else
using host_launch = RAJA::seq_launch_t;
using host_loop = RAJA::seq_exec;
#endif

#if defined(RAJA_ENABLE_CUDA)
using device_launch = RAJA::cuda_launch_t<false>;
using device_loop = RAJA::cuda_global_thread_x;
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::hip_launch_t<false>;
using device_loop = RAJA::hip_global_thread_x;
#endif

using launch_policy = RAJA::LaunchPolicy<host_launch
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                         , device_launch
#endif
                                        >;

using loop_pol = RAJA::LoopPolicy<host_loop
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                  , device_loop
#endif
                                 >;

#if defined(RAJA_ENABLE_CUDA)
using reduce_policy = RAJA::cuda_reduce;
#elif defined(RAJA_ENABLE_HIP)
using reduce_policy = RAJA::hip_reduce;
#elif defined(RAJA_ENABLE_OPENMP)
using reduce_policy = RAJA::omp_reduce;
#else
using reduce_policy = RAJA::seq_reduce;
#endif


int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./launch_reductions host or ./launch_reductions device");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument (host or device).
  // Example usage ./launch_reductions host or ./launch_reductions device
  //
  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./launch_reductions host or ./launch_reductions device");
    return 0;
  }

  RAJA::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::HOST; printf("Running RAJA-Launch reductions example on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::DEVICE; printf("Running RAJA-Launch reductions example on the device \n"); }

  // _reductions_array_init_start
//
// Define array length
//
  const int N = 1000000;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  int* a = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    if ( i % 2 == 0 ) {
      a[i] = 1;
    } else {
      a[i] = -1;
    }
  }

//
// Set min and max loc values
//
  const int minloc_ref = N / 2;
  a[minloc_ref] = -100;

  const int maxloc_ref = N / 2 + 1;
  a[maxloc_ref] = 100;
  // _reductions_array_init_end

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
//  - the sum will be zero
//  - the min will be -100
//  - the max will be 100
//  - the min loc will be N/2
//  - the max loc will be N/2 + 1
//
//

//
// Define index range for iterating over a elements in all examples
//
  // _reductions_range_start
  RAJA::RangeSegment arange(0, N);
  // _reductions_range_end

//----------------------------------------------------------------------------//

  RAJA::ReduceSum<reduce_policy, int> kernel_sum(0);
  RAJA::ReduceMin<reduce_policy, int> kernel_min(std::numeric_limits<int>::max());
  RAJA::ReduceMax<reduce_policy, int> kernel_max(std::numeric_limits<int>::min());
  RAJA::ReduceMinLoc<reduce_policy, int> kernel_minloc(std::numeric_limits<int>::max(), -1);
  RAJA::ReduceMaxLoc<reduce_policy, int> kernel_maxloc(std::numeric_limits<int>::min(), -1);

  const int TEAM_SZ = 256;
  const int GRID_SZ = RAJA_DIVIDE_CEILING_INT(N,TEAM_SZ);

  RAJA::launch<launch_policy>
    (select_cpu_or_gpu,
     RAJA::LaunchParams(RAJA::Teams(GRID_SZ),
                        RAJA::Threads(TEAM_SZ)),
     RAJA::Name("Launch Reductions"),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) 
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           
           kernel_sum += a[i];
           
           kernel_min.min(a[i]);
           kernel_max.max(a[i]);
           
           kernel_minloc.minloc(a[i], i);
           kernel_maxloc.maxloc(a[i], i);
         });
       
    });

  std::cout << "\tsum = " << kernel_sum.get() << std::endl;
  std::cout << "\tmin = " << kernel_min.get() << std::endl;
  std::cout << "\tmax = " << kernel_max.get() << std::endl;
  std::cout << "\tmin, loc = " << kernel_minloc.get() << " , "
                               << kernel_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << kernel_maxloc.get() << " , "
                               << kernel_maxloc.getLoc() << std::endl;

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);

  std::cout << "\n DONE!...\n";

  return 0;
}
