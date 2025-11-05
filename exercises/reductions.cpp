//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <limits>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Reduction Example
 *
 *  This example illustrates use of the RAJA reduction types: min, max,
 *  sum, min-loc, and max-loc.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA)
//constexpr int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
constexpr int HIP_BLOCK_SIZE = 256;
#endif

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA reductions example...\n";

  // _reductions_array_init_start
//
// Define array length
//
  constexpr int N = 1000000;

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
  constexpr int minloc_ref = N / 2;
  a[minloc_ref] = -100;

  constexpr int maxloc_ref = N / 2 + 1;
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
//RAJA::TypedRangeSegment<int> arange(0, N);
  // _reductions_range_end

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential reductions...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Define EXEC_POL1 and REDCUE_POL1 for executing sequentially.
  ///
 
  /// TODO...
  ///
  /// EXERCISE: Remove comments for remainder of sequential section.
  ///
  ///           Uncomment 'arange' variable above so it can be used in kernel.
  ///
  /*
  RAJA::ReduceSum<REDUCE_POL1, int> seq_sum(0);
  RAJA::ReduceMin<REDUCE_POL1, int> seq_min(std::numeric_limits<int>::max());
  RAJA::ReduceMax<REDUCE_POL1, int> seq_max(std::numeric_limits<int>::min());
  RAJA::ReduceMinLoc<REDUCE_POL1, int> seq_minloc(std::numeric_limits<int>::max(), -1);
  RAJA::ReduceMaxLoc<REDUCE_POL1, int> seq_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL1>(arange, [=](int i) {
    
    seq_sum += a[i];

    seq_min.min(a[i]);
    seq_max.max(a[i]);

    seq_minloc.minloc(a[i], i);
    seq_maxloc.maxloc(a[i], i);

  });

  std::cout << "\tsum = " << seq_sum.get() << std::endl;
  std::cout << "\tmin = " << seq_min.get() << std::endl;
  std::cout << "\tmax = " << seq_max.get() << std::endl;
  std::cout << "\tmin, loc = " << seq_minloc.get() << " , " 
                               << seq_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << seq_maxloc.get() << " , " 
                               << seq_maxloc.getLoc() << std::endl;
  */
  

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP reductions...\n";

  // _reductions_raja_omppolicy_start
  /*
  using EXEC_POL2   = RAJA::omp_parallel_for_exec;
  using REDUCE_POL2 = RAJA::omp_reduce;
  */
  // _reductions_raja_omppolicy_end

  ///
  /// TODO...
  ///
  /// EXERCISE: Define Reduce(Sum, Min, Max, MinLoc, MaxLoc) to complete this exercise.
  ///
  ///           Uncomment 'arange' variable above so it can be used in kernel.
  ///

  /// TODO...
  ///
  /// EXERCISE: Remove comments for remainder of openmp section and uncomment
  ///           policy types above to use in kernel.
  ///
  /*
  RAJA::forall<EXEC_POL2>(arange, [=](int i) {

    omp_sum += a[i];

    omp_min.min(a[i]);
    omp_max.max(a[i]);

    omp_minloc.minloc(a[i], i);
    omp_maxloc.maxloc(a[i], i);

  });

  std::cout << "\tsum = " << omp_sum.get() << std::endl;
  std::cout << "\tmin = " << omp_min.get() << std::endl;
  std::cout << "\tmax = " << omp_max.get() << std::endl;
  std::cout << "\tmin, loc = " << omp_minloc.get() << " , "
                               << omp_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << omp_maxloc.get() << " , "
                               << omp_maxloc.getLoc() << std::endl; 
  */
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA reductions...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Define EXEC_POL3 and REDCUE_POL3 for executing on a CUDA device.
  ///

  ///
  /// TODO...
  ///
  /// EXERCISE: Define Reduce(Sum, Min, Max, MinLoc, MaxLoc) to complete this exercise.
  ///
  ///           Uncomment 'arange' variable above so it can be used in kernel.
  ///

  /// TODO...
  ///
  /// EXERCISE: Remove comments for remainder of cuda section.
  ///
  /*
  RAJA::forall<EXEC_POL3>(arange, [=] RAJA_DEVICE (int i) {

    cuda_sum += a[i];

    cuda_min.min(a[i]);
    cuda_max.max(a[i]);

    cuda_minloc.minloc(a[i], i);
    cuda_maxloc.maxloc(a[i], i);

  });

  std::cout << "\tsum = " << cuda_sum.get() << std::endl;
  std::cout << "\tmin = " << cuda_min.get() << std::endl;
  std::cout << "\tmax = " << cuda_max.get() << std::endl;
  std::cout << "\tmin, loc = " << cuda_minloc.get() << " , "
                               << cuda_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << cuda_maxloc.get() << " , "
                               << cuda_maxloc.getLoc() << std::endl;
  */
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP reductions...\n";

  RAJA::TypedRangeSegment<int> arange1(0, N);

  int* d_a = memoryManager::allocate_gpu<int>(N);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N * sizeof(int), hipMemcpyHostToDevice);

  // _reductions_raja_hippolicy_start
  using EXEC_POL3   = RAJA::hip_exec<HIP_BLOCK_SIZE>;
  using REDUCE_POL3 = RAJA::hip_reduce;
  // _reductions_raja_hippolicy_end

  RAJA::ReduceSum<REDUCE_POL3, int> hip_sum(0);
  RAJA::ReduceMin<REDUCE_POL3, int> hip_min(std::numeric_limits<int>::max());
  RAJA::ReduceMax<REDUCE_POL3, int> hip_max(std::numeric_limits<int>::min());
  RAJA::ReduceMinLoc<REDUCE_POL3, int> hip_minloc(std::numeric_limits<int>::max(), -1);
  RAJA::ReduceMaxLoc<REDUCE_POL3, int> hip_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL3>(arange1, [=] RAJA_DEVICE (int i) {

    hip_sum += d_a[i];

    hip_min.min(d_a[i]);
    hip_max.max(d_a[i]);

    hip_minloc.minloc(d_a[i], i);
    hip_maxloc.maxloc(d_a[i], i);

  });

  std::cout << "\tsum = " << hip_sum.get() << std::endl;
  std::cout << "\tmin = " << hip_min.get() << std::endl;
  std::cout << "\tmax = " << hip_max.get() << std::endl;
  std::cout << "\tmin, loc = " << hip_minloc.get() << " , "
                               << hip_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << hip_maxloc.get() << " , "
                               << hip_maxloc.getLoc() << std::endl;

  memoryManager::deallocate_gpu(d_a);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);

  std::cout << "\n DONE!...\n";
 
  return 0;
}
