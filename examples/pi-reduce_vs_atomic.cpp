//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Pi Example
 *
 *  Computes an approximation to pi. It illustrates that RAJA reduction
 *  and atomic operations are used similarly for different progamming
 *  model backends.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    - Index range segment
 *    - Sum reduction
 *    - Atomic add
 *
 *  If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA pi example...\n";

//
// Define RangeSegment to enumerate "bins" and "bin step" size used in
// Riemann integral sum to approximate pi,
// and memory location for atomic add operation.
//
  const int num_bins = 512 * 512;
  const double dx = 1.0 / double(num_bins); 

  RAJA::RangeSegment bins(0, num_bins); 

  double* atomic_pi = memoryManager::allocate<double>(1);

// Set precision for printing pi
  int prec = 16;


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style sequential pi approximation...\n";
 
  double c_pi = 0.0;

  for (int i = 0; i < num_bins; ++i) {
      double x = (double(i) + 0.5) * dx;
      c_pi += dx / (1.0 + x * x);
  }
  c_pi *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec) 
            << c_pi << std::endl;


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential pi approximation (reduction)...\n";

  using EXEC_POL1   = RAJA::seq_exec;
  using REDUCE_POL1 = RAJA::seq_reduce; 

  RAJA::ReduceSum<REDUCE_POL1, double> seq_pi(0.0);

  RAJA::forall<EXEC_POL1>(bins, [=](int i) {
      double x = (double(i) + 0.5) * dx;
      seq_pi += dx / (1.0 + x * x);
  });
  double seq_pi_val = seq_pi.get() * 4.0;

  std::cout << "\tpi = " << std::setprecision(prec) 
            << seq_pi_val << std::endl;


  std::cout << "\n Running RAJA sequential pi approximation (atomic)...\n";

  using ATOMIC_POL1 = RAJA::seq_atomic;

  *atomic_pi = 0.0;

  RAJA::forall<EXEC_POL1>(bins, [=](int i) {
      double x = (double(i) + 0.5) * dx;
      RAJA::atomicAdd<ATOMIC_POL1>(atomic_pi, 
                                   dx / (1.0 + x * x));
  });
  *atomic_pi *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec) 
            << *atomic_pi << std::endl;


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running RAJA OpenMP pi approximation (reduction)...\n";

  using EXEC_POL2   = RAJA::omp_parallel_for_exec;
  using REDUCE_POL2 = RAJA::omp_reduce;

  RAJA::ReduceSum<REDUCE_POL2, double> omp_pi(0.0);

  RAJA::forall<EXEC_POL2>(bins, [=](int i) {
      double x = (double(i) + 0.5) * dx;
      omp_pi += dx / (1.0 + x * x);
  });
  double omp_pi_val = omp_pi.get() * 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << omp_pi_val << std::endl;


  std::cout << "\n Running RAJA OpenMP pi approximation (atomic)...\n";

  using ATOMIC_POL2 = RAJA::omp_atomic;

  *atomic_pi = 0.0;

  RAJA::forall<EXEC_POL2>(bins, [=](int i) {
      double x = (double(i) + 0.5) * dx;
      RAJA::atomicAdd<ATOMIC_POL2>(atomic_pi, 
                                   dx / (1.0 + x * x));
  });
  *atomic_pi *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << *atomic_pi << std::endl;

#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running RAJA CUDA pi approximation (reduction)...\n";

  using EXEC_POL3   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using REDUCE_POL3 = RAJA::cuda_reduce;

  RAJA::ReduceSum<REDUCE_POL3, double> cuda_pi(0.0);

  RAJA::forall<EXEC_POL3>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) * dx;
      cuda_pi += dx / (1.0 + x * x);
  });
  double cuda_pi_val = cuda_pi.get() * 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << cuda_pi_val << std::endl;


  std::cout << "\n Running RAJA CUDA pi approximation (atomic)...\n";

  using ATOMIC_POL3 = RAJA::cuda_atomic;

  *atomic_pi = 0.0;

  RAJA::forall<EXEC_POL3>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) * dx;
      RAJA::atomicAdd<ATOMIC_POL3>(atomic_pi, dx / (1.0 + x * x));
  });
  *atomic_pi *= 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << *atomic_pi << std::endl;

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running RAJA HIP pi approximation (reduction)...\n";

  using EXEC_POL4   = RAJA::hip_exec<HIP_BLOCK_SIZE>;
  using REDUCE_POL4 = RAJA::hip_reduce;

  RAJA::ReduceSum<REDUCE_POL4, double> hip_pi(0.0);

  RAJA::forall<EXEC_POL4>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) * dx;
      hip_pi += dx / (1.0 + x * x);
  });
  double hip_pi_val = hip_pi.get() * 4.0;

  std::cout << "\tpi = " << std::setprecision(prec)
            << hip_pi_val << std::endl;

  std::cout << "\n Running RAJA HIP pi approximation (atomic)...\n";

  *atomic_pi = 0;
  double* d_atomic_pi = memoryManager::allocate_gpu<double>(1);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_atomic_pi, atomic_pi, 1 * sizeof(double), hipMemcpyHostToDevice);

  using ATOMIC_POL4 = RAJA::hip_atomic;

  RAJA::forall<EXEC_POL4>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) * dx;
      RAJA::atomicAdd<ATOMIC_POL4>(d_atomic_pi, dx / (1.0 + x * x));
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, atomic_pi, d_atomic_pi, 1 * sizeof(double), hipMemcpyDeviceToHost);
  *atomic_pi *= 4.0; 
  std::cout << "\tpi = " << std::setprecision(prec)
            << *atomic_pi << std::endl;

  memoryManager::deallocate_gpu(d_atomic_pi);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(atomic_pi);

  std::cout << "\n DONE!...\n";

  return 0;
}
