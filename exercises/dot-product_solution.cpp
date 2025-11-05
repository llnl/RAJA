//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Vector Dot Product Exercise
 *
 *  Computes dot = (a,b), where a, b are vectors of 
 *  doubles and dot is a scalar double. It illustrates how RAJA
 *  supports a portable parallel reduction opertion in a way that 
 *  the code looks like it does in a sequential implementation.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
//  Function to check dot product result.
//
void checkResult(double compdot, double refdot);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise: vector dot product...\n";

//
// Define vector length
//
  constexpr int N = 1000000;

//
// Allocate and initialize vector data
//
  double *a = memoryManager::allocate<double>(N);
  double *b = memoryManager::allocate<double>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = 1.0;
    b[i] = 1.0;
  }

//----------------------------------------------------------------------------//

//
// C-style dot product operation.
//
  std::cout << "\n Running C-version of dot product...\n";

  // _csytle_dotprod_start
  double dot = 0.0;

  for (int i = 0; i < N; ++i) {
    dot += a[i] * b[i];
  }

  std::cout << "\t (a, b) = " << dot << std::endl;
  // _csytle_dotprod_end

  double dot_ref = dot;

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential dot product...\n";

  dot = 0.0;

  // _rajaseq_dotprod_start
  RAJA::ReduceSum<RAJA::seq_reduce, double> seqdot(0.0);

  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, N), [=] (int i) { 
    seqdot += a[i] * b[i]; 
  });

  dot = seqdot.get();
  // _rajaseq_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP dot product...\n";

  dot = 0.0;

  // _rajaomp_dotprod_start
  RAJA::ReduceSum<RAJA::omp_reduce, double> ompdot(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    ompdot += a[i] * b[i]; 
  }); 

  dot = ompdot.get();
  // _rajaomp_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  const int CUDA_BLOCK_SIZE = 256;

  std::cout << "\n Running RAJA CUDA dot product...\n";

  dot = 0.0;

  // _rajacuda_dotprod_start
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();
  // _rajacuda_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  const int HIP_BLOCK_SIZE = 256;

  std::cout << "\n Running RAJA HIP dot product...\n";

  dot = 0.0;

  double *d_a = memoryManager::allocate_gpu<double>(N);
  double *d_b = memoryManager::allocate_gpu<double>(N);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_b, b, N * sizeof(double), hipMemcpyHostToDevice);

  // _rajahip_dotprod_start
  RAJA::ReduceSum<RAJA::hip_reduce, double> hpdot(0.0);

  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    hpdot += d_a[i] * d_b[i];
  });

  dot = hpdot.get();
  // _rajahip_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_SYCL)

  const int SYCL_BLOCK_SIZE = 256;

  std::cout << "\n Running RAJA SYCL dot product...\n";

  dot = 0.0;

  // _rajasycl_dotprod_start
  RAJA::ReduceSum<RAJA::sycl_reduce, double> hpdot(0.0);

  RAJA::forall<RAJA::sycl_exec<SYCL_BLOCK_SIZE, false>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    hpdot += a[i] * b[i];
  });

  dot = static_cast<double>(hpdot.get());
  // _rajasycl_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);

#endif

//----------------------------------------------------------------------------//


  memoryManager::deallocate(a);
  memoryManager::deallocate(b);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
//  Function to check computed dot product and report P/F.
//
void checkResult(double compdot, double refdot)
{
  if ( compdot == refdot ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

