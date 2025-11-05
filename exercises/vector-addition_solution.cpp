//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 *  Vector Addition Exercise
 *
 *  In this exercise, you will compute c = a + b, where a, b, c are 
 *  integer vectors.
 *
 *  This file contains sequential and OpenMP variants of the vector addition
 *  using C-style for-loops. You will fill in RAJA versions of these variants,
 *  plus a RAJA CUDA version if you have access to an NVIDIA GPU and a CUDA
 *  compiler, in empty code sections indicated by comments.
 *
 *  The exercise shows you how to use RAJA in its simplest form and 
 *  illustrates similarities between a C-style for-loop and a RAJA forall loop.
 *
 *  RAJA features you will use:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *
 * Note: if CUDA is enabled, CUDA unified memory is used. 
 */

/*
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA)
constexpr int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
constexpr int HIP_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_SYCL)
constexpr int SYCL_BLOCK_SIZE = 256;
#endif

//
// Functions for checking and printing arrays
//
void checkResult(int* c, int* c_ref, int len); 
void printArray(int* v, int len);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise: RAJA Vector Addition...\n";

//
// Define vector length
//
  constexpr int N = 1000000;

//
// Allocate and initialize vector data to random numbers in [1, 10].
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);
  int *c_ref = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = rand() % 10 + 1;
    b[i] = rand() % 10 + 1;
  }


//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::memset(c_ref, 0, N * sizeof(int));

  std::cout << "\n Running C-style sequential vector addition...\n";

  // _cstyle_vector_add_start
  for (int i = 0; i < N; ++i) {
    c_ref[i] = a[i] + b[i];
  }
  // _cstyle_vector_add_end

//printArray(c_ref, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces strictly sequential execution.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA sequential vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      c[i] = a[i] + b[i];
    }
  );
  // _rajaseq_vector_add_end

  checkResult(c, c_ref, N);
//printArray(c, N);


//----------------------------------------------------------------------------//
// RAJA::simd_exec policy attempts to force the compiler to generate SIMD
// vectorization optimizations.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA SIMD vector addition...\n";

  RAJA::forall<RAJA::simd_exec>(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) { 
      c[i] = a[i] + b[i]; 
    }
  );    

  checkResult(c, c_ref, N);
//printArray(c, N);


//----------------------------------------------------------------------------//
// C-style OpenMP multithreading variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running C-style OpenMP vector addition...\n";

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

  checkResult(c, c_ref, N);
//printArray(c, N);

#endif


//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_exec policy runs the loop in parallel using
// OpenMP multithreading.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA OpenMP multithreaded vector addition...\n";

  // _rajaomp_vector_add_start
  RAJA::forall< RAJA::omp_parallel_for_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) { 
      c[i] = a[i] + b[i]; 
    }
  ); 
  // _rajaomp_vector_add_end

  checkResult(c, c_ref, N);
//printArray(c, N);
#endif


//----------------------------------------------------------------------------//
// RAJA::cuda_exec policy runs the loop as a CUDA kernel on a GPU device.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA CUDA vector addition...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);
  int *d_c = memoryManager::allocate_gpu<int>(N);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpy, d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpy, d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // _rajacuda_vector_add_start
  RAJA::forall< RAJA::cuda_exec<CUDA_BLOCK_SIZE> >(RAJA::TypedRangeSegment<int>(0, N), 
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });
  // _rajacuda_vector_add_end

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpy, c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  checkResult(c, c_ref, N);
//printArray(c, N);

//----------------------------------------------------------------------------//
// RAJA::cuda_exec policy runs the loop as a CUDA kernel asynchronously on a 
// GPU device with 2 blocks per SM.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA CUDA explicit (2 blocks per SM) vector addition...\n";

  // _rajacuda_explicit_vector_add_start
  const bool Asynchronous = true;

  RAJA::forall<RAJA::cuda_exec_explicit<CUDA_BLOCK_SIZE, 2, Asynchronous>>(RAJA::TypedRangeSegment<int>(0, N), 
    [=] RAJA_DEVICE (int i) { 
    d_c[i] = d_a[i] + d_b[i]; 
  });    
  // _rajacuda_explicit_vector_add_end

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpy, c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  checkResult(c, c_ref, N);
//printResult(c, N);
#endif

//----------------------------------------------------------------------------//
// RAJA::hip_exec policy runs the loop as a HIP kernel on a GPU device.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP vector addition...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);
  int *d_c = memoryManager::allocate_gpu<int>(N);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_a, a, N * sizeof(int), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_b, b, N * sizeof(int), hipMemcpyHostToDevice);

  // _rajahip_vector_add_start
  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::TypedRangeSegment<int>(0, N),
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });
  // _rajahip_vector_add_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, c, d_c, N * sizeof(int), hipMemcpyDeviceToHost);

  checkResult(c, c_ref, N);
//printResult(c, N);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
  memoryManager::deallocate_gpu(d_c);
#endif

//----------------------------------------------------------------------------//
// RAJA::sycl_exec policy runs the loop as a SYCL kernel.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_SYCL)
  std::cout << "\n Running RAJA SYCL vector addition...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);
  int *d_c = memoryManager::allocate_gpu<int>(N);

  memoryManager::sycl_res->memcpy(d_a, a, N * sizeof(int));
  memoryManager::sycl_res->memcpy(d_b, b, N * sizeof(int));

  // _rajasycl_vector_add_start
  RAJA::forall<RAJA::sycl_exec<SYCL_BLOCK_SIZE>>(RAJA::TypedRangeSegment<int>(0, N),
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });
  // _rajasycl_vector_add_end

  memoryManager::sycl_res->memcpy(c, d_c, N * sizeof(int));

  checkResult(c, c_ref, N);
//printResult(c, N);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
  memoryManager::deallocate_gpu(d_c);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);
  memoryManager::deallocate(c_ref);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* c, int* c_ref, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && c[i] != c_ref[i] ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print array.
//
void printArray(int* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}

