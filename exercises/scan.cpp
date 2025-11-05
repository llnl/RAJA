//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define OP_PLUS_INT RAJA::operators::plus<int>
#define OP_MIN_INT RAJA::operators::minimum<int>
#define OP_MAX_INT RAJA::operators::maximum<int>
#define CHECK_INC_SCAN_RESULTS(X) checkInclusiveScanResult<X>(in, out, N);
#define CHECK_EXC_SCAN_RESULTS(X) checkExclusiveScanResult<X>(in, out, N);

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Scan Exercise
 *
 *  This exercise demonstrates RAJA inclusive and exclusive scan operations
 *  for integer arrays, including in-place, using different operators.
 *  Other array data types, operators, etc. are similar
 *
 *  RAJA features shown:
 *    - `RAJA::inclusive_scan` and `RAJA::inclusive_scan_inplace` methods
 *    - `RAJA::exclusive_scan` and `RAJA::exclusive_scan_inplace` methods
 *    -  RAJA operators
 *    -  Execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA)
//constexpr int CUDA_BLOCK_SIZE = 16;
#endif

#if defined(RAJA_ENABLE_HIP)
//constexpr int HIP_BLOCK_SIZE = 16;
#endif

//
// Functions for checking results and printing vectors
//
template <typename Function, typename T>
void checkInclusiveScanResult(const T* in, const T* out, int N);
//
template <typename Function, typename T>
void checkExclusiveScanResult(const T* in, const T* out, int N);
//
template <typename T>
void printArray(const T* v, int N);


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA scan example...\n";

  // _scan_array_init_start
//
// Define array length
//
  constexpr int N = 20;

//
// Allocate and initialize vector data
//
  int* in = memoryManager::allocate<int>(N);
  int* out = memoryManager::allocate<int>(N);

  std::iota(in, in + N, -1);

  std::cout << "\n in values...\n";
  printArray(in, N);
  std::cout << "\n";
  // _scan_array_init_end



//----------------------------------------------------------------------------//
// Perform various sequential scans to illustrate inclusive/exclusive,
// in-place, default scans with different operators
//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan (default)...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive RAJA scan with RAJA::seq_exec
  ///           execution policy type. 
  ///
  /// NOTE: We've done this one for you to help you get started...
  ///

  // _scan_inclusive_seq_start
  RAJA::inclusive_scan<RAJA::seq_exec>(RAJA::make_span(in, N),
                                       RAJA::make_span(out, N));
  // _scan_inclusive_seq_end

  CHECK_INC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive RAJA scan with RAJA::seq_exec
  ///           execution policy type and an explicit plus operator. 
  ///

  CHECK_INC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential exclusive_scan (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive RAJA scan with RAJA::seq_exec
  ///           execution policy type and an explicit plus operator. 
  ///

  CHECK_EXC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan_inplace (minimum)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive inplace RAJA scan with RAJA::seq_exec
  ///           execution policy type and an explicit minimum operator. 
  ///

  CHECK_INC_SCAN_RESULTS(OP_MIN_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential exclusive_scan_inplace (maximum)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive inplace RAJA scan with RAJA::seq_exec
  ///           execution policy type and an explicit maximum operator. 
  ///

  CHECK_EXC_SCAN_RESULTS(OP_MAX_INT)
  printArray(out, N);
  std::cout << "\n";


#if defined(RAJA_ENABLE_OPENMP)

//----------------------------------------------------------------------------//
// Perform a couple of OpenMP scans...
//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP inclusive_scan (plus)...\n";

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive RAJA scan with RAJA::omp_parallel_for_exec
  ///           execution policy type and an explicit plus operator. 
  ///

  CHECK_INC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP exclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive inplace RAJA scan with RAJA::omp_parallel_for_exec
  ///           execution policy type and an explicit plus operator. 
  ///

  CHECK_EXC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

//----------------------------------------------------------------------------//
// Perform a couple of CUDA scans...
//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA inclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive inplace RAJA scan with RAJA::cuda_exec
  ///           execution policy type and an explicit plus operator.
  ///
  ///           NOTE: You will have to uncomment 'CUDA_BLOCK_SIZE' near the top
  ///                 of the file if you want to use it here.
  ///

  CHECK_INC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA exclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive inplace RAJA scan with RAJA::cuda_exec
  ///           execution policy type and an explicit plus operator.
  ///
  ///           NOTE: You will have to uncomment 'CUDA_BLOCK_SIZE' near the top
  ///                 of the file if you want to use it here.
  ///

  CHECK_EXC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA exclusive_scan (plus)...\n";

  std::copy_n(in, N, out);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive RAJA scan with RAJA::cuda_exec
  ///           execution policy type and an explicit plus operator.
  ///
  ///           NOTE: You will have to uncomment 'CUDA_BLOCK_SIZE' near the top
  ///                 of the file if you want to use it here.
  ///

  CHECK_EXC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

#endif

//----------------------------------------------------------------------------//


#if defined(RAJA_ENABLE_HIP)

//----------------------------------------------------------------------------//
// Perform a couple of HIP scans...
//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP inclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);
  int* d_in = memoryManager::allocate_gpu<int>(N);
  int* d_out = memoryManager::allocate_gpu<int>(N);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_out, out, N * sizeof(int), hipMemcpyHostToDevice);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an inclusive inplace RAJA scan with RAJA::hip_exec
  ///           execution policy type and an explicit plus operator. 
  ///
  ///           NOTE: You will have to uncomment 'HIP_BLOCK_SIZE' near the top
  ///                 of the file if you want to use it here.
  ///

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

  CHECK_INC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP exclusive_scan (plus)...\n";

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_in, in, N * sizeof(int), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_out, out, N * sizeof(int), hipMemcpyHostToDevice);

  ///
  /// TODO...
  ///
  /// EXERCISE: Implement an exclusive RAJA scan with RAJA::hip_exec
  ///           execution policy type and an explicit plus operator. 
  ///
  ///           NOTE: You will have to uncomment 'HIP_BLOCK_SIZE' near the top
  ///                 of the file if you want to use it here.
  ///

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

  CHECK_EXC_SCAN_RESULTS(OP_PLUS_INT)
  printArray(out, N);
  std::cout << "\n";

  memoryManager::deallocate_gpu(d_in);
  memoryManager::deallocate_gpu(d_out);

#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(in);
  memoryManager::deallocate(out);

  std::cout << "\n DONE!...\n";

  return 0;
}


//
// Function to check inclusive scan result
//
template <typename Function, typename T>
void checkInclusiveScanResult(const T* in, const T* out, int N)
{
  T val = Function::identity();
  for (int i = 0; i < N; ++i) {
    val = Function()(val, in[i]);
    if (out[i] != val) {
      std::cout << "\n\t result -- WRONG\n";
      std::cout << "\t" << out[i] << " != " << val
                << " (at index " << i << ")\n";
    }
  }
  std::cout << "\n\t result -- CORRECT\n";
}

//
// Function to check exclusive scan result
//
template <typename Function, typename T>
void checkExclusiveScanResult(const T* in, const T* out, int N)
{
  T val = Function::identity();
  for (int i = 0; i < N; ++i) {
    if (out[i] != val) {
      std::cout << "\n\t result -- WRONG\n";
      std::cout << "\t" << out[i] << " != " << val
                << " (at index " << i << ")\n";
    }
    val = Function()(val, in[i]);
  }
  std::cout << "\n\t result -- CORRECT\n";
}

//
// Function to print vector.
//
template <typename T>
void printArray(const T* v, int N)
{
  std::cout << std::endl;
  for (int i = 0; i < N; ++i) { std::cout << " " << v[i]; }
  std::cout << std::endl;
}
