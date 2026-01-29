//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Vector Addition Example with dynamic policy selection
 *
 *  Computes c = a + b, where a, b, c are vectors of ints.
 *  a policy selected at run-time
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Functions for checking and printing results
//
void checkResult(int* res, int len);
void printResult(int* res, int len);

using policy_list = camp::list<RAJA::seq_exec
                               ,RAJA::simd_exec
#if defined(RAJA_ENABLE_OPENMP)
                               ,RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                               ,RAJA::cuda_exec<256>
                               ,RAJA::cuda_exec<512>
#endif
                               >;

int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./dynamic-forall N, where N is the index of the policy to run");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument
  // Example usage ./dynamic_forall policy N
  //

  const int pol = std::stoi(argv[1]);

  std::cout << "\n\nRAJA vector addition example...\n";
  std::cout << "Using policy # "<<pol<<std::endl;

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i;
  }


//----------------------------------------------------------------------------//

  std::cout << "\n Running dynamic forall vector addition and reductions...\n";

  int sum = 0;
  using VAL_INT_SUM = RAJA::expt::ValOp<int, RAJA::operators::plus>;
  
  RAJA::RangeSegment range(0, N); 
  
  //policy is chosen from the list
  RAJA::dynamic_forall<policy_list>(pol, range,
    RAJA::expt::Reduce<RAJA::operators::plus>(&sum),
      RAJA::Name("RAJA dynamic forall"),
      [=] RAJA_HOST_DEVICE (int i, VAL_INT_SUM &_sum) {
      
      c[i] = a[i] + b[i];
      _sum += 1;
  });
  // _rajaseq_vector_add_end

  std::cout << "Sum = " << sum << ", expected sum: " << N << std::endl;
  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* res, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( res[i] != 0 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t Vector sum result -- PASS\n";
  } else {
    std::cout << "\n\t Vector sum result -- FAIL\n";
  }
}

//
// Function to print result.
//
void printResult(int* res, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << res[i] << std::endl;
  }
  std::cout << std::endl;
}
