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
 *  Vector Addition Example with resource + dynamic policy selection
 *
 *  Computes c = a + b, where a, b, c are vectors of ints using
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
#if defined(RAJA_ENABLE_CUDA)
                               ,RAJA::cuda_exec<256>
                               ,RAJA::cuda_exec<512>
#endif

#if defined(RAJA_ENABLE_HIP)
                               ,RAJA::hip_exec<256>
                               ,RAJA::hip_exec<512>
#endif
                               >;


int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./cuda-dynamic-forall N, where N is the index of  the policy to run");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument
  // Example usage ./dynamic_forall policy N
  //

  const int pol = std::stoi(argv[1]);

  RAJA::ExecPlace select_cpu_or_gpu;
  if(pol < 2) {
    select_cpu_or_gpu = RAJA::ExecPlace::HOST;
  } else {
    select_cpu_or_gpu = RAJA::ExecPlace::DEVICE;
  }

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

  std::cout << "\n Running C-style vector addition...\n";

  // _cstyle_vector_add_start
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
  // _cstyle_vector_add_end

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
// Example of dynamic policy selection for forall
//----------------------------------------------------------------------------//

  RAJA::resources::Host host_res;
#if defined(RAJA_ENABLE_CUDA)
  RAJA::resources::Cuda device_res;
#endif
#if defined(RAJA_ENABLE_HIP)
  RAJA::resources::Hip device_res;
#endif
#if defined(RAJA_ENABLE_SYCL)
  RAJA::resources::Sycl device_res;
#endif  

  //Get typed erased resource - it will internally store if we are running on the host or device
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
  RAJA::resources::Resource res = RAJA::Get_Runtime_Resource(host_res, device_res, select_cpu_or_gpu);
#else
  RAJA::resources::Resource res = RAJA::Get_Host_Resource(host_res, select_cpu_or_gpu);
#endif

  RAJA::dynamic_forall<policy_list>
  (res, pol, RAJA::RangeSegment(0, N), [=] RAJA_HOST_DEVICE (int i)   {

    c[i] = a[i] + b[i];

  });

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
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
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
