//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef EXAMPLES_MEMORYMANAGER_HPP
#define EXAMPLES_MEMORYMANAGER_HPP

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif

/*
  As RAJA does not manage memory we include a general purpose memory
  manager which may be used to perform c++ style allocation/deallocation
  or allocate/deallocate CUDA unified memory. The type of memory allocated
  is dependent on how RAJA was configured.
*/
namespace memoryManager
{

template <typename T>
T *allocate(RAJA::Index_type size)
{
  T *ptr;
#if defined(RAJA_ENABLE_CUDA)
  CAMP_CUDA_API_INVOKE_AND_CHECK(
      cudaMallocManaged, (void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal);
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  template <typename T>
  T *allocate_gpu(RAJA::Index_type size)
  {
    T *ptr;
#if defined(RAJA_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMalloc, (void **)&ptr, sizeof(T) * size);
#elif defined(RAJA_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, (void **)&ptr, sizeof(T) * size);
#endif
    return ptr;
  }

  template <typename T>
  void deallocate_gpu(T *&ptr)
  {
    if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
      CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
#elif defined(RAJA_ENABLE_HIP)
      CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, ptr);
#endif
      ptr = nullptr;
    }
  }
#endif

};  // namespace memoryManager
#endif
