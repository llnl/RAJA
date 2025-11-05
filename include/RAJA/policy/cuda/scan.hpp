/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_cuda_HPP
#define RAJA_scan_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iterator>
#include <type_traits>

#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename InputIter,
         typename Function>
RAJA_INLINE resources::EventProxy<resources::Cuda> inclusive_inplace(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    InputIter begin,
    InputIter end,
    Function binary_op)
{
  cudaStream_t stream = cuda_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin,
                                 begin, binary_op, len, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin,
                                 begin, binary_op, len, stream);
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename InputIter,
         typename Function,
         typename T>
RAJA_INLINE resources::EventProxy<resources::Cuda> exclusive_inplace(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    InputIter begin,
    InputIter end,
    Function binary_op,
    T init)
{
  cudaStream_t stream = cuda_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin,
                                 begin, binary_op, init, len, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin,
                                 begin, binary_op, init, len, stream);
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename InputIter,
         typename OutputIter,
         typename Function>
RAJA_INLINE resources::EventProxy<resources::Cuda> inclusive(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op)
{
  cudaStream_t stream = cuda_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin, out,
                                 binary_op, len, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin, out,
                                 binary_op, len, stream);
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename InputIter,
         typename OutputIter,
         typename Function,
         typename T>
RAJA_INLINE resources::EventProxy<resources::Cuda> exclusive(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op,
    T init)
{
  cudaStream_t stream = cuda_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin, out,
                                 binary_op, init, len, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                 d_temp_storage, temp_storage_bytes, begin, out,
                                 binary_op, init, len, stream);
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
