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

#ifndef RAJA_scan_hip_HPP
#define RAJA_scan_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iterator>
#include <type_traits>

#if defined(__HIPCC__)
// Tell rocprim to provide its HIP API
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_scan.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

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
         bool Async,
         typename InputIter,
         typename Function>
RAJA_INLINE resources::EventProxy<resources::Hip> inclusive_inplace(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    InputIter begin,
    InputIter end,
    Function binary_op)
{
  hipStream_t stream = hip_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::inclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, begin, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin,
                                begin, binary_op, len, stream);
#endif

  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::inclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, begin, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin,
                                begin, binary_op, len, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename InputIter,
         typename Function,
         typename T>
RAJA_INLINE resources::EventProxy<resources::Hip> exclusive_inplace(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    InputIter begin,
    InputIter end,
    Function binary_op,
    T init)
{
  hipStream_t stream = hip_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, begin, init, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin,
                                begin, binary_op, init, len, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, begin, init, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin,
                                begin, binary_op, init, len, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename InputIter,
         typename OutputIter,
         typename Function>
RAJA_INLINE resources::EventProxy<resources::Hip> inclusive(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op)
{
  hipStream_t stream = hip_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::inclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, out, len, binary_op,
                                stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin, out,
                                binary_op, len, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::inclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, out, len, binary_op,
                                stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::InclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin, out,
                                binary_op, len, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename InputIter,
         typename OutputIter,
         typename Function,
         typename T>
RAJA_INLINE resources::EventProxy<resources::Hip> exclusive(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op,
    T init)
{
  hipStream_t stream = hip_res.get_stream();

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, out, init, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin, out,
                                binary_op, init, len, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan, d_temp_storage,
                                temp_storage_bytes, begin, out, init, len,
                                binary_op, stream);
#elif defined(__CUDACC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
                                d_temp_storage, temp_storage_bytes, begin, out,
                                binary_op, init, len, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
