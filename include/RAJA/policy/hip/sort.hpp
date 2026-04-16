/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_hip_HPP
#define RAJA_sort_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <climits>
#include <iterator>
#include <type_traits>

#if defined(__HIPCC__)
// Tell rocprim to provide its HIP API
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_transform.hpp"
#include "rocprim/device/device_radix_sort.hpp"
#include "rocprim/device/device_merge_sort.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_merge_sort.cuh"
#endif

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{

#if defined(__HIPCC__)
template<typename R>
using double_buffer = ::rocprim::double_buffer<R>;
#elif defined(__CUDACC__)
template<typename R>
using double_buffer = ::cub::DoubleBuffer<R>;
#endif

template<typename R>
R* get_current(double_buffer<R>& d_bufs)
{
#if defined(__HIPCC__)
  return d_bufs.current();
#elif defined(__CUDACC__)
  return d_bufs.Current();
#endif
}

/*!
        \brief sort given range
*/
template<bool Stable,
         typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter,
         typename Compare>
resources::EventProxy<resources::Hip> sort(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    Iter begin,
    Iter end,
    Compare comp)
{
  RAJA_UNUSED_VAR(Stable);

  hipStream_t stream = hip_res.get_stream();

  using R = RAJA::detail::IterVal<Iter>;

  const int len           = std::distance(begin, end);
  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(R) * CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = hip::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  double_buffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<R>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys, d_temp_storage,
                                  temp_storage_bytes, d_keys, len, begin_bit,
                                  end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
#endif
  }
  else if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<R>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys_desc,
                                  d_temp_storage, temp_storage_bytes, d_keys,
                                  len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
#endif
  }
  else
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::merge_sort, d_temp_storage,
                                  temp_storage_bytes, d_keys, len, comp,
                                  stream);
#elif defined(__CUDACC__)
    if constexpr (Stable)
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::StableSortKeys,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     len, comp, stream);
    }
    else
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::SortKeys,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     len, comp, stream);
    }
#endif
  }
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<R>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys, d_temp_storage,
                                  temp_storage_bytes, d_keys, len, begin_bit,
                                  end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
#endif
  }
  else if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<R>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys_desc,
                                  d_temp_storage, temp_storage_bytes, d_keys,
                                  len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
#endif
  }
  else
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::merge_sort, d_temp_storage,
                                  temp_storage_bytes, d_keys, len, comp,
                                  stream);
#elif defined(__CUDACC__)
    if constexpr (Stable)
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::StableSortKeys,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     len, comp, stream);
    }
    else
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::SortKeys,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     len, comp, stream);
    }
#endif
  }
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (get_current(d_keys) == d_out)
  {

    // copy
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, begin, d_out, len * sizeof(R),
                                  hipMemcpyDefault, stream);
  }

  hip::device_mempool_type::getInstance().free(d_out);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief stable sort given range of pairs in order of keys
*/
template<bool Stable,
         typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
resources::EventProxy<resources::Hip> sort_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  RAJA_UNUSED_VAR(Stable);

  hipStream_t stream = hip_res.get_stream();

  using K = RAJA::detail::IterVal<KeyIter>;
  using V = RAJA::detail::IterVal<ValIter>;

  const int len           = std::distance(keys_begin, keys_end);
  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(K) * CHAR_BIT;

  // Allocate temporary storage for the output arrays
  K* d_keys_out = hip::device_mempool_type::getInstance().malloc<K>(len);
  V* d_vals_out = hip::device_mempool_type::getInstance().malloc<V>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the keys_begin and vals_begin buffers
  double_buffer<K> d_keys(keys_begin, d_keys_out);
  double_buffer<V> d_vals(vals_begin, d_vals_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                std::is_pointer_v<ValIter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<K>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs, d_temp_storage,
                                  temp_storage_bytes, d_keys, d_vals, len,
                                  begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
#endif
  }
  else if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                     std::is_pointer_v<ValIter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<K>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs_desc,
                                  d_temp_storage, temp_storage_bytes, d_keys,
                                  d_vals, len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
#endif
  }
  else
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::merge_sort, d_temp_storage,
                                  temp_storage_bytes, d_keys, d_vals, len, comp,
                                  stream);
#elif defined(__CUDACC__)
    if constexpr (Stable)
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::StableSortPairs,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     d_vals, len, comp, stream);
    }
    else
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::SortPairs,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     d_vals, len, comp, stream);
    }
#endif
  }
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                std::is_pointer_v<ValIter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<K>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs, d_temp_storage,
                                  temp_storage_bytes, d_keys, d_vals, len,
                                  begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
#endif
  }
  else if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                     std::is_pointer_v<ValIter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<K>>)
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs_desc,
                                  d_temp_storage, temp_storage_bytes, d_keys,
                                  d_vals, len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
#endif
  }
  else
  {
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::merge_sort, d_temp_storage,
                                  temp_storage_bytes, d_keys, d_vals, len, comp,
                                  stream);
#elif defined(__CUDACC__)
    if constexpr (Stable)
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::StableSortPairs,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     d_vals, len, comp, stream);
    }
    else
    {
      CAMP_CUDA_API_INVOKE_AND_CHECK(cub::DeviceMergeSort::SortPairs,
                                     d_temp_storage, temp_storage_bytes, d_keys,
                                     d_vals, len, comp, stream);
    }
#endif
  }
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (get_current(d_keys) == d_keys_out)
  {

    // copy keys
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, keys_begin, d_keys_out,
                                  len * sizeof(K), hipMemcpyDefault, stream);
  }
  if (get_current(d_vals) == d_vals_out)
  {

    // copy vals
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, vals_begin, d_vals_out,
                                  len * sizeof(V), hipMemcpyDefault, stream);
  }

  hip::device_mempool_type::getInstance().free(d_keys_out);
  hip::device_mempool_type::getInstance().free(d_vals_out);

  hip::launch(hip_res, Async);

  return resources::EventProxy<resources::Hip>(hip_res);
}

}  // namespace detail

/*!
        \brief stable sort given range
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter,
         typename Compare>
resources::EventProxy<resources::Hip> stable(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    Iter begin,
    Iter end,
    Compare comp)
{
  constexpr bool stable = true;
  return detail::sort<stable>(hip_res, p, begin, end, comp);
}

/*!
        \brief sort given range
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter,
         typename Compare>
resources::EventProxy<resources::Hip> unstable(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    Iter begin,
    Iter end,
    Compare comp)
{
  constexpr bool stable = true;
  return detail::sort<!stable>(hip_res, p, begin, end, comp);
}

/*!
        \brief stable sort given range of pairs in order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
resources::EventProxy<resources::Hip> stable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  constexpr bool stable = true;
  return detail::sort_pairs<stable>(hip_res, p, keys_begin, keys_end,
                                    vals_begin, comp);
}

/*!
        \brief stable sort given range of pairs in order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
resources::EventProxy<resources::Hip> unstable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  constexpr bool stable = true;
  return detail::sort_pairs<!stable>(hip_res, p, keys_begin, keys_end,
                                     vals_begin, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
