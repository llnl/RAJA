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

#ifndef RAJA_sort_cuda_HPP
#define RAJA_sort_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <climits>
#include <iterator>
#include <type_traits>

#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_merge_sort.cuh"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

/*!
        \brief stable sort given range
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter,
         typename Compare>
resources::EventProxy<resources::Cuda> stable(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    Iter begin,
    Iter end,
    Compare comp)
{
  cudaStream_t stream = cuda_res.get_stream();

  using R = RAJA::detail::IterVal<Iter>;

  int len       = std::distance(begin, end);
  int begin_bit = 0;
  int end_bit   = sizeof(R) * CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = cuda::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  cub::DoubleBuffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<R>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
  }
  else if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<R>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
  }
  else
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceMergeSort::StableSortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, comp, stream);
  }
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<R>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
  }
  else if constexpr (std::is_arithmetic_v<R> && std::is_pointer_v<Iter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<R>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, begin_bit, end_bit, stream);
  }
  else
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceMergeSort::StableSortKeys,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   len, comp, stream);
  }
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_out)
  {

    // copy
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpyAsync, begin, d_out,
                                   len * sizeof(R), cudaMemcpyDefault, stream);
  }

  cuda::device_mempool_type::getInstance().free(d_out);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief sort given range
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter,
         typename Compare>
resources::EventProxy<resources::Cuda> unstable(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async> p,
    Iter begin,
    Iter end,
    Compare comp)
{
  return stable(cuda_res, p, begin, end, comp);
}

/*!
        \brief stable sort given range of pairs in order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
resources::EventProxy<resources::Cuda> stable_pairs(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async>,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  cudaStream_t stream = cuda_res.get_stream();

  using K = RAJA::detail::IterVal<KeyIter>;
  using V = RAJA::detail::IterVal<ValIter>;

  int len       = std::distance(keys_begin, keys_end);
  int begin_bit = 0;
  int end_bit   = sizeof(K) * CHAR_BIT;

  // Allocate temporary storage for the output arrays
  K* d_keys_out = cuda::device_mempool_type::getInstance().malloc<K>(len);
  V* d_vals_out = cuda::device_mempool_type::getInstance().malloc<V>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the keys_begin and vals_begin buffers
  cub::DoubleBuffer<K> d_keys(keys_begin, d_keys_out);
  cub::DoubleBuffer<V> d_vals(vals_begin, d_vals_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                std::is_pointer_v<ValIter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<K>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
  }
  else if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                     std::is_pointer_v<ValIter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<K>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
  }
  else
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceMergeSort::StableSortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, comp, stream);
  }
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                std::is_pointer_v<ValIter> &&
                std::is_same_v<std::decay_t<Compare>, operators::less<K>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
  }
  else if constexpr (std::is_arithmetic_v<K> && std::is_pointer_v<KeyIter> &&
                     std::is_pointer_v<ValIter> &&
                     std::is_same_v<std::decay_t<Compare>,
                                    operators::greater<K>>)
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, begin_bit, end_bit, stream);
  }
  else
  {
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceMergeSort::StableSortPairs,
                                   d_temp_storage, temp_storage_bytes, d_keys,
                                   d_vals, len, comp, stream);
  }
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_keys_out)
  {

    // copy keys
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpyAsync, keys_begin, d_keys_out,
                                   len * sizeof(K), cudaMemcpyDefault, stream);
  }
  if (d_vals.Current() == d_vals_out)
  {

    // copy vals
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpyAsync, vals_begin, d_vals_out,
                                   len * sizeof(V), cudaMemcpyDefault, stream);
  }

  cuda::device_mempool_type::getInstance().free(d_keys_out);
  cuda::device_mempool_type::getInstance().free(d_vals_out);

  cuda::launch(cuda_res, Async);

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief stable sort given range of pairs in order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
resources::EventProxy<resources::Cuda> unstable_pairs(
    resources::Cuda cuda_res,
    ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                             IterationGetter,
                                             Concretizer,
                                             BLOCKS_PER_SM,
                                             Async> p,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  return stable_pairs(cuda_res, p, keys_begin, keys_end, vals_begin, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
