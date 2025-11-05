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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
        \brief static assert unimplemented stable sort
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
        std::is_pointer<Iter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<Iter>>>,
            camp::is_same<Compare,
                          operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
stable(resources::Cuda cuda_res,
       ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                IterationGetter,
                                                Concretizer,
                                                BLOCKS_PER_SM,
                                                Async>,
       Iter,
       Iter,
       Compare)
{
  static_assert(std::is_pointer<Iter>::value,
                "stable_sort<cuda_exec> is only implemented for pointers");
  using iterval = RAJA::detail::IterVal<Iter>;
  static_assert(
      type_traits::is_arithmetic<iterval>::value,
      "stable_sort<cuda_exec> is only implemented for arithmetic types");
  static_assert(concepts::any_of<
                    camp::is_same<Compare, operators::less<iterval>>,
                    camp::is_same<Compare, operators::greater<iterval>>>::value,
                "stable_sort<cuda_exec> is only implemented for "
                "RAJA::operators::less or RAJA::operators::greater");

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief stable sort given range in ascending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Cuda>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
stable(resources::Cuda cuda_res,
       ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                IterationGetter,
                                                Concretizer,
                                                BLOCKS_PER_SM,
                                                Async>,
       Iter begin,
       Iter end,
       operators::less<RAJA::detail::IterVal<Iter>>)
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
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
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
        \brief stable sort given range in descending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Cuda>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
stable(resources::Cuda cuda_res,
       ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                IterationGetter,
                                                Concretizer,
                                                BLOCKS_PER_SM,
                                                Async>,
       Iter begin,
       Iter end,
       operators::greater<RAJA::detail::IterVal<Iter>>)
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
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
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
        \brief static assert unimplemented sort
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
        std::is_pointer<Iter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<Iter>>>,
            camp::is_same<Compare,
                          operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
unstable(resources::Cuda cuda_res,
         ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                  IterationGetter,
                                                  Concretizer,
                                                  BLOCKS_PER_SM,
                                                  Async>,
         Iter,
         Iter,
         Compare)
{
  static_assert(std::is_pointer<Iter>::value,
                "sort<cuda_exec> is only implemented for pointers");
  using iterval = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_arithmetic<iterval>::value,
                "sort<cuda_exec> is only implemented for arithmetic types");
  static_assert(concepts::any_of<
                    camp::is_same<Compare, operators::less<iterval>>,
                    camp::is_same<Compare, operators::greater<iterval>>>::value,
                "sort<cuda_exec> is only implemented for RAJA::operators::less "
                "or RAJA::operators::greater");

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief sort given range in ascending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Cuda>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
unstable(resources::Cuda cuda_res,
         ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                  IterationGetter,
                                                  Concretizer,
                                                  BLOCKS_PER_SM,
                                                  Async> p,
         Iter begin,
         Iter end,
         operators::less<RAJA::detail::IterVal<Iter>> comp)
{
  return stable(cuda_res, p, begin, end, comp);
}

/*!
        \brief sort given range in descending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Cuda>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
unstable(resources::Cuda cuda_res,
         ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                  IterationGetter,
                                                  Concretizer,
                                                  BLOCKS_PER_SM,
                                                  Async> p,
         Iter begin,
         Iter end,
         operators::greater<RAJA::detail::IterVal<Iter>> comp)
{
  return stable(cuda_res, p, begin, end, comp);
}

/*!
        \brief static assert unimplemented stable sort pairs
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
        std::is_pointer<KeyIter>,
        std::is_pointer<ValIter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<KeyIter>>>,
            camp::is_same<
                Compare,
                operators::greater<RAJA::detail::IterVal<KeyIter>>>>>>>
stable_pairs(resources::Cuda cuda_res,
             ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                      IterationGetter,
                                                      Concretizer,
                                                      BLOCKS_PER_SM,
                                                      Async>,
             KeyIter,
             KeyIter,
             ValIter,
             Compare)
{
  static_assert(
      std::is_pointer<KeyIter>::value,
      "stable_sort_pairs<cuda_exec> is only implemented for pointers");
  static_assert(
      std::is_pointer<ValIter>::value,
      "stable_sort_pairs<cuda_exec> is only implemented for pointers");
  using K = RAJA::detail::IterVal<KeyIter>;
  static_assert(
      type_traits::is_arithmetic<K>::value,
      "stable_sort_pairs<cuda_exec> is only implemented for arithmetic types");
  static_assert(
      concepts::any_of<camp::is_same<Compare, operators::less<K>>,
                       camp::is_same<Compare, operators::greater<K>>>::value,
      "stable_sort_pairs<cuda_exec> is only implemented for "
      "RAJA::operators::less or RAJA::operators::greater");

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief stable sort given range of pairs in ascending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
stable_pairs(resources::Cuda cuda_res,
             ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                      IterationGetter,
                                                      Concretizer,
                                                      BLOCKS_PER_SM,
                                                      Async>,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             operators::less<RAJA::detail::IterVal<KeyIter>>)
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
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
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
        \brief stable sort given range of pairs in descending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
stable_pairs(resources::Cuda cuda_res,
             ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                      IterationGetter,
                                                      Concretizer,
                                                      BLOCKS_PER_SM,
                                                      Async>,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             operators::greater<RAJA::detail::IterVal<KeyIter>>)
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
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
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
        \brief static assert unimplemented sort pairs
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
        std::is_pointer<KeyIter>,
        std::is_pointer<ValIter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<KeyIter>>>,
            camp::is_same<
                Compare,
                operators::greater<RAJA::detail::IterVal<KeyIter>>>>>>>
unstable_pairs(resources::Cuda cuda_res,
               ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                        IterationGetter,
                                                        Concretizer,
                                                        BLOCKS_PER_SM,
                                                        Async>,
               KeyIter,
               KeyIter,
               ValIter,
               Compare)
{
  static_assert(std::is_pointer<KeyIter>::value,
                "sort_pairs<cuda_exec> is only implemented for pointers");
  static_assert(std::is_pointer<ValIter>::value,
                "sort_pairs<cuda_exec> is only implemented for pointers");
  using K = RAJA::detail::IterVal<KeyIter>;
  static_assert(
      type_traits::is_arithmetic<K>::value,
      "sort_pairs<cuda_exec> is only implemented for arithmetic types");
  static_assert(
      concepts::any_of<camp::is_same<Compare, operators::less<K>>,
                       camp::is_same<Compare, operators::greater<K>>>::value,
      "sort_pairs<cuda_exec> is only implemented for RAJA::operators::less or "
      "RAJA::operators::greater");

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

/*!
        \brief stable sort given range of pairs in ascending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
unstable_pairs(resources::Cuda cuda_res,
               ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                        IterationGetter,
                                                        Concretizer,
                                                        BLOCKS_PER_SM,
                                                        Async> p,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               operators::less<RAJA::detail::IterVal<KeyIter>> comp)
{
  return stable_pairs(cuda_res, p, keys_begin, keys_end, vals_begin, comp);
}

/*!
        \brief stable sort given range of pairs in descending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         size_t BLOCKS_PER_SM,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Cuda>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
unstable_pairs(resources::Cuda cuda_res,
               ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping,
                                                        IterationGetter,
                                                        Concretizer,
                                                        BLOCKS_PER_SM,
                                                        Async> p,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               operators::greater<RAJA::detail::IterVal<KeyIter>> comp)
{
  return stable_pairs(cuda_res, p, keys_begin, keys_end, vals_begin, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
