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
#elif defined(__CUDACC__)
#include "cub/device/device_radix_sort.cuh"
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

}  // namespace detail

/*!
        \brief static assert unimplemented stable sort
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
        std::is_pointer<Iter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<Iter>>>,
            camp::is_same<Compare,
                          operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
stable(resources::Hip hip_res,
       ::RAJA::policy::hip::
           hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
       Iter,
       Iter,
       Compare)
{
  static_assert(
      concepts::all_of<
          type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
          std::is_pointer<Iter>,
          concepts::any_of<
              camp::is_same<Compare,
                            operators::less<RAJA::detail::IterVal<Iter>>>,
              camp::is_same<Compare, operators::greater<
                                         RAJA::detail::IterVal<Iter>>>>>::value,
      "RAJA stable_sort<hip_exec> is only implemented for pointers to "
      "arithmetic types and RAJA::operators::less and "
      "RAJA::operators::greater.");

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief stable sort given range in ascending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Hip>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
stable(resources::Hip hip_res,
       ::RAJA::policy::hip::
           hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
       Iter begin,
       Iter end,
       operators::less<RAJA::detail::IterVal<Iter>>)
{
  hipStream_t stream = hip_res.get_stream();

  using R = RAJA::detail::IterVal<Iter>;

  int len       = std::distance(begin, end);
  int begin_bit = 0;
  int end_bit   = sizeof(R) * CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = hip::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  detail::double_buffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys, d_temp_storage,
                                temp_storage_bytes, d_keys, len, begin_bit,
                                end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys, d_temp_storage,
                                temp_storage_bytes, d_keys, len, begin_bit,
                                end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeys,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (detail::get_current(d_keys) == d_out)
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
        \brief stable sort given range in descending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Hip>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
stable(resources::Hip hip_res,
       ::RAJA::policy::hip::
           hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
       Iter begin,
       Iter end,
       operators::greater<RAJA::detail::IterVal<Iter>>)
{
  hipStream_t stream = hip_res.get_stream();

  using R = RAJA::detail::IterVal<Iter>;

  int len       = std::distance(begin, end);
  int begin_bit = 0;
  int end_bit   = sizeof(R) * CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = hip::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  detail::double_buffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys_desc, d_temp_storage,
                                temp_storage_bytes, d_keys, len, begin_bit,
                                end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_keys_desc, d_temp_storage,
                                temp_storage_bytes, d_keys, len, begin_bit,
                                end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortKeysDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 len, begin_bit, end_bit, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (detail::get_current(d_keys) == d_out)
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
        \brief static assert unimplemented sort
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    concepts::negate<concepts::all_of<
        type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
        std::is_pointer<Iter>,
        concepts::any_of<
            camp::is_same<Compare,
                          operators::less<RAJA::detail::IterVal<Iter>>>,
            camp::is_same<Compare,
                          operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
unstable(resources::Hip hip_res,
         ::RAJA::policy::hip::
             hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
         Iter,
         Iter,
         Compare)
{
  static_assert(
      concepts::all_of<
          type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
          std::is_pointer<Iter>,
          concepts::any_of<
              camp::is_same<Compare,
                            operators::less<RAJA::detail::IterVal<Iter>>>,
              camp::is_same<Compare, operators::greater<
                                         RAJA::detail::IterVal<Iter>>>>>::value,
      "RAJA sort<hip_exec> is only implemented for pointers to arithmetic "
      "types and RAJA::operators::less and RAJA::operators::greater.");

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief sort given range in ascending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Hip>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
unstable(resources::Hip hip_res,
         ::RAJA::policy::hip::
             hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
         Iter begin,
         Iter end,
         operators::less<RAJA::detail::IterVal<Iter>> comp)
{
  return stable(hip_res, p, begin, end, comp);
}

/*!
        \brief sort given range in descending order
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename Iter>
concepts::enable_if_t<resources::EventProxy<resources::Hip>,
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>>
unstable(resources::Hip hip_res,
         ::RAJA::policy::hip::
             hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
         Iter begin,
         Iter end,
         operators::greater<RAJA::detail::IterVal<Iter>> comp)
{
  return stable(hip_res, p, begin, end, comp);
}

/*!
        \brief static assert unimplemented stable sort pairs
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
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
stable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    KeyIter,
    KeyIter,
    ValIter,
    Compare)
{
  static_assert(std::is_pointer<KeyIter>::value,
                "stable_sort_pairs<hip_exec> is only implemented for pointers");
  static_assert(std::is_pointer<ValIter>::value,
                "stable_sort_pairs<hip_exec> is only implemented for pointers");
  using K = RAJA::detail::IterVal<KeyIter>;
  static_assert(
      type_traits::is_arithmetic<K>::value,
      "stable_sort_pairs<hip_exec> is only implemented for arithmetic types");
  static_assert(
      concepts::any_of<camp::is_same<Compare, operators::less<K>>,
                       camp::is_same<Compare, operators::greater<K>>>::value,
      "stable_sort_pairs<hip_exec> is only implemented for "
      "RAJA::operators::less or RAJA::operators::greater");

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief stable sort given range of pairs in ascending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
stable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    operators::less<RAJA::detail::IterVal<KeyIter>>)
{
  hipStream_t stream = hip_res.get_stream();

  using K = RAJA::detail::IterVal<KeyIter>;
  using V = RAJA::detail::IterVal<ValIter>;

  int len       = std::distance(keys_begin, keys_end);
  int begin_bit = 0;
  int end_bit   = sizeof(K) * CHAR_BIT;

  // Allocate temporary storage for the output arrays
  K* d_keys_out = hip::device_mempool_type::getInstance().malloc<K>(len);
  V* d_vals_out = hip::device_mempool_type::getInstance().malloc<V>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the keys_begin and vals_begin buffers
  detail::double_buffer<K> d_keys(keys_begin, d_keys_out);
  detail::double_buffer<V> d_vals(vals_begin, d_vals_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs, d_temp_storage,
                                temp_storage_bytes, d_keys, d_vals, len,
                                begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs, d_temp_storage,
                                temp_storage_bytes, d_keys, d_vals, len,
                                begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairs,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (detail::get_current(d_keys) == d_keys_out)
  {

    // copy keys
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, keys_begin, d_keys_out,
                                  len * sizeof(K), hipMemcpyDefault, stream);
  }
  if (detail::get_current(d_vals) == d_vals_out)
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

/*!
        \brief stable sort given range of pairs in descending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
stable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    operators::greater<RAJA::detail::IterVal<KeyIter>>)
{
  hipStream_t stream = hip_res.get_stream();

  using K = RAJA::detail::IterVal<KeyIter>;
  using V = RAJA::detail::IterVal<ValIter>;

  int len       = std::distance(keys_begin, keys_end);
  int begin_bit = 0;
  int end_bit   = sizeof(K) * CHAR_BIT;

  // Allocate temporary storage for the output arrays
  K* d_keys_out = hip::device_mempool_type::getInstance().malloc<K>(len);
  V* d_vals_out = hip::device_mempool_type::getInstance().malloc<V>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the keys_begin and vals_begin buffers
  detail::double_buffer<K> d_keys(keys_begin, d_keys_out);
  detail::double_buffer<V> d_vals(vals_begin, d_vals_out);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs_desc,
                                d_temp_storage, temp_storage_bytes, d_keys,
                                d_vals, len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::radix_sort_pairs_desc,
                                d_temp_storage, temp_storage_bytes, d_keys,
                                d_vals, len, begin_bit, end_bit, stream);
#elif defined(__CUDACC__)
  CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceRadixSort::SortPairsDescending,
                                 d_temp_storage, temp_storage_bytes, d_keys,
                                 d_vals, len, begin_bit, end_bit, stream);
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (detail::get_current(d_keys) == d_keys_out)
  {

    // copy keys
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpyAsync, keys_begin, d_keys_out,
                                  len * sizeof(K), hipMemcpyDefault, stream);
  }
  if (detail::get_current(d_vals) == d_vals_out)
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

/*!
        \brief static assert unimplemented sort pairs
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
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
unstable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async>,
    KeyIter,
    KeyIter,
    ValIter,
    Compare)
{
  static_assert(std::is_pointer<KeyIter>::value,
                "sort_pairs<hip_exec> is only implemented for pointers");
  static_assert(std::is_pointer<ValIter>::value,
                "sort_pairs<hip_exec> is only implemented for pointers");
  using K = RAJA::detail::IterVal<KeyIter>;
  static_assert(
      type_traits::is_arithmetic<K>::value,
      "sort_pairs<hip_exec> is only implemented for arithmetic types");
  static_assert(
      concepts::any_of<camp::is_same<Compare, operators::less<K>>,
                       camp::is_same<Compare, operators::greater<K>>>::value,
      "sort_pairs<hip_exec> is only implemented for RAJA::operators::less or "
      "RAJA::operators::greater");

  return resources::EventProxy<resources::Hip>(hip_res);
}

/*!
        \brief stable sort given range of pairs in ascending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
unstable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    operators::less<RAJA::detail::IterVal<KeyIter>> comp)
{
  return stable_pairs(hip_res, p, keys_begin, keys_end, vals_begin, comp);
}

/*!
        \brief stable sort given range of pairs in descending order of keys
*/
template<typename IterationMapping,
         typename IterationGetter,
         typename Concretizer,
         bool Async,
         typename KeyIter,
         typename ValIter>
concepts::enable_if_t<
    resources::EventProxy<resources::Hip>,
    type_traits::is_arithmetic<RAJA::detail::IterVal<KeyIter>>,
    std::is_pointer<KeyIter>,
    std::is_pointer<ValIter>>
unstable_pairs(
    resources::Hip hip_res,
    ::RAJA::policy::hip::
        hip_exec<IterationMapping, IterationGetter, Concretizer, Async> p,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    operators::greater<RAJA::detail::IterVal<KeyIter>> comp)
{
  return stable_pairs(hip_res, p, keys_begin, keys_end, vals_begin, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
