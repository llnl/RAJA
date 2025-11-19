/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for CUDA
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_atomic_HPP
#define RAJA_policy_cuda_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <stdexcept>
#include <type_traits>

#if __CUDA__ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 11 &&                     \
    __CUDACC_VER_MINOR__ >= 6
#define RAJA_ENABLE_CUDA_ATOMIC_REF
#endif

#if defined(RAJA_ENABLE_CUDA_ATOMIC_REF)
#include <cuda/atomic>
#endif

#include "camp/list.hpp"

#include "RAJA/policy/sequential/atomic.hpp"
#include "RAJA/policy/atomic_builtin.hpp"
#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/atomic.hpp"
#endif

#include "RAJA/util/EnableIf.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"

// TODO: When we can use if constexpr in C++17, this file can be cleaned up


namespace RAJA
{


namespace detail
{


/*!
 * Type trait for determining if atomic operators should be implemented
 * using builtin functions. This type trait can be used for a lot of atomic
 * operators. More specific type traits are added when needed, such as
 * cuda_useBuiltinExchange below.
 */
template<typename T>
struct cuda_useBuiltinCommon
{
  static constexpr bool value = std::is_same<T, int>::value ||
                                std::is_same<T, unsigned int>::value ||
                                std::is_same<T, unsigned long long>::value;
};

/*!
 * Type trait for determining if atomic operators should be implemented
 * by reinterpreting inputs to types that the builtin functions support.
 * This type trait can be used for a lot of atomic operators. More specific
 * type traits are added when needed, such as cuda_useReinterpretExchange
 * below.
 */
template<typename T>
struct cuda_useReinterpretCommon
{
  static constexpr bool value = !cuda_useBuiltinCommon<T>::value &&
                                (sizeof(T) == sizeof(unsigned int) ||
                                 sizeof(T) == sizeof(unsigned long long));

  using type = std::conditional_t<sizeof(T) == sizeof(unsigned int),
                                  unsigned int,
                                  unsigned long long>;
};

/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using cuda_useReinterpretCommon_t = typename cuda_useReinterpretCommon<T>::type;

/*!
 * Performs an atomic bitwise or using a builtin function. Stores the new value
 * in the given address and returns the old value.
 *
 * This overload using builtin functions is used to implement atomic loads
 * under some build configurations.
 */
template<typename T,
         std::enable_if_t<cuda_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicOr(T* acc, T value)
{
  return ::atomicOr(acc, value);
}

/*!
 * Atomic exchange
 */

/*!
 * Type trait for determining if the exchange operator should be implemented
 * using a builtin
 */
template<typename T>
struct cuda_useBuiltinExchange
{
  static constexpr bool value = std::is_same<T, int>::value ||
                                std::is_same<T, unsigned int>::value ||
                                std::is_same<T, unsigned long long>::value ||
                                std::is_same<T, float>::value;
};

/*!
 * Type trait for determining if the exchange operator should be implemented
 * by reinterpreting inputs to types that the builtin exchange supports
 */
template<typename T>
struct cuda_useReinterpretExchange
{
  static constexpr bool value = !cuda_useBuiltinExchange<T>::value &&
                                (sizeof(T) == sizeof(unsigned int) ||
                                 sizeof(T) == sizeof(unsigned long long));

  using type = std::conditional_t<sizeof(T) == sizeof(unsigned int),
                                  unsigned int,
                                  unsigned long long>;
};

/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using cuda_useReinterpretExchange_t =
    typename cuda_useReinterpretExchange<T>::type;

/*!
 * Performs an atomic exchange using a builtin function. Stores the new value
 * in the given address and returns the old value.
 */
template<typename T,
         std::enable_if_t<cuda_useBuiltinExchange<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicExchange(T* acc, T value)
{
  return ::atomicExch(acc, value);
}

/*!
 * Performs an atomic exchange using a reinterpret cast. Stores the new value
 * in the given address and returns the old value.
 */
template<typename T,
         std::enable_if_t<cuda_useReinterpretExchange<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicExchange(T* acc, T value)
{
  using R = cuda_useReinterpretExchange_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(cuda_atomicExchange(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(value)));
}

/*!
 * Atomic load and store
 */
#if defined(RAJA_ENABLE_CUDA_ATOMIC_REF)

template<typename T>
RAJA_INLINE __device__ T cuda_atomicLoad(T* acc)
{
  return cuda::atomic_ref<T, cuda::thread_scope_device>(*acc).load(
      cuda::memory_order_relaxed {});
}

template<typename T>
RAJA_INLINE __device__ void cuda_atomicStore(T* acc, T value)
{
  cuda::atomic_ref<T, cuda::thread_scope_device>(*acc).store(
      value, cuda::memory_order_relaxed {});
}

#else

template<typename T,
         std::enable_if_t<cuda_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicLoad(T* acc)
{
  return cuda_atomicOr(acc, static_cast<T>(0));
}

template<typename T,
         std::enable_if_t<cuda_useReinterpretCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicLoad(T* acc)
{
  using R = cuda_useReinterpretCommon_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(
      cuda_atomicLoad(reinterpret_cast<R*>(acc)));
}

template<typename T>
RAJA_INLINE __device__ void cuda_atomicStore(T* acc, T value)
{
  cuda_atomicExchange(acc, value);
}

#endif


/*!
 * Atomic compare and swap
 */

/*!
 * Type trait for determining if the compare and swap operator should be
 * implemented using a builtin
 */
template<typename T>
struct cuda_useBuiltinCAS
{
  static constexpr bool value =
#if __CUDA_ARCH__ >= 700
      std::is_same<T, unsigned short int>::value ||
#endif
      std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
      std::is_same<T, unsigned long long>::value;
};

/*!
 * Type trait for determining if the compare and swap operator should be
 * implemented by reinterpreting inputs to types that the builtin compare
 * and swap supports
 */
template<typename T>
struct cuda_useReinterpretCAS
{
  static constexpr bool value = !cuda_useBuiltinCAS<T>::value &&
                                (
#if __CUDA_ARCH__ >= 700
                                    sizeof(T) == sizeof(unsigned short) ||
#endif
                                    sizeof(T) == sizeof(unsigned int) ||
                                    sizeof(T) == sizeof(unsigned long long));

  using type =
#if __CUDA_ARCH__ >= 700
      std::conditional_t<sizeof(T) == sizeof(unsigned short),
                         unsigned short,
#endif
                         std::conditional_t<sizeof(T) == sizeof(unsigned int),
                                            unsigned int,
                                            unsigned long long>
#if __CUDA_ARCH__ >= 700
                         >
#endif
      ;
};

/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using cuda_useReinterpretCAS_t = typename cuda_useReinterpretCAS<T>::type;

template<typename T,
         std::enable_if_t<cuda_useBuiltinCAS<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T* acc, T compare, T value)
{
  return ::atomicCAS(acc, compare, value);
}

template<typename T,
         std::enable_if_t<cuda_useReinterpretCAS<T>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T* acc, T compare, T value)
{
  using R = cuda_useReinterpretCAS_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(cuda_atomicCAS(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(compare),
      RAJA::util::reinterp_A_as_B<T, R>(value)));
}

/*!
 * Equality comparison for compare and swap loop. Converts to the underlying
 * integral type to avoid cases where the values will never compare equal
 * (most notably, NaNs).
 */
template<typename T,
         std::enable_if_t<cuda_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ bool cuda_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

template<typename T,
         std::enable_if_t<cuda_useReinterpretCommon<T>::value, bool> = true>
RAJA_INLINE __device__ bool cuda_atomicCAS_equal(const T& a, const T& b)
{
  using R = cuda_useReinterpretCommon_t<T>;

  return cuda_atomicCAS_equal(RAJA::util::reinterp_A_as_B<T, R>(a),
                              RAJA::util::reinterp_A_as_B<T, R>(b));
}

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator.
 * Implementation uses the existing CUDA supplied unsigned 32-bit or 64-bit CAS
 * operator. Returns the OLD value that was replaced by the result of this
 * operation.
 */
template<typename T, typename Oper>
RAJA_INLINE __device__ T cuda_atomicCAS_loop(T* acc, Oper&& oper)
{
  T old = cuda_atomicLoad(acc);
  T expected;

  do
  {
    expected = old;
    old      = cuda_atomicCAS(acc, expected, oper(expected));
  } while (!cuda_atomicCAS_equal(old, expected));

  return old;
}

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator with
 * short-circuiting. Implementation uses the existing CUDA supplied unsigned
 * 32-bit or 64-bit CAS operator. Returns the OLD value that was replaced by the
 * result of this operation.
 */
template<typename T, typename Oper, typename ShortCircuit>
RAJA_INLINE __device__ T cuda_atomicCAS_loop(T* acc,
                                             Oper&& oper,
                                             ShortCircuit&& sc)
{
  T old = cuda_atomicLoad(acc);

  if (sc(old))
  {
    return old;
  }

  T expected;

  do
  {
    expected = old;
    old      = cuda_atomicCAS(acc, expected, oper(expected));
  } while (!cuda_atomicCAS_equal(old, expected) && !sc(old));

  return old;
}

/*!
 * Atomic addition
 */
using cuda_atomicAdd_builtin_types = ::camp::list<int,
                                                  unsigned int,
                                                  unsigned long long int,
                                                  float
#if __CUDA_ARCH__ >= 600
                                                  ,
                                                  double
#endif
                                                  >;

template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicAdd_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicAdd(T* acc, T value)
{
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old + value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAdd(T* acc, T value)
{
  return ::atomicAdd(acc, value);
}

/*!
 * Atomic subtract
 */
using cuda_atomicSub_builtin_types = cuda_atomicAdd_builtin_types;

using cuda_atomicSub_via_Sub_builtin_types = ::camp::list<int, unsigned int>;

using cuda_atomicSub_via_Add_builtin_types =
    ::camp::list<unsigned long long int,
                 float
#if __CUDA_ARCH__ >= 600
                 ,
                 double
#endif
                 >;

template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicSub_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T* acc, T value)
{
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old - value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicSub_via_Sub_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T* acc, T value)
{
  return ::atomicSub(acc, value);
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicSub_via_Add_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T* acc, T value)
{
  return ::atomicAdd(acc, -value);
}

/*!
 * Atomic min/max
 */
using cuda_atomicMinMax_builtin_types = ::camp::list<int,
                                                     unsigned int
#if __CUDA_ARCH__ >= 500
                                                     ,
                                                     long long int,
                                                     unsigned long long int
#endif
                                                     >;


/*!
 * Atomic min
 */
template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicMinMax_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicMin(T* acc, T value)
{
  return cuda_atomicCAS_loop(
      acc,
      [value](T old) {
        return value < old ? value : old;
      },
      [value](T current) {
        return current <= value;
      });
}

template<typename T,
         RAJA::util::enable_if_is_any_of<T, cuda_atomicMinMax_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicMin(T* acc, T value)
{
  return ::atomicMin(acc, value);
}

/*!
 * Atomic max
 */
template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicMinMax_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicMax(T* acc, T value)
{
  return cuda_atomicCAS_loop(
      acc,
      [value](T old) {
        return old < value ? value : old;
      },
      [value](T current) {
        return value <= current;
      });
}

template<typename T,
         RAJA::util::enable_if_is_any_of<T, cuda_atomicMinMax_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicMax(T* acc, T value)
{
  return ::atomicMax(acc, value);
}

/*!
 * Atomic increment/decrement with reset
 */
using cuda_atomicIncDecReset_builtin_types = ::camp::list<unsigned int>;

/*!
 * Atomic increment with reset
 */
template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, cuda_atomicIncDecReset_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicInc(T* acc, T value)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicIncDecReset_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicInc(T* acc, T value)
{
  return ::atomicInc(acc, value);
}

/*!
 * Atomic increment (implemented in terms of atomic addition)
 */
template<typename T>
RAJA_INLINE __device__ T cuda_atomicInc(T* acc)
{
  return cuda_atomicAdd(acc, static_cast<T>(1));
}

/*!
 * Atomic decrement with reset
 */
template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, cuda_atomicIncDecReset_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicDec(T* acc, T value)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old == static_cast<T>(0) || value < old ? value
                                                   : old - static_cast<T>(1);
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicIncDecReset_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T cuda_atomicDec(T* acc, T value)
{
  return ::atomicDec(acc, value);
}

/*!
 * Atomic decrement (implemented in terms of atomic subtraction)
 */
template<typename T>
RAJA_INLINE __device__ T cuda_atomicDec(T* acc)
{
  return cuda_atomicSub(acc, static_cast<T>(1));
}

/*!
 * Atomic bitwise functions (and, or, xor)
 */
using cuda_atomicBit_builtin_types =
    ::camp::list<int, unsigned int, unsigned long long int>;

/*!
 * Atomic and
 */
template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicAnd(T* acc, T value)
{
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old & value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAnd(T* acc, T value)
{
  return ::atomicAnd(acc, value);
}

/*!
 * Atomic or
 */
template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicOr(T* acc, T value)
{
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old | value;
  });
}

/*!
 * Atomic or via builtin functions was implemented much earlier since atomicLoad
 * may depend on it.
 */


/*!
 * Atomic xor
 */
template<typename T,
         RAJA::util::enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* =
             nullptr>
RAJA_INLINE __device__ T cuda_atomicXor(T* acc, T value)
{
  return cuda_atomicCAS_loop(acc, [value](T old) {
    return old ^ value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicXor(T* acc, T value)
{
  return ::atomicXor(acc, value);
}


}  // namespace detail

/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in cuda device code and non-atomic otherwise
 */
RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicLoad(cuda_atomic_explicit<host_policy>,
                                          T* acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicLoad(acc);
#else
  return RAJA::atomicLoad(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE void atomicStore(cuda_atomic_explicit<host_policy>,
                                              T* acc,
                                              T value)
{
#ifdef __CUDA_ARCH__
  detail::cuda_atomicStore(acc, value);
#else
  RAJA::atomicStore(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAdd(acc, value);
#else
  return RAJA::atomicAdd(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicSub(acc, value);
#else
  return RAJA::atomicSub(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMin(acc, value);
#else
  return RAJA::atomicMin(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMax(acc, value);
#else
  return RAJA::atomicMax(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return detail::cuda_atomicInc(acc, value);
#else
  return RAJA::atomicInc(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(cuda_atomic_explicit<host_policy>,
                                         T* acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicInc(acc);
#else
  return RAJA::atomicInc(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return detail::cuda_atomicDec(acc, value);
#else
  return RAJA::atomicDec(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(cuda_atomic_explicit<host_policy>,
                                         T* acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicDec(acc);
#else
  return RAJA::atomicDec(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAnd(acc, value);
#else
  return RAJA::atomicAnd(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(cuda_atomic_explicit<host_policy>,
                                        T* acc,
                                        T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicOr(acc, value);
#else
  return RAJA::atomicOr(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(cuda_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicXor(acc, value);
#else
  return RAJA::atomicXor(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicExchange(cuda_atomic_explicit<host_policy>,
                                              T* acc,
                                              T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicExchange(acc, value);
#else
  return RAJA::atomicExchange(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(cuda_atomic_explicit<host_policy>, T* acc, T compare, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicCAS(acc, compare, value);
#else
  return RAJA::atomicCAS(host_policy {}, acc, compare, value);
#endif
}

}  // namespace RAJA


#endif  // RAJA_ENABLE_CUDA
#endif  // guard
