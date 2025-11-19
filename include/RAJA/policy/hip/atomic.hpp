/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for HIP
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hip_atomic_HPP
#define RAJA_policy_hip_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "hip/hip_runtime.h"

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

using hip_atomicCommon_builtin_types =
    ::camp::list<int, unsigned int, unsigned long long>;

/*!
 * Type trait for determining if atomic operators should be implemented
 * using builtin functions. This type trait can be used for a lot of atomic
 * operators. More specific type traits are added when needed, such as
 * hip_useBuiltinExchange below.
 */
template<typename T>
struct hip_useBuiltinCommon
{
  static constexpr bool value = std::is_same<T, int>::value ||
                                std::is_same<T, unsigned int>::value ||
                                std::is_same<T, unsigned long long>::value;
};

/*!
 * Type trait for determining if atomic operators should be implemented
 * by reinterpreting inputs to types that the builtin functions support.
 * This type trait can be used for a lot of atomic operators. More specific
 * type traits are added when needed, such as hip_useReinterpretExchange
 * below.
 */
template<typename T>
struct hip_useReinterpretCommon
{
  static constexpr bool value = !hip_useBuiltinCommon<T>::value &&
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
using hip_useReinterpretCommon_t = typename hip_useReinterpretCommon<T>::type;

/*!
 * Performs an atomic bitwise or using a builtin function. Stores the new value
 * in the given address and returns the old value.
 *
 * This overload using builtin functions is used to implement atomic loads
 * under some build configurations.
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicOr(T* acc, T value)
{
  return ::atomicOr(acc, value);
}

/*!
 * Type trait for determining if the exchange operator should be implemented
 * using a builtin
 */
template<typename T>
struct hip_useBuiltinExchange
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
struct hip_useReinterpretExchange
{
  static constexpr bool value = !hip_useBuiltinExchange<T>::value &&
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
using hip_useReinterpretExchange_t =
    typename hip_useReinterpretExchange<T>::type;

/*!
 * Performs an atomic exchange using a builtin function. Stores the new value
 * in the given address and returns the old value.
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinExchange<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T* acc, T value)
{
  return ::atomicExch(acc, value);
}

/*!
 * Performs an atomic exchange using a reinterpret cast. Stores the new value
 * in the given address and returns the old value.
 */
template<typename T,
         std::enable_if_t<hip_useReinterpretExchange<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T* acc, T value)
{
  using R = hip_useReinterpretExchange_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(hip_atomicExchange(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(value)));
}


#if defined(__has_builtin) &&                                                  \
    (__has_builtin(__hip_atomic_load) || __has_builtin(__hip_atomic_store))

/*!
 * Type trait for determining if the operator should be implemented
 * using an intrinsic
 */
template<typename T>
struct hip_useBuiltinLoad
{
  static constexpr bool value =
      (std::is_integral<T>::value || std::is_enum<T>::value) &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};

template<typename T>
using hip_useBuiltinStore = hip_useBuiltinLoad<T>;

/*!
 * Type trait for determining if the operator should be implemented
 * by reinterpreting inputs to types that intrinsics support
 */
template<typename T>
struct hip_useReinterpretLoad
{
  static constexpr bool value = !std::is_integral<T>::value &&
                                !std::is_enum<T>::value &&
                                ((sizeof(T) == 1
#if !defined(UINT8_MAX)
                                  && sizeof(unsigned char) == 1
#endif
                                  ) ||
                                 (sizeof(T) == 2
#if !defined(UINT16_MAX)
                                  && sizeof(unsigned short) == 2
#endif
                                  ) ||
                                 (sizeof(T) == 4
#if !defined(UINT32_MAX)
                                  && sizeof(unsigned int) == 4
#endif
                                  ) ||
                                 (sizeof(T) == 8
#if !defined(UINT64_MAX)
                                  && sizeof(unsigned long long) == 8
#endif
                                  ));

  using type =
      std::conditional_t<sizeof(T) == 1,
#if defined(UINT8_MAX)
                         uint8_t,
#else
                         unsigned char,
#endif
                         std::conditional_t<sizeof(T) == 2,
#if defined(UINT16_MAX)
                                            uint16_t,
#else
                                            unsigned short,
#endif
                                            std::conditional_t<sizeof(T) == 4,
#if defined(UINT32_MAX)
                                                               uint32_t,
#else
                                                               unsigned int,
#endif
#if defined(UINT64_MAX)
                                                               uint64_t>>>;
#else
                                                               unsigned long long>>>;
#endif
};

template<typename T>
using hip_useReinterpretStore = hip_useReinterpretLoad<T>;

#else

template<typename T>
using hip_useBuiltinLoad = hip_useBuiltinCommon<T>;

template<typename T>
using hip_useBuiltinStore = hip_useBuiltinExchange<T>;

/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using hip_useReinterpretLoad = hip_useReinterpretCommon<T>;

template<typename T>
using hip_useReinterpretStore = hip_useReinterpretExchange<T>;

#endif

/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using hip_useReinterpretLoad_t = typename hip_useReinterpretLoad<T>::type;

template<typename T>
using hip_useReinterpretStore_t = typename hip_useReinterpretStore<T>::type;

/*!
 * Atomic load
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinLoad<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T* acc)
{
#if defined(__has_builtin) && __has_builtin(__hip_atomic_load)
  return __hip_atomic_load(acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return hip_atomicOr(acc, static_cast<T>(0));
#endif
}

template<typename T,
         std::enable_if_t<hip_useReinterpretLoad<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T* acc)
{
  using R = hip_useReinterpretLoad_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(
      hip_atomicLoad(reinterpret_cast<R*>(acc)));
}

/*!
 * Atomic store
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinStore<T>::value, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T* acc, T value)
{
#if defined(__has_builtin) && __has_builtin(__hip_atomic_store)
  __hip_atomic_store(acc, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  hip_atomicExchange(acc, value);
#endif
}

template<typename T,
         std::enable_if_t<hip_useReinterpretStore<T>::value, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T* acc, T value)
{
  using R = hip_useReinterpretStore_t<T>;

  hip_atomicStore(reinterpret_cast<R*>(acc),
                  RAJA::util::reinterp_A_as_B<T, R>(value));
}

/*!
 * Hip atomicCAS using builtin function
 *
 * Returns the old value in memory before this operation.
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicCAS(T* acc, T compare, T value)
{
  return ::atomicCAS(acc, compare, value);
}

/*!
 * Hip atomicCAS using reinterpret cast
 *
 * Returns the old value in memory before this operation.
 */
template<typename T,
         std::enable_if_t<hip_useReinterpretCommon<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicCAS(T* acc, T compare, T value)
{
  using R = hip_useReinterpretCommon_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(hip_atomicCAS(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(compare),
      RAJA::util::reinterp_A_as_B<T, R>(value)));
}

/*!
 * Equality comparison for compare and swap loop. Converts to the underlying
 * integral type to avoid cases where the values will never compare equal
 * (most notably, NaNs).
 */
template<typename T,
         std::enable_if_t<hip_useBuiltinCommon<T>::value, bool> = true>
RAJA_INLINE __device__ bool hip_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

template<typename T,
         std::enable_if_t<hip_useReinterpretCommon<T>::value, bool> = true>
RAJA_INLINE __device__ bool hip_atomicCAS_equal(const T& a, const T& b)
{
  using R = hip_useReinterpretCommon_t<T>;

  return hip_atomicCAS_equal(RAJA::util::reinterp_A_as_B<T, R>(a),
                             RAJA::util::reinterp_A_as_B<T, R>(b));
}

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator.
 * Implementation uses the existing HIP supplied unsigned 32-bit or 64-bit CAS
 * operator. Returns the OLD value that was replaced by the result of this
 * operation.
 */
template<typename T, typename Oper>
RAJA_INLINE __device__ T hip_atomicCAS_loop(T* acc, Oper&& oper)
{
  T old = hip_atomicLoad(acc);
  T expected;

  do
  {
    expected = old;
    old      = hip_atomicCAS(acc, expected, oper(expected));
  } while (!hip_atomicCAS_equal(old, expected));

  return old;
}

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator with
 * short-circuiting. Implementation uses the existing HIP supplied unsigned
 * 32-bit or 64-bit CAS operator. Returns the OLD value that was replaced by the
 * result of this operation.
 */
template<typename T, typename Oper, typename ShortCircuit>
RAJA_INLINE __device__ T hip_atomicCAS_loop(T* acc,
                                            Oper&& oper,
                                            ShortCircuit&& sc)
{
  T old = hip_atomicLoad(acc);

  if (sc(old))
  {
    return old;
  }

  T expected;

  do
  {
    expected = old;
    old      = hip_atomicCAS(acc, expected, oper(expected));
  } while (!hip_atomicCAS_equal(old, expected) && !sc(old));

  return old;
}

/*!
 * Atomic addition
 */

/*!
 * List of types where HIP builtin atomics are used to implement atomicAdd.
 */
using hip_atomicAdd_builtin_types = ::camp::list<int,
                                                 unsigned int,
                                                 unsigned long long,
                                                 float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
                                                 ,
                                                 double
#endif
                                                 >;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return old + value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T* acc, T value)
{
  return ::atomicAdd(acc, value);
}

/*!
 * Atomic subtraction
 */

/*!
 * List of types where HIP builtin atomics are used to implement atomicSub.
 */
using hip_atomicSub_builtin_types = ::camp::list<int,
                                                 unsigned int,
                                                 unsigned long long,
                                                 float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
                                                 ,
                                                 double
#endif
                                                 >;

/*!
 * List of types where HIP builtin atomicSub is used to implement atomicSub.
 *
 * Avoid multiple definition errors by including the previous list type here
 * to ensure these lists have different types.
 */
using hip_atomicSub_via_Sub_builtin_types = ::camp::list<int, unsigned int>;

/*!
 * List of types where HIP builtin atomicAdd is used to implement atomicSub.
 *
 * Avoid multiple definition errors by including the previous list type here
 * to ensure these lists have different types.
 */
using hip_atomicSub_via_Add_builtin_types = ::camp::list<unsigned long long,
                                                         float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
                                                         ,
                                                         double
#endif
                                                         >;

/*!
 * HIP atomicSub compare and swap loop implementation.
 */
template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicSub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return old - value;
  });
}

/*!
 * HIP atomicSub builtin implementation.
 */
template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicSub_via_Sub_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T* acc, T value)
{
  return ::atomicSub(acc, value);
}

/*!
 * HIP atomicSub via atomicAdd builtin implementation.
 */
template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicSub_via_Add_builtin_types>* =
        nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T* acc, T value)
{
  return ::atomicAdd(acc, -value);
}

/*!
 * Atomic minimum
 */
using hip_atomicMin_builtin_types = hip_atomicCommon_builtin_types;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T* acc, T value)
{
  return hip_atomicCAS_loop(
      acc,
      [value](T old) {
        return value < old ? value : old;
      },
      [value](T current) {
        return current <= value;
      });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T* acc, T value)
{
  return ::atomicMin(acc, value);
}

/*!
 * Atomic maximum
 */
using hip_atomicMax_builtin_types = hip_atomicCommon_builtin_types;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T* acc, T value)
{
  return hip_atomicCAS_loop(
      acc,
      [value](T old) {
        return old < value ? value : old;
      },
      [value](T current) {
        return value <= current;
      });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T* acc, T value)
{
  return ::atomicMax(acc, value);
}

/*!
 * Atomic increment with reset
 */
template<typename T>
RAJA_INLINE __device__ T hip_atomicInc(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

/*!
 * Atomic increment (implemented in terms of atomic addition)
 */
template<typename T>
RAJA_INLINE __device__ T hip_atomicInc(T* acc)
{
  return hip_atomicAdd(acc, static_cast<T>(1));
}

/*!
 * Atomic decrement with reset
 */
template<typename T>
RAJA_INLINE __device__ T hip_atomicDec(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return old == static_cast<T>(0) || value < old ? value
                                                   : old - static_cast<T>(1);
  });
}

/*!
 * Atomic decrement (implemented in terms of atomic subtraction)
 */
template<typename T>
RAJA_INLINE __device__ T hip_atomicDec(T* acc)
{
  return hip_atomicSub(acc, static_cast<T>(1));
}

/*!
 * Atomic and
 */
using hip_atomicAnd_builtin_types = hip_atomicCommon_builtin_types;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return old & value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T* acc, T value)
{
  return ::atomicAnd(acc, value);
}

/*!
 * Atomic or
 */
using hip_atomicOr_builtin_types = hip_atomicCommon_builtin_types;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
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
using hip_atomicXor_builtin_types = hip_atomicCommon_builtin_types;

template<
    typename T,
    RAJA::util::enable_if_is_none_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T* acc, T value)
{
  return hip_atomicCAS_loop(acc, [value](T old) {
    return old ^ value;
  });
}

template<
    typename T,
    RAJA::util::enable_if_is_any_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T* acc, T value)
{
  return ::atomicXor(acc, value);
}


}  // namespace detail

/*!
 * Catch-all policy passes off to HIP's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in hip device code and non-atomic otherwise
 */

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicLoad(hip_atomic_explicit<host_policy>,
                                          T* acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicLoad(acc);
#else
  return RAJA::atomicLoad(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE void atomicStore(hip_atomic_explicit<host_policy>,
                                              T* acc,
                                              T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  detail::hip_atomicStore(acc, value);
#else
  RAJA::atomicStore(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicAdd(acc, value);
#else
  return RAJA::atomicAdd(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicSub(acc, value);
#else
  return RAJA::atomicSub(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicMin(acc, value);
#else
  return RAJA::atomicMin(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicMax(acc, value);
#else
  return RAJA::atomicMax(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicInc(acc, value);
#else
  return RAJA::atomicInc(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(hip_atomic_explicit<host_policy>,
                                         T* acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicInc(acc);
#else
  return RAJA::atomicInc(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicDec(acc, value);
#else
  return RAJA::atomicDec(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(hip_atomic_explicit<host_policy>,
                                         T* acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicDec(acc);
#else
  return RAJA::atomicDec(host_policy {}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicAnd(acc, value);
#else
  return RAJA::atomicAnd(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(hip_atomic_explicit<host_policy>,
                                        T* acc,
                                        T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicOr(acc, value);
#else
  return RAJA::atomicOr(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(hip_atomic_explicit<host_policy>,
                                         T* acc,
                                         T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicXor(acc, value);
#else
  return RAJA::atomicXor(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T atomicExchange(hip_atomic_explicit<host_policy>,
                                              T* acc,
                                              T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicExchange(acc, value);
#else
  return RAJA::atomicExchange(host_policy {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(hip_atomic_explicit<host_policy>, T* acc, T compare, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicCAS(acc, compare, value);
#else
  return RAJA::atomicCAS(host_policy {}, acc, compare, value);
#endif
}

}  // namespace RAJA


#endif  // RAJA_ENABLE_HIP
#endif  // guard
