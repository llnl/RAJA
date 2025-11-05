/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic and builtin atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_builtin_HPP
#define RAJA_policy_atomic_builtin_HPP

#include "RAJA/config.hpp"

#include <cstdint>

#if defined(RAJA_COMPILER_MSVC) ||                                             \
    ((defined(_WIN32) || defined(_WIN64)) && defined(__INTEL_COMPILER))
#include <intrin.h>
#endif

#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"


#if defined(RAJA_ENABLE_HIP)
#define RAJA_DEVICE_HIP RAJA_HOST_DEVICE
#else
#define RAJA_DEVICE_HIP
#endif

namespace RAJA
{


//! Atomic policy that uses the compilers builtin __atomic_XXX routines
struct builtin_atomic
{};

namespace detail
{


#if defined(RAJA_COMPILER_MSVC) ||                                             \
    ((defined(_WIN32) || defined(_WIN64)) && defined(__INTEL_COMPILER))


/*!
 * Type trait for determining if the operator should be implemented
 * using an intrinsic
 */
template<typename T>
struct builtin_useIntrinsic
{
  static constexpr bool value =
      std::is_same<T, char>::value || std::is_same<T, short>::value ||
      std::is_same<T, long>::value || std::is_same<T, long long>::value;
};

/*!
 * Type trait for determining if the operator should be implemented
 * by reinterpreting inputs to types that intrinsics support
 */
template<typename T>
struct builtin_useReinterpret
{
  static constexpr bool value =
      !builtin_useIntrinsic<T>::value &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);

  using type = std::conditional_t<
      sizeof(T) == 1,
      char,
      std::conditional_t<sizeof(T) == 2,
                         short,
                         std::conditional_t<sizeof(T) == 4, long, long long>>>;
};

/*!
 * Type trait for determining if the operator should be implemented
 * using a compare and swap loop
 */
template<typename T>
struct builtin_useCAS
{
  static constexpr bool value =
      !builtin_useIntrinsic<T>::value &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};

/*!
 * Atomics implemented using intrinsics
 */


/*!
 * Atomic or using intrinsics
 */
RAJA_INLINE char builtin_atomicOr(char* acc, char value)
{
  return _InterlockedOr8(acc, value);
}

RAJA_INLINE uint8_t builtin_atomicOr(uint8_t* acc, uint8_t value)
{
  char return_val = _InterlockedOr8((char*)acc, *((char*)&value));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicOr(short* acc, short value)
{
  return _InterlockedOr16(acc, value);
}

RAJA_INLINE uint16_t builtin_atomicOr(uint16_t* acc, uint16_t value)
{
  short return_val = _InterlockedOr16((short*)acc, *((short*)&value));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicOr(long* acc, long value)
{
  return _InterlockedOr(acc, value);
}

RAJA_INLINE uint32_t builtin_atomicOr(uint32_t* acc, uint32_t value)
{
  long return_val = _InterlockedOr((long*)acc, *((long*)&value));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicOr(long long* acc, long long value)
{
  return _InterlockedOr64(acc, value);
}

RAJA_INLINE uint64_t builtin_atomicOr(uint64_t* acc, uint64_t value)
{
  long long return_val =
      _InterlockedOr64((long long*)acc, *((long long*)&value));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif

/*!
 * Atomic load using atomic or
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_INLINE T builtin_atomicLoad(T* acc)
{
  return builtin_atomicOr(acc, static_cast<T>(0));
}

/*!
 * Atomic exchange using intrinsics
 */
RAJA_INLINE char builtin_atomicExchange(char* acc, char value)
{
  return _InterlockedExchange8(acc, value);
}

RAJA_INLINE uint8_t builtin_atomicExchange(uint8_t* acc, uint8_t value)
{
  char return_val = _InterlockedExchange8((char*)acc, *((char*)&value));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicExchange(short* acc, short value)
{
  return _InterlockedExchange16(acc, value);
}

RAJA_INLINE uint16_t builtin_atomicExchange(uint16_t* acc, uint16_t value)
{
  short return_val = _InterlockedExchange16((short*)acc, *((short*)&value));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicExchange(long* acc, long value)
{
  return _InterlockedExchange(acc, value);
}

RAJA_INLINE uint32_t builtin_atomicExchange(uint32_t* acc, uint32_t value)
{
  long return_val = _InterlockedExchange((long*)acc, *((long*)&value));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicExchange(long long* acc, long long value)
{
  return _InterlockedExchange64(acc, value);
}

RAJA_INLINE uint64_t builtin_atomicExchange(uint64_t* acc, uint64_t value)
{
  long long return_val =
      _InterlockedExchange64((long long*)acc, *((long long*)&value));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


/*!
 * Atomic store using atomic exchange
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_INLINE void builtin_atomicStore(T* acc, T value)
{
  builtin_atomicExchange(acc, value);
}

/*!
 * Atomic compare and swap using intrinsics
 */
RAJA_INLINE char builtin_atomicCAS(char* acc, char compare, char value)
{
  return _InterlockedCompareExchange8(acc, value, compare);
}

RAJA_INLINE uint8_t builtin_atomicCAS(uint8_t* acc,
                                      uint8_t compare,
                                      uint8_t value)
{
  char return_val = _InterlockedCompareExchange8((char*)acc, *((char*)&value),
                                                 *((char*)&compare));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicCAS(short* acc, short compare, short value)
{
  return _InterlockedCompareExchange16(acc, value, compare);
}

RAJA_INLINE uint16_t builtin_atomicCAS(uint16_t* acc,
                                       uint16_t compare,
                                       uint16_t value)
{
  short return_val = _InterlockedCompareExchange16(
      (short*)acc, *((short*)&value), *((short*)&compare));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicCAS(long* acc, long compare, long value)
{
  return _InterlockedCompareExchange(acc, value, compare);
}

RAJA_INLINE uint32_t builtin_atomicCAS(uint32_t* acc,
                                       uint32_t compare,
                                       uint32_t value)
{
  long return_val = _InterlockedCompareExchange((long*)acc, *((long*)&value),
                                                *((long*)&compare));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicCAS(long long* acc,
                                        long long compare,
                                        long long value)
{
  return _InterlockedCompareExchange64(acc, value, compare);
}

RAJA_INLINE uint64_t builtin_atomicCAS(uint64_t* acc,
                                       uint64_t compare,
                                       uint64_t value)
{
  long long return_val = _InterlockedCompareExchange64(
      (long long*)acc, *((long long*)&value), *((long long*)&compare));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


/*!
 * Atomic addition using intrinsics
 */
RAJA_INLINE char builtin_atomicAdd(char* acc, char value)
{
  return _InterlockedExchangeAdd8(acc, value);
}

RAJA_INLINE uint8_t builtin_atomicAdd(uint8_t* acc, uint8_t value)
{
  char return_val = _InterlockedExchangeAdd8((char*)acc, *((char*)&value));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicAdd(short* acc, short value)
{
  return _InterlockedExchangeAdd16(acc, value);
}

RAJA_INLINE uint16_t builtin_atomicAdd(uint16_t* acc, uint16_t value)
{
  short return_val = _InterlockedExchangeAdd16((short*)acc, *((short*)&value));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicAdd(long* acc, long value)
{
  return _InterlockedExchangeAdd(acc, value);
}

RAJA_INLINE uint32_t builtin_atomicAdd(uint32_t* acc, uint32_t value)
{
  long return_val = _InterlockedExchangeAdd((long*)acc, *((long*)&value));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicAdd(long long* acc, long long value)
{
  return _InterlockedExchangeAdd64(acc, value);
}

RAJA_INLINE uint64_t builtin_atomicAdd(uint64_t* acc, uint64_t value)
{
  long long return_val =
      _InterlockedExchangeAdd64((long long*)acc, *((long long*)&value));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


/*!
 * Atomic subtraction using intrinsics
 */
RAJA_INLINE char builtin_atomicSub(char* acc, char value)
{
  return _InterlockedExchangeAdd8(acc, -value);
}

RAJA_INLINE uint8_t builtin_atomicSub(uint8_t* acc, uint8_t value)
{
  char neg_value  = -(*((char*)&(value)));
  char return_val = _InterlockedExchangeAdd8((char*)acc, neg_value);
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicSub(short* acc, short value)
{
  return _InterlockedExchangeAdd16(acc, -value);
}

RAJA_INLINE uint16_t builtin_atomicSub(uint16_t* acc, uint16_t value)
{
  short neg_value  = -(*((short*)&(value)));
  short return_val = _InterlockedExchangeAdd16((short*)acc, neg_value);
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicSub(long* acc, long value)
{
  return _InterlockedExchangeAdd(acc, -value);
}

RAJA_INLINE uint32_t builtin_atomicSub(uint32_t* acc, uint32_t value)
{
  long neg_value  = -(*((long*)&(value)));
  long return_val = _InterlockedExchangeAdd((long*)acc, neg_value);
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicSub(long long* acc, long long value)
{
  return _InterlockedExchangeAdd64(acc, -value);
}

RAJA_INLINE uint64_t builtin_atomicSub(uint64_t* acc, uint64_t value)
{
  long long neg_value  = -(*((long long*)&(value)));
  long long return_val = _InterlockedExchangeAdd64((long long*)acc, neg_value);
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


/*!
 * Atomic and using intrinsics
 */
RAJA_INLINE char builtin_atomicAnd(char* acc, char value)
{
  return _InterlockedAnd8(acc, value);
}

RAJA_INLINE uint8_t builtin_atomicAnd(uint8_t* acc, uint8_t value)
{
  char return_val = _InterlockedAnd8((char*)acc, *((char*)&value));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicAnd(short* acc, short value)
{
  return _InterlockedAnd16(acc, value);
}

RAJA_INLINE uint16_t builtin_atomicAnd(uint16_t* acc, uint16_t value)
{
  short return_val = _InterlockedAnd16((short*)acc, *((short*)&value));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicAnd(long* acc, long value)
{
  return _InterlockedAnd(acc, value);
}

RAJA_INLINE uint32_t builtin_atomicAnd(uint32_t* acc, uint32_t value)
{
  long return_val = _InterlockedAnd((long*)acc, *((long*)&value));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicAnd(long long* acc, long long value)
{
  return _InterlockedAnd64(acc, value);
}

RAJA_INLINE uint64_t builtin_atomicAnd(uint64_t* acc, uint64_t value)
{
  long long return_val =
      _InterlockedAnd64((long long*)acc, *((long long*)&value));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


/*!
 * Atomic xor using intrinsics
 */
RAJA_INLINE char builtin_atomicXor(char* acc, char value)
{
  return _InterlockedXor8(acc, value);
}

RAJA_INLINE uint8_t builtin_atomicXor(uint8_t* acc, uint8_t value)
{
  char return_val = _InterlockedXor8((char*)acc, *((char*)&value));
  return *(reinterpret_cast<uint8_t*>(&return_val));
}

RAJA_INLINE short builtin_atomicXor(short* acc, short value)
{
  return _InterlockedXor16(acc, value);
}

RAJA_INLINE uint16_t builtin_atomicXor(uint16_t* acc, uint16_t value)
{
  short return_val = _InterlockedXor16((short*)acc, *((short*)&value));
  return *(reinterpret_cast<uint16_t*>(&return_val));
}

RAJA_INLINE long builtin_atomicXor(long* acc, long value)
{
  return _InterlockedXor(acc, value);
}

RAJA_INLINE uint32_t builtin_atomicXor(uint32_t* acc, uint32_t value)
{
  long return_val = _InterlockedXor((long*)acc, *((long*)&value));
  return *(reinterpret_cast<uint32_t*>(&return_val));
}

#if defined(_WIN64)

RAJA_INLINE long long builtin_atomicXor(long long* acc, long long value)
{
  return _InterlockedXor64(acc, value);
}

RAJA_INLINE uint64_t builtin_atomicXor(uint64_t* acc, uint64_t value)
{
  long long return_val =
      _InterlockedXor64((long long*)acc, *((long long*)&value));
  return *(reinterpret_cast<uint64_t*>(&return_val));
}

#endif


#else  // else clause from if defined(RAJA_COMPILER_MSVC) ||
       // ((defined(_WIN32) || defined(_WIN64)) && defined(__INTEL_COMPILER))


/*!
 * Type trait for determining if the operator should be implemented
 * using an intrinsic
 */
template<typename T>
struct builtin_useIntrinsic
{
  static constexpr bool value =
      (std::is_integral<T>::value || std::is_enum<T>::value) &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};

/*!
 * Type trait for determining if the operator should be implemented
 * by reinterpreting inputs to types that intrinsics support
 */
template<typename T>
struct builtin_useReinterpret
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

/*!
 * Type trait for determining if the operator should be implemented
 * using a compare and swap loop
 */
template<typename T>
struct builtin_useCAS
{
  static constexpr bool value =
      !std::is_integral<T>::value && !std::is_enum<T>::value &&
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};

/*!
 * Atomics implemented using intrinsics
 */


/*!
 * Atomic load using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicLoad(T* acc)
{
  return __atomic_load_n(acc, __ATOMIC_RELAXED);
}

/*!
 * Atomic store using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE void builtin_atomicStore(T* acc, T value)
{
  __atomic_store_n(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic exchange using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicExchange(T* acc, T value)
{
  return __atomic_exchange_n(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic compare and swap using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T* acc, T compare, T value)
{
  __atomic_compare_exchange_n(acc, &compare, value, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
  return compare;
}

/*!
 * Atomic addition using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T* acc, T value)
{
  return __atomic_fetch_add(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic subtraction using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T* acc, T value)
{
  return __atomic_fetch_sub(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic and using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T* acc, T value)
{
  return __atomic_fetch_and(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic or using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T* acc, T value)
{
  return __atomic_fetch_or(acc, value, __ATOMIC_RELAXED);
}

/*!
 * Atomic xor using intrinsic
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T* acc, T value)
{
  return __atomic_fetch_xor(acc, value, __ATOMIC_RELAXED);
}


#endif  // RAJA_COMPILER_MSVC


/*!
 * Atomics implemented using reinterpret cast
 */


/*!
 * Alias for determining the integral type of the same size as the given type
 */
template<typename T>
using builtin_useReinterpret_t = typename builtin_useReinterpret<T>::type;

/*!
 * Atomic load using reinterpret cast
 */
template<typename T,
         std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicLoad(T* acc)
{
  using R = builtin_useReinterpret_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(
      builtin_atomicLoad(reinterpret_cast<R*>(acc)));
}

/*!
 * Atomic store using reinterpret cast
 */
template<typename T,
         std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE void builtin_atomicStore(T* acc, T value)
{
  using R = builtin_useReinterpret_t<T>;

  builtin_atomicStore(reinterpret_cast<R*>(acc),
                      RAJA::util::reinterp_A_as_B<T, R>(value));
}

/*!
 * Atomic exchange using reinterpret cast
 */
template<typename T,
         std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicExchange(T* acc, T value)
{
  using R = builtin_useReinterpret_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(builtin_atomicExchange(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(value)));
}

/*!
 * Atomic compare and swap using reinterpret cast
 */
template<typename T,
         std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T* acc, T compare, T value)
{
  using R = builtin_useReinterpret_t<T>;

  return RAJA::util::reinterp_A_as_B<R, T>(builtin_atomicCAS(
      reinterpret_cast<R*>(acc), RAJA::util::reinterp_A_as_B<T, R>(compare),
      RAJA::util::reinterp_A_as_B<T, R>(value)));
}

/*!
 * Implementation of compare and swap loop
 */


/*!
 * Equality comparison for compare and swap loop using types supported by
 * intrinsics.
 */
template<typename T,
         std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE bool builtin_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

/*!
 * Equality comparison for compare and swap loop using reinterpret cast.
 * Converts to the underlying integral type to avoid cases where the values
 * will never compare equal (most notably, NaNs).
 */
template<typename T,
         std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE bool builtin_atomicCAS_equal(const T& a, const T& b)
{
  using R = builtin_useReinterpret_t<T>;

  return builtin_atomicCAS_equal(RAJA::util::reinterp_A_as_B<T, R>(a),
                                 RAJA::util::reinterp_A_as_B<T, R>(b));
}

/*!
 * Generic impementation of any atomic 8, 16, 32, or 64 bit operator
 * that can be implemented using a builtin compare and swap primitive.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template<typename T, typename Oper>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS_loop(T* acc, Oper&& oper)
{
  T old = builtin_atomicLoad(acc);
  T expected;

  do
  {
    expected = old;
    old      = builtin_atomicCAS(acc, expected, oper(expected));
  } while (!builtin_atomicCAS_equal(old, expected));

  return old;
}

/*!
 * Generic impementation of any atomic 8, 16, 32, or 64 bit operator
 * that can be implemented using a builtin compare and swap primitive.
 * Uses short-circuiting for improved efficiency. Returns the OLD value
 * that was replaced by the result of this operation.
 */
template<typename T, typename Oper, typename ShortCircuit>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS_loop(T* acc,
                                                     Oper&& oper,
                                                     ShortCircuit&& sc)
{
  T old = builtin_atomicLoad(acc);

  if (sc(old))
  {
    return old;
  }

  T expected;

  do
  {
    expected = old;
    old      = builtin_atomicCAS(acc, expected, oper(expected));
  } while (!builtin_atomicCAS_equal(old, expected) && !sc(old));

  return old;
}

/*!
 * Atomics implemented using compare and swap loop
 */


/*!
 * Atomic addition using compare and swap loop
 */
template<typename T, std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T* acc, T value)
{
  return builtin_atomicCAS_loop(acc, [value](T old) {
    return old + value;
  });
}

/*!
 * Atomic subtraction using compare and swap loop
 */
template<typename T, std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T* acc, T value)
{
  return builtin_atomicCAS_loop(acc, [value](T old) {
    return old - value;
  });
}

/*!
 * Atomic and using compare and swap loop
 */
template<typename T, std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T* acc, T value)
{
  return builtin_atomicCAS_loop(acc, [value](T old) {
    return old & value;
  });
}

/*!
 * Atomic or using compare and swap loop
 */
template<typename T, std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T* acc, T value)
{
  return builtin_atomicCAS_loop(acc, [value](T old) {
    return old | value;
  });
}

/*!
 * Atomic xor using compare and swap loop
 */
template<typename T, std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T* acc, T value)
{
  return builtin_atomicCAS_loop(acc, [value](T old) {
    return old ^ value;
  });
}


}  // namespace detail

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicLoad(builtin_atomic, T* acc)
{
  return detail::builtin_atomicLoad(acc);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE void atomicStore(builtin_atomic, T* acc, T value)
{
  detail::builtin_atomicStore(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAdd(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicAdd(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicSub(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicSub(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMin(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicCAS_loop(
      acc,
      [value](T old) {
        return value < old ? value : old;
      },
      [value](T current) {
        return current <= value;
      });
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMax(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicCAS_loop(
      acc,
      [value](T old) {
        return old < value ? value : old;
      },
      [value](T current) {
        return value <= current;
      });
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T* acc)
{
  return detail::builtin_atomicAdd(acc, static_cast<T>(1));
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicCAS_loop(acc, [value](T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T* acc)
{
  return detail::builtin_atomicSub(acc, static_cast<T>(1));
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicCAS_loop(acc, [value](T old) {
    return old == static_cast<T>(0) || value < old ? value
                                                   : old - static_cast<T>(1);
  });
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAnd(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicAnd(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicOr(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicOr(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicXor(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicXor(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicExchange(builtin_atomic, T* acc, T value)
{
  return detail::builtin_atomicExchange(acc, value);
}

template<typename T>
RAJA_DEVICE_HIP RAJA_INLINE T
atomicCAS(builtin_atomic, T* acc, T compare, T value)
{
  return detail::builtin_atomicCAS(acc, compare, value);
}


}  // namespace RAJA


#endif
