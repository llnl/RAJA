//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_desul_HPP
#define RAJA_policy_atomic_desul_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_DESUL_ATOMICS)

#include <cstdint>
#include <type_traits>
#include <utility>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/TypeConvert.hpp"

#include "RAJA/policy/atomic_builtin.hpp"

#include "desul/atomics.hpp"

// Default desul options for RAJA
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

namespace RAJA
{

namespace detail
{

template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE bool desul_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

template<typename T,
         std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
RAJA_HOST_DEVICE RAJA_INLINE bool desul_atomicCAS_equal(const T& a, const T& b)
{
  using R = std::conditional_t<sizeof(T) == sizeof(std::uint32_t),
                               std::uint32_t,
                               std::uint64_t>;
  static_assert(sizeof(T) == sizeof(std::uint32_t) ||
                sizeof(T) == sizeof(std::uint64_t),
                "desul_atomicCAS_equal only supports 32/64-bit floating point");

  return RAJA::util::reinterp_A_as_B<T, R>(a) ==
         RAJA::util::reinterp_A_as_B<T, R>(b);
}

}  // namespace detail

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicLoad(AtomicPolicy, T* acc)
{
  return desul::atomic_load(acc, raja_default_desul_order {},
                            raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE void atomicStore(AtomicPolicy, T* acc, T value)
{
  desul::atomic_store(acc, value, raja_default_desul_order {},
                      raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAdd(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_add(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicSub(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_sub(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMin(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_min(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMax(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_max(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(AtomicPolicy, T* acc)
{
  return desul::atomic_fetch_inc(acc, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(AtomicPolicy, T* acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return desul::atomic_fetch_inc_mod(acc, val, raja_default_desul_order {},
                                     raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(AtomicPolicy, T* acc)
{
  return desul::atomic_fetch_dec(acc, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(AtomicPolicy, T* acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return desul::atomic_fetch_dec_mod(acc, val, raja_default_desul_order {},
                                     raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAnd(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_and(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicOr(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_or(acc, value, raja_default_desul_order {},
                                raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicXor(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_fetch_xor(acc, value, raja_default_desul_order {},
                                 raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicExchange(AtomicPolicy, T* acc, T value)
{
  return desul::atomic_exchange(acc, value, raja_default_desul_order {},
                                raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T
atomicCAS(AtomicPolicy, T* acc, T compare, T value)
{
  return desul::atomic_compare_exchange(acc, compare, value,
                                        raja_default_desul_order {},
                                        raja_default_desul_scope {});
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy, typename T, typename Operation>
RAJA_HOST_DEVICE RAJA_INLINE T
atomicOperation(AtomicPolicy, T* acc, Operation&& operation)
{
  T expected = desul::atomic_load(acc,
                                  raja_default_desul_order {},
                                  raja_default_desul_scope {});

  while (true) {
    const T desired = operation(expected);

    if (desul_atomicCAS_equal(desired, expected)) {
      return expected; // no-op
    }

    const T old = desul::atomic_compare_exchange(acc, expected, desired,
                                                 raja_default_desul_order {},
                                                 raja_default_desul_scope {});

    if (desul_atomicCAS_equal(old, expected)) {
      return old; // success
    }

    expected = old; // CAS failed, old is the latest observed value
  }
}

}  // namespace RAJA

#endif  // RAJA_ENABLE_DESUL_ATOMICS
#endif  // guard
