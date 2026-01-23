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

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/atomic_builtin.hpp"

#include "desul/atomics.hpp"

// Default desul options for RAJA
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

namespace RAJA
{

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

}  // namespace RAJA

#endif  // RAJA_ENABLE_DESUL_ATOMICS
#endif  // guard
