/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining OpenMP atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_atomic_HPP
#define RAJA_policy_openmp_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

// Relies on builtin_atomic when OpenMP can't do the job
#if !defined(RAJA_COMPILER_MSVC)

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicLoad(omp_atomic, T* acc)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc += (T)0;
  }
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE void atomicStore(omp_atomic, T* acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret  = *acc;
    *acc = value;
  }
  RAJA_UNUSED_VAR(ret);
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAdd(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc += value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicSub(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc -= value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMin(omp_atomic, T* acc, T value)
{
#if defined(RAJA_USE_OPENMP_5_1_ATOMICS)
  T old;
#pragma omp atomic capture compare
  {
    old = *acc;
    if (value < *acc)
    {
      *acc = value;
    }
  }
  return old;
#else
  // OpenMP doesn't define atomic ternary operators so use builtin atomics
  return atomicMin(builtin_atomic {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMax(omp_atomic, T* acc, T value)
{
#if defined(RAJA_USE_OPENMP_5_1_ATOMICS)
  T old;
#pragma omp atomic capture compare
  {
    old = *acc;
    if (value > *acc)
    {
      *acc = value;
    }
  }
  return old;
#else
  // OpenMP doesn't define atomic ternary operators so use builtin atomics
  return atomicMax(builtin_atomic {}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(omp_atomic, T* acc)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc += T(1);
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(omp_atomic, T* acc, T value)
{
  // OpenMP doesn't define needed operations, so use builtin atomics
  return RAJA::atomicInc(builtin_atomic {}, acc, value);
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(omp_atomic, T* acc)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc -= T(1);
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(omp_atomic, T* acc, T value)
{
  // OpenMP doesn't define needed operations, so use builtin atomics
  return RAJA::atomicDec(builtin_atomic {}, acc, value);
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAnd(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc &= value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicOr(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc |= value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicXor(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old = *acc;  // capture old for return value
    *acc ^= value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicExchange(omp_atomic, T* acc, T value)
{
  T old;
#pragma omp atomic capture
  {
    old  = *acc;  // capture old for return value
    *acc = value;
  }
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicCAS(omp_atomic, T* acc, T compare, T value)
{
  // OpenMP doesn't define atomic ternary operators so use builtin atomics
  return RAJA::atomicCAS(builtin_atomic {}, acc, compare, value);
}

#endif  // not defined RAJA_COMPILER_MSVC


}  // namespace RAJA

#endif  // RAJA_ENABLE_OPENMP
#endif  // guard
