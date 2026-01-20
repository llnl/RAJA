/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining sequential atomic operations.
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

#ifndef RAJA_policy_sequential_atomic_HPP
#define RAJA_policy_sequential_atomic_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicLoad(seq_atomic, T* acc)
{
  return *acc;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE void atomicStore(seq_atomic, T* acc, T value)
{
  *acc = value;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAdd(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc += value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicSub(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc -= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMin(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc  = ret < value ? ret : value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMax(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc  = value < ret ? ret : value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(seq_atomic, T* acc)
{
  T ret = *acc;
  (*acc) += T(1);
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(seq_atomic, T* acc, T val)
{
  T old = *acc;
  *acc  = val <= old ? T(0) : old + T(1);
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(seq_atomic, T* acc)
{
  T ret = *acc;
  (*acc) -= T(1);
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(seq_atomic, T* acc, T val)
{
  T old = *acc;
  *acc  = old == T(0) || val < old ? val : old - T(1);
  return old;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAnd(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc &= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicOr(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc |= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicXor(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc ^= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicExchange(seq_atomic, T* acc, T value)
{
  T ret = *acc;
  *acc  = value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicCAS(seq_atomic, T* acc, T compare, T value)
{
  T ret = *acc;
  *acc  = ret == compare ? value : ret;
  return ret;
}


}  // namespace RAJA


#endif  // guard
