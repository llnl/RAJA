//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types for atomic ref tests.
//
// These are used to break apart atomic ref tests for shorter compilation times.
//

#ifndef __RAJA_test_atomic_ref_types_HPP__
#define __RAJA_test_atomic_ref_types_HPP__

#include "RAJA/RAJA.hpp"

#include <type_traits>

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 1, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 2, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 4, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 8, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  val |= val >> 32 ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 16, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  val |= val >> 32 ;
  val |= val >> 64 ;
  return val;
}

// Assist return type conditional overloading of testAtomicRefLogicalOp
struct int_op {}; // represents underlying op type = integral
struct all_op {}; // these op types can accept integral or float


#endif // __RAJA_test_atomic_ref_types_HPP__
