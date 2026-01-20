/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA math templates.
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

#ifndef RAJA_util_math_HPP
#define RAJA_util_math_HPP

#include "RAJA/config.hpp"

#include <type_traits>
#include <climits>

namespace RAJA
{

/*!
    \brief evaluate log base 2 of n

    For positive n calculate log base 2 of n, and round the result down to the
    nearest integer.
    For zero or negative n return 0

*/
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T log2(T n) noexcept
{
  T result = 0;
  if (n > 0)
  {
    while (n >>= 1)
    {
      ++result;
    }
  }
  return result;
}

/*!
    \brief "round up" to the next greatest power of 2

    For a integer n,
      if n is non-negative,
        if n is a power of 2, return n
        if n is not a power of 2, return the next greater power of 2
      if n is negative, return 0
*/
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T next_pow2(T n) noexcept
{
  --n;
  for (size_t s = 1; s < CHAR_BIT * sizeof(T); s *= 2)
  {
    n |= n >> s;
  }
  ++n;
  return n;
}

/*!
    \brief "round down" to the largest power of 2 that is less than or equal to
   n

    For an integer n,
      if n is negative, return 0
      else
        if n is a power of 2, return n
        else return the largest power of 2 that is less than n
*/
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T prev_pow2(T n) noexcept
{
  if (n < 0) return 0;
  for (size_t s = 1; s < CHAR_BIT * sizeof(T); s *= 2)
  {
    n |= n >> s;
  }
  return n - (n >> 1);
}

/*!
    \brief compute lhs mod rhs where lhs is non-negative and rhs is a power of 2
*/
template<typename L,
         typename R,
         std::enable_if_t<std::is_integral<L>::value &&
                          std::is_integral<R>::value>* = nullptr>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto power_of_2_mod(L lhs,
                                                           R rhs) noexcept
{
  return lhs & (rhs - R(1));
}

}  // namespace RAJA

#endif
