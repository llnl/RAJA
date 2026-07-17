/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for reinterpreting type conversions.
 *
 *          These conversions are needed to pass N-bit floating point values
 *          as integral types for certain API's that have limited type support.
 *          These conversions are used heavily by the atomic operators.
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

#ifndef RAJA_util_TypeConvert_HPP
#define RAJA_util_TypeConvert_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/array.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace RAJA
{
namespace util
{


// TODO: Investigate std::bit_cast in C++20.
//       Currently breaks the ROCm 5.7.1 build.

/*!
 * Reinterpret any datatype as another datatype of the same size
 */
template<typename A, typename B>
RAJA_INLINE RAJA_HOST_DEVICE constexpr B reinterp_A_as_B(A const& a)
{
  static_assert(sizeof(A) == sizeof(B), "A and B must be the same size");
  // TODO: Consider requiring A and B to be trivially copyable

  B b;
  memcpy(&b, &a, sizeof(A));
  return b;
}

/*!
 * Compare two values by their object representation.
 */
template<typename T>
RAJA_INLINE RAJA_HOST_DEVICE constexpr bool bit_equal(T const& a, T const& b)
{
  if constexpr (sizeof(T) == 1)
  {
    return reinterp_A_as_B<T, std::uint8_t>(a) ==
           reinterp_A_as_B<T, std::uint8_t>(b);
  }
  else if constexpr (sizeof(T) == 2)
  {
    return reinterp_A_as_B<T, std::uint16_t>(a) ==
           reinterp_A_as_B<T, std::uint16_t>(b);
  }
  else if constexpr (sizeof(T) == 4)
  {
    return reinterp_A_as_B<T, std::uint32_t>(a) ==
           reinterp_A_as_B<T, std::uint32_t>(b);
  }
  else if constexpr (sizeof(T) == 8)
  {
    return reinterp_A_as_B<T, std::uint64_t>(a) ==
           reinterp_A_as_B<T, std::uint64_t>(b);
  }
  else
  {
    return reinterp_A_as_B<T, camp::array<unsigned char, sizeof(T)>>(a) ==
           reinterp_A_as_B<T, camp::array<unsigned char, sizeof(T)>>(b);
  }
}


}  // namespace util
}  // namespace RAJA

#endif  // closing endif for header file include guard
