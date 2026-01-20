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

#include <string.h>

namespace RAJA
{
namespace util
{


/*!
 * Reinterpret any datatype as another datatype of the same size
 */
template<typename A, typename B>
RAJA_INLINE RAJA_HOST_DEVICE constexpr B reinterp_A_as_B(A const& a)
{
  static_assert(sizeof(A) == sizeof(B), "A and B must be the same size");

  B b;
  memcpy(&b, &a, sizeof(A));
  return b;
}


}  // namespace util
}  // namespace RAJA

#endif  // closing endif for header file include guard
