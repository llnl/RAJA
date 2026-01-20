/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of std align.
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

#ifndef RAJA_ALIGN_HPP
#define RAJA_ALIGN_HPP

#include "RAJA/config.hpp"

namespace RAJA
{

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Taken from libc++
// See libc++ license in docs/Licenses/libc++ License
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
RAJA_INLINE
void* align(size_t alignment, size_t size, void*& ptr, size_t& space)
{

#ifdef RAJA_COMPILER_MSVC
#pragma warning(disable : 4146)  // Force msvc to ignore subtracting from signed
                                 // number warning
#endif
  void* r = nullptr;
  if (size <= space)
  {
    char* p1 = static_cast<char*>(ptr);
    char* p2 = reinterpret_cast<char*>(
        reinterpret_cast<size_t>(p1 + (static_cast<ptrdiff_t>(alignment) - 1)) &
        -alignment);
    size_t d = static_cast<size_t>(p2 - p1);
    if (d <= space - size)
    {
      r   = p2;
      ptr = r;
      space -= d;
    }
  }
  return r;

#ifdef RAJA_COMPILER_MSVC
#pragma warning(default : 4146)  // Force msvc to ignore subtracting from signed
                                 // number warning
#endif
}

}  // end namespace RAJA

#endif
