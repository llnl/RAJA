/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for handling builtin function differences for
 *          different compilers.
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

#ifndef RAJA_util_builtin_compat_HPP
#define RAJA_util_builtin_compat_HPP

#if defined(_MSC_VER)
#include <cstring>
#define RAJA_BUILTIN_MEMCPY(a, b, c) std::memcpy(a, b, c)
   
#else
#define RAJA_BUILTIN_MEMCPY(a, b, c) __builtin_memcpy(a, b, c)
#endif

#endif  // RAJA_util_builtin_compat_HPP
