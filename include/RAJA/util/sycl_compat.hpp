/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for handling different SYCL header include paths
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

#ifndef RAJA_util_sycl_compat_HPP
#define RAJA_util_sycl_compat_HPP

#if defined(RAJA_SYCL_ACTIVE)
#if (__INTEL_CLANG_COMPILER && __INTEL_CLANG_COMPILER < 20230000)
// older version, use legacy header locations
#include <CL/sycl.hpp>
#else
// SYCL 2020 standard header
#include <sycl/sycl.hpp>
#endif
#endif

#endif  // RAJA_util_sycl_compat_HPP
