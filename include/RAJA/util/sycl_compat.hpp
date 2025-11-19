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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
