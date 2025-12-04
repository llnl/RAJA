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
RAJA_SUPPRESS_HD_WARN
template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_max_threads(omp_atomic)
{
  return omp_get_max_threads();
}

RAJA_SUPPRESS_HD_WARN
template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_thread_num(omp_atomic)
{
  return omp_get_thread_num();
}
  
}  // namespace RAJA

#endif  // RAJA_ENABLE_OPENMP
#endif  // guard
