/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining OpenMP thread operations.
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

#ifndef RAJA_policy_openmp_thread_HPP
#define RAJA_policy_openmp_thread_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_OPENMP_ACTIVE)

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/thread_auto.hpp"

namespace RAJA
{

template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_max_threads(omp_thread)
{
  return omp_get_max_threads();
}

template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_thread_num(omp_thread)
{
  return omp_get_thread_num();
}

}  // namespace RAJA

#endif  // RAJA_ENABLE_OPENMP
#endif  // guard
