/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic thread operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_thread_auto_HPP
#define RAJA_policy_thread_auto_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#ifdef RAJA_ENABLE_OPENMP
#include "RAJA/policy/openmp/policy.hpp"
#endif

namespace RAJA
{

// SGS builtin?
RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy>
RAJA_HOST_DEVICE RAJA_INLINE int get_max_threads(AtomicPolicy)
{
  return 1;
}

RAJA_SUPPRESS_HD_WARN
template<typename AtomicPolicy>
RAJA_HOST_DEVICE RAJA_INLINE int get_thread_num(AtomicPolicy)
{
  return 0;
}
}  

#include "RAJA/policy/sequential/thread.hpp"

/*!
 * Provides priority between thread policies that should do the "right thing"
 *
 * If OpenMP is active we always use the omp_thread.
 *
 * Fallbac to seq_thread, which performs non-thread operations
 * assumes there is no thread safety issues
 */
#if defined(RAJA_ENABLE_OPENMP)

#define RAJA_AUTO_THREAD                                                       \
  RAJA::omp_thread {}
#else
#define RAJA_AUTO_THREAD                                                       \
  RAJA::seq_thread {}
#endif

namespace RAJA
{

//! Thread policy that automatically does "the right thing"
struct auto_thread
{};

template<>
RAJA_INLINE RAJA_HOST_DEVICE int get_max_threads(auto_thread)
{
  return get_max_threads(RAJA_AUTO_THREAD);
}

template<>
RAJA_INLINE RAJA_HOST_DEVICE int get_thread_num(auto_thread)
{
  return get_thread_num(RAJA_AUTO_THREAD);
}

}  // namespace RAJA

// make sure this define doesn't bleed out of this header
#undef RAJA_AUTO_THREAD

#endif
