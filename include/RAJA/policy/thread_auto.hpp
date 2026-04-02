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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_thread_auto_HPP
#define RAJA_policy_thread_auto_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#ifdef RAJA_OPENMP_ACTIVE
#include "RAJA/policy/openmp/policy.hpp"
#endif

#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{

namespace detail
{
/*!
 * Provides priority between thread policies that should do the "right thing"
 *
 * If OpenMP is active we always use the omp_thread.
 *
 * Fallback to seq_thread, which performs non-thread operations
 * assumes there is no thread safety issues
 */
#if defined(RAJA_OPENMP_ACTIVE)
using active_auto_thread = RAJA::omp_thread;
#else
using active_auto_thread = RAJA::seq_thread;
#endif

}  // namespace detail

template<typename AtomicPolicy>
RAJA_HOST_DEVICE RAJA_INLINE int get_max_threads(AtomicPolicy);

template<typename AtomicPolicy>
RAJA_HOST_DEVICE RAJA_INLINE int get_thread_num(AtomicPolicy);

}  // namespace RAJA

#endif
