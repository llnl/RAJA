/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining thread operations.
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

#ifndef RAJA_pattern_thread_HPP
#define RAJA_pattern_thread_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/thread_auto.hpp"

namespace RAJA
{
/*!
 * \file
 * Thread operation functions in the namespace RAJA::thread
 *
 * The dispatch of all of these is:
 *
 * int get_thread_num<Policy>()      -- User facing API
 *
 * calls
 *
 * int get_thread_num(Policy{})    -- Policy specific implementation
 *
 *
 * With the exception of the auto_thread policy which then calls the
 * "appropriate" policy implementation.
 *
 *
 * Current supported policies include:
 *
 *   auto_thread       -- Attempts to do "the right thing"
 *
 *   omp_thread        -- Available (and default) when OpenMP is active
 *                        these are safe inside and outside of OMP parallel
 *                        regions
 *
 *   seq_thread        -- Non-thread
 *
 *
 * The implementation code lives in:
 * RAJA/policy/thread_auto.hpp     -- for auto_thread
 * RAJA/policy/XXX/thread.hpp      -- for omp_thread
 *
 */

/*!
 * @brief Get maximum number of threads

 * This is based on OpenMP threading model. This value is also an
 * upper bound on the number of threads that could be used to form a
 * new team if a parallel region without a num_threads clause were
 * encountered after execution returns from this routine.
 *
 * Returns 1 if OMP is not active.
 * @return Maximum number of threads
 */
template<typename Policy>
RAJA_INLINE RAJA_HOST_DEVICE int get_max_threads()
{
  return RAJA::get_max_threads(Policy {});
}

/*!
 * @brief Get current thread number
 * This is based on the OpenMP threading model.  Within a parallel team
 * executing a parallel region the threads are numbered 0-N. Returns 0 if called
 * in sequential part of a program or OMP is not active
 * @return Current thread number
 */
template<typename Policy>
RAJA_INLINE RAJA_HOST_DEVICE int get_thread_num()
{
  return RAJA::get_thread_num(Policy {});
}

}  // namespace RAJA

#endif
