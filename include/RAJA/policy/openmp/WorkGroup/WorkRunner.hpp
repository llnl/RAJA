/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkRunner class specializations.
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

#ifndef RAJA_openmp_WorkGroup_WorkRunner_HPP
#define RAJA_openmp_WorkGroup_WorkRunner_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template<typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<RAJA::omp_work,
                  RAJA::ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...>
    : WorkRunnerForallOrdered<RAJA::omp_parallel_for_exec,
                              RAJA::omp_work,
                              RAJA::ordered,
                              DISPATCH_POLICY_T,
                              ALLOCATOR_T,
                              INDEX_T,
                              Args...>
{};

/*!
 * Runs work in a storage container in reverse order
 * and returns any per run resources
 */
template<typename DISPATCH_POLICY_T,
         typename ALLOCATOR_T,
         typename INDEX_T,
         typename... Args>
struct WorkRunner<RAJA::omp_work,
                  RAJA::reverse_ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...>
    : WorkRunnerForallReverse<RAJA::omp_parallel_for_exec,
                              RAJA::omp_work,
                              RAJA::reverse_ordered,
                              DISPATCH_POLICY_T,
                              ALLOCATOR_T,
                              INDEX_T,
                              Args...>
{};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
