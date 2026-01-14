/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  RAJA header for execution synchronization template.
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

#ifndef RAJA_synchronize_HPP
#define RAJA_synchronize_HPP

namespace RAJA
{

/*!
 * \brief Synchronize all current RAJA executions for the specified policy.
 *
 * The type of synchronization performed depends on the execution policy. For
 * example, to syncrhonize the current CUDA device, use:
 *
 * \code
 *
 * RAJA::synchronize<RAJA::cuda_synchronize>();
 *
 * \endcode
 *
 * \tparam Policy synchronization policy
 *
 * \see RAJA::policy::omp::synchronize_impl
 * \see RAJA::policy::cuda::synchronize_impl
 */
template<typename Policy>
void synchronize()
{
  synchronize_impl(Policy {});
}
}  // namespace RAJA

#endif  // RAJA_synchronize_HPP
