/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for OpenMP synchronization.
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

#ifndef RAJA_synchronize_openmp_HPP
#define RAJA_synchronize_openmp_HPP

namespace RAJA
{

namespace policy
{

namespace omp
{

/*!
 * \brief Synchronize all OpenMP threads and tasks.
 */
RAJA_INLINE
void synchronize_impl(const omp_synchronize&)
{
#pragma omp barrier
}


}  // end of namespace omp
}  // namespace policy
}  // end of namespace RAJA

#endif  // RAJA_synchronize_openmp_HPP
