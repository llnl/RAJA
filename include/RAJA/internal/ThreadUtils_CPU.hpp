/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing prototypes and methods for managing
 *          CPU threading operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ThreadUtils_CPU_HPP
#define RAJA_ThreadUtils_CPU_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

/*!
*************************************************************************
*
* Return max number of available OpenMP threads.
*
*************************************************************************
*/
RAJA_INLINE
int getMaxOMPThreadsCPU()
{
  int nthreads = 1;

#if defined(RAJA_ENABLE_OPENMP)
  nthreads = omp_get_max_threads();
#endif

  return nthreads;
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
