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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ThreadUtils_CPU_HPP
#define RAJA_ThreadUtils_CPU_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/thread.hpp"
#include "RAJA/policy/openmp/thread.hpp"
#include "RAJA/policy/sequential/thread.hpp"

namespace RAJA
{

/*!
*************************************************************************
*
* Return max number of available OpenMP threads.
*
*************************************************************************
*/
template<typename ThreadPolicy = RAJA::detail::active_auto_thread>
RAJA_INLINE int getMaxOMPThreadsCPU()
{
  return RAJA::get_max_threads<ThreadPolicy>();
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
