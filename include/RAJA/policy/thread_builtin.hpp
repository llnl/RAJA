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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_thread_builtin_HPP
#define RAJA_policy_thread_builtin_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

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

}  // namespace RAJA


#endif
