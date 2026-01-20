/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining sequential thread operations.
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

#ifndef RAJA_policy_sequential_thread_HPP
#define RAJA_policy_sequential_thread_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/thread_auto.hpp"

namespace RAJA
{
template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_max_threads(seq_thread)
{
  return 1;
}

template<>
RAJA_HOST_DEVICE RAJA_INLINE int get_thread_num(seq_thread)
{
  return 0;
}

}  // namespace RAJA


#endif  // guard
