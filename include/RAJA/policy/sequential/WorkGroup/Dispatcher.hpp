/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
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

#ifndef RAJA_sequential_WorkGroup_Dispatcher_HPP
#define RAJA_sequential_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Populate and return a Dispatcher object
 */
template<typename T, typename Dispatcher_T>
inline const Dispatcher_T* get_Dispatcher(seq_work const&)
{
  static Dispatcher_T dispatcher {Dispatcher_T::template makeDispatcher<T>()};
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
