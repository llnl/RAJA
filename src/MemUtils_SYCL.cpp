/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for SYCL reductions and other operations.
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

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"

namespace RAJA
{

namespace sycl
{

namespace detail
{
//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

//! State of the host code globally
syclInfo g_status;

//! State of the host code in this thread
thread_local syclInfo tl_status;

//! State of raja sycl queue synchronization for sycl reducer objects
std::unordered_map<::sycl::queue, bool> g_queue_info_map {
    {::sycl::queue(), true}};

}  // namespace detail

}  // namespace sycl

}  // namespace RAJA


#endif  // if defined(RAJA_ENABLE_SYCL)
