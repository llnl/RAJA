/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for SYCL execution.
 *
 *          These methods work only on platforms that support SYCL.
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

#ifndef RAJA_sycl_HPP
#define RAJA_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_SYCL_ACTIVE)

#include "RAJA/util/sycl_compat.hpp"

#include "RAJA/policy/sycl/forall.hpp"
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/reduce.hpp"
//#include "RAJA/policy/sycl/multi_reduce.hpp"
//#include "RAJA/policy/sycl/scan.hpp"
//#include "RAJA/policy/sycl/sort.hpp"
#include "RAJA/policy/sycl/kernel.hpp"
//#include "RAJA/policy/sycl/synchronize.hpp"
#include "RAJA/policy/sycl/launch.hpp"
//#include "RAJA/policy/sycl/WorkGroup.hpp"
#include "RAJA/policy/sycl/params/reduce.hpp"
#include "RAJA/policy/sycl/params/kernel_name.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_SYCL)

#endif  // closing endif for header file include guard
