/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for sequential execution.
 *
 *          These methods work on all platforms.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sequential_HPP
#define RAJA_sequential_HPP

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/sequential/atomic.hpp"
#endif

#include "RAJA/policy/sequential/forall.hpp"
#include "RAJA/policy/sequential/kernel.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/policy/sequential/reduce.hpp"
#include "RAJA/policy/sequential/multi_reduce.hpp"
#include "RAJA/policy/sequential/scan.hpp"
#include "RAJA/policy/sequential/sort.hpp"
#include "RAJA/policy/sequential/thread.hpp"
#include "RAJA/policy/sequential/launch.hpp"
#include "RAJA/policy/sequential/WorkGroup.hpp"
#include "RAJA/policy/sequential/params/reduce.hpp"
#include "RAJA/policy/sequential/params/kernel_name.hpp"

#endif  // closing endif for header file include guard
