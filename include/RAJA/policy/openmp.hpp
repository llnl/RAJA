/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for OpenMP execution.
 *
 *          These methods work only on platforms that support OpenMP.
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


#ifndef RAJA_openmp_HPP
#define RAJA_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>
#include <thread>

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/openmp/atomic.hpp"
#endif

#include "RAJA/policy/openmp/forall.hpp"
#include "RAJA/policy/openmp/kernel.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/openmp/reduce.hpp"
#include "RAJA/policy/openmp/multi_reduce.hpp"
#include "RAJA/policy/openmp/region.hpp"
#include "RAJA/policy/openmp/scan.hpp"
#include "RAJA/policy/openmp/sort.hpp"
#include "RAJA/policy/openmp/synchronize.hpp"
#include "RAJA/policy/openmp/thread.hpp"
#include "RAJA/policy/openmp/launch.hpp"
#include "RAJA/policy/openmp/WorkGroup.hpp"
#include "RAJA/policy/openmp/params/reduce.hpp"
#include "RAJA/policy/openmp/params/kernel_name.hpp"


#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
