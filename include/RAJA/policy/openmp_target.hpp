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

#ifndef RAJA_openmp_target_HPP
#define RAJA_openmp_target_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RAJA_ENABLE_TARGET_OPENMP)

#include <omp.h>

#include "RAJA/policy/openmp_target/policy.hpp"
#include "RAJA/policy/openmp_target/kernel.hpp"
#include "RAJA/policy/openmp_target/forall.hpp"
#include "RAJA/policy/openmp_target/reduce.hpp"
//#include "RAJA/policy/openmp_target/multi_reduce.hpp"
#include "RAJA/policy/openmp_target/WorkGroup.hpp"
#include "RAJA/policy/openmp_target/params/reduce.hpp"
#include "RAJA/policy/openmp_target/params/kernel_name.hpp"


#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP) &&
        // defined(RAJA_ENABLE_TARGET_OPENMP)

#endif  // closing endif for header file include guard
