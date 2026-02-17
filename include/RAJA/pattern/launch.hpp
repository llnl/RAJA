/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing headers for RAJA::Launch backends
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

#ifndef RAJA_pattern_launch_HPP
#define RAJA_pattern_launch_HPP

#include "RAJA/pattern/launch/launch_core.hpp"

//
// All platforms must support host execution.
//
#include "RAJA/policy/sequential/launch.hpp"
#include "RAJA/policy/simd/launch.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/launch.hpp"
#endif

#if defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/launch.hpp"
#endif

#if defined(RAJA_OPENMP_ACTIVE)
#include "RAJA/policy/openmp/launch.hpp"
#endif

#if defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/launch.hpp"
#endif

#endif /* RAJA_pattern_launch_HPP */
