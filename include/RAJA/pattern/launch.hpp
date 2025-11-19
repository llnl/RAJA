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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/launch.hpp"
#endif

#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl/launch.hpp"
#endif

#endif /* RAJA_pattern_launch_HPP */
