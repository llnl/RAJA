/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic and builtin atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_auto_HPP
#define RAJA_policy_atomic_auto_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/sequential/atomic.hpp"
#endif

namespace RAJA
{
/*!
 * Provides priority between atomic policies that should do the "right thing"
 *
 * If we are in a CUDA __device__ function, then it always uses the cuda_atomic
 * policy.
 *
 * Next, if OpenMP is enabled we always use the omp_atomic, which should
 * generally work everywhere.
 *
 * Finally, we fallback on the seq_atomic, which performs non-atomic operations
 * because we assume there is no thread safety issues (no parallel model)
 */
#if defined(__CUDA_ARCH__) && defined(RAJA_CUDA_ACTIVE)
  using auto_atomic = RAJA::cuda_atomic;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(RAJA_HIP_ACTIVE)
  using auto_atomic = RAJA::hip_atomic;
#elif defined(__SYCL_DEVICE_ONLY__)
  using auto_atomic = RAJA::sycl_atomic;
#elif defined(RAJA_OPENMP_ACTIVE)
  using auto_atomic = RAJA::omp_atomic;
#else
  using auto_atomic = RAJA::seq_atomic;
#endif

}

#endif
