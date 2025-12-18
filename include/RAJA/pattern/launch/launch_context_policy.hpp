/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing template types of RAJA::LaunchContextT
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_context_policy_HPP
#define RAJA_pattern_context_policy_HPP

namespace RAJA
{

class LaunchContextDefaultPolicy;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE)
class LaunchContextDim3Policy;
#endif

}  // namespace RAJA
#endif
