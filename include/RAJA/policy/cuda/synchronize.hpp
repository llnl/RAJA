/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Header file for CUDA synchronize method.
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

#ifndef RAJA_synchronize_cuda_HPP
#define RAJA_synchronize_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

namespace RAJA
{

namespace policy
{

namespace cuda
{

/*!
 * \brief Synchronize the current CUDA device.
 */
RAJA_INLINE
void synchronize_impl(const cuda_synchronize&)
{
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
}


}  // end of namespace cuda
}  // namespace policy
}  // end of namespace RAJA

#endif  // defined(RAJA_ENABLE_CUDA)

#endif  // RAJA_synchronize_cuda_HPP
