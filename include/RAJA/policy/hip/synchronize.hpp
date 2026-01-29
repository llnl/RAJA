/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Header file for HIP synchronize method.
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

#ifndef RAJA_synchronize_hip_HPP
#define RAJA_synchronize_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace policy
{

namespace hip
{

/*!
 * \brief Synchronize the current HIP device.
 */
RAJA_INLINE
void synchronize_impl(const hip_synchronize&)
{
  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
}


}  // end of namespace hip
}  // namespace policy
}  // end of namespace RAJA

#endif  // defined(RAJA_ENABLE_HIP)

#endif  // RAJA_synchronize_hip_HPP
