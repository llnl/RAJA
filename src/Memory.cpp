/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for RAJA memory utility routines.
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

#include "RAJA/util/memory.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#endif

#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#endif

namespace RAJA
{

size_t release_unused_internal_memory()
{
  size_t released = 0;

#if defined(RAJA_ENABLE_CUDA)
  released += ::RAJA::cuda::release_unused_internal_memory();
#endif
#if defined(RAJA_ENABLE_HIP)
  released += ::RAJA::hip::release_unused_internal_memory();
#endif
#if defined(RAJA_ENABLE_SYCL)
  released += ::RAJA::sycl::release_unused_internal_memory();
#endif

  return released;
}

}  // namespace RAJA
