//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Reduction policies used for reduction tests
//

#ifndef __RAJA_test_reducepol_HPP__
#define __RAJA_test_reducepol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential reduction policy types
using SequentialReducePols = camp::list< RAJA::seq_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPReducePols =
#if 0 // is ordered reduction broken???
  camp::list< RAJA::omp_reduce,
              RAJA::omp_reduce_ordered >;
#else
  camp::list< RAJA::omp_reduce >;
#endif
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducePols =
  camp::list< RAJA::omp_target_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReducePols = camp::list< RAJA::cuda_reduce_device_fence,
                                   RAJA::cuda_reduce_block_fence,
                                   RAJA::cuda_reduce_atomic_device_init_device_fence,
                                   RAJA::cuda_reduce_atomic_device_init_block_fence,
                                   RAJA::cuda_reduce_atomic_host_init_device_fence,
                                   RAJA::cuda_reduce_atomic_host_init_block_fence >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipReducePols = camp::list< RAJA::hip_reduce_device_fence,
                                  RAJA::hip_reduce_block_fence,
                                  RAJA::hip_reduce_atomic_device_init_device_fence,
                                  RAJA::hip_reduce_atomic_device_init_block_fence,
                                  RAJA::hip_reduce_atomic_host_init_device_fence,
                                  RAJA::hip_reduce_atomic_host_init_block_fence >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclReducePols = camp::list< RAJA::sycl_reduce >;
#endif

#endif  // __RAJA_test_reducepol_HPP__
