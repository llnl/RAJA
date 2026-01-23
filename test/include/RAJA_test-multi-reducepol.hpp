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

#ifndef __RAJA_test_multi_reducepol_HPP__
#define __RAJA_test_multi_reducepol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential reduction policy types
using SequentialMultiReducePols = camp::list< RAJA::seq_multi_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPMultiReducePols =
  camp::list< RAJA::omp_multi_reduce,
              RAJA::omp_multi_reduce_ordered >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaMultiReducePols =
  camp::list< RAJA::cuda_multi_reduce_atomic_block_then_atomic_grid_host_init,
              RAJA::cuda_multi_reduce_atomic_block_then_atomic_grid_host_init_fallback_testing,
              RAJA::cuda_multi_reduce_atomic_global_host_init,
              RAJA::cuda_multi_reduce_atomic_global_no_replication_host_init >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipMultiReducePols =
  camp::list< RAJA::hip_multi_reduce_atomic_block_then_atomic_grid_host_init,
              RAJA::hip_multi_reduce_atomic_block_then_atomic_grid_host_init_fallback_testing,
              RAJA::hip_multi_reduce_atomic_global_host_init,
              RAJA::hip_multi_reduce_atomic_global_no_replication_host_init  >;
#endif

#endif  // __RAJA_test_multi_reducepol_HPP__
