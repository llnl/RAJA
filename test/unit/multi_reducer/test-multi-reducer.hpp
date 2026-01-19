//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_MULTI_REDUCER_UTILS_HPP__
#define __TEST_MULTI_REDUCER_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-forone.hpp"
#include "RAJA_test-multi-reduce-abstractor.hpp"

//
// Data types
//
using DataTypeList = camp::list< int,
                                 float,
                                 double >;

using SequentialMultiReducerPolicyList = camp::list< RAJA::seq_multi_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPMultiReducerPolicyList = camp::list< RAJA::omp_multi_reduce,
                                                 RAJA::omp_multi_reduce_ordered >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaMultiReducerPolicyList =
  camp::list< RAJA::cuda_multi_reduce_atomic_block_then_atomic_grid_host_init,
              RAJA::cuda_multi_reduce_atomic_block_then_atomic_grid_host_init_fallback_testing,
              RAJA::cuda_multi_reduce_atomic_global_host_init,
              RAJA::cuda_multi_reduce_atomic_global_no_replication_host_init >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipMultiReducerPolicyList =
  camp::list< RAJA::hip_multi_reduce_atomic_block_then_atomic_grid_host_init,
              RAJA::hip_multi_reduce_atomic_block_then_atomic_grid_host_init_fallback_testing,
              RAJA::hip_multi_reduce_atomic_global_host_init,
              RAJA::hip_multi_reduce_atomic_global_no_replication_host_init  >;
#endif

#endif  // __TEST_MULTI_REDUCER_UTILS_HPP__
