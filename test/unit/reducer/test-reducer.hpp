//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_REDUCER_UTILS_HPP__
#define __TEST_REDUCER_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-forone.hpp"

//
// Data types
//
using DataTypeList = camp::list< int,
                                 float,
                                 double >;

using SequentialReducerPolicyList = camp::list< RAJA::seq_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPReducerPolicyList = camp::list< RAJA::omp_reduce,
                                            RAJA::omp_reduce_ordered >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducerPolicyList = camp::list< RAJA::omp_target_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReducerPolicyList = camp::list< RAJA::cuda_reduce >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipReducerPolicyList = camp::list< RAJA::hip_reduce >;
#endif

#endif  // __TEST_REDUCER_UTILS_HPP__
