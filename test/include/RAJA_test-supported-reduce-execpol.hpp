//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Supported execution-policy lists for reducer tests.
//

#ifndef __RAJA_test_supported_reduce_execpol_HPP__
#define __RAJA_test_supported_reduce_execpol_HPP__

#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-reducepol.hpp"

using SequentialReduceSupportedForallExecPols = SequentialForallReduceExecPols;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPReduceSupportedForallExecPols =
    typename camp::join<SequentialForallReduceExecPols,
                        OpenMPForallReduceExecPols>::type;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReduceSupportedForallExecPols =
    typename camp::join<SequentialForallReduceExecPols,
                        OpenMPTargetForallReduceExecPols>::type;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReduceSupportedForallExecPols =
    typename camp::join<SequentialForallReduceExecPols,
#if defined(RAJA_ENABLE_OPENMP)
                        OpenMPForallReduceExecPols,
#endif
                        CudaForallReduceExecPols>::type;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipReduceSupportedForallExecPols =
    typename camp::join<SequentialForallReduceExecPols,
#if defined(RAJA_ENABLE_OPENMP)
                        OpenMPForallReduceExecPols,
#endif
                        HipForallReduceExecPols>::type;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclReduceSupportedForallExecPols =
    typename camp::join<SequentialForallReduceExecPols,
                        SyclForallReduceExecPols>::type;
#endif

#endif  // __RAJA_test_supported_reduce_execpol_HPP__
