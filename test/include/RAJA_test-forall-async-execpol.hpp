//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout forall tests
//

#ifndef __RAJA_test_forall_async_execpol_HPP__
#define __RAJA_test_forall_async_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

#include "RAJA_test-forall-execpol.hpp"

// Sequential execution policy types
using SequentialAsyncForallExecPols = SequentialForallExecPols;
using SequentialAsyncForallReduceExecPols = SequentialForallReduceExecPols;
using SequentialAsyncForallAtomicExecPols = SequentialForallAtomicExecPols;

#if defined(RAJA_ENABLE_OPENMP)

using OpenMPAsyncForallExecPols = OpenMPForallExecPols;
using OpenMPAsyncForallReduceExecPols = OpenMPForallReduceExecPols;
using OpenMPAsyncForallAtomicExecPols = OpenMPForallAtomicExecPols;

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAsyncForallExecPols = OpenMPTargetForallExecPols;
using OpenMPTargetAsyncForallReduceExecPols = OpenMPTargetForallReduceExecPols;
using OpenMPTargetAsyncForallAtomicExecPols = OpenMPTargetForallAtomicExecPols;

#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaAsyncForallExecPols = camp::list< RAJA::cuda_exec<128, true>,
                                       RAJA::cuda_exec<256, true>,
                                       RAJA::cuda_exec_explicit<256,2, true> >;

using CudaAsyncForallReduceExecPols = CudaForallExecPols;

using CudaAsyncForallAtomicExecPols = CudaForallExecPols;

#endif

#if defined(RAJA_ENABLE_HIP)
using HipAsyncForallExecPols = camp::list< RAJA::hip_exec<128, true>,
                                      RAJA::hip_exec<256, true>  >;

using HipAsyncForallReduceExecPols = HipForallExecPols;

using HipAsyncForallAtomicExecPols = HipForallExecPols;

#endif

#endif  // __RAJA_test_forall_execpol_HPP__
