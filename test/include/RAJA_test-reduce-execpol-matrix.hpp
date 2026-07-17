//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Supported execution-policy to reduction-policy combinations for reducer tests.
//

#ifndef __RAJA_test_reduce_execpol_matrix_HPP__
#define __RAJA_test_reduce_execpol_matrix_HPP__

#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-reducepol.hpp"

#include "test-forall-basic-reduce-interface.hpp"

using SequentialSupportedReducePols =
    typename camp::join<
        SequentialReducePols
#if defined(RAJA_ENABLE_OPENMP)
        ,
        OpenMPReducePols
#endif
#if defined(RAJA_ENABLE_CUDA)
        ,
        CudaReducePols
#endif
#if defined(RAJA_ENABLE_HIP)
        ,
        HipReducePols
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
        ,
        OpenMPTargetReducePols
#endif
#if defined(RAJA_ENABLE_SYCL)
        ,
        SyclReducePols
#endif
        >::type;

using SequentialSupportedReduceExecPairs =
    camp::cartesian_product<SequentialForallReduceExecPols,
                            SequentialSupportedReducePols>;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPSupportedReducePols =
    typename camp::join<
        OpenMPReducePols
#if defined(RAJA_ENABLE_CUDA)
        ,
        CudaReducePols
#endif
#if defined(RAJA_ENABLE_HIP)
        ,
        HipReducePols
#endif
        >::type;

using OpenMPSupportedReduceExecPairs =
    camp::cartesian_product<OpenMPForallReduceExecPols,
                            OpenMPSupportedReducePols>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetSupportedReduceExecPairs =
    camp::cartesian_product<OpenMPTargetForallReduceExecPols,
                            OpenMPTargetReducePols>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaSupportedReduceExecPairs =
    camp::cartesian_product<CudaForallReduceExecPols, CudaReducePols>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipSupportedReduceExecPairs =
    camp::cartesian_product<HipForallReduceExecPols, HipReducePols>;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclSupportedReduceExecPairs =
    camp::cartesian_product<SyclForallReduceExecPols, SyclReducePols>;
#endif

using SupportedReduceExecPairs =
    typename camp::join<
        SequentialSupportedReduceExecPairs
#if defined(RAJA_ENABLE_OPENMP)
        ,
        OpenMPSupportedReduceExecPairs
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
        ,
        OpenMPTargetSupportedReduceExecPairs
#endif
#if defined(RAJA_ENABLE_CUDA)
        ,
        CudaSupportedReduceExecPairs
#endif
#if defined(RAJA_ENABLE_HIP)
        ,
        HipSupportedReduceExecPairs
#endif
#if defined(RAJA_ENABLE_SYCL)
        ,
        SyclSupportedReduceExecPairs
#endif
        >::type;

#endif  // __RAJA_test_reduce_execpol_matrix_HPP__
