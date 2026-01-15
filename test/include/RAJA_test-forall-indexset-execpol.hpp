//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __RAJA_test_forall_indexset_execpol_HPP__
#define __RAJA_test_forall_indexset_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential execution policy types
using SequentialForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec> >;

//
// Sequential execution policy types for reduction tests.
//
// Note: RAJA::simd_exec does not work with these.
//
using SequentialForallIndexSetReduceExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec> >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPForallIndexSetExecPols =  
  camp::list< RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec> >;

using OpenMPForallIndexSetReduceExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec> >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit,
                               RAJA::omp_target_parallel_for_exec<8>>,
              RAJA::ExecPolicy<RAJA::seq_segit, 
                               RAJA::omp_target_parallel_for_exec_nt> >;

using OpenMPTargetForallIndexSetReduceExecPols = 
      OpenMPTargetForallIndexSetExecPols;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<256>> >;

using CudaForallIndexSetReduceExecPols = CudaForallIndexSetExecPols;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<256>> >;

using HipForallIndexSetReduceExecPols = HipForallIndexSetExecPols;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::sycl_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::sycl_exec<256>> >;

using SyclForallIndexSetReduceExecPols = SyclForallIndexSetExecPols;
#endif

#endif  // __RAJA_test_forall_indexset_execpol_HPP__
