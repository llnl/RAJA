//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_ATOMICPOL__
#define __TEST_ATOMICPOL__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

using SequentialAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
              RAJA::builtin_atomic,
#endif
#if defined(RAJA_ENABLE_CUDA)
              RAJA::cuda_atomic_explicit<RAJA::seq_atomic>,
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::cuda_atomic_explicit<RAJA::builtin_atomic>,
#endif
#endif
#if defined(RAJA_ENABLE_HIP)
              RAJA::hip_atomic_explicit<RAJA::seq_atomic>,
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::hip_atomic_explicit<RAJA::builtin_atomic>,
#endif
#endif
              RAJA::seq_atomic
            >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::omp_atomic,
              RAJA::builtin_atomic,
#endif
#if defined(RAJA_ENABLE_CUDA)
              RAJA::cuda_atomic_explicit<RAJA::omp_atomic>,
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::cuda_atomic_explicit<RAJA::builtin_atomic>,
#endif
#endif
#if defined(RAJA_ENABLE_HIP)
              RAJA::hip_atomic_explicit<RAJA::omp_atomic>,
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::hip_atomic_explicit<RAJA::builtin_atomic>,
#endif
#endif
              RAJA::auto_atomic
            >;
#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
using CudaAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
              RAJA::cuda_atomic_explicit<RAJA::seq_atomic>,
              RAJA::cuda_atomic_explicit<RAJA::builtin_atomic>,
#if defined(RAJA_ENABLE_OPENMP)
              RAJA::cuda_atomic_explicit<RAJA::omp_atomic>,
#endif
#endif
              RAJA::cuda_atomic
            >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)
using HipAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
               RAJA::auto_atomic,
               RAJA::hip_atomic_explicit<RAJA::seq_atomic>,
               RAJA::hip_atomic_explicit<RAJA::builtin_atomic>,
#if defined(RAJA_ENABLE_OPENMP)
               RAJA::hip_atomic_explicit<RAJA::omp_atomic>,
#endif
#endif
               RAJA::hip_atomic
            >;
#endif  // RAJA_ENABLE_HIP

#if defined(RAJA_ENABLE_SYCL)
using SyclAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
               RAJA::auto_atomic,
               RAJA::sycl_atomic_explicit<RAJA::seq_atomic>,
               RAJA::sycl_atomic_explicit<RAJA::builtin_atomic>,
#if defined(RAJA_ENABLE_OPENMP)
               RAJA::sycl_atomic_explicit<RAJA::omp_atomic>,
#endif
#endif
               RAJA::sycl_atomic
            >;
#endif  // RAJA_ENABLE_SYCL

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAtomicPols = OpenMPAtomicPols;
#endif

#endif  // __TEST_ATOMICPOL__
