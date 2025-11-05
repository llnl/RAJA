//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing test types for atomic ref unit tests
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

using basic_types =
    ::testing::Types<
                      std::tuple<int, RAJA::builtin_atomic>,
                      std::tuple<int, RAJA::seq_atomic>,
                      std::tuple<unsigned int, RAJA::builtin_atomic>,
                      std::tuple<unsigned int, RAJA::seq_atomic>,
                      std::tuple<unsigned long long int, RAJA::builtin_atomic>,
                      std::tuple<unsigned long long int, RAJA::seq_atomic>,
                      std::tuple<float, RAJA::builtin_atomic>,
                      std::tuple<float, RAJA::seq_atomic>,
                      std::tuple<double, RAJA::builtin_atomic>,
                      std::tuple<double, RAJA::seq_atomic>
#if defined(RAJA_COMPILER_MSVC) ||  \
    ((defined(_WIN32) || defined(_WIN64)) && defined(__INTEL_COMPILER))
// Test unsigned fixed width integer types for Windows builds
                      , 
                      std::tuple<uint8_t, RAJA::builtin_atomic>,
                      std::tuple<uint16_t, RAJA::builtin_atomic>,
                      std::tuple<uint32_t, RAJA::builtin_atomic>,
                      std::tuple<uint64_t, RAJA::builtin_atomic>
#endif
#if defined(RAJA_ENABLE_OPENMP)
                      ,
                      std::tuple<int, RAJA::omp_atomic>,
                      std::tuple<unsigned int, RAJA::omp_atomic>,
                      std::tuple<unsigned long long int, RAJA::omp_atomic>,
                      std::tuple<float, RAJA::omp_atomic>,
                      std::tuple<double, RAJA::omp_atomic>
#endif
#if defined(RAJA_ENABLE_CUDA)
                      ,
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<float, RAJA::cuda_atomic>,
                      std::tuple<double, RAJA::auto_atomic>,
                      std::tuple<double, RAJA::cuda_atomic>
#endif
#if defined(RAJA_ENABLE_HIP)
                      ,
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::hip_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::hip_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::hip_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<float, RAJA::hip_atomic>,
                      std::tuple<double, RAJA::auto_atomic>,
                      std::tuple<double, RAJA::hip_atomic>
#endif
                    >;

#if defined(RAJA_ENABLE_CUDA)
using CUDA_types =
    ::testing::Types<
                      std::tuple<int, RAJA::auto_atomic>,
                      std::tuple<int, RAJA::cuda_atomic>,
                      std::tuple<unsigned int, RAJA::auto_atomic>,
                      std::tuple<unsigned int, RAJA::cuda_atomic>,
                      std::tuple<unsigned long long int, RAJA::auto_atomic>,
                      std::tuple<unsigned long long int, RAJA::cuda_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<float, RAJA::auto_atomic>,
                      std::tuple<double, RAJA::cuda_atomic>,
                      std::tuple<double, RAJA::cuda_atomic>
                    >;
#endif

