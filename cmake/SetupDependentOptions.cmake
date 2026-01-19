###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

##
## Here are the CMake dependent options in RAJA.
##

set(RAJA_DEPENDENT_OPTIONS ENABLE_OPENMP ENABLE_CUDA ENABLE_HIP ENABLE_CLANG_CUDA ENABLE_COVERAGE ENABLE_TESTS ENABLE_EXAMPLES ENABLE_BENCHMARKS)
foreach (option ${RAJA_DEPENDENT_OPTIONS})
   if (${RAJA_${option}})
      if (NOT ${option})
         if (RAJA_ALLOW_INCONSISTENT_OPTIONS)
            message(WARNING "RAJA_${option} set to On, but ${option} is Off. Please set ${option} to On to enable this feature.")
         else ()
            message(FATAL_ERROR "RAJA_${option} set to On, but ${option} is Off. Please set ${option} to On enable this feature.")
         endif ()
      endif ()
   endif ()
endforeach ()

cmake_dependent_option(RAJA_ENABLE_OPENMP "Build with OpenMP support" On "ENABLE_OPENMP" Off)
cmake_dependent_option(RAJA_ENABLE_CUDA "Build with CUDA support" On "ENABLE_CUDA" Off)
cmake_dependent_option(RAJA_ENABLE_HIP "Build with HIP support" On "ENABLE_HIP" Off)
cmake_dependent_option(RAJA_ENABLE_CLANG_CUDA "Build with Clang CUDA support" On "ENABLE_CLANG_CUDA" Off)

if (RAJA_ENABLE_CUDA)
   set(RAJA_ENABLE_EXTERNAL_CUB VersionDependent CACHE STRING "Build with external cub")
   set_property(CACHE RAJA_ENABLE_EXTERNAL_CUB PROPERTY STRINGS VersionDependent ON OFF)
else()
   set(RAJA_ENABLE_EXTERNAL_CUB OFF CACHE STRING "Build with external cub")
   set_property(CACHE RAJA_ENABLE_EXTERNAL_CUB PROPERTY STRINGS OFF)
endif()

if (RAJA_ENABLE_HIP)
   set(RAJA_ENABLE_EXTERNAL_ROCPRIM VersionDependent CACHE STRING "Build with external ROCPRIM")
   set_property(CACHE RAJA_ENABLE_EXTERNAL_ROCPRIM PROPERTY STRINGS VersionDependent ON OFF)
else()
   set(RAJA_ENABLE_EXTERNAL_ROCPRIM OFF CACHE STRING "Build with external ROCPRIM")
   set_property(CACHE RAJA_ENABLE_EXTERNAL_ROCPRIM PROPERTY STRINGS OFF)
endif()

cmake_dependent_option(RAJA_ENABLE_COVERAGE "Enable coverage (only supported with GCC)" On "ENABLE_COVERAGE" Off)
cmake_dependent_option(RAJA_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)
cmake_dependent_option(RAJA_ENABLE_EXAMPLES "Build simple examples" On "ENABLE_EXAMPLES" off)
cmake_dependent_option(RAJA_ENABLE_BENCHMARKS "Build benchmarks" On "ENABLE_BENCHMARKS" Off)
