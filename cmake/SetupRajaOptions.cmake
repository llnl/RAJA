###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

set(RAJA_ENABLE_WARNINGS_AS_ERRORS Off CACHE BOOL "")
set(ENABLE_GTEST_DEATH_TESTS On CACHE BOOL "Enable tests asserting failure.")

option(RAJA_ENABLE_NVTX "Build with NVTX support" Off)
option(RAJA_ENABLE_ROCTX "Build with ENABLE_ROCTX support" Off)

option(RAJA_ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
option(RAJA_ENABLE_SYCL "Build SYCL support" Off)

option(RAJA_ENABLE_VECTORIZATION "Build experimental vectorization support" On)

option(RAJA_ENABLE_OPENMP_TASK "Build OpenMP task variants of certain algorithms" Off)

option(RAJA_ALLOW_OPENMP_5_1_ATOMICS "Enable OpenMP 5.1 atomics for min/max operations" Off)

option(RAJA_ENABLE_REPRODUCERS "Build issue reproducers" Off)

option(RAJA_ENABLE_EXERCISES "Build exercises " On)
option(RAJA_ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(RAJA_ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
option(RAJA_ENABLE_FORCEINLINE_RECURSIVE "Enable Forceinline recursive (only supported with Intel compilers)" On)

option(RAJA_DEPRECATED_TESTS "Test deprecated features" Off)
option(RAJA_ENABLE_BOUNDS_CHECK "Enable bounds checking in RAJA::Views/Layouts" Off)
option(RAJA_TEST_EXHAUSTIVE "Build RAJA exhaustive tests" Off)
option(RAJA_TEST_OPENMP_TARGET_SUBSET "Build subset of RAJA OpenMP target tests when it is enabled" On)
option(RAJA_ENABLE_RUNTIME_PLUGINS "Enable support for loading plugins at runtime" Off)
option(RAJA_ALLOW_INCONSISTENT_OPTIONS "Enable inconsistent values for ENABLE_X and RAJA_ENABLE_X options" Off)

option(RAJA_ENABLE_DESUL_ATOMICS "Enable support of desul atomics" Off)
set(DESUL_ENABLE_TESTS Off CACHE BOOL "")

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")
