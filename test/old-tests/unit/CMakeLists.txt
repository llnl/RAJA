###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-sharedmem
  SOURCES test-sharedmem.cpp)

raja_add_test(
  NAME test-simd
  SOURCES test-simd.cpp)

add_subdirectory(cpu)

if(RAJA_ENABLE_CUDA)
  add_subdirectory(cuda)
endif(RAJA_ENABLE_CUDA)
