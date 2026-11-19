###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# Add backend names here as their synchronization traits are implemented.
set(SYNC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND SYNC_BACKENDS OpenMP)

  raja_add_test(
    NAME test-synchronize-OpenMP-barrier
    SOURCES test-openmp-barrier.cpp
    NUM_OMP_THREADS 4)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND SYNC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND SYNC_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND SYNC_BACKENDS OpenMPTarget)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND SYNC_BACKENDS Sycl)
endif()

foreach(BACKEND ${SYNC_BACKENDS})
  configure_file(test-synchronize.cpp.in
                 test-synchronize-${BACKEND}.cpp)

  raja_add_test(
    NAME test-synchronize-${BACKEND}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-synchronize-${BACKEND}.cpp)

  target_include_directories(test-synchronize-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
