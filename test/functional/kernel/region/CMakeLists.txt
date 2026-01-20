###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND KERNEL_REGION_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND KERNEL_REGION_BACKENDS OpenMP)
endif()


#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( REGION_BACKEND ${KERNEL_REGION_BACKENDS} )
  configure_file( test-kernel-region.cpp.in
                  test-kernel-region-${REGION_BACKEND}.cpp )
  raja_add_test( NAME test-kernel-region-${REGION_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-region-${REGION_BACKEND}.cpp )

  target_include_directories(test-kernel-region-${REGION_BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset( KERNEL_REGION_BACKENDS )


#
# Generate kernel region sync tests for OpenMP.
#
if(RAJA_ENABLE_OPENMP)

set(REGION_BACKEND OpenMP)

configure_file( test-kernel-region-sync.cpp.in
                test-kernel-region-sync-${REGION_BACKEND}.cpp )
raja_add_test( NAME test-kernel-region-sync-${REGION_BACKEND}
               SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-region-sync-${REGION_BACKEND}.cpp )

target_include_directories(test-kernel-region-sync-${REGION_BACKEND}.exe
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

unset(REGION_BACKEND)

endif()
