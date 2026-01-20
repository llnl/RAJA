###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND SCAN_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND SCAN_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND SCAN_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND SCAN_BACKENDS Hip)
endif()


set(SCAN_TYPES Exclusive ExclusiveInplace Inclusive InclusiveInplace)

#
# Generate scan tests for each enabled RAJA back-end.
#
foreach( SCAN_BACKEND ${SCAN_BACKENDS} )
  foreach( SCAN_TYPE ${SCAN_TYPES} )
    configure_file( test-scan.cpp.in
                    test-${SCAN_TYPE}-scan-${SCAN_BACKEND}.cpp )
    raja_add_test( NAME test-${SCAN_TYPE}-scan-${SCAN_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-${SCAN_TYPE}-scan-${SCAN_BACKEND}.cpp )

    target_include_directories(test-${SCAN_TYPE}-scan-${SCAN_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  endforeach()
endforeach()

unset( SCAN_TYPES )
unset( SCAN_BACKENDS )
