###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate kernel hyperplane tests for each enabled RAJA back-end.
#
set(TESTTYPES 2D 3D)
set(REDUCER_TYPES ParamReduce CaptureReduce)

foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( TEST_TYPE ${TESTTYPES} )
    foreach( REDUCER_TYPE ${REDUCER_TYPES} )
      # Removing Sycl backend, implementation of Hyperplane does not exist
      if( NOT ((BACKEND STREQUAL "Sycl")) AND NOT ((BACKEND STREQUAL "OpenMPTarget")) )
        configure_file( test-kernel-hyperplane-${TEST_TYPE}.cpp.in
                        test-kernel-hyperplane-${TEST_TYPE}-${REDUCER_TYPE}-${BACKEND}.cpp )
        raja_add_test( NAME test-kernel-hyperplane-${TEST_TYPE}-${REDUCER_TYPE}-${BACKEND}
                      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-hyperplane-${TEST_TYPE}-${REDUCER_TYPE}-${BACKEND}.cpp )

        target_include_directories(test-kernel-hyperplane-${TEST_TYPE}-${REDUCER_TYPE}-${BACKEND}.exe
                                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      endif()
    endforeach()
  endforeach()
endforeach()

unset( REDUCER_TYPES )
unset( TESTTYPES )
