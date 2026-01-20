###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of test types for future expansion, if needed. 
#
set(TESTTYPES Segments)

set( USE_RESOURCE "-resource-" "-" )

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( TESTTYPE ${TESTTYPES} )
      configure_file( test-kernel-basic-single-loop.cpp.in
                      test-kernel${RESOURCE}basic-single-loop-${TESTTYPE}-${BACKEND}.cpp )
      raja_add_test( NAME test-kernel${RESOURCE}basic-single-loop-${TESTTYPE}-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}basic-single-loop-${TESTTYPE}-${BACKEND}.cpp )
      target_include_directories(test-kernel${RESOURCE}basic-single-loop-${TESTTYPE}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endforeach()

unset( USE_RESOURCE )
unset( TESTTYPES )
