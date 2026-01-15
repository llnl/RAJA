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
set(TESTTYPES PermutedView OffsetView PermutedOffsetView)
set(DIMTYPES 2D 3D)

#
# Generate tests for each enabled RAJA back-end.
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( TESTTYPE ${TESTTYPES} )
    foreach( DIM ${DIMTYPES} )
      configure_file( test-kernel-nested-loop-view.cpp.in
                      test-kernel-nested-loop-${TESTTYPE}${DIM}-${BACKEND}.cpp )
      raja_add_test( NAME test-kernel-nested-loop-${TESTTYPE}${DIM}-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${TESTTYPE}${DIM}-${BACKEND}.cpp )
      target_include_directories(test-kernel-nested-loop-${TESTTYPE}${DIM}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endforeach()

unset( DIMTYPES )
unset( TESTTYPES )
