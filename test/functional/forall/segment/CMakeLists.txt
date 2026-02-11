###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(SEGTYPES ListSegment RangeSegment RangeStrideSegment)

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( SEGTYPE ${SEGTYPES} )
    configure_file( test-forall-segment.cpp.in
                    test-forall-${SEGTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${SEGTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${SEGTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-${SEGTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGTYPES )
