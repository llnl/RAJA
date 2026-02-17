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
set(NESTEDTYPES DirectUnchecked)

#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_DIRECT_UNCHECKED_BACKENDS} )
  foreach( NESTEDTYPES ${NESTEDTYPES} )
    configure_file( test-launch-nested.cpp.in
                    test-launch-nested-${NESTEDTYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-${NESTEDTYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-${NESTEDTYPES}-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-${NESTEDTYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( NESTEDTYPES )
