###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
    configure_file( test-launch-nested-tile-loop.cpp.in
                    test-launch-nested-Tile-Loop-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-Tile-Loop-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-Tile-Loop-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-Tile-Loop-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset( NESTEDTYPES )
