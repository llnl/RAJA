###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of indexset test types for generating test files.
#
set(INDEXSETTESTTYPES IndexSet IcountIndexSet)

#
# Generate tests for each enabled RAJA back-end.
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( INDEXSETTESTTYPE ${INDEXSETTESTTYPES} )
    configure_file( test-forall-indexset.cpp.in
                    test-forall-${INDEXSETTESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${INDEXSETTESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${INDEXSETTESTTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-${INDEXSETTESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( INDEXSETTESTTYPES )
