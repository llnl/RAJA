###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of experimental reduction types for generating test files.
#
set(DATATYPES CoreMsgDataTypeList)


#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${FORALL_BACKENDS} )
  configure_file( test-forall-basic-msg.cpp.in
                  test-forall-basic-msg-${BACKEND}.cpp )
  raja_add_test( NAME test-forall-basic-msg-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-msg-${BACKEND}.cpp )

  target_include_directories(test-forall-basic-msg-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
