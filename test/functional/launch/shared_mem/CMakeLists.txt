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
set(SHARED_MEM_TYPES DynamicMem StaticMem)


#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
  foreach( SHARED_MEM_TYPES ${SHARED_MEM_TYPES} )
    if( NOT ((BACKEND STREQUAL "Sycl") AND (SHARED_MEM_TYPES STREQUAL "StaticMem")))
    configure_file( test-launch-shared-mem.cpp.in
                    test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.cpp )

    target_include_directories(test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif()
  endforeach()
endforeach()

unset( SHARED_MEM_TYPES )
