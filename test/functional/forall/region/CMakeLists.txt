###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND FORALL_REGION_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND FORALL_REGION_BACKENDS OpenMP)
endif()


#
# Generate tests for each enabled RAJA back-end.
#
foreach( REGION_BACKEND ${FORALL_REGION_BACKENDS} )
  configure_file( test-forall-region.cpp.in
                  test-forall-region-${REGION_BACKEND}.cpp )
  raja_add_test( NAME test-forall-region-${REGION_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-region-${REGION_BACKEND}.cpp )

  target_include_directories(test-forall-region-${REGION_BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset( FORALL_REGION_BACKENDS )
