###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of test types for generating test files.
#
set(TESTTYPES Depends MultiStream AsyncTime BasicAsyncSemantics JoinAsyncSemantics)

list(APPEND RESOURCE_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND RESOURCE_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND RESOURCE_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND RESOURCE_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND RESOURCE_BACKENDS OpenMPTarget)
endif()

#
# Generate tests for each enabled RAJA back-end. 
# 
foreach( BACKEND ${RESOURCE_BACKENDS} )
  foreach( TESTTYPE ${TESTTYPES} )
    configure_file( test-resource.cpp.in
                    test-resource-${TESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-resource-${TESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-resource-${TESTTYPE}-${BACKEND}.cpp )

    target_include_directories(test-resource-${TESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( TESTTYPES )
