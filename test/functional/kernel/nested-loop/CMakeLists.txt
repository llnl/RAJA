###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES Basic)

set( USE_RESOURCE "-resource-" "-" )

#
# Generate kernel basic tests for each enabled RAJA back-end.
#
foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      configure_file( test-kernel-nested-loop.cpp.in
                      test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

      raja_add_test( NAME test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

      target_include_directories(test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    endforeach()
  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )


#
# Now, do multi-lambda tests...
#

set(NESTED_LOOPTYPES MultiLambda MultiLambdaParam)

#
# If building SYCL tests, remove the back-end from
# from the list of tests to generate here for the
# multiple lambda tests.
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      configure_file( test-kernel-nested-loop.cpp.in
          test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )
      raja_add_test( NAME test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

      target_include_directories(test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    endforeach()
  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )
