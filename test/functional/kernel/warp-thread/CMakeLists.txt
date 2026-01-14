###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES WarpLoop ReduceMask ReduceWarp)

set( USE_RESOURCE "-resource-" "-" )

#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      if( (${NESTED_LOOP_BACKEND} STREQUAL "Cuda" OR ${NESTED_LOOP_BACKEND} STREQUAL "Hip" ) # allow only device tests
        )
        configure_file( test-kernel-warp-thread.cpp.in
                        test-kernel${RESOURCE}warp-thread-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        raja_add_test( NAME test-kernel${RESOURCE}warp-thread-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}warp-thread-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        target_include_directories(test-kernel${RESOURCE}warp-thread-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      endif()

    endforeach()
  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )
