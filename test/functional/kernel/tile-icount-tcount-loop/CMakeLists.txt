###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of test types for future expansion, if needed.
#
set(TESTTYPES ForICount TileTCount)
set(TILESIZES 8 32)
set(REDUCER_TYPES ParamReduce CaptureReduce)

#
# Generate tests for each enabled RAJA back-end.
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${KERNEL_BACKENDS} )
  # using omp target crashes the compiler with this one
  if( NOT ((BACKEND STREQUAL "OpenMPTarget")) )
    foreach( TESTTYPE ${TESTTYPES} )
      foreach( TILESIZE ${TILESIZES} )
        foreach( REDUCER_TYPE ${REDUCER_TYPES} )
          # The sycl backend does not support parameter based reductions.
          if ((BACKEND STREQUAL "Sycl") AND (REDUCER_TYPE STREQUAL "ParamReduce"))
            continue()
          endif()
          configure_file( test-kernel-tile-count-loop.cpp.in
                          test-kernel-${TESTTYPE}-${TILESIZE}-${REDUCER_TYPE}-${BACKEND}-loop.cpp )
          raja_add_test( NAME test-kernel-${TESTTYPE}-${TILESIZE}-${REDUCER_TYPE}-${BACKEND}-loop
                        SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-${TESTTYPE}-${TILESIZE}-${REDUCER_TYPE}-${BACKEND}-loop.cpp )
          target_include_directories(test-kernel-${TESTTYPE}-${TILESIZE}-${REDUCER_TYPE}-${BACKEND}-loop.exe
                                    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        endforeach()
      endforeach()
    endforeach()
  endif()
endforeach()

unset( REDUCER_TYPES )
unset( TILESIZES )
unset( TESTTYPES )
