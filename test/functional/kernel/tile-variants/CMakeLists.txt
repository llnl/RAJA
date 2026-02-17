###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate kernel fixed tile tests for each enabled RAJA back-end.
#
set(TILETYPES Fixed2D Fixed2DSum Fixed2DMinMax)

foreach( TILE_BACKEND ${KERNEL_BACKENDS} )
  foreach( TILE_TYPE ${TILETYPES} )
    #
    # OpenMPTarget tests fail for traditional RAJA reductions
    #
    if( (TILE_TYPE STREQUAL "Fixed2D") OR NOT((TILE_BACKEND STREQUAL "OpenMPTarget")) )
      configure_file( test-kernel-tilefixed.cpp.in
                      test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )
      raja_add_test( NAME test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )

      target_include_directories(test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.exe
                                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif()
  endforeach()
endforeach()

unset( TILETYPES )

#
# Generate kernel dynamic tile tests for each enabled RAJA back-end.
#
set(TILETYPES Dynamic2D)

foreach( TILE_BACKEND ${KERNEL_BACKENDS} )
  foreach( TILE_TYPE ${TILETYPES} )
    #
    # Dynamic tiling not yet implemented for CUDA, HIP, or SYCL
    #
    if( NOT ((TILE_BACKEND STREQUAL "Cuda") OR (TILE_BACKEND STREQUAL "Hip") OR (TILE_BACKEND STREQUAL "Sycl")) )
      configure_file( test-kernel-tiledyn.cpp.in
                      test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )
      raja_add_test( NAME test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )

      target_include_directories(test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif()
  endforeach()
endforeach()

unset( TILETYPES )

#
# Generate kernel local array tile tests for each enabled RAJA back-end.
#
set(TILETYPES LocalArray2D)

foreach( TILE_BACKEND ${KERNEL_BACKENDS} )
  foreach( TILE_TYPE ${TILETYPES} )
    # Removing OpenMPTarget because local array capability does not exist
    if( NOT ((TILE_BACKEND STREQUAL "OpenMPTarget") OR (TILE_BACKEND STREQUAL "Sycl")) )
      configure_file( test-kernel-tilelocal.cpp.in
                      test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )
      raja_add_test( NAME test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.cpp )

      target_include_directories(test-kernel-tile-${TILE_TYPE}-${TILE_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif()
  endforeach()
endforeach()

unset( TILETYPES )
