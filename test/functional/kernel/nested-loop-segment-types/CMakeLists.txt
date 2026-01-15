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
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

#
# Eventually, enable SYCL tests when we know they work
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

foreach( BACKEND ${KERNEL_BACKENDS} )
  configure_file( test-kernel-nested-loop-segments.cpp.in
                  test-kernel-nested-loop-segments-${BACKEND}.cpp )
  raja_add_test( NAME test-kernel-nested-loop-segments-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-segments-${BACKEND}.cpp )

  target_include_directories(test-kernel-nested-loop-segments-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
