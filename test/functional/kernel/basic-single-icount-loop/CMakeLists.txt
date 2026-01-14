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

foreach( BACKEND ${KERNEL_BACKENDS} )
  configure_file( test-kernel-basic-single-icount-loop.cpp.in
                  test-kernel-basic-single-icount-loop-${BACKEND}.cpp )
  raja_add_test( NAME test-kernel-basic-single-icount-loop-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-basic-single-icount-loop-${BACKEND}.cpp )
  target_include_directories(test-kernel-basic-single-icount-loop-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
