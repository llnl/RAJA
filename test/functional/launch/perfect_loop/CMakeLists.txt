###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

foreach(BACKEND ${LAUNCH_BACKENDS})
  configure_file(test-launch-perfect-loop.cpp.in
                 test-launch-perfect-loop-${BACKEND}.cpp)

  raja_add_test(NAME test-launch-perfect-loop-${BACKEND}
                SOURCES
                  ${CMAKE_CURRENT_BINARY_DIR}/test-launch-perfect-loop-${BACKEND}.cpp)

  target_include_directories(test-launch-perfect-loop-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
