###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: PLUGIN_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-forall.cpp.in
                  test-plugin-forall-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-forall-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-forall-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-forall-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-kernel.cpp.in
                  test-plugin-kernel-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-kernel-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-kernel-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-kernel-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-launch.cpp.in
                  test-plugin-launch-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-launch-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-launch-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-launch-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-resource-launch.cpp.in
                  test-plugin-resource-launch-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-resource-launch-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-resource-launch-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-resource-launch-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

set(DISPATCHERS Direct IndirectFunction IndirectVirtual)

foreach( BACKEND ${PLUGIN_BACKENDS} )
  foreach( DISPATCHER ${DISPATCHERS} )

    configure_file( test-plugin-workgroup.cpp.in
                    test-plugin-workgroup-${DISPATCHER}-${BACKEND}.cpp )

    raja_add_test( NAME test-plugin-workgroup-${DISPATCHER}-${BACKEND}
               SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-workgroup-${DISPATCHER}-${BACKEND}.cpp
                       plugin_to_test.cpp )

    target_include_directories(test-plugin-workgroup-${DISPATCHER}-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  endforeach()
endforeach()

unset(DISPATCHERS)
