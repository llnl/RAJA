###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND PLUGIN_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND PLUGIN_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND PLUGIN_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND PLUGIN_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
  endif()
endif()

add_subdirectory(plugin)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  if(NOT WIN32)
  raja_add_test(
    NAME test-plugin-dynamic
    SOURCES test_plugin_dynamic.cpp)

  raja_add_plugin_library(NAME dynamic_plugin
                          SHARED TRUE
                          SOURCES plugin_for_test_dynamic.cpp)

  raja_add_test(
    NAME test-plugin-old-dynamic
    SOURCES test_plugin_old_dynamic.cpp)

  raja_add_plugin_library(NAME dynamic_plugin_old
                          SHARED TRUE
                          SOURCES plugin_old_for_test_dynamic.cpp)

  raja_add_test(
    NAME test-plugin-kokkos
    SOURCES test_plugin_kokkos.cpp)

  raja_add_plugin_library(NAME kokkos_plugin
                          SHARED TRUE
                          SOURCES plugin_for_test_kokkos.cpp)

  set_tests_properties(test-plugin-kokkos.exe PROPERTIES
                      ENVIRONMENT "KOKKOS_PLUGINS=${CMAKE_BINARY_DIR}/lib/libkokkos_plugin.so")
  endif()
endif ()
