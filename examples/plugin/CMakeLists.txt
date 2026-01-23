###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if(RAJA_ENABLE_CALIPER)
  raja_add_executable(
    NAME raja-forall-caliper
    SOURCES raja-forall-caliper.cpp)
  raja_add_executable(
    NAME raja-launch-caliper
    SOURCES raja-launch-caliper.cpp)
endif()

raja_add_executable(
  NAME plugin-example
  SOURCES test-plugin.cpp counter-plugin.cpp)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  raja_add_executable(
    NAME plugin-example-dynamic
    SOURCES test-plugin-dynamic.cpp)

  raja_add_plugin_library(NAME timer_plugin
                          SHARED TRUE
                          SOURCES timer-plugin.cpp)
endif ()
