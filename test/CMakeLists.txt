###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(include)

add_subdirectory(integration)

add_subdirectory(functional)

add_subdirectory(unit)

add_subdirectory(old-tests)

add_subdirectory(install)

configure_file(${PROJECT_SOURCE_DIR}/test/CTestCustom.cmake ${CMAKE_BINARY_DIR})
