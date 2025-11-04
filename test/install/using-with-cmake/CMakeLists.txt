###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

cmake_minimum_required(VERSION 3.23)

project(using_with_cmake)

if(NOT DEFINED RAJA_DIR OR NOT EXISTS ${RAJA_DIR}/lib/cmake/raja/raja-config.cmake)
    message(FATAL_ERROR "Missing required 'RAJA_DIR' variable pointing to an installed RAJA")
endif()

find_package(RAJA REQUIRED
             NO_DEFAULT_PATH 
             PATHS ${RAJA_DIR}/lib/cmake/raja)

add_executable(using-with-cmake using-with-cmake.cpp)
target_link_libraries(using-with-cmake RAJA)
 
