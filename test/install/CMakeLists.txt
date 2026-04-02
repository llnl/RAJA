###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

configure_file(
     using-with-cmake/host-config.cmake.in
     ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake)

 install( FILES
   using-with-cmake/CMakeLists.txt
   using-with-cmake/using-with-cmake.cpp
   ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake
   DESTINATION examples/RAJA/using-with-cmake)
