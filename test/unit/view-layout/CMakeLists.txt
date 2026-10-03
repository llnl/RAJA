###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-standard-layout
  SOURCES test-standard-layout.cpp)

raja_add_test(
  NAME test-typedlayout
  SOURCES test-typedlayout.cpp)

raja_add_test(
  NAME test-typedview
  SOURCES test-typedview.cpp)

raja_add_test(
  NAME test-makelayout
  SOURCES test-makelayout.cpp)

raja_add_test(
  NAME test-multiview
  SOURCES test-multiview.cpp)

raja_add_test(
  NAME test-indexlayout
  SOURCES test-indexlayout.cpp)

raja_add_test(
  NAME test-subview
  SOURCES test-subview.cpp)

set(SUBVIEW_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND SUBVIEW_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND SUBVIEW_BACKENDS OpenMPTarget)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND SUBVIEW_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND SUBVIEW_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND SUBVIEW_BACKENDS Sycl)
endif()

foreach(SUBVIEW_BACKEND ${SUBVIEW_BACKENDS})
  configure_file(test-subview-backend.cpp.in
                 test-subview-backend-${SUBVIEW_BACKEND}.cpp)
  raja_add_test(
    NAME test-subview-backend-${SUBVIEW_BACKEND}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-subview-backend-${SUBVIEW_BACKEND}.cpp)
  target_include_directories(test-subview-backend-${SUBVIEW_BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset(SUBVIEW_BACKENDS)
