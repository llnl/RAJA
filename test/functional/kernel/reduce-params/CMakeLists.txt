###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#set(LOCTYPES Min2D Max2D Min2DView Max2DView Min2DViewTuple Max2DViewTuple)

#
# If building a subset of openmp target tests, remove the back-end from
# from the list of tests to generate here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
  endif()
endif()

#
# Remove SYCL until kernel reduction support is added
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( LOC_BACKEND ${KERNEL_BACKENDS} )
  #foreach( LOC_TYPE ${LOCTYPES} )
    configure_file( test-kernel-basic-param.cpp.in
                    test-kernel-basic-param-${LOC_BACKEND}.cpp )
    raja_add_test( NAME test-kernel-basic-param-${LOC_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-basic-param-${LOC_BACKEND}.cpp )

    target_include_directories(test-kernel-basic-param-${LOC_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  #endforeach()
endforeach()

#unset( LOCTYPES )
