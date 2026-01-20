###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of core reduction types for generating test files.
#
set(REDUCETYPES Sum Min Max BitAnd BitOr)

##
## Disable traditional RAJA reductions test creation for OpenMP Target
##
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
endif()

#
# If building SYCL tests, remove the back-end to
# from the list of tests to generate here.
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-kernel-nested-multi-reduce.cpp.in
                    test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( REDUCETYPES )


#
# If building a subset of openmp target tests, add tests to build here.
#
#if(RAJA_ENABLE_TARGET_OPENMP)
#  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
#
#    set(BACKEND OpenMPTarget)
#    set(REDUCETYPES ReduceSum)
#
#    foreach( REDUCETYPE ${REDUCETYPES} )
#      configure_file( test-kernel-nested-multi-reduce.cpp.in
#                      test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )
#      raja_add_test( NAME test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}
#                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )
#
#      target_include_directories(test-kernel-nested-MultiReduce${REDUCETYPE}-${BACKEND}.exe
#                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
#    endforeach()
#
#  endif()
#endif()

unset( REDUCETYPES )
