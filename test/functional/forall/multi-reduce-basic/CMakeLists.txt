###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

function(raja_add_multi_reduce_tests TESTTYPE)
  foreach(REDUCE_BACKEND ${FORALL_BACKENDS})
    #
    # Do not create tests for OpenMP Target, support not currently implemented
    #
    if(REDUCE_BACKEND STREQUAL OpenMPTarget)
      continue()
    endif()

    #
    # Do not create tests for SYCL, support not currently implemented
    #
    if(REDUCE_BACKEND STREQUAL Sycl)
      continue()
    endif()

    foreach(REDUCETYPE ${REDUCETYPES})
      configure_file(test-forall-basic-multi-reduce.cpp.in
                     test-forall-basic-${TESTTYPE}-MultiReduce${REDUCETYPE}-${REDUCE_BACKEND}.cpp)
      raja_add_test(NAME test-forall-basic-${TESTTYPE}-MultiReduce${REDUCETYPE}-${REDUCE_BACKEND}
                    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-${TESTTYPE}-MultiReduce${REDUCETYPE}-${REDUCE_BACKEND}.cpp)

      target_include_directories(test-forall-basic-${TESTTYPE}-MultiReduce${REDUCETYPE}-${REDUCE_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endfunction()

#
# List of core reduction types for generating test files.
#
set(REDUCETYPES Sum Min Max BitAnd BitOr)

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#

raja_add_multi_reduce_tests(Legacy)
raja_add_multi_reduce_tests(Runtime)

unset( REDUCETYPES )
