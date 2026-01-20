###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of reduction types for generating test files.
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)

##
## Do not create OpenMP Target tests for "traditional" RAJA reduction interface
##
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
endif()

#
# If building SYCL tests, remove the back-end from
# from the list of tests to generate here for the 
# reduce tests.
#
if(RAJA_ENABLE_SYCL)
        list(REMOVE_ITEM REDUCETYPES ReduceMaxLoc)
        list(REMOVE_ITEM REDUCETYPES ReduceMinLoc)
endif()

#
# Generate tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-segment-multiple-reduce.cpp.in
                    test-forall-segment-multiple-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-segment-multiple-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-segment-multiple-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-segment-multiple-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( REDUCETYPES )
