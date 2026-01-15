###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES ReduceSum BlockReduceSum)

set( USE_RESOURCE "-resource-" "-" )

#
# Intel OneAPI emits a clang front-end error for the OpenMP version of these
# tests. So don't generate them.
#
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(RAJA_ENABLE_OPENMP)
    list(REMOVE_ITEM KERNEL_BACKENDS OpenMP)
  endif()
endif()

##
## Don't create OpenMP Target tests for "traditional" RAJA reduce interface
## 
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
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
foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      if( ${NESTED_LOOP_TYPE} STREQUAL "ReduceSum" OR # allow all ReduceSum tests
          ((${NESTED_LOOP_BACKEND} STREQUAL "Sequential" OR ${NESTED_LOOP_BACKEND} STREQUAL "Cuda" OR ${NESTED_LOOP_BACKEND} STREQUAL "Hip" ) AND ${NESTED_LOOP_TYPE} STREQUAL "BlockReduceSum") # allow only certain BlockReduceSum tests
        )
        # Note on BlockReduceSum: Inherent kernel reduction functionality does not exist for - OpenMP and OpenMPTarget.
        configure_file( test-kernel-nested-loop.cpp.in
                        test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        raja_add_test( NAME test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        target_include_directories(test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      endif()

    endforeach()
  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )
