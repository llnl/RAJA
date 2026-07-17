###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of experimental reduction types for generating test files.
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMaxLoc ReduceMinLoc ReduceMaxLocAlt ReduceMinLocAlt)

set(DATATYPES CoreReductionDataTypeList)


#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-basic-expt-reduce.cpp.in
                    test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-basic-expt-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )

function(raja_add_supported_reduce_tests TESTPREFIX SUITEPREFIX APITAGLIST)
  foreach(BACKEND ${REDUCE_MATRIX_BACKENDS})
    foreach(REDUCETYPE ${REDUCETYPES})
      configure_file(test-forall-basic-supported-reduce.cpp.in
                     ${TESTPREFIX}${REDUCETYPE}-${BACKEND}.cpp)
      raja_add_test(NAME ${TESTPREFIX}${REDUCETYPE}-${BACKEND}
                    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${TESTPREFIX}${REDUCETYPE}-${BACKEND}.cpp)

      target_include_directories(${TESTPREFIX}${REDUCETYPE}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endfunction()

#
# Supported reduce/exec matrix coverage for the legacy and runtime-policy
# reducer interfaces.
#
set(REDUCE_MATRIX_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND REDUCE_MATRIX_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND REDUCE_MATRIX_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND REDUCE_MATRIX_BACKENDS Hip)
endif()

set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)
set(DATATYPES CoreReductionDataTypeList)

raja_add_supported_reduce_tests(test-forall-basic-legacy-
                                LegacyInterface
                                LegacyReducerApiList)
raja_add_supported_reduce_tests(test-forall-basic-runtime-policy-
                                RuntimePolicy
                                RuntimePolicyReducerApiList)

unset( DATATYPES )
unset( REDUCETYPES )

set(REDUCETYPES ReduceBitAnd ReduceBitOr)
set(DATATYPES BitwiseReductionDataTypeList)

raja_add_supported_reduce_tests(test-forall-basic-legacy-
                                LegacyInterface
                                LegacyReducerApiList)
raja_add_supported_reduce_tests(test-forall-basic-runtime-policy-
                                RuntimePolicy
                                RuntimePolicyReducerApiList)

unset( DATATYPES )
unset( REDUCETYPES )
unset( REDUCE_MATRIX_BACKENDS )

#
# List of bitwise reduction types for generating test files.
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)

set(DATATYPES BitwiseReductionDataTypeList)

#
# If building SYCL tests, remove the back-end from
# from the list of tests to generate here for the 
# expt-reduce tests.

if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM FORALL_BACKENDS Sycl)
endif()

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-basic-expt-reduce.cpp.in
                    test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-basic-expt-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-basic-expt-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )

#
# List of core reduction types for generating test files.
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)

set(DATATYPES CoreReductionDataTypeList)

##
## Do not create OpenMP Target tests for "traditional" RAJA reduction interface
##
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
endif()

#
# If building SYCL tests, re-add the back-end to
# from the list of tests to generate here.
#
if(RAJA_ENABLE_SYCL)
  list(APPEND FORALL_BACKENDS Sycl)
endif()

#
# If building SYCL tests, remove the back-end from
# from the list of tests to generate here for the
# core reduction tests.
#
if(RAJA_ENABLE_SYCL)
        list(REMOVE_ITEM REDUCETYPES ReduceMaxLoc)
        list(REMOVE_ITEM REDUCETYPES ReduceMinLoc)
endif()

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-basic-reduce.cpp.in
                    test-forall-basic-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )

#
# List of bitwise reduction types for generating test files.
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)

set(DATATYPES BitwiseReductionDataTypeList)

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-basic-reduce.cpp.in
                    test-forall-basic-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )
