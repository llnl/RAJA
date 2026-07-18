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
  #
  # If building SYCL tests, remove the back-end from
  # from the list of tests to generate here for the
  # expt-reduce tests.
  #if(BACKEND STREQUAL Sycl)
  #  continue()
  #endif()
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


function(raja_add_reduce_tests TESTTYPE)
  foreach(REDUCE_BACKEND ${FORALL_BACKENDS})
    ##
    ## Do not create OpenMP Target tests for "traditional" RAJA reduction interface
    ##
    #if(REDUCE_BACKEND STREQUAL OpenMPTarget)
    #  continue()
    #endif()
    foreach(REDUCETYPE ${REDUCETYPES})
      if(REDUCE_BACKEND STREQUAL Sycl AND
         (REDUCETYPE STREQUAL ReduceMinLoc OR REDUCETYPE STREQUAL ReduceMaxLoc))
        continue()
      endif()

      configure_file(test-forall-basic-reduce.cpp.in
                     test-forall-basic-${TESTTYPE}-${REDUCETYPE}-${REDUCE_BACKEND}Reduce.cpp)
      raja_add_test(NAME test-forall-basic-${TESTTYPE}-${REDUCETYPE}-${REDUCE_BACKEND}Reduce
                    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-${TESTTYPE}-${REDUCETYPE}-${REDUCE_BACKEND}Reduce.cpp)

      target_include_directories(test-forall-basic-${TESTTYPE}-${REDUCETYPE}-${REDUCE_BACKEND}Reduce.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endfunction()

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)

set(DATATYPES CoreReductionDataTypeList)


raja_add_reduce_tests(Legacy)

unset( DATATYPES )
unset( REDUCETYPES )

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)

set(DATATYPES BitwiseReductionDataTypeList)

raja_add_reduce_tests(Legacy)

unset( DATATYPES )
unset( REDUCETYPES )

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)
set(DATATYPES CoreReductionDataTypeList)

raja_add_reduce_tests(Runtime)

unset( DATATYPES )
unset( REDUCETYPES )

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)
set(DATATYPES BitwiseReductionDataTypeList)

raja_add_reduce_tests(Runtime)

unset( DATATYPES )
unset( REDUCETYPES )
