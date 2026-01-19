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
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMaxLoc ReduceMinLoc)

set(DATATYPES CoreReductionDataTypeList)

#
# List of core reduction types for generating test files.
# Only does a couple as they are covered in the forall tests
set(REDUCETYPES ReduceSum ReduceMin)

set(DATATYPES CoreReductionDataTypeList)


#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: LAUNCH_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${LAUNCH_BACKENDS} )
  if( NOT (BACKEND STREQUAL "Sycl"))
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-reduce.cpp.in
                    test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp)
    raja_add_test( NAME test-launch-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
  endif()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )



#
# List of bitwise reduction types for generating test files.
# ReduceBitAnd ReduceBitOr -- only does one as they are covered
# in the forall tests
set(REDUCETYPES ReduceBitAnd)

set(DATATYPES BitwiseReductionDataTypeList)

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: LAUNCH_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${LAUNCH_BACKENDS} )
  if( NOT (BACKEND STREQUAL "Sycl"))
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-reduce.cpp.in
                    test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
  endif()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )
