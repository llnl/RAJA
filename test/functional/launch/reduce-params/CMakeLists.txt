###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

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
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-param-expt-reduce.cpp.in
                    test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.cpp)
    raja_add_test( NAME test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
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
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-param-expt-reduce.cpp.in
                    test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-param-expt-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )
