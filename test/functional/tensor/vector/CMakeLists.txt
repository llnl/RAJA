###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set(TENSOR_VECTOR_TESTS
      CtorGetSet
      MinMax
      SumDot
      FmaFms
      ForallVectorRef1d
      ForallVectorRef2d
   )
				

#
# Generate tensor register tests for each element type, and each register policy
#
foreach( TENSOR_ELEMENT_TYPE ${TENSOR_ELEMENT_TYPES} )
	foreach( TENSOR_VECTOR_TEST ${TENSOR_VECTOR_TESTS} )
	
		set(TEST_NAME test-tensor-vector-${TENSOR_ELEMENT_TYPE}-${TENSOR_VECTOR_TEST})
	
		configure_file( test-tensor-vector.cpp.in  ${TEST_NAME}.cpp )
		
		raja_add_test( NAME ${TEST_NAME} SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp )

		target_include_directories(${TEST_NAME}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

		unset( TEST_NAME )
		
	endforeach()
endforeach()

unset( TENSOR_VECTOR_TESTS )
