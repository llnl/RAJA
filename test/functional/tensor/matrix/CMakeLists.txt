###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set(TENSOR_MATRIX_TESTS
				CtorGetSet
				Load_RowMajor
				Load_ColMajor
				Store_RowMajor
				#Transpose     # AJK:  Disabled, feature not complete yet
                Store_ColMajor
                ET_LoadStore
                ET_Add
                ET_Subtract
                ET_Divide
                ET_MatrixVector
                ET_MatrixMatrixMultiply
                ET_MatrixMatrixMultiplyAdd
                ET_Negate
                #ET_Transpose    # AJK:  Disabled, feature not complete yet
                )
				
set(TENSOR_MATRIX_LAYOUTS
                RowMajor
                ColMajor)

#
# Generate tensor register tests for each element type, and each register policy
#
foreach( TENSOR_ELEMENT_TYPE ${TENSOR_ELEMENT_TYPES} )
	foreach( TENSOR_MATRIX_LAYOUT ${TENSOR_MATRIX_LAYOUTS} )
		foreach( TENSOR_MATRIX_TEST ${TENSOR_MATRIX_TESTS} )
		
			set(TEST_NAME test-tensor-matrix-${TENSOR_ELEMENT_TYPE}-${TENSOR_MATRIX_LAYOUT}-${TENSOR_MATRIX_TEST})
		
			configure_file( test-tensor-matrix.cpp.in  ${TEST_NAME}.cpp )
			
			raja_add_test( NAME ${TEST_NAME} SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp )
	
	        target_include_directories(${TEST_NAME}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
			target_include_directories(${TEST_NAME}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
	
			unset( TEST_NAME )
			
		endforeach()
	endforeach()
endforeach()

unset( TENSOR_MATRIX_LAYOUTS )
unset( TENSOR_MATRIX_TESTS )
