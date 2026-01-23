###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set(TENSOR_REGISTER_TESTS
                GetSet
				Load
				Store
				Gather
				Scatter
				Add
				Subtract
				Multiply
				Divide
				DotProduct
				FMA
				FMS
				Max
				Min
				SegmentedDotProduct
			    SegmentedBroadcastInner
			    SegmentedBroadcastOuter
				SegmentedSumInner
				SegmentedSumOuter)

#
# Generate tensor register tests for each element type, and each register policy
#
foreach( TENSOR_ELEMENT_TYPE ${TENSOR_ELEMENT_TYPES} )
	foreach( TENSOR_REGISTER_TEST ${TENSOR_REGISTER_TESTS} )
	
		set(TEST_NAME test-tensor-register-${TENSOR_ELEMENT_TYPE}-${TENSOR_REGISTER_TEST})
	
		configure_file( test-tensor-register.cpp.in  ${TEST_NAME}.cpp )
		
		raja_add_test( NAME ${TEST_NAME} SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp )

		target_include_directories(${TEST_NAME}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

		unset( TEST_NAME )
		
	endforeach()
endforeach()

unset( TENSOR_REGISTER_TESTS )
