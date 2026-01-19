###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# macro that generates test file and build target for each
# sub-test and backend
# this must be a macro or the linker variable set by FindHIP won't be set in
# the right scope and linking will fail with a weird error from
# hipcc_cmake_linker_helper because it expects the path to hipcc as the first
# argument
#
macro( buildfunctionalworkgrouptest TESTNAME SUBTESTNAMES DISPATCHERS BACKENDS )
  foreach( BACKEND ${BACKENDS} )
    foreach( DISPATCHER ${DISPATCHERS} )
      foreach( SUBTESTNAME ${SUBTESTNAMES} )

        configure_file( test-workgroup-${TESTNAME}.cpp.in
                        test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.cpp )

        raja_add_test( NAME test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}
                       SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.cpp )

        target_include_directories( test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.exe
                                    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests )

      endforeach()
    endforeach()
  endforeach()
endmacro()


set(BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND BACKENDS Hip)
endif()

##
## Do not create OpenMP Target or SYCL tests for workgroup, since not supported yet 
##


set(DISPATCHERS IndirectFunction IndirectVirtual Direct)


set(Ordered_SUBTESTS Single MultipleReuse)
buildfunctionalworkgrouptest(Ordered "${Ordered_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

set(Unordered_SUBTESTS Single MultipleReuse)
buildfunctionalworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

unset(DISPATCHERS)
unset(BACKENDS)
unset(Ordered_SUBTESTS)
unset(Unordered_SUBTESTS)
