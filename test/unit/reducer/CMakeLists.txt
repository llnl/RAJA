###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# macro that generates test file and build target for each backend
# this must be a macro or the linker variable set by FindHIP won't be set in
# the right scope and linking will fail with a weird error from
# hipcc_cmake_linker_helper because it expects the path to hipcc as the first
# argument
#
macro( buildunitreducetest TESTNAME BACKENDS )
  foreach( BACKEND ${BACKENDS} )

    configure_file( test-reducer-${TESTNAME}.cpp.in
                    test-reducer-${TESTNAME}-${BACKEND}.cpp )

    raja_add_test( NAME test-reducer-${TESTNAME}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-reducer-${TESTNAME}-${BACKEND}.cpp )

    target_include_directories( test-reducer-${TESTNAME}-${BACKEND}.exe
                                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests )

  endforeach()
endmacro()


set(BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND BACKENDS OpenMPTarget)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND BACKENDS Sycl)
endif()



buildunitreducetest(constructors "${BACKENDS}")

buildunitreducetest(reset "${BACKENDS}")



unset(BACKENDS)
