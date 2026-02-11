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
macro( buildunitworkgrouptest TESTNAME SUBTESTNAMES DISPATCHERS BACKENDS )
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

# only need to test WorkStorage once
set(WorkStorage_BACKENDS Sequential)

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

set(DISPATCHERS Direct IndirectFunction IndirectVirtual)


# reduce travis build times with intel compiler
if(RAJA_TEST_EXHAUSTIVE OR NOT RAJA_COMPILER MATCHES "RAJA_COMPILER_Intel")
  set(Constructor_SUBTESTS Single)
  buildunitworkgrouptest(Constructor "${Constructor_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

  if(RAJA_ENABLE_TARGET_OPENMP)
    # WorkGroup dispatcher for OpenMPTarget not implemented yet
    list(REMOVE_ITEM BACKENDS OpenMPTarget)
  endif()

  set(Enqueue_SUBTESTS Single Multiple)
  buildunitworkgrouptest(Enqueue     "${Enqueue_SUBTESTS}"     "${DISPATCHERS}" "${BACKENDS}")

  unset(Constructor_SUBTESTS)
  unset(Enqueue_SUBTESTS)
endif()

set(Dispatcher_SUBTESTS Single)

if(RAJA_ENABLE_TARGET_OPENMP)
  # WorkGroup dispatcher for OpenMPTarget not implemented yet
  list(REMOVE_ITEM BACKENDS OpenMPTarget)
endif()

buildunitworkgrouptest(Dispatcher    "${Dispatcher_SUBTESTS}"  "${DISPATCHERS}" "${BACKENDS}")

set(WorkStorage_SUBTESTS Constructor Iterator InsertCall Multiple)
buildunitworkgrouptest(WorkStorage "${WorkStorage_SUBTESTS}"   "${DISPATCHERS}" "${WorkStorage_BACKENDS}")

unset(Dispatcher_SUBTESTS)
unset(WorkStorage_SUBTESTS)

unset(DISPATCHERS)

unset(BACKENDS)
unset(WorkStorage_BACKENDS)
