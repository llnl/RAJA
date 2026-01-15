###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND SORT_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND SORT_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND SORT_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND SORT_BACKENDS Hip)
endif()

##
## OpenMP Target back-end support missing for these tests
##
# if(RAJA_ENABLE_TARGET_OPENMP)
#   list(APPEND SORT_BACKENDS OpenMPTarget)
# endif()


#
# Generate tests for each enabled RAJA back-end.
#
foreach( SORT_BACKEND ${SORT_BACKENDS} )
  configure_file( test-algorithm-sort.cpp.in
                  test-algorithm-sort-${SORT_BACKEND}.cpp )
  raja_add_test( NAME test-algorithm-sort-${SORT_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-sort-${SORT_BACKEND}.cpp )

  target_include_directories(test-algorithm-sort-${SORT_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( SORT_BACKEND ${SORT_BACKENDS} )
  configure_file( test-algorithm-stable-sort.cpp.in
                  test-algorithm-stable-sort-${SORT_BACKEND}.cpp )
  raja_add_test( NAME test-algorithm-stable-sort-${SORT_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-stable-sort-${SORT_BACKEND}.cpp )

  target_include_directories(test-algorithm-stable-sort-${SORT_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()



macro(RAJA_GENERATE_ALGORITHM_UTIL_TESTS ALG ALG_BACKEND_in ALG_SIZE_in UTIL_ALGS)
  set( ALG_BACKEND ${ALG_BACKEND_in} )
  set( ALG_SIZE ${ALG_SIZE_in} )
  foreach( UTIL_ALG ${UTIL_ALGS} )
    configure_file( test-algorithm-util-${ALG}.cpp.in
                    test-algorithm-util-${ALG}-${UTIL_ALG}-${ALG_BACKEND}.cpp )

    raja_add_test( NAME test-algorithm-util-${ALG}-${UTIL_ALG}-${ALG_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-util-${ALG}-${UTIL_ALG}-${ALG_BACKEND}.cpp )

    target_include_directories(test-algorithm-util-${ALG}-${UTIL_ALG}-${ALG_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  endforeach()
  unset( ALG_SIZE )
  unset( ALG_BACKEND )
endmacro()


set( SEQUENTIAL_UTIL_SORTS Shell Heap Intro Merge )
set( CUDA_UTIL_SORTS       Shell Heap Intro )
set( HIP_UTIL_SORTS        Shell Heap Intro )

RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Sequential Default "${SEQUENTIAL_UTIL_SORTS}" )
RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Sequential Small "Insertion" )

if(RAJA_ENABLE_CUDA)
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Cuda Small "${CUDA_UTIL_SORTS}" )
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Cuda Tiny "Insertion" )
endif()

if(RAJA_ENABLE_HIP)
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Hip Small "${HIP_UTIL_SORTS}" )
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( sort Hip Tiny "Insertion" )
endif()


set( UTIL_REDUCES BinaryTree Accumulate Kahan )

RAJA_GENERATE_ALGORITHM_UTIL_TESTS( reduce Sequential Default "${UTIL_REDUCES}" )

if(RAJA_ENABLE_CUDA)
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( reduce Cuda Small "${UTIL_REDUCES}" )
endif()

if(RAJA_ENABLE_HIP)
  RAJA_GENERATE_ALGORITHM_UTIL_TESTS( reduce Hip Small "${UTIL_REDUCES}" )
endif()


unset( SORT_BACKENDS )
unset( SEQUENTIAL_UTIL_SORTS )
unset( CUDA_UTIL_SORTS )
unset( HIP_UTIL_SORTS )
unset( UTIL_REDUCES )


raja_add_test(
  NAME test-algorithm-util-for_each
  SOURCES test-algorithm-util-for_each.cpp)
