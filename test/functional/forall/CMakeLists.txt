###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND FORALL_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND FORALL_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND FORALL_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND FORALL_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND FORALL_BACKENDS Sycl)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND FORALL_BACKENDS OpenMPTarget)
endif()

add_subdirectory(indexset)
add_subdirectory(indexset-view)

add_subdirectory(segment)
add_subdirectory(segment-view)

add_subdirectory(reduce-basic)
add_subdirectory(reduce-multiple-segment)
add_subdirectory(reduce-multiple-indexset)

add_subdirectory(multi-reduce-basic)

add_subdirectory(resource-indexset)
add_subdirectory(resource-segment)

add_subdirectory(CombiningAdapter)

unset( FORALL_BACKENDS )


#
# Note: Forall atomic tests use their own backend list since atomic
#       constructs are defined for only some of the RAJA back-ends.
list(APPEND FORALL_FAIL_ATOMIC_BACKENDS Sequential)
list(APPEND FORALL_ATOMIC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND FORALL_ATOMIC_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND FORALL_ATOMIC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND FORALL_ATOMIC_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND FORALL_ATOMIC_BACKENDS Sycl)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND FORALL_ATOMIC_BACKENDS OpenMPTarget)
endif()

add_subdirectory(atomic-basic)
add_subdirectory(atomic-view)
add_subdirectory(atomic-ref)

unset( FORALL_ATOMIC_BACKENDS )
unset( FORALL_FAIL_ATOMIC_BACKENDS )

#
# Note: Forall region tests define their backend list in the region
#       test directory since region constructs are defined for only
#       some of the RAJA back-ends.
#
add_subdirectory(region)
