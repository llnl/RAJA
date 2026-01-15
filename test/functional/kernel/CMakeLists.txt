###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND KERNEL_BACKENDS Sequential)
set(KERNEL_DIRECT_UNCHECKED_BACKENDS "")

if(RAJA_ENABLE_OPENMP)
  list(APPEND KERNEL_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND KERNEL_BACKENDS Cuda)
  list(APPEND KERNEL_DIRECT_UNCHECKED_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND KERNEL_BACKENDS Hip)
  list(APPEND KERNEL_DIRECT_UNCHECKED_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND KERNEL_BACKENDS Sycl)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND KERNEL_BACKENDS OpenMPTarget)
endif()

add_subdirectory(basic-single-loop)

add_subdirectory(basic-single-icount-loop)

add_subdirectory(basic-fission-fusion-loop)

add_subdirectory(conditional-fission-fusion-loop)

add_subdirectory(hyperplane)

add_subdirectory(multi-reduce-nested)

add_subdirectory(nested-loop)

add_subdirectory(nested-loop-reducesum)

add_subdirectory(nested-loop-segment-types)

add_subdirectory(nested-loop-view-types)

add_subdirectory(reduce-loc)

add_subdirectory(reduce-params-multi-lambda)

add_subdirectory(reduce-params)

add_subdirectory(tile-icount-tcount-direct-unchecked)

add_subdirectory(tile-icount-tcount-direct)

add_subdirectory(tile-icount-tcount-loop)

add_subdirectory(tile-variants)

add_subdirectory(warp-thread)

unset( KERNEL_BACKENDS )

#
# Note: Kernel region tests define their backend list in the region
#       test directory since region constructs are defined for only
#       some of the RAJA back-ends.
#
add_subdirectory(region)

