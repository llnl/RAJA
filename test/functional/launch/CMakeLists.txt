###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND LAUNCH_BACKENDS Sequential)
set(LAUNCH_DIRECT_UNCHECKED_BACKENDS "")

if(RAJA_ENABLE_OPENMP)
  list(APPEND LAUNCH_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND LAUNCH_BACKENDS Cuda)
  list(APPEND LAUNCH_DIRECT_UNCHECKED_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND LAUNCH_BACKENDS Hip)
  list(APPEND LAUNCH_DIRECT_UNCHECKED_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND LAUNCH_BACKENDS Sycl)
endif()

add_subdirectory(run-time-switch)

#Adapted from forall test
add_subdirectory(multi-reduce-nested)

add_subdirectory(reduce-basic)

add_subdirectory(reduce-params)

add_subdirectory(segment)

add_subdirectory(shared_mem)

add_subdirectory(nested_direct_unchecked)

add_subdirectory(nested_direct)

add_subdirectory(nested_loop)

add_subdirectory(tile_icount_tcount_direct_unchecked)

add_subdirectory(tile_icount_tcount_direct)

add_subdirectory(tile_icount_tcount_loop)

add_subdirectory(nested_tile_direct_unchecked)

add_subdirectory(nested_tile_direct)

add_subdirectory(nested_tile_loop)

unset( LAUNCH_BACKENDS )
unset( LAUNCH_DIRECT_UNCHECKED_BACKENDS )

