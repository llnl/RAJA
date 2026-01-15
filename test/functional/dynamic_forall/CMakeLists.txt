###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND DYNAMIC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND DYNAMIC_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND DYNAMIC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND DYNAMIC_BACKENDS Hip)
endif()

add_subdirectory(segment)

add_subdirectory(resource-segment)

unset( DYNAMIC_BACKENDS )
