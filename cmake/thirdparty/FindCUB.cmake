###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include (FindPackageHandleStandardArgs)

find_path(CUB_INCLUDE_DIRS
  NAMES cub/cub.cuh
  HINTS
    ${CUB_DIR}/
    ${CUB_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include)

find_package_handle_standard_args(
  CUB
  DEFAULT_MSG
  CUB_INCLUDE_DIRS)

if (CUB_INCLUDE_DIRS)
  set(CUB_FOUND True)
else ()
  set(CUB_FOUND False)
endif()
