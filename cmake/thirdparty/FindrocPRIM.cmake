###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#[=======================================================================[.rst:

FindrocPRIM
-------

Finds the rocPRIM package.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``rocPRIM_FOUND``
True if the system has the rocPRIM library.
``rocPRIM_INCLUDE_DIRS``
Include directories needed to use rocPRIM.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``rocPRIM_INCLUDE_DIR``
The directory containing ``rocprim.hpp``.

#]=======================================================================]

include (FindPackageHandleStandardArgs)

find_path(rocPRIM_INCLUDE_DIR
  NAMES rocprim/rocprim.hpp
  HINTS
    ${ROCPRIM_DIR}/
    ${HIP_ROOT_DIR}/../
  PATH_SUFFIXES
    include
    rocprim
    rocprim/include)

find_package_handle_standard_args(
  rocPRIM
  DEFAULT_MSG
  rocPRIM_INCLUDE_DIR)

if (rocPRIM_FOUND)
  set(rocPRIM_INCLUDE_DIRS ${rocPRIM_INCLUDE_DIR})
endif()
