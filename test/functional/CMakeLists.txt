###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

add_subdirectory(forall)

add_subdirectory(dynamic_forall)

add_subdirectory(indexset-build)

add_subdirectory(kernel)

add_subdirectory(scan)

add_subdirectory(workgroup)

add_subdirectory(launch)

if (RAJA_ENABLE_VECTORIZATION)
  add_subdirectory(tensor)
endif()

add_subdirectory(util)
