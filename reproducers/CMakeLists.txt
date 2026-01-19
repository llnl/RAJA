###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

if(RAJA_ENABLE_CLANG_CUDA)
  add_subdirectory(clangcuda)
endif(RAJA_ENABLE_CLANG_CUDA)

add_subdirectory(openmp-target)
