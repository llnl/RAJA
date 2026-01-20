###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set( TENSOR_ELEMENT_TYPES  int32_t int64_t float double )

add_subdirectory(register)
add_subdirectory(vector)
add_subdirectory(matrix)


unset( TENSOR_ELEMENT_TYPES )
