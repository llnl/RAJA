###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_ENABLE_DOCUMENTATION)
  if (SPHINX_FOUND)
    blt_add_sphinx_target( raja_docs )
  endif()

  if (DOXYGEN_FOUND)
    add_subdirectory(doxygen)
  endif()

  if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
    message(WARNING "RAJA_ENABLE_DOCUMENTATION=On, but Sphinx or Doxygen not found. \
      Documentation won't be built.")
  endif ()
endif()
