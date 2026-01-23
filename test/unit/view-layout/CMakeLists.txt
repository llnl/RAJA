###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-standard-layout
  SOURCES test-standard-layout.cpp)

raja_add_test(
  NAME test-typedlayout
  SOURCES test-typedlayout.cpp)

raja_add_test(
  NAME test-typedview
  SOURCES test-typedview.cpp)

raja_add_test(
  NAME test-makelayout
  SOURCES test-makelayout.cpp)

raja_add_test(
  NAME test-multiview
  SOURCES test-multiview.cpp)

raja_add_test(
  NAME test-indexlayout
  SOURCES test-indexlayout.cpp)
