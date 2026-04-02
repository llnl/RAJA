###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_reproducer(
  NAME reproducer-openmptarget
  SOURCES reproducer-openmptarget.cpp)

raja_add_reproducer(
  NAME reproducer-openmptarget-issue1
  SOURCES reproducer-openmptarget-issue1.cpp)

raja_add_reproducer(
  NAME reproducer-openmptarget-issue2
  SOURCES reproducer-openmptarget-issue2.cpp)

raja_add_reproducer(
  NAME reproducer-openmptarget-issue3
  SOURCES reproducer-openmptarget-issue3.cpp)

