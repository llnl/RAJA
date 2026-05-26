###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-atomic-incdec
  SOURCES test-atomic-incdec.cpp)

raja_add_test(
  NAME test-atomic-ref-constructor
  SOURCES test-atomic-ref-constructor.cpp)

raja_add_test(
  NAME test-atomic-ref-accessors
  SOURCES test-atomic-ref-accessors.cpp)

raja_add_test(
  NAME test-atomic-ref-exchanges
  SOURCES test-atomic-ref-exchanges.cpp)

raja_add_test(
  NAME test-atomic-ref-addsub
  SOURCES test-atomic-ref-addsub.cpp)

raja_add_test(
  NAME test-atomic-ref-minmax
  SOURCES test-atomic-ref-minmax.cpp)

raja_add_test(
  NAME test-atomic-ref-bitwise
  SOURCES test-atomic-ref-bitwise.cpp)
