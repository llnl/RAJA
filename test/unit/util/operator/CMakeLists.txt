###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-operators-math
  SOURCES test-operators-math.cpp)

raja_add_test(
  NAME test-operators-logical
  SOURCES test-operators-logical.cpp)

raja_add_test(
  NAME test-operators-equivalence
  SOURCES test-operators-equivalence.cpp)

raja_add_test(
  NAME test-operators-bitwise-modulus
  SOURCES test-operators-bitwise-modulus.cpp)

raja_add_test(
  NAME test-operators-identity
  SOURCES test-operators-identity.cpp)

