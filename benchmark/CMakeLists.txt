###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_ENABLE_CUDA)
  raja_add_benchmark(
    NAME benchmark-host-device-lambda
    SOURCES host-device-lambda-benchmark.cpp)
endif()

if (RAJA_ENABLE_DESUL)
raja_add_benchmark(
  NAME benchmark-atomic
  SOURCES benchmark-atomic.cpp)
endif()

raja_add_benchmark(
  NAME ltimes
  SOURCES ltimes.cpp)

raja_add_benchmark(
  NAME raja_view_blur
  SOURCES raja_view_blur.cpp)
