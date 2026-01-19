###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME atomic-histogram
  SOURCES atomic-histogram.cpp)
raja_add_executable(
  NAME atomic-histogram_solution
  SOURCES atomic-histogram_solution.cpp)

raja_add_executable(
  NAME dot-product
  SOURCES dot-product.cpp)
raja_add_executable(
  NAME dot-product_solution
  SOURCES dot-product_solution.cpp)

raja_add_executable(
  NAME kernelintro-execpols
  SOURCES kernelintro-execpols.cpp)
raja_add_executable(
  NAME kernelintro-execpols_solution
  SOURCES kernelintro-execpols_solution.cpp)

raja_add_executable(
  NAME launchintro-execpols
  SOURCES launchintro-execpols.cpp)
raja_add_executable(
  NAME launchintro-execpols_solution
  SOURCES launchintro-execpols_solution.cpp)

raja_add_executable(
  NAME kernelintro-nested-loop-reorder
  SOURCES kernelintro-nested-loop-reorder.cpp)
raja_add_executable(
  NAME kernelintro-nested-loop-reorder_solution
  SOURCES kernelintro-nested-loop-reorder_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose
  SOURCES kernel-matrix-transpose.cpp)
raja_add_executable(
  NAME kernel-matrix-transpose_solution
  SOURCES kernel-matrix-transpose_solution.cpp)

raja_add_executable(
  NAME launch-matrix-transpose
  SOURCES launch-matrix-transpose.cpp)
raja_add_executable(
  NAME launch-matrix-transpose_solution
  SOURCES launch-matrix-transpose_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose-tiled
  SOURCES kernel-matrix-transpose-tiled.cpp)
raja_add_executable(
  NAME kernel-matrix-transpose-tiled_solution
  SOURCES kernel-matrix-transpose-tiled_solution.cpp)

raja_add_executable(
  NAME launch-matrix-transpose-tiled
  SOURCES launch-matrix-transpose-tiled.cpp)
raja_add_executable(
  NAME launch-matrix-transpose-tiled_solution
  SOURCES launch-matrix-transpose-tiled_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose-local-array
  SOURCES kernel-matrix-transpose-local-array.cpp)
raja_add_executable(
  NAME kernel-matrix-transpose-local-array_solution
  SOURCES kernel-matrix-transpose-local-array_solution.cpp)

raja_add_executable(
  NAME launch-matrix-transpose-local-array
  SOURCES launch-matrix-transpose-local-array.cpp)
raja_add_executable(
  NAME launch-matrix-transpose-local-array_solution
  SOURCES launch-matrix-transpose-local-array_solution.cpp)

raja_add_executable(
  NAME offset-layout-stencil
  SOURCES offset-layout-stencil.cpp)
raja_add_executable(
  NAME offset-layout-stencil_solution
  SOURCES offset-layout-stencil_solution.cpp)

raja_add_executable(
  NAME permuted-layout-batch-matrix-multiply
  SOURCES permuted-layout-batch-matrix-multiply.cpp) 
raja_add_executable(
  NAME permuted-layout-batch-matrix-multiply_solution
  SOURCES permuted-layout-batch-matrix-multiply_solution.cpp) 

raja_add_executable(
  NAME reductions
  SOURCES reductions.cpp)
raja_add_executable(
  NAME reductions_solution
  SOURCES reductions_solution.cpp)

raja_add_executable(
  NAME scan
  SOURCES scan.cpp)
raja_add_executable(
  NAME scan_solution
  SOURCES scan_solution.cpp)

raja_add_executable(
  NAME segment-indexset-basics
  SOURCES segment-indexset-basics.cpp)
raja_add_executable(
  NAME segment-indexset-basics_solution
  SOURCES segment-indexset-basics_solution.cpp)

raja_add_executable(
  NAME sort
  SOURCES sort.cpp)
raja_add_executable(
  NAME sort_solution
  SOURCES sort_solution.cpp)

raja_add_executable(
  NAME vector-addition
  SOURCES vector-addition.cpp)
raja_add_executable(
  NAME vector-addition_solution
  SOURCES vector-addition_solution.cpp)

raja_add_executable(
  NAME vertexsum-indexset
  SOURCES vertexsum-indexset.cpp)
raja_add_executable(
  NAME vertexsum-indexset_solution
  SOURCES vertexsum-indexset_solution.cpp)

raja_add_executable(
  NAME view-layout
  SOURCES view-layout.cpp)
raja_add_executable(
  NAME view-layout_solution
  SOURCES view-layout_solution.cpp)

add_subdirectory(tutorial_halfday)
