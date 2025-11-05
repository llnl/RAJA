# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME make_permuted_view
  SOURCES make_permuted_view.cpp)

raja_add_executable(
  NAME tut_launch_basic
  SOURCES tut_launch_basic.cpp)

raja_add_executable(
  NAME resource-forall
  SOURCES resource-forall.cpp)

raja_add_executable(
  NAME messages-forall
  SOURCES messages-forall.cpp)

raja_add_executable(
  NAME dynamic-forall
  SOURCES dynamic-forall.cpp)

raja_add_executable(
  NAME forall-param-reductions
  SOURCES forall-param-reductions.cpp)

raja_add_executable(
  NAME forall_multi-reductions
  SOURCES forall_multi-reductions.cpp)

raja_add_executable(
  NAME launch-param-reductions
  SOURCES launch-param-reductions.cpp)

raja_add_executable(
  NAME resource-dynamic-forall
  SOURCES resource-dynamic-forall.cpp)

raja_add_executable(
  NAME tut_daxpy
  SOURCES tut_daxpy.cpp)

raja_add_executable(
  NAME dynamic_mat_transpose
  SOURCES dynamic_mat_transpose.cpp)

raja_add_executable(
  NAME tut_matrix-multiply
  SOURCES tut_matrix-multiply.cpp)

raja_add_executable(
  NAME launch_flatten
  SOURCES launch_flatten.cpp)

raja_add_executable(
  NAME launch_reductions
  SOURCES launch_reductions.cpp)

raja_add_executable(
  NAME resource-runtime-launch
  SOURCES resource-runtime-launch.cpp)

raja_add_executable(
  NAME tut_halo-exchange
  SOURCES tut_halo-exchange.cpp)

raja_add_executable(
  NAME pi-reduce_vs_atomic
  SOURCES pi-reduce_vs_atomic.cpp)

raja_add_executable(
  NAME raja-launch
  SOURCES raja-launch.cpp)

raja_add_executable(
  NAME launch_matrix-multiply
  SOURCES launch_matrix-multiply.cpp)

raja_add_executable(
  NAME jacobi
  SOURCES jacobi.cpp)

raja_add_executable(
  NAME red-black-gauss-seidel
  SOURCES red-black-gauss-seidel.cpp)

raja_add_executable(
  NAME wave-eqn
  SOURCES wave-eqn.cpp)

raja_add_executable(
  NAME multiview
  SOURCES multiview.cpp)

if(RAJA_ENABLE_TARGET_OPENMP)
  #  raja_add_executable(
  #    NAME target-kernel
  #    SOURCES omp-target-kernel.cpp)
  #
  #  raja_add_executable(
  #    NAME omp-target-ltimes
  #    SOURCES omp-target-ltimes.cpp)
endif()

raja_add_executable(
  NAME kernel-dynamic-tile
  SOURCES kernel-dynamic-tile.cpp)

raja_add_executable(
  NAME kernel-reduction
  SOURCES kernel-reduction.cpp)

raja_add_executable(
  NAME resource-kernel
  SOURCES resource-kernel.cpp)

raja_add_executable(
  NAME resource-launch
  SOURCES resource-launch.cpp)

add_subdirectory(plugin)
