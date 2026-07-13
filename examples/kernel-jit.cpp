//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// This example mirrors forall-jit.cpp, but uses RAJA::kernel to drive the
// outer parallel loop.

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
using kernel_policy = RAJA::KernelPolicy<
    RAJA::statement::CudaKernelFixed<256,
        RAJA::statement::Tile<0, RAJA::tile_fixed<256>,
                              RAJA::cuda_block_x_direct,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>>>>>;
#elif defined(RAJA_ENABLE_HIP)
using kernel_policy = RAJA::KernelPolicy<
    RAJA::statement::HipKernelFixed<256,
        RAJA::statement::Tile<0, RAJA::tile_fixed<256>,
                              RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<0>>>>>;
#elif defined(RAJA_ENABLE_OPENMP)
using kernel_policy = RAJA::KernelPolicy<
    RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::Lambda<0>>>;
#else
using kernel_policy = RAJA::KernelPolicy<
    RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0>>>;
#endif

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout
        << "Expected ./bin/kernel-jit <num rows> <num cols> <num matrices> "
           "<boolean accum>, for example ./bin/kernel-jit 24 16 1000000 1\n";
    return 1;
  }

  int a = std::stoi(argv[1]);
  int b = std::stoi(argv[2]);
  int N = std::stoi(argv[3]);
  bool accum = std::stoi(argv[4]);

#ifdef RAJA_ENABLE_CUDA
  using resource_policy = RAJA::cuda_exec<256>;
#elif defined(RAJA_ENABLE_HIP)
  using resource_policy = RAJA::hip_exec<256>;
#elif defined(RAJA_ENABLE_OPENMP)
  using resource_policy = RAJA::omp_parallel_for_exec;
#else
  using resource_policy = RAJA::seq_exec;
#endif

  auto res = RAJA::resources::get_default_resource<resource_policy>();
  double *A_ptr = res.template allocate<double>(N * a * b);
  auto A = RAJA::make_permuted_view<RAJA::layout_right>(A_ptr, N, a, b);
  double *B_ptr = res.template allocate<double>(N * a * b);
  auto B = RAJA::make_permuted_view<RAJA::layout_right>(B_ptr, N, b, a);
  double *C_ptr = res.template allocate<double>(N * a * a);
  auto C = RAJA::make_permuted_view<RAJA::layout_right>(C_ptr, N, a, a);

  auto Seg = RAJA::RangeSegment(0, N);

  RAJA::kernel<kernel_policy>(RAJA::make_tuple(Seg), [=](int i) {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        A(i, row, col) = 0;
        B(i, row, col) = 0;
        C(i, row, col) = 0;
      }
    }
  });

  RAJA::Timer aot_timer;
  proteus::disable();
  aot_timer.start();

  RAJA::kernel<kernel_policy>(RAJA::make_tuple(Seg), [=](int i) {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        A(i, row, col) = i % row;
        B(i, row, col) = i % col;
        C(i, row, col) = i % col;
      }
    }
  });

  RAJA::kernel<kernel_policy>(RAJA::make_tuple(Seg), [=](int i) {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        if (!accum) {
          C(i, row, col) = A(i, row, col) * B(i, col, row);
        } else {
          C(i, row, col) += A(i, row, col) * B(i, col, row);
        }
      }
    }
  });

  aot_timer.stop();
  proteus::enable();
  std::cout << "aot total time = " << aot_timer.elapsed() << "\n";

  RAJA::Timer jit_timer;
  jit_timer.start();

  RAJA::kernel<kernel_policy>(
      RAJA::make_tuple(Seg),
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b)](int i)
          RAJA_JIT_COMPILE {
            for (int row = 0; row < a; ++row) {
              for (int col = 0; col < b; ++col) {
                A(i, row, col) = i % row;
                B(i, row, col) = i % col;
                C(i, row, col) = i % col;
              }
            }
          });

  RAJA::kernel<kernel_policy>(
      RAJA::make_tuple(Seg),
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b),
       accum = RAJA_JIT_VARIABLE(accum)](int i) RAJA_JIT_COMPILE {
        for (int row = 0; row < a; ++row) {
          for (int col = 0; col < b; ++col) {
            if (!accum) {
              C(i, row, col) = A(i, row, col) * B(i, col, row);
            } else {
              C(i, row, col) += A(i, row, col) * B(i, col, row);
            }
          }
        }
      });

  jit_timer.stop();

  std::cout << "jit total time = " << jit_timer.elapsed() << "\n";
  std::cout << "speedup = " << aot_timer.elapsed() / jit_timer.elapsed()
            << "\n";

  return 0;
}
