//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// This example mirrors forall-jit.cpp, but uses RAJA::launch and
// RAJA::loop to drive the outer parallel loop.

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

using host_launch_policy =
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::omp_launch_t;
#else
    RAJA::seq_launch_t;
#endif

#if defined(RAJA_ENABLE_CUDA)
using launch_policy =
    RAJA::LaunchPolicy<host_launch_policy, RAJA::cuda_launch_t<false>>;
#elif defined(RAJA_ENABLE_HIP)
using launch_policy =
    RAJA::LaunchPolicy<host_launch_policy, RAJA::hip_launch_t<false>>;
#else
using launch_policy = RAJA::LaunchPolicy<host_launch_policy>;
#endif

using loop_policy = RAJA::seq_exec;

#if defined(RAJA_ENABLE_CUDA)
using gpu_global_thread_x_policy = RAJA::cuda_global_thread_x;
#elif defined(RAJA_ENABLE_HIP)
using gpu_global_thread_x_policy = RAJA::hip_global_thread_x;
#elif defined(RAJA_ENABLE_OPENMP)
using gpu_global_thread_x_policy = RAJA::omp_for_exec;
#endif

using global_thread_x = RAJA::LoopPolicy<
    loop_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) ||                       \
    defined(RAJA_ENABLE_OPENMP)
    ,
    gpu_global_thread_x_policy
#endif
    >;

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout
        << "Expected ./bin/launch-jit <num rows> <num cols> <num matrices> "
           "<boolean accum>, for example ./bin/launch-jit 24 16 1000000 1\n";
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
  RAJA::LaunchParams Params(
      RAJA::Teams((N + 255) / 256 == 0 ? 1 : (N + 255) / 256),
      RAJA::Threads(256));

  RAJA::launch<launch_policy>(Params, [=] RAJA_HOST_DEVICE(
                                          RAJA::LaunchContext ctx) {
    RAJA::loop<global_thread_x>(ctx, Seg, [&](int i) {
      for (int row = 0; row < a; ++row) {
        for (int col = 0; col < b; ++col) {
          A(i, row, col) = 0;
          B(i, row, col) = 0;
          C(i, row, col) = 0;
        }
      }
    });
  });

  RAJA::Timer aot_timer;
  proteus::disable();
  aot_timer.start();

  RAJA::launch<launch_policy>(Params, [=] RAJA_HOST_DEVICE(
                                          RAJA::LaunchContext ctx) {
    RAJA::loop<global_thread_x>(ctx, Seg, [&](int i) {
      for (int row = 0; row < a; ++row) {
        for (int col = 0; col < b; ++col) {
          A(i, row, col) = (row == 0) ? 0 : (i % row);
          B(i, row, col) = (col == 0) ? 0 : (i % col);
          C(i, row, col) = (col == 0) ? 0 : (i % col);
        }
      }
    });
  });

  RAJA::launch<launch_policy>(Params, [=] RAJA_HOST_DEVICE(
                                          RAJA::LaunchContext ctx) {
    RAJA::loop<global_thread_x>(ctx, Seg, [&](int i) {
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
  });

  aot_timer.stop();
  std::cout << "aot total time = " << aot_timer.elapsed() << "\n";
  proteus::enable();

  RAJA::Timer jit_timer;
  jit_timer.start();

  RAJA::launch<launch_policy>(
      Params,
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b),
        accum = RAJA_JIT_VARIABLE(accum)]
        RAJA_JIT_COMPILE RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {
        RAJA::loop<global_thread_x>(
            ctx, Seg,
            [&](int i)  {
          for (int row = 0; row < a; ++row) {
            for (int col = 0; col < b; ++col) {
              A(i, row, col) = (row == 0) ? 0 : (i % row);
              B(i, row, col) = (col == 0) ? 0 : (i % col);
              C(i, row, col) = (col == 0) ? 0 : (i % col);
            }
          }
        });
      });

  RAJA::launch<launch_policy>(
      Params,
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b),
       accum = RAJA_JIT_VARIABLE(accum)] RAJA_JIT_COMPILE RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {
        RAJA::loop<global_thread_x>(
            ctx, Seg,
            [&](int i)  {
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
      });

  jit_timer.stop();

  std::cout << "jit total time = " << jit_timer.elapsed() << "\n";
  std::cout << "speedup = " << aot_timer.elapsed() / jit_timer.elapsed()
            << "\n";

  return 0;
}
