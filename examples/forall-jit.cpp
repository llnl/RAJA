//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// This file demonstrates usage of RAJA with Proteus JIT compilation to accelerate
// types of GPU kernels.  In this example, we show how propagating variables
// as runtime constants with proteus::jit_variable in the capture list of a lambda
// can lead create speedups from branch elimination and better loop scheduling analysis.
// usage example:
// ./bin/forall-jit 24 16             100000               1
//                   ^  ^ matrix dims      ^ problem size  ^ branch conditions

#include <cstdlib>
#include <iostream>
#include <limits>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include "proteus/JitInterface.h"

int main (int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Expected integer threshold passed to example through command line\n";
    return 1;
  }
  int a = std::stoi(argv[1]);
  int b = std::stoi(argv[2]);
  int N =  std::stoi(argv[3]);
  bool accum = std::stoi(argv[4]);
  #ifdef RAJA_ENABLE_CUDA
  using policy = RAJA::cuda_exec<256>;
  #elif defined(RAJA_ENABLE_HIP)
  using policy = RAJA::hip_exec<256>;
  #elif defined(RAJA_ENABLE_OPENMP)
  using policy = RAJA::omp_parallel_for_exec;
  #else
  using policy = RAJA::seq_exec;
  #endif

  auto res = RAJA::resources::get_default_resource<policy>();
  // layout i x a x b
  double *A_ptr = res.template allocate<double>(N * a * b);
  auto A = RAJA::make_permuted_view<RAJA::layout_right>(A_ptr, N, a, b);
  // layout i x b x a
  double *B_ptr = res.template allocate<double>(N * a * b);
  auto B = RAJA::make_permuted_view<RAJA::layout_right>(B_ptr, N, b, a);
  // layout i x a x a.  matrix products of A[i]B[i]
  double *C_ptr = res.template allocate<double>(N * a * a);
  auto C = RAJA::make_permuted_view<RAJA::layout_right>(C_ptr, N, a, a);


  RAJA::Timer aot_timer;
  aot_timer.start();
  // data setup
  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=] (int i) {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        A(i, row, col) = i % row;
        B(i, row, col) = i % col;
        C(i, row, col) = i % col;
      }
    }
  });

  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=] (int i) {
    for (int row = 0; row < a; ++row){
      for (int col = 0; col < b; ++col) {
        if (accum) {
          C(i, row, col) = A(i, row, col) * B(i, col, row);
        }
        else {
          C(i, row, col) += A(i, row, col) * B(i, col, row);
        }
      }
    }
  });
  aot_timer.stop();
  double t = aot_timer.elapsed();
  std::cout << "aot total time = " << t << "\n";
  RAJA::Timer jit_timer;
  jit_timer.start();

  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=,
    a = proteus::jit_variable(a),
    b = proteus::jit_variable(b)
  ]  (int i) RAJA_JIT_COMPILE {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        A(i, row, col) = i % row;
        B(i, row, col) = i % col;
        C(i, row, col) = i % col;
      }
    }
  });

  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=,
    a = proteus::jit_variable(a),
    b = proteus::jit_variable(b),
    accum = proteus::jit_variable(accum)
  ]  (int i) RAJA_JIT_COMPILE {
    for (int row = 0; row < a; ++row){
      for (int col = 0; col < b; ++col) {
        if (accum) {
          C(i, row, col) = A(i, row, col) * B(i, col, row);
        }
        else {
          C(i, row, col) += A(i, row, col) * B(i, col, row);
        }
      }
    }
  });
  jit_timer.stop();

  std::cout << "jit total time = " << jit_timer.elapsed() << "\n";
  std::cout << "speedup = " << aot_timer.elapsed() / jit_timer.elapsed() << "\n";

  return 0;
}
