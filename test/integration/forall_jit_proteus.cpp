//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

#include <cstdio>

// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/test-forall-jit-proteus | %FILECHECK %s

static volatile int runtime_a = 24;
static volatile int runtime_b = 16;
static volatile int runtime_accum = 1;

int main()
{
#if defined(RAJA_ENABLE_HIP)
  using policy = RAJA::hip_exec<256>;
#else
  return 0;
#endif

  const int a = runtime_a;
  const int b = runtime_b;
  constexpr int N = 1;
  const bool accum = runtime_accum != 0;

  auto res = RAJA::resources::get_default_resource<policy>();
  double *C_ptr = res.template allocate<double>(N * a * b);
  auto C = RAJA::make_permuted_view<RAJA::layout_right>(C_ptr, N, a, b);

  RAJA::forall<policy>(RAJA::RangeSegment(0, N), [=](int i) {
    for (int row = 0; row < a; ++row) {
      for (int col = 0; col < b; ++col) {
        C(i, row, col) = 0.0;
      }
    }
  });

  proteus::enable();
  RAJA::forall<policy>(
      RAJA::RangeSegment(0, N),
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b),
       accum = RAJA_JIT_VARIABLE(accum)](int i) RAJA_JIT_COMPILE {
        for (int row = 0; row < a; ++row) {
          for (int col = 0; col < b; ++col) {
            double v = static_cast<double>(i + row + col);
            if (accum) {
              C(i, row, col) += v;
            } else {
              C(i, row, col) = v;
            }
          }
        }
      });

  res.wait();
  double host_result = -1.0;
  res.memcpy(&host_result, C_ptr + (N * a * b - 1), sizeof(double));
  res.wait();
  std::printf("forall result=%.1f\n", host_result);
  res.deallocate(C_ptr);
  return host_result == 38.0 ? 0 : 1;
}

// CHECK-DAG: [LambdaSpec] Replacing slot 2 with i8 1
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 16
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 24
// CHECK: forall result=38.0
