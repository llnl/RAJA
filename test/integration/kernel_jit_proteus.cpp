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

// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/test-kernel-jit-proteus | %FILECHECK %s

static volatile int runtime_a = 24;
static volatile int runtime_b = 16;
static volatile int runtime_accum = 1;

int main()
{
#if defined(RAJA_ENABLE_HIP)
  using exec_policy = RAJA::hip_exec<256>;
  using kernel_policy = RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
          RAJA::statement::For<0, RAJA::hip_thread_x_loop,
                               RAJA::statement::Lambda<0>>>>;
#else
  return 0;
#endif

  const int a = runtime_a;
  const int b = runtime_b;
  const bool accum = runtime_accum != 0;

  auto res = RAJA::resources::get_default_resource<exec_policy>();
  int *out_ptr = res.template allocate<int>(1);
  int zero = 0;
  int host_result = -1;

  res.memcpy(out_ptr, &zero, sizeof(int));
  res.wait();

  proteus::enable();
  RAJA::kernel<kernel_policy>(
      RAJA::make_tuple(RAJA::RangeSegment(0, 1)),
      [=, a = RAJA_JIT_VARIABLE(a), b = RAJA_JIT_VARIABLE(b),
       accum = RAJA_JIT_VARIABLE(accum)](int i) RAJA_JIT_COMPILE {
        out_ptr[i] = accum ? (a + b + i) : (a - b);
      });

  res.memcpy(&host_result, out_ptr, sizeof(int));
  res.wait();
  std::printf("kernel result=%d\n", host_result);
  res.deallocate(out_ptr);

  return host_result == 40 ? 0 : 1;
}

// CHECK-DAG: [LambdaSpec] Replacing slot 2 with i8 1
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 16
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 24
// CHECK: kernel result=40
