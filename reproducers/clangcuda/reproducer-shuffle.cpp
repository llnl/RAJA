//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing clangcuda __shfl/__shfl_sync issue reproducer.
///
/// Immediate Issue: Warp reductions using shuffle intrinsics give wrong results.
///
/// Larger Issue: RAJA reducer tests fail.
///
/// Seen when compiling for sm_60 on Sierra EA systems (IBM p8, Nvidia P100)
/// Seen when compiling for sm_70 on Sierra systems (IBM p9, Nvidia V100)
///
/// Notes: The instructions splitting the double into two 32-bit registers
/// used by shfl are missing in the PTX.
///

#include <cassert>
#include <cstdio>

__global__ void shuffle_warp_reduce(double* d)
{
   double val = d[threadIdx.x];

   for (int laneMask = 1; laneMask < 32; laneMask *= 2) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
      val += __shfl_xor_sync(0xffffffffu, val, laneMask);
#else
      // use multiple __shfl instructions to __shfl double
      constexpr size_t ints_per_double = (sizeof(double)+sizeof(int)-1)/sizeof(int);
      union double_int {
        double d;
        int i[ints_per_double];
      };
      double_int di {.d = val};
      for (size_t ip = 0; ip < ints_per_double; ++ip) {
         di.i[ip] = __shfl_xor(di.i[ip], laneMask);
      }
      val += di.d;
#endif
   }

   d[threadIdx.x] = val;
}

int main(int, char**)
{

   double* d;
   assert(cudaHostAlloc(&d, 32*sizeof(double), cudaHostAllocDefault) == cudaSuccess);

   for (int i = 0; i < 32; ++i) {
      d[i] = 1.0;
   }

   shuffle_warp_reduce<<<1,32>>>(d);
   assert(cudaDeviceSynchronize() == cudaSuccess);

   int wrong = 0;
   for (int i = 0; i < 32; ++i) {
      if (d[i] != 32.0) wrong = 1;
   }

   assert(cudaFreeHost(d) == cudaSuccess);

   if (wrong) {
      printf("Got wrong answer.\n");
   } else {
      printf("Got correct answer.\n");
   }

   return wrong;
}
