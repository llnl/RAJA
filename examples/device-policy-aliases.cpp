//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Device policy aliases example
//
// This example demonstrates:
//   - Backend-generic GPU execution policies via RAJA::device_*
//   - CUDA-like x/y/z naming for SYCL (x->2, y->1, z->0) in those aliases
//   - RAJA::Teams::sycl_order / RAJA::Threads::sycl_order for expressing a
//     launch grid in SYCL's (dim0,dim1,dim2) ordering.
//
// Notes:
//   - RAJA::device_* aliases are defined only in a GPU device compile pass
//     (RAJA_GPU_ACTIVE). If you compile this file as host-only, it will fall
//     back to a sequential path.
//

#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/resource.hpp"

static void printDims(const char* label, const int v[3])
{
  std::cout << label << " (x,y,z)=(" << v[0] << "," << v[1] << "," << v[2]
            << ")\n";
}

int main(int, char**)
{
  constexpr int N = 1024;

  // Demonstrate the (dim0,dim1,dim2)->(x,y,z) mapping helpers for RAJA::launch.
  {
    const auto threads_xyz = RAJA::Threads(8, 4, 2);
    const auto threads_sycl = RAJA::Threads::sycl_order(/*dim0=*/2,
                                                        /*dim1=*/4,
                                                        /*dim2=*/8);
    printDims("Threads(x,y,z)", threads_xyz.value);
    printDims("Threads::sycl_order(dim0,dim1,dim2)", threads_sycl.value);
  }

  RAJA::resources::Host host {};
  int* a = host.allocate<int>(N);
  int* b = host.allocate<int>(N);
  int* c = host.allocate<int>(N);

  for (int i = 0; i < N; ++i)
  {
    a[i] = i;
    b[i] = 2 * i;
    c[i] = 0;
  }

#if defined(RAJA_GPU_ACTIVE)
  // Use RAJA::device_exec to select the active GPU backend (CUDA/HIP/SYCL).
  //
  // The matching resource type is inferred from the execution policy.
  auto device_res = RAJA::resources::get_default_resource<RAJA::device_exec<256>>();

  int* d_a = device_res.allocate<int>(N);
  int* d_b = device_res.allocate<int>(N);
  int* d_c = device_res.allocate<int>(N);

  device_res.memcpy(d_a, a, sizeof(int) * N);
  device_res.memcpy(d_b, b, sizeof(int) * N);

  RAJA::forall<RAJA::device_exec<256>>(device_res, RAJA::RangeSegment(0, N),
                                      [=] RAJA_DEVICE(int i) {
                                        d_c[i] = d_a[i] + d_b[i];
                                      });

  device_res.memcpy(c, d_c, sizeof(int) * N);
  device_res.wait();

  device_res.deallocate(d_a);
  device_res.deallocate(d_b);
  device_res.deallocate(d_c);

  std::cout << "Computed on device using RAJA::device_exec<256>\n";
#else
  RAJA::forall<RAJA::seq_exec>(host, RAJA::RangeSegment(0, N), [=](int i) {
    c[i] = a[i] + b[i];
  });
  std::cout << "Computed on host (no RAJA_GPU_ACTIVE)\n";
#endif

  // Quick correctness check.
  bool ok = true;
  for (int i = 0; i < N; ++i)
  {
    if (c[i] != a[i] + b[i])
    {
      ok = false;
      std::cerr << "Mismatch at i=" << i << " got " << c[i]
                << " expected " << (a[i] + b[i]) << "\n";
      break;
    }
  }

  host.deallocate(a);
  host.deallocate(b);
  host.deallocate(c);

  return ok ? 0 : 1;
}

