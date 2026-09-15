//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// RAJA::launch with device policy aliases
//
// This example shows how to write a single RAJA::launch kernel using the
// RAJA::device_* aliases so the same source can target CUDA, HIP, or SYCL
// without downstream #if/#ifdef in user code.
//
// It also demonstrates Teams/Threads ordering helpers:
//   - RAJA::Teams(x,y,z) and RAJA::Threads(x,y,z) use RAJA's canonical x/y/z
//     ordering.
//   - RAJA::Teams::sycl_order(dim0,dim1,dim2) and Threads::sycl_order(...)
//     express SYCL's dim0/dim1/dim2 ordering and map explicitly as
//     x = dim2, y = dim1, z = dim0.
//

#include <iostream>
#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/resource.hpp"

/*
 * Define host/device launch policies.
 *
 * When compiling for a GPU backend, the RAJA::device_* aliases resolve to the
 * active backend (CUDA/HIP/SYCL). When compiling host-only, we fall back to
 * pure host policies.
 */
using host_launch = RAJA::seq_launch_t;

#if defined(RAJA_GPU_ACTIVE)
using device_launch = RAJA::device_launch_t<false>;
#endif

using launch_policy = RAJA::LaunchPolicy<
  host_launch
#if defined(RAJA_GPU_ACTIVE)
  , device_launch
#endif
  >;

/*
 * Define team and thread loop policies.
 *
 * These use RAJA::device_* aliases on GPU and expand to seq_exec on host.
 */
using teams_x = RAJA::LoopPolicy<
  RAJA::seq_exec
#if defined(RAJA_GPU_ACTIVE)
  , RAJA::device_block_x_direct
#endif
  >;

using teams_y = RAJA::LoopPolicy<
  RAJA::seq_exec
#if defined(RAJA_GPU_ACTIVE)
  , RAJA::device_block_y_direct
#endif
  >;

using threads_x = RAJA::LoopPolicy<
  RAJA::seq_exec
#if defined(RAJA_GPU_ACTIVE)
  , RAJA::device_thread_x_direct
#endif
  >;

using threads_y = RAJA::LoopPolicy<
  RAJA::seq_exec
#if defined(RAJA_GPU_ACTIVE)
  , RAJA::device_thread_y_direct
#endif
  >;

int main(int argc, char** argv)
{
  constexpr int nteams_x   = 2;
  constexpr int nteams_y   = 3;
  constexpr int nthreads_x = 4;
  constexpr int nthreads_y = 5;

  constexpr int len = nteams_y * nteams_x * nthreads_y * nthreads_x;

  RAJA::ExecPlace place = RAJA::ExecPlace::HOST;

#if defined(RAJA_GPU_ACTIVE)
  if (argc != 2) {
    RAJA_ABORT_OR_THROW("Usage: launch-device-policy-aliases host|device");
  }

  std::string exec_space = argv[1];
  if (!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0)) {
    RAJA_ABORT_OR_THROW("Usage: launch-device-policy-aliases host|device");
  }

  if (exec_space.compare("host") == 0) { place = RAJA::ExecPlace::HOST; }
  if (exec_space.compare("device") == 0) { place = RAJA::ExecPlace::DEVICE; }
#else
  (void)argc;
  (void)argv;
#endif

  // Default host resource for host allocations/checks.
  RAJA::resources::Host host_res {};
  int* out_host = host_res.allocate<int>(len);

  for (int i = 0; i < len; ++i) {
    out_host[i] = -1;
  }

#if defined(RAJA_GPU_ACTIVE)
  auto device_res = RAJA::resources::get_default_resource<device_launch>();
  int* out_device = nullptr;

  if (place == RAJA::ExecPlace::DEVICE) {
    out_device = device_res.allocate<int>(len);
    device_res.memcpy(out_device, out_host, sizeof(int) * len);
    device_res.wait();
  }

  int* out_ptr = (out_device != nullptr) ? out_device : out_host;
#else
  int* out_ptr = out_host;
#endif

  // RAJA launch grid configuration is specified in (x,y,z). For SYCL-minded
  // users, sycl_order(dim0,dim1,dim2) provides an explicit x=dim2, y=dim1,
  // z=dim0 mapping.
  RAJA::LaunchParams params_raja(RAJA::Teams(nteams_x, nteams_y, 1),
                                 RAJA::Threads(nthreads_x, nthreads_y, 1));

  RAJA::LaunchParams params_sycl(
      RAJA::Teams::sycl_order(/*dim0=*/1, /*dim1=*/nteams_y, /*dim2=*/nteams_x),
      RAJA::Threads::sycl_order(/*dim0=*/1, /*dim1=*/nthreads_y,
                                /*dim2=*/nthreads_x));

  // Use the RAJA-ordered params for execution (it is backend-independent).
  // params_sycl is included to demonstrate equivalent spelling.
  (void)params_sycl;

  RAJA::launch<launch_policy>(place,
                              params_raja,
                              [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

    RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, nteams_y), [&] (int by) {
      RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, nteams_x), [&] (int bx) {

        RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, nthreads_y), [&] (int ty) {
          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, nthreads_x), [&] (int tx) {

            int i = (((by * nteams_x) + bx) * nthreads_y + ty) * nthreads_x + tx;
            out_ptr[i] = i;

          });
        });

      });
    });

  });

#if defined(RAJA_GPU_ACTIVE)
  if (place == RAJA::ExecPlace::DEVICE) {
    device_res.memcpy(out_host, out_device, sizeof(int) * len);
    device_res.wait();
  }

  if (out_device != nullptr) {
    device_res.deallocate(out_device);
  }
#endif

  bool ok = true;
  for (int i = 0; i < len; ++i) {
    if (out_host[i] != i) {
      ok = false;
      std::cerr << "Mismatch at i=" << i << " got " << out_host[i]
                << " expected " << i << "\n";
      break;
    }
  }

  host_res.deallocate(out_host);

  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
