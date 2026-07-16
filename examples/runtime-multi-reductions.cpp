//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <string>

#include "RAJA/RAJA.hpp"

/*
 *  Runtime MultiReduction Example
 *
 *  This example illustrates runtime selection between host and device
 *  execution while selecting the same multi-reduction policy for both paths.
 *
 *  RAJA features shown:
 *    - `dynamic_forall` loop iteration template method
 *    -  Index range segment
 *    -  Runtime resource selection
 *    -  MultiReduction types
 *
 */

constexpr int N = 1000000;
constexpr int NumBins = 10;
constexpr int BlockSize = 256;

#if defined(RAJA_ENABLE_OPENMP)
using host_exec_policy = RAJA::omp_parallel_for_exec;
#else
using host_exec_policy = RAJA::seq_exec;
#endif

#if defined(RAJA_ENABLE_CUDA)
using device_exec_policy = RAJA::cuda_exec_async<BlockSize>;
#elif defined(RAJA_ENABLE_HIP)
using device_exec_policy = RAJA::hip_exec_async<BlockSize>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using multi_reduce_policy = RAJA::cuda_multi_reduce_atomic;
#elif defined(RAJA_ENABLE_HIP)
using multi_reduce_policy = RAJA::hip_multi_reduce_atomic;
#elif defined(RAJA_ENABLE_OPENMP)
using multi_reduce_policy = RAJA::omp_multi_reduce;
#else
using multi_reduce_policy = RAJA::seq_multi_reduce;
#endif

using exec_policy_list = camp::list<host_exec_policy
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                                    , device_exec_policy
#endif
                                    >;

int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./runtime-multi-reductions host or ./runtime-multi-reductions device");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument (host or device).
  // Example usage ./runtime-multi-reductions host or ./runtime-multi-reductions device
  //
  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./runtime-multi-reductions host or ./runtime-multi-reductions device");
    return 0;
  }

  RAJA::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::HOST; printf("Running RAJA runtime multi-reductions example on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::ExecPlace::DEVICE; printf("Running RAJA runtime multi-reductions example on the device \n"); }

  // _multi_reductions_array_init_start
//
// Define array length
//
  const int num_values = N;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  RAJA::resources::Host host_res;

  int* host_bins = host_res.allocate<int>(num_values);
  int* host_vals = host_res.allocate<int>(num_values);

  for (int i = 0; i < num_values; ++i) {
    host_bins[i] = i % NumBins;
    host_vals[i] = (i % (2 * NumBins)) - NumBins;
  }
  // _multi_reductions_array_init_end

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
// for bin in [0, num_bins)
//  - the sum will be (bin - num_bins/2) * N / num_bins
//  - the min will be bin - num_bins
//  - the max will be bin
//  - the and will be min & max
//  - the or  will be min | max
//

//
// Define index range for iterating over a elements in all examples
//
  // _multi_reductions_range_start
  RAJA::RangeSegment arange(0, num_values);
  // _multi_reductions_range_end

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  RAJA::resources::Cuda device_res;
#elif defined(RAJA_ENABLE_HIP)
  RAJA::resources::Hip device_res;
#elif defined(RAJA_ENABLE_SYCL)
  RAJA::resources::Sycl device_res;
#endif

  // The reducer policy is selected at compile time, but the storage backend
  // follows the runtime resource chosen here.
  RAJA::resources::Resource res =
#if defined(RAJA_ENABLE_CUDA)
      RAJA::Get_Runtime_Resource(host_res, device_res, select_cpu_or_gpu);
#elif defined(RAJA_ENABLE_HIP)
      RAJA::Get_Runtime_Resource(host_res, device_res, select_cpu_or_gpu);
#elif defined(RAJA_ENABLE_SYCL)
      RAJA::Get_Runtime_Resource(host_res, device_res, select_cpu_or_gpu);
#else
      RAJA::Get_Host_Resource(host_res, select_cpu_or_gpu);
#endif

  // Memory follows the selected runtime resource:
  // host path uses host allocations, device path uses device allocations.
  // These are the input buffers read by the kernel on the selected backend.
  int* bins = res.allocate<int>(num_values);
  int* vals = res.allocate<int>(num_values);
  res.memcpy(bins, host_bins, sizeof(int) * num_values);
  res.memcpy(vals, host_vals, sizeof(int) * num_values);

  bool ok = false;

  int policy_index = 0;
  char const* label = "host runtime path";
  if (select_cpu_or_gpu == RAJA::ExecPlace::DEVICE)
  {
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    policy_index = 1;
    label        = "device runtime path";
#else
    RAJA_ABORT_OR_THROW("Device runtime path requires a GPU-enabled build");
#endif
  }

  // Passing `res` keeps the reducer storage on the same host/device backend
  // as the input buffers, so the device path uses device-resident tally
  // storage while the host path stays on host memory.
  //
  // Device-side flow:
  //   1. `res.allocate(...)` places `bins` and `vals` in device memory.
  //   2. `MultiReduce(..., res, NumBins)` allocates its tally storage from
  //      the same backend; CUDA uses managed memory with device preference
  //      for the tally pool, while HIP uses device allocation.
  //   3. The kernel reads the device buffers and updates the device-side
  //      reducer state.
  //   4. `get()` synchronizes and returns the reduced values to the host.
  RAJA::MultiReduceSum<multi_reduce_policy, int> multi_reduce_sum(res,
                                                                  NumBins);
  RAJA::MultiReduceMin<multi_reduce_policy, int> multi_reduce_min(res,
                                                                  NumBins);
  RAJA::MultiReduceMax<multi_reduce_policy, int> multi_reduce_max(res,
                                                                  NumBins);
  RAJA::MultiReduceBitAnd<multi_reduce_policy, int> multi_reduce_and(
      res, NumBins);
  RAJA::MultiReduceBitOr<multi_reduce_policy, int> multi_reduce_or(res,
                                                                   NumBins);

  std::cout << "Running " << label << '\n';

  RAJA::dynamic_forall<exec_policy_list>(
      policy_index, arange, [=] RAJA_HOST_DEVICE(int i) {
        int bin = bins[i];

        multi_reduce_sum[bin] += vals[i];
        multi_reduce_min[bin].min(vals[i]);
        multi_reduce_max[bin].max(vals[i]);
        multi_reduce_and[bin] &= vals[i];
        multi_reduce_or[bin] |= vals[i];
      });

  ok = true;
  const int expected_sum_scale = N / NumBins;
  for (int bin = 0; bin < NumBins; ++bin) {
    const int expected_sum = (bin - (NumBins / 2)) * expected_sum_scale;
    const int expected_min = bin - NumBins;
    const int expected_max = bin;
    const int expected_and = expected_min & expected_max;
    const int expected_or = expected_min | expected_max;

    ok = ok && (multi_reduce_sum.get(bin) == expected_sum);
    ok = ok && (multi_reduce_min.get(bin) == expected_min);
    ok = ok && (multi_reduce_max.get(bin) == expected_max);
    ok = ok && (multi_reduce_and.get(bin) == expected_and);
    ok = ok && (multi_reduce_or.get(bin) == expected_or);
  }

  std::cout << (ok ? "\tresult -- PASS\n" : "\tresult -- FAIL\n");

//----------------------------------------------------------------------------//

//
// Clean up.
//
  res.deallocate(vals);
  res.deallocate(bins);
  host_res.deallocate(host_vals);
  host_res.deallocate(host_bins);

  std::cout << "\n DONE!...\n";
  return ok ? 0 : 1;
}
