//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_TEST_UNIT_SYNC_TEST_SYNCHRONIZE_HPP
#define RAJA_TEST_UNIT_SYNC_TEST_SYNCHRONIZE_HPP

#include "RAJA_test-base.hpp"

// A case supplies an execution policy, a resource, and its wait operation.
// Backends with a synchronize policy can use it in synchronize().
struct SequentialSyncCase
{
  using ExecPolicy = RAJA::seq_exec;
  using Resource = camp::resources::Host;

  static void synchronize(Resource& resource) { resource.wait(); }
};

#if defined(RAJA_ENABLE_OPENMP)
struct OpenMPSyncCase
{
  using ExecPolicy = RAJA::omp_parallel_for_exec;
  using Resource = camp::resources::Host;

  // OpenMP forall completes before returning. A barrier is tested separately
  // from inside a parallel region.
  static void synchronize(Resource& resource) { resource.wait(); }
};
#endif

#if defined(RAJA_ENABLE_CUDA)
struct CudaSyncCase
{
  using ExecPolicy = RAJA::cuda_exec_async<256>;
  using Resource = camp::resources::Cuda;

  static void synchronize(Resource&)
  {
    RAJA::synchronize<RAJA::cuda_synchronize>();
  }
};
#endif

#if defined(RAJA_ENABLE_HIP)
struct HipSyncCase
{
  using ExecPolicy = RAJA::hip_exec_async<256>;
  using Resource = camp::resources::Hip;

  static void synchronize(Resource&)
  {
    RAJA::synchronize<RAJA::hip_synchronize>();
  }
};
#endif

template <typename SyncCase>
void testSynchronize()
{
  constexpr RAJA::Index_type length = 50;
  using ExecPolicy = typename SyncCase::ExecPolicy;
  using Resource = typename SyncCase::Resource;

  auto host = camp::resources::Host::get_default();
  auto working = Resource::get_default();
  int* host_data = host.allocate<int>(length);
  int* working_data = working.template allocate<int>(length);

  for (RAJA::Index_type i = 0; i < length; ++i) {
    host_data[i] = -1;
  }
  working.memcpy(working_data, host_data, sizeof(int) * length);
  working.wait();

  RAJA::forall<ExecPolicy>(working, RAJA::RangeSegment(0, length),
                           [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
                             working_data[i] = static_cast<int>(i * 3 + 1);
                           });

  SyncCase::synchronize(working);
  working.memcpy(host_data, working_data, sizeof(int) * length);
  working.wait();

  for (RAJA::Index_type i = 0; i < length; ++i) {
    EXPECT_EQ(host_data[i], i * 3 + 1) << "index " << i;
  }

  working.deallocate(working_data);
  host.deallocate(host_data);
}

#endif  // RAJA_TEST_UNIT_SYNC_TEST_SYNCHRONIZE_HPP
