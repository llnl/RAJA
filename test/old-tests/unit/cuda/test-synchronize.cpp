//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "RAJA_gtest.hpp"

GPU_TEST(SynchronizeTest, CUDA)
{

  double* managed_data;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &managed_data, sizeof(double) * 50);

  RAJA::forall<RAJA::cuda_exec_async<256>>( RAJA::RangeSegment(0, 50),
    [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    managed_data[i] = 1.0 * i;
  });
  RAJA::synchronize<RAJA::cuda_synchronize>();

  RAJA::forall<RAJA::seq_exec>( RAJA::RangeSegment(0, 50),
    [=](RAJA::Index_type i) {
    EXPECT_EQ(managed_data[i], 1.0 * i);
  });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, managed_data);
}
