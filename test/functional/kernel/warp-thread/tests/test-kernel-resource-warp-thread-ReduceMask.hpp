//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_WARP_THREAD_RESOURCE_REDUCEMASK_HPP__
#define __TEST_WARP_THREAD_RESOURCE_REDUCEMASK_HPP__

#include "warp-thread-ReduceMask-impl.hpp"

//
//
// Setup the Warp Reduction ReduceMask g-tests.
//
//
TYPED_TEST_SUITE_P(KernelWarpThreadReduceMaskTest);
template <typename T>
class KernelWarpThreadReduceMaskTest : public ::testing::Test {};

TYPED_TEST_P(KernelWarpThreadReduceMaskTest, WarpThreadReduceMaskKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using REDUCE_POL = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<2>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename WarpThreadExec<LOOP_TYPE, REDUCE_POL, LOOP_POLS>::type;

  constexpr bool USE_RES = true;

  // For double nested loop tests the third arg is ignored.
  KernelWarpThreadTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RES>( LOOP_TYPE(), 64, 4*123 );
}

REGISTER_TYPED_TEST_SUITE_P(KernelWarpThreadReduceMaskTest,
                            WarpThreadReduceMaskKernel);

#endif  // __TEST_WARP_THREAD_RESOURCE_REDUCEMASK_HPP__
