//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_HPP__
#define __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_HPP__

#include "nested-loop-BlockReduceSum-impl.hpp"

//
//
// Setup the Nested Loop Multi Lambda g-tests.
//
//
TYPED_TEST_SUITE_P(KernelNestedLoopBlockReduceSumTest);
template <typename T>
class KernelNestedLoopBlockReduceSumTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopBlockReduceSumTest, NestedLoopBlockKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using REDUCE_POL = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<2>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename BlockNestedLoopExec<LOOP_TYPE, REDUCE_POL, LOOP_POLS>::type;

  constexpr bool USE_RES = true;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RES>(LOOP_TYPE(), 1023);
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, REDUCE_POL, USE_RES>(LOOP_TYPE(), 2345);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopBlockReduceSumTest,
                            NestedLoopBlockKernel);

#endif  // __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_REDUCE_SUM_HPP__
