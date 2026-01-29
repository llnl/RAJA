//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_HPP__
#define __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_HPP__

#include "nested-loop-MultiLambdaParam-impl.hpp"

//
//
// Setup the Nested Loop Multi Lambda g-tests.
//
//
TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaParamTest);
template <typename T>
class KernelNestedLoopMultiLambdaParamTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopMultiLambdaParamTest, NestedLoopMultiLambdaParamKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<1>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename MultiLambdaParamNestedLoopExec<LOOP_TYPE, LOOP_POLS>::type;

  constexpr bool USE_RES = true;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RES>();
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaParamTest,
                            NestedLoopMultiLambdaParamKernel);

#endif  // __TEST_KERNEL_RESOURCE_NESTED_LOOP_MULTI_LAMBDA_PARAM_HPP__
