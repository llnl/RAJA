//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_RESOURCE_BASIC_HPP__
#define __TEST_KERNEL_NESTED_LOOP_RESOURCE_BASIC_HPP__

#include "nested-loop-Basic-impl.hpp"

//
// Setup the Nested Loop Basic g-tests.
//
//
TYPED_TEST_SUITE_P(KernelNestedLoopBasicTest);
template <typename T>
class KernelNestedLoopBasicTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopBasicTest, NestedLoopBasicKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<1>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename BasicNestedLoopExec<LOOP_TYPE, LOOP_POLS>::type;

  constexpr bool USE_RES = true;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RES>( LOOP_TYPE(), 1,1,1);
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RES>( LOOP_TYPE(), 40,30,20);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopBasicTest,
                            NestedLoopBasicKernel);

#endif  // __TEST_KERNEL_NESTED_LOOP_RESOURCE_BASIC_HPP__
