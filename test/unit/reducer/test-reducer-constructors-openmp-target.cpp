//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "tests/test-reducer-constructors.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#if 0 
// Tests cannot be created since OpenMP Target reduction type constructor is
// explicitly marked deleted, which is inconsistent with other back-ends --RDH
using OpenMPTargetBasicReducerConstructorTypes =
  Test< camp::cartesian_product< OpenMPTargetReducerPolicyList,
                                 DataTypeList,
                                 OpenMPTargetResourceList > >::Types;
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTargetBasicTest,
                               ReducerBasicConstructorUnitTest,
                               OpenMPTargetBasicReducerConstructorTypes);
#else
// This is needed to suppress a runtime test error for uninstantiated test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ReducerBasicConstructorUnitTest);
#endif

using OpenMPTargetInitReducerConstructorTypes = 
  Test< camp::cartesian_product< OpenMPTargetReducerPolicyList,
                                 DataTypeList,
                                 OpenMPTargetResourceList,
                                 SequentialUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTargetInitTest,
                               ReducerInitConstructorUnitTest,
                               OpenMPTargetInitReducerConstructorTypes);
#endif

