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

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< OpenMPReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList > >::Types;

using OpenMPInitReducerConstructorTypes = 
  Test< camp::cartesian_product< OpenMPReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPBasicTest,
                               ReducerBasicConstructorUnitTest,
                               OpenMPBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPInitTest,
                               ReducerInitConstructorUnitTest,
                               OpenMPInitReducerConstructorTypes);
#endif

