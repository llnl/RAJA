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

#if defined(RAJA_ENABLE_HIP)
using HipBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< HipReducerPolicyList,
                                 DataTypeList,
                                 HipResourceList > >::Types;

using HipInitReducerConstructorTypes = 
  Test< camp::cartesian_product< HipReducerPolicyList,
                                 DataTypeList,
                                 HipResourceList,
                                 HipUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(HipBasicTest,
                               ReducerBasicConstructorUnitTest,
                               HipBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(HipInitTest,
                               ReducerInitConstructorUnitTest,
                               HipInitReducerConstructorTypes);
#endif

