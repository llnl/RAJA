//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer reset.
///

#include "tests/test-reducer-reset.hpp"

using SequentialReducerResetTypes = 
  Test< camp::cartesian_product< SequentialReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialUnitTestPolicyList > >::Types;


INSTANTIATE_TYPED_TEST_SUITE_P(SequentialResetTest,
                               ReducerResetUnitTest,
                               SequentialReducerResetTypes);

