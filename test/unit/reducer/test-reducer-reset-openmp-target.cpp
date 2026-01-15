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

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducerResetTypes = 
  Test< camp::cartesian_product< OpenMPTargetReducerPolicyList,
                                 DataTypeList,
                                 OpenMPTargetResourceList,
                                 SequentialUnitTestPolicyList > >::Types;


INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTargetResetTest,
                               ReducerResetUnitTest,
                               OpenMPTargetReducerResetTypes);
#endif
