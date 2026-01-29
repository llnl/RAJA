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

#if defined(RAJA_ENABLE_CUDA)
using CudaBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< CudaReducerPolicyList,
                                 DataTypeList,
                                 CudaResourceList > >::Types;

using CudaInitReducerConstructorTypes = 
  Test< camp::cartesian_product< CudaReducerPolicyList,
                                 DataTypeList,
                                 CudaResourceList,
                                 CudaUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(CudaBasicTest,
                               ReducerBasicConstructorUnitTest,
                               CudaBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(CudaInitTest,
                               ReducerInitConstructorUnitTest,
                               CudaInitReducerConstructorTypes);
#endif

