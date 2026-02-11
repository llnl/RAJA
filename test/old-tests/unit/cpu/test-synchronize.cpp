//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#if defined(RAJA_ENABLE_OPENMP)

TEST(SynchronizeTest, omp)
{

  double test_val = 0.0;

#pragma omp parallel shared(test_val)
  {
    if (omp_get_thread_num() == 0) {
      test_val = 5.0;
    }

    RAJA::synchronize<RAJA::omp_synchronize>();

    EXPECT_EQ(test_val, 5.0);
  }
}

#endif
