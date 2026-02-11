//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  double* a = new double[10];

  for (int ii = 0; ii < 10; ii++)
  {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
      a[i] = 0;
    });
  }
}
