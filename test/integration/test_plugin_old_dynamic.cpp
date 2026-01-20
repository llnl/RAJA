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

TEST(PluginTestDynamic, Exception)
{
  RAJA::util::init_plugins("../../lib/libdynamic_plugin_old.so");
  int* a = new int[10];

  ASSERT_ANY_THROW({
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10),
                               [=](int i) { a[i] = 0; });
  });

  delete[] a;
}
