//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RAJAVec
///

#include "RAJA_test-base.hpp"

TEST(RAJAVecUnitTest, basic_test)
{
  RAJA::RAJAVec<int> a;

  ASSERT_TRUE(a.empty());
  ASSERT_EQ(0lu, a.size());
  a.push_back(5);
  ASSERT_FALSE(a.empty());
  ASSERT_EQ(5, *a.begin());
  ASSERT_EQ(5, *(a.end() - 1));
  a.push_front(10);
  ASSERT_EQ(10, *a.begin());
  ASSERT_EQ(5, *(a.end() - 1));

  RAJA::RAJAVec<int> a1(a);
  ASSERT_EQ(a.size(), a1.size());
  int* a_data = a.data(); 
  int* a1_data = a1.data(); 
  ASSERT_EQ(a_data[0], a1_data[0]);
  ASSERT_EQ(a_data[1], a1_data[1]);

  a.resize(5, 20);
  ASSERT_EQ(20, a[2]);
  ASSERT_EQ(20, a[3]);
  ASSERT_EQ(20, a[4]);
  ASSERT_EQ(5lu, a.size());

  a.resize(1);
  ASSERT_EQ(1lu, a.size());

  a.resize(0);
  for (int i = 0; i < 100; ++i)
    a.push_back(i);
  ASSERT_EQ(100lu, a.size());

  auto b = a;
  b.resize(0);
  ASSERT_EQ(0lu, b.size());
  ASSERT_EQ(100lu, a.size());

  a.swap(b);
  ASSERT_EQ(0lu, a.size());
  ASSERT_EQ(100lu, b.size());

  RAJA::RAJAVec<int> c;
  for (int i = 0; i < 100; ++i)
    c.push_front(i);
  for (int i = 0; i < 100; ++i)
    ASSERT_EQ(c[i], b[99 - i]);
  ASSERT_EQ(c.data() + c.size(), c.end());
  ASSERT_EQ(c.data(), c.begin());
}
