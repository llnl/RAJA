//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for IndexSet class.
///

#include "RAJA_test-base.hpp"

#include "RAJA/index/IndexSetBuilders.hpp" 

#include "camp/resource.hpp"

#include <numeric>
#include <vector>

TEST(IndexSetBuild, Aligned)
{
  const RAJA::Index_type range_min_length = 8;
  const RAJA::Index_type range_align = 2;

  using RSType = RAJA::RangeSegment;
  using LSType = RAJA::ListSegment;

  //
  // Create index vector containing indices:
  // {0, 1, ..., 15,  17, 18,  20, 21, ..., 27,  29,  30, 31}
  //
  std::vector<RAJA::Index_type> indices(16);
  std::iota(indices.begin(), indices.end(), 0);

  indices.push_back(17);
  indices.push_back(18);

  for (RAJA::Index_type i = 20; i < 28; ++i) {
    indices.push_back(i);
  }

  indices.push_back(29);
  indices.push_back(30);
  indices.push_back(31);

  camp::resources::Resource res{camp::resources::Host::get_default()};
 
  RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment> iset;

  RAJA::buildIndexSetAligned(iset, 
                             res,
                             &indices[0],
                             static_cast<RAJA::Index_type>(indices.size()),
                             range_min_length,
                             range_align);

  ASSERT_EQ(iset.getLength(), indices.size());

  ASSERT_EQ(iset.size(), 5);

  const RSType& s0 = iset.getSegment<const RSType>(0);
  ASSERT_EQ(s0.size(), 16);
  ASSERT_EQ(*s0.begin(), 0);

  const LSType& s1 = iset.getSegment<const LSType>(1);
  ASSERT_EQ(s1.size(), 2);
  ASSERT_EQ(*s1.begin(), 17);

  const RSType& s2 = iset.getSegment<const RSType>(2);
  ASSERT_EQ(s2.size(), 8);
  ASSERT_EQ(*s2.begin(), 20);

  const LSType& s3 = iset.getSegment<const LSType>(3);
  ASSERT_EQ(s3.size(), 1);
  ASSERT_EQ(*s3.begin(), 29);

  const RSType& s4 = iset.getSegment<const RSType>(4);
  ASSERT_EQ(s4.size(), 2);
  ASSERT_EQ(*s4.begin(), 30);
}
