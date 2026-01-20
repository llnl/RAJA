//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for CombiningAdapter class.
///

#include "RAJA_test-base.hpp"

#include "RAJA/util/Span.hpp"
#include "RAJA/util/CombiningAdapter.hpp"

#include "camp/resource.hpp"

#include <numeric>
#include <vector>

template < typename Perm, typename IndexType, typename Segment >
void test_PermutedCombiningAdapter_1D(Segment const& seg0)
{
  using std::begin; using std::end; using std::distance;
  auto seg0_begin = begin(seg0);

  size_t counters[1] = {0};
  auto adapter = RAJA::make_PermutedCombiningAdapter<Perm>([&](IndexType i0) {
    ASSERT_EQ(seg0_begin[counters[0]], i0);
    counters[camp::seq_at<0, Perm>::value] += 1;
  }, seg0);

  ASSERT_EQ(adapter.size(), seg0.size());

  auto range = adapter.getRange();

  ASSERT_EQ(distance(begin(range), end(range)), seg0.size());

  auto range_end = end(range);
  for (auto idx = begin(range); idx != range_end; ++idx) {
    adapter(*idx);
  }
}

template < typename Perm, typename IndexType >
void test_types_PermutedCombiningAdapter_1D(IndexType ibegin0, IndexType iend0)
{
  RAJA::TypedRangeSegment<IndexType> rseg0(ibegin0, iend0);
  test_PermutedCombiningAdapter_1D<Perm, IndexType>(rseg0);
}

TEST(PermutedCombiningAdapter, test1D)
{
  test_types_PermutedCombiningAdapter_1D<RAJA::PERM_I, int>(0, 0);

  test_types_PermutedCombiningAdapter_1D<RAJA::PERM_I, int>(0, 15);
  test_types_PermutedCombiningAdapter_1D<RAJA::PERM_I, int>(-8, 16);
}
