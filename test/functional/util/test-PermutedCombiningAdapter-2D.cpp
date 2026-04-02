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
void test_PermutedCombiningAdapter_2D(Segment const& seg0, Segment const& seg1)
{
  using std::begin; using std::end; using std::distance;
  auto seg0_begin = begin(seg0);
  auto seg1_begin = begin(seg1);
  size_t seg_lens[2] = {static_cast<size_t>(seg0.size()),
                        static_cast<size_t>(seg1.size())};

  size_t counters[2] = {0, 0};
  auto adapter = RAJA::make_PermutedCombiningAdapter<Perm>([&](IndexType i0, IndexType i1) {
    ASSERT_EQ(seg0_begin[counters[0]], i0);
    ASSERT_EQ(seg1_begin[counters[1]], i1);
    counters[camp::seq_at<1, Perm>::value] += 1;
    if (counters[camp::seq_at<1, Perm>::value] == seg_lens[camp::seq_at<1, Perm>::value]) {
      counters[camp::seq_at<1, Perm>::value] = 0;
      counters[camp::seq_at<0, Perm>::value] += 1;
    }
  }, seg0, seg1);

  ASSERT_EQ(adapter.size(), seg0.size()*seg1.size());

  auto range = adapter.getRange();

  ASSERT_EQ(distance(begin(range), end(range)), seg0.size()*seg1.size());

  auto range_end = end(range);
  for (auto idx = begin(range); idx != range_end; ++idx) {
    adapter(*idx);
  }
}

template < typename Perm, typename IndexType >
void test_types_PermutedCombiningAdapter_2D(IndexType ibegin0, IndexType iend0,
                                            IndexType ibegin1, IndexType iend1)
{
  RAJA::TypedRangeSegment<IndexType> rseg0(ibegin0, iend0);
  RAJA::TypedRangeSegment<IndexType> rseg1(ibegin1, iend1);
  test_PermutedCombiningAdapter_2D<Perm, IndexType>(rseg0, rseg1);
}

TEST(PermutedCombiningAdapter, test2D)
{
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_IJ, int>(0, 0, 0, 0);
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_JI, int>(0, 5, 0, 0);
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_IJ, int>(0, 0, 0, 5);

  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_JI, int>(0, 3, 0, 4);
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_IJ, long>(-3, 5, 0, 6);
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_JI, long>(4, 13, -2, 7);
  test_types_PermutedCombiningAdapter_2D<RAJA::PERM_IJ, long>(-8, -2, -5, 3);
}
