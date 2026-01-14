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

template < typename SegIndexType0, typename SegIndexType1,
           typename Segment0, typename Segment1 >
void test_CombiningAdapter_2D(Segment0 const& seg0, Segment1 const& seg1)
{
  using std::begin; using std::end; using std::distance;
  auto seg0_begin = begin(seg0);
  auto seg1_begin = begin(seg1);
  size_t seg1_len = static_cast<size_t>(seg1.size());

  size_t counter0 = 0;
  size_t counter1 = 0;
  auto adapter = RAJA::make_CombiningAdapter([&](SegIndexType0 i0, SegIndexType1 i1) {
    ASSERT_EQ(seg0_begin[counter0], i0);
    ASSERT_EQ(seg1_begin[counter1], i1);
    counter1 += 1;
    if (counter1 == seg1_len) {
      counter1 = 0;
      counter0 += 1;
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

template < typename SegIndexType0, typename SegIndexType1 >
void test_types_CombiningAdapter_2D(SegIndexType0 ibegin0, SegIndexType0 iend0,
                                    SegIndexType1 ibegin1, SegIndexType1 iend1)
{
  RAJA::TypedRangeSegment<SegIndexType0> rseg0(ibegin0, iend0);
  RAJA::TypedRangeSegment<SegIndexType1> rseg1(ibegin1, iend1);
  test_CombiningAdapter_2D<SegIndexType0, SegIndexType1>(rseg0, rseg1);
}

TEST(CombiningAdapter, test2D)
{
  test_types_CombiningAdapter_2D<int, int>(0, 0, 0, 0);
  test_types_CombiningAdapter_2D<int, int>(0, 5, 0, 0);
  test_types_CombiningAdapter_2D<int, int>(0, 0, 0, 5);

  test_types_CombiningAdapter_2D<int, int>(0, 3, 0, 4);
  test_types_CombiningAdapter_2D<int, long>(-3, 5, 0, 6);
  test_types_CombiningAdapter_2D<long, int>(4, 13, -2, 7);
  test_types_CombiningAdapter_2D<long, long>(-8, -2, -5, 3);
}
