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

template < typename SegIndexType, typename Segment0 >
void test_CombiningAdapter_1D(Segment0 const& seg0)
{
  using std::begin; using std::end; using std::distance;
  auto seg0_begin = begin(seg0);

  size_t counter0 = 0;
  auto adapter = RAJA::make_CombiningAdapter([&](SegIndexType i0) {
    ASSERT_EQ(seg0_begin[counter0], i0);
    counter0 += 1;
  }, seg0);

  ASSERT_EQ(adapter.size(), seg0.size());

  auto range = adapter.getRange();

  ASSERT_EQ(distance(begin(range), end(range)), seg0.size());

  auto range_end = end(range);
  for (auto idx = begin(range); idx != range_end; ++idx) {
    adapter(*idx);
  }
}

template < typename SegIndexType >
void test_types_CombiningAdapter_1D(SegIndexType ibegin0, SegIndexType iend0)
{
  RAJA::TypedRangeSegment<SegIndexType> rseg0(ibegin0, iend0);
  test_CombiningAdapter_1D<SegIndexType>(rseg0);
}

TEST(CombiningAdapter, test1D)
{
  test_types_CombiningAdapter_1D<int>(0, 0);

  test_types_CombiningAdapter_1D<int>(0, 15);
  test_types_CombiningAdapter_1D<long>(-8, 16);
}
