//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for IndexSet class.
///

#include "RAJA_test-base.hpp"

#include "camp/resource.hpp"

//
// Resource object used to construct list segment objects with indices
// living in host (CPU) memory. Used in all tests.
//
  camp::resources::Resource host_res{camp::resources::Host::get_default()};


TEST(IndexSetUnitTest, Empty)
{
  RAJA::TypedIndexSet<> is;
  ASSERT_EQ(0, is.size());
  ASSERT_EQ(is.begin(), is.end());

  RAJA::TypedIndexSet<> is2;
  ASSERT_EQ(is2.size(), is.size());
  is.swap(is2);
  ASSERT_EQ(is2.size(), is.size());
}

TEST(IndexSetUnitTest, ConstructAndCompareSegments)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType isr;
  ASSERT_EQ((size_t)1, isr.getNumTypes());
  isr.push_back(RangeSegType(1, 3));
  isr.push_front(RangeSegType(0, 1));
  ASSERT_EQ(2, isr.size()); 
  ASSERT_EQ(size_t(3), isr.getLength());
  const RangeSegType& rs0 = isr.getSegment<const RangeSegType>(0);
  const RangeSegType& rs1 = isr.getSegment<const RangeSegType>(1);
  ASSERT_EQ(1, rs0.size());
  ASSERT_EQ(2, rs1.size());
  ASSERT_TRUE(isr.compareSegmentById(0, isr));
  ASSERT_TRUE(isr.compareSegmentById(1, isr));

  RIndexSetType isr2;
  isr2.push_back(RangeSegType(0, 3));
  ASSERT_TRUE(isr != isr2);
  ASSERT_FALSE(isr == isr2);
  ASSERT_NE(isr.size(), isr2.size());
  ASSERT_EQ(isr.getLength(), isr2.getLength());

  using ListSegType = RAJA::TypedListSegment<int>; 
  using RLIndexSetType = RAJA::TypedIndexSet<RangeSegType, ListSegType>;
  RLIndexSetType isrl;
  ASSERT_EQ(size_t(2), isrl.getNumTypes());
  int idx[ ] = {0, 2, 4, 5};
  ListSegType lseg(idx, 4, host_res); 
  isrl.push_back(lseg);
  isrl.push_back(RangeSegType(6, 8));
  ASSERT_EQ(2, isrl.size()); 
  ASSERT_EQ(size_t(6), isrl.getLength());
  const ListSegType ls0 = isrl.getSegment<const ListSegType>(0);
  const RangeSegType rs11 = isrl.getSegment<const RangeSegType>(1);
  ASSERT_EQ(4, ls0.size());
  ASSERT_EQ(2, rs11.size());

  ASSERT_FALSE(isrl.compareSegmentById(0, isr));
  ASSERT_FALSE(isr.compareSegmentById(1, isrl));

  RIndexSetType isr3(isr);
  RLIndexSetType isrl3 = isrl;
  ASSERT_TRUE(isr == isr3);
  ASSERT_FALSE(isrl != isrl3);
  ASSERT_FALSE(isr3 == isrl3);
  ASSERT_TRUE(isr3 != isrl3);
}

TEST(IndexSetUnitTest, Swap)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType iset1;
  RangeSegType range(0, 10);
  iset1.push_back(range);
  iset1.push_back_nocopy(&range);
  iset1.push_front(range);
  iset1.push_front_nocopy(&range);
  RIndexSetType iset2;

  ASSERT_EQ(4, iset1.size());
  ASSERT_EQ(size_t(40), iset1.getLength());
  ASSERT_EQ(0, iset2.size());
  ASSERT_EQ(size_t(0), iset2.getLength());

  iset1.swap(iset2);

  ASSERT_EQ(4, iset2.size());
  ASSERT_EQ(size_t(40), iset2.getLength());
  ASSERT_EQ(0, iset1.size());
  ASSERT_EQ(size_t(0), iset1.getLength());
}

TEST(IndexSetUnitTest, Slice)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType iset1;
  RangeSegType range1(0, 2);
  RangeSegType range2(2, 4);
  RangeSegType range3(4, 6);
  RangeSegType range4(6, 8);
  RangeSegType range5(8, 10);
  iset1.push_back(range1);
  iset1.push_back(range2);
  iset1.push_back(range3);
  iset1.push_back(range4);
  iset1.push_back(range5);
  ASSERT_EQ(5, iset1.size());
  ASSERT_EQ(size_t(10), iset1.getLength());

  RIndexSetType iset2 = iset1.createSlice(2, 5);
  ASSERT_EQ(3, iset2.size());
  ASSERT_EQ(size_t(6), iset2.getLength());
  const RangeSegType rs20 = iset2.getSegment<const RangeSegType>(0);
  ASSERT_EQ(4, *rs20.begin());
  ASSERT_EQ(6, *rs20.end());
  const RangeSegType rs21 = iset2.getSegment<const RangeSegType>(1);
  ASSERT_EQ(6, *rs21.begin());
  ASSERT_EQ(8, *rs21.end());
  const RangeSegType rs22 = iset2.getSegment<const RangeSegType>(2);
  ASSERT_EQ(8, *rs22.begin());
  ASSERT_EQ(10, *rs22.end());

  int segs[ ] = {0, 3};
  RIndexSetType iset3 = iset1.createSlice(segs, 2);
  ASSERT_EQ(2, iset3.size());
  ASSERT_EQ(size_t(4), iset3.getLength());
  const RangeSegType rs30 = iset3.getSegment<const RangeSegType>(0);
  ASSERT_EQ(0, *rs30.begin());
  ASSERT_EQ(2, *rs30.end());
  const RangeSegType rs31 = iset3.getSegment<const RangeSegType>(1);
  ASSERT_EQ(6, *rs31.begin());
  ASSERT_EQ(8, *rs31.end());

  std::vector<int> segvec;
  segvec.push_back(3);
  segvec.push_back(2);
  RIndexSetType iset4 = iset1.createSlice(segvec);
  ASSERT_EQ(2, iset4.size());
  ASSERT_EQ(size_t(4), iset4.getLength());
  const RangeSegType rs40 = iset4.getSegment<const RangeSegType>(0);
  ASSERT_EQ(6, *rs40.begin());
  ASSERT_EQ(8, *rs40.end());
  const RangeSegType rs41 = iset4.getSegment<const RangeSegType>(1);
  ASSERT_EQ(4, *rs41.begin());
  ASSERT_EQ(6, *rs41.end());
}

TEST(IndexSetUnitTest, ConditionalEvenIndices)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using ListSegType = RAJA::TypedListSegment<int>; 
  using RLIndexSetType = RAJA::TypedIndexSet<RangeSegType, ListSegType>;
  RLIndexSetType iset;

  iset.push_back(RangeSegType(0, 6));
  int idx[ ] = {7, 8, 10, 11};
  ListSegType lseg(idx, 4, host_res); 
  iset.push_back(lseg);
  iset.push_back(RangeSegType(13, 17));

  RAJA::RAJAVec<int> ref_even_indices;
  ref_even_indices.push_back(0); 
  ref_even_indices.push_back(2);
  ref_even_indices.push_back(4);
  ref_even_indices.push_back(8);
  ref_even_indices.push_back(10);
  ref_even_indices.push_back(14);
  ref_even_indices.push_back(16);

  RAJA::RAJAVec<int> even_indices;
  getIndicesConditional(even_indices, iset, [] (int i) {
    return !(i % 2);
  });

  EXPECT_EQ(even_indices.size(), ref_even_indices.size());
  for (size_t i = 0; i < ref_even_indices.size(); ++i) {
    EXPECT_EQ(even_indices[i], ref_even_indices[i]);
  }
}

TEST(IndexSetUnitTest, ConditionalLessThan100Indices)
{
  using RangeSegType = RAJA::TypedRangeSegment<int>;
  using RIndexSetType = RAJA::TypedIndexSet<RangeSegType>;
  RIndexSetType iset;

  iset.push_back(RangeSegType(92, 97));
  iset.push_back(RangeSegType(98, 103));

  RAJA::RAJAVec<int> ref_lt100_indices;
  ref_lt100_indices.push_back(92);
  ref_lt100_indices.push_back(93);
  ref_lt100_indices.push_back(94);
  ref_lt100_indices.push_back(95);
  ref_lt100_indices.push_back(96);
  ref_lt100_indices.push_back(98);
  ref_lt100_indices.push_back(99);

  RAJA::RAJAVec<int> lt100_indices;
  getIndicesConditional(lt100_indices, iset, [] (int idx) {
    return (idx < 100);
  });

  EXPECT_EQ(lt100_indices.size(), ref_lt100_indices.size());
  for (size_t i = 0; i < ref_lt100_indices.size(); ++i) {
    EXPECT_EQ(lt100_indices[i], ref_lt100_indices[i]);
  }
}
